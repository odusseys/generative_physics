import gc
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from huggingface_hub import try_to_load_from_cache
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .airfoil import (
    AIRFOIL_COLOR_MODE,
    AIRFOIL_FLOW_SPEED,
    AIRFOIL_MAX_POINTS,
    AIRFOIL_MIN_POINTS,
    AIRFOIL_X_STRETCH,
    AIRFOIL_Y_STRETCH,
)
from .config import TrainingConfig
from .data import (
    StreamingPdePairDataset,
    cpu_sim_worker_count,
    make_pde_records_with_workers,
    simulation_grid_size,
    simulations_run_on_cpu,
)
from .eikonal import (
    EIKONAL_CMAP_NAME,
    EIKONAL_MAX_COMPONENTS,
    EIKONAL_MIN_COMPONENTS,
    EIKONAL_REFRACTIVE_INDEX_CONTRAST,
    EIKONAL_REFRACTIVE_INDEX_MAX,
    EIKONAL_REFRACTIVE_INDEX_MEAN,
    EIKONAL_REFRACTIVE_INDEX_MIN,
    EIKONAL_SIGMA_MAX,
    EIKONAL_SIGMA_MIN,
    EIKONAL_SOLUTION_CMAP_NAME,
    EIKONAL_SOLUTION_GAMMA,
    EIKONAL_SOLUTION_MIDPOINT_CONTRAST,
    EIKONAL_SOLUTION_VMAX,
    EIKONAL_WEIGHT_SIGMA,
)
from .elasticity import (
    ELASTICITY_FAR_FIELD_STRESS,
    ELASTICITY_HANDLE_SCALE,
    ELASTICITY_HOLE_BOX_FRACTION,
    ELASTICITY_HORIZONTAL_STRESS_MAX,
    ELASTICITY_HORIZONTAL_STRESS_MIN,
    ELASTICITY_LAMBDA,
    ELASTICITY_LAMBDA_MAX,
    ELASTICITY_LAMBDA_MIN,
    ELASTICITY_MAX_POINTS,
    ELASTICITY_MIN_POINTS,
    ELASTICITY_MU,
    ELASTICITY_MU_MAX,
    ELASTICITY_MU_MIN,
    ELASTICITY_PLANE_STRESS,
    ELASTICITY_SAMPLES_PER_SEGMENT,
    ELASTICITY_STRESS_PERCENTILE,
    ELASTICITY_VERTICAL_STRESS_MAX,
    ELASTICITY_VERTICAL_STRESS_MIN,
)
from .elliptic import (
    ELLIPTIC_A_MAX,
    ELLIPTIC_A_MIN,
    ELLIPTIC_MAX_CYCLES,
    ELLIPTIC_REACTION_MAX,
    ELLIPTIC_REACTION_MIN,
)
from .flux_lora import (
    compile_flux2_regions,
    flux2_klein_lora_targets,
    pde_lora_loss,
    trainable_parameter_count,
    unwrap_flux2_regions,
)
from .fracture import (
    FRACTURE_BIAXIAL_STRAIN_MAX,
    FRACTURE_BIAXIAL_STRAIN_MIN,
    FRACTURE_GC_MAX,
    FRACTURE_GC_MIN,
    FRACTURE_HIGH_STRAIN_MAX,
    FRACTURE_HIGH_STRAIN_MIN,
    FRACTURE_LOW_STRAIN_MAX,
    FRACTURE_LOW_STRAIN_MIN,
    FRACTURE_MAX_DEFECTS,
    FRACTURE_MIN_DEFECTS,
    FRACTURE_NU_MAX,
    FRACTURE_NU_MIN,
)
from .heat import HEAT_FORCING_NUM_MODES, HEAT_FORCING_SCALE
from .ks import (
    KS_BURN_T,
    KS_CMAP_NAME,
    KS_DOMAIN_LENGTH,
    KS_HORIZON_T,
    KS_INITIAL_DECAY,
    KS_INITIAL_NUM_MODES,
    KS_STEPS_PER_FRAME,
    KS_VALUE_BOUNDS,
    ks_effective_discretization,
)
from .navier_stokes import (
    NAVIER_STOKES_CFL,
    NAVIER_STOKES_DENSITY_MAX,
    NAVIER_STOKES_DENSITY_MIN,
    NAVIER_STOKES_DOMAIN_SIZE,
    NAVIER_STOKES_INITIAL_SPEED_MAX,
    NAVIER_STOKES_INITIAL_SPEED_MIN,
    NAVIER_STOKES_MAX_COMPONENTS,
    NAVIER_STOKES_MIN_COMPONENTS,
    NAVIER_STOKES_MULTIPLE_TIMES,
    NAVIER_STOKES_SIGMA_MAX,
    NAVIER_STOKES_SIGMA_MIN,
    NAVIER_STOKES_VISCOSITY_MAX,
    NAVIER_STOKES_VISCOSITY_MIN,
)
from .ot import (
    OT_COST_PATH_SAMPLES,
    OT_COST_SIGMA_MULTIPLIER,
    OT_COST_STRENGTH,
    OT_EPSILON,
    OT_LOGPROB_VMAX,
    OT_LOGPROB_VMIN,
    OT_MAX_COMPONENTS,
    OT_MIN_COMPONENTS,
    OT_POTENTIAL_CMAP_NAME,
    OT_POTENTIAL_VMAX,
    OT_POTENTIAL_VMIN,
    OT_SIGMA_MAX,
    OT_SIGMA_MIN,
    OT_SINKHORN_ITERS,
)
from .poisson import POISSON_NUM_GAUSSIAN_MODES, POISSON_SOURCE_SCALE
from .thermal_modulation import (
    attach_condition_latent_adaln_zero,
    attach_condition_latent_cross_attention,
    attach_scalar_parameter_modulation,
    attach_thermal_coefficient_modulation,
)
from .visualization import (
    show_ks_vae_roundtrip_physical_error,
    show_ks_timewise_error,
    show_navier_stokes_multiple_inference_grid,
    show_random_inference_grid,
    show_smoothed_loss,
)


def _count_parameters(parameters):
    return sum(p.numel() for p in parameters)


def _fixed_or_range(fixed, min_value, max_value):
    if fixed is not None:
        return f"{fixed:g}"
    return f"{min_value:g}..{max_value:g}"


def _cosine_decay_lr(initial_lr, step, max_steps):
    if max_steps <= 0:
        return float(initial_lr)
    progress = min(max(float(step) / float(max_steps), 0.0), 1.0)
    return float(initial_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _stage(message):
    print(message, flush=True)


def resolve_pretrained_model_source(model_id):
    """Prefer an existing model snapshot without requiring Hub authentication."""
    local_path = Path(model_id).expanduser()
    if local_path.exists():
        return local_path.resolve()

    cached_model_index = try_to_load_from_cache(model_id, "model_index.json")
    if isinstance(cached_model_index, str):
        return Path(cached_model_index).parent.resolve()
    return model_id


def resolve_vae_lora_checkpoint(config, vae_lora_checkpoint=None):
    if vae_lora_checkpoint is not None:
        return Path(vae_lora_checkpoint).expanduser().resolve()
    if config.pde_kind != "ks":
        return None

    checkpoint_dir = Path(config.ks_vae_lora_dir).expanduser()
    candidates = [
        checkpoint
        for checkpoint in checkpoint_dir.glob("*.safetensors")
        if checkpoint.is_file()
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda checkpoint: (checkpoint.stat().st_mtime_ns, checkpoint.name),
    ).resolve()


def _show_validation_debug(
    pipe,
    records,
    prompt_embeds,
    text_ids,
    device,
    config,
    seed,
    negative_prompt_embeds=None,
    negative_text_ids=None,
):
    if config.pde_kind == "navier_stokes_multiple":
        show_navier_stokes_multiple_inference_grid(
            pipe,
            records,
            prompt_embeds,
            text_ids,
            device=device,
            train_image_size=config.train_image_size,
            num_inference_steps=config.inference_num_steps,
            n=config.navier_stokes_multiple_debug_samples,
            seed=seed,
        )
        return

    show_random_inference_grid(
        pipe,
        records,
        prompt_embeds,
        text_ids,
        device=device,
        pde_name=config.pde_name,
        train_image_size=config.train_image_size,
        num_inference_steps=config.inference_num_steps,
        guidance_scale=config.inference_guidance_scale,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_text_ids=negative_text_ids,
        n=config.validation_num_images,
        seed=seed,
    )
    if config.pde_kind == "ks" and config.ks_debug_num_samples:
        show_ks_timewise_error(
            pipe,
            records,
            prompt_embeds,
            text_ids,
            device=device,
            train_image_size=config.train_image_size,
            num_inference_steps=config.inference_num_steps,
            guidance_scale=config.inference_guidance_scale,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            n=config.ks_debug_num_samples,
            seed=seed,
        )


def print_parameter_report(pipe):
    transformer_params = list(pipe.transformer.named_parameters())
    lora_trainable = _count_parameters(p for name, p in transformer_params if p.requires_grad and "lora_" in name)
    scalar_modulation_trainable = _count_parameters(
        p for name, p in transformer_params if p.requires_grad and name.startswith("thermal_modulation.")
    )
    condition_modulation_trainable = _count_parameters(
        p
        for name, p in transformer_params
        if p.requires_grad and name.startswith("condition_latent_adaln_zero.")
    )
    condition_cross_attention_trainable = _count_parameters(
        p
        for name, p in transformer_params
        if p.requires_grad and name.startswith("condition_latent_cross_attention.")
    )
    other_transformer_trainable = _count_parameters(
        p
        for name, p in transformer_params
        if p.requires_grad
        and "lora_" not in name
        and not name.startswith("thermal_modulation.")
        and not name.startswith("condition_latent_adaln_zero.")
        and not name.startswith("condition_latent_cross_attention.")
    )
    frozen_transformer = _count_parameters(p for _, p in transformer_params if not p.requires_grad)
    frozen_vae = _count_parameters(p for p in pipe.vae.parameters() if not p.requires_grad)
    frozen_text_encoder = _count_parameters(p for p in pipe.text_encoder.parameters() if not p.requires_grad)

    trainable_total = (
        lora_trainable
        + scalar_modulation_trainable
        + condition_modulation_trainable
        + condition_cross_attention_trainable
        + other_transformer_trainable
    )
    frozen_total = frozen_transformer + frozen_vae + frozen_text_encoder
    print("parameter report:")
    print(f"  trainable: {trainable_total:,}")
    print(f"    LoRA adapters: {lora_trainable:,}")
    print(f"    scalar AdaLN modulators: {scalar_modulation_trainable:,}")
    print(f"    condition-latent AdaLN-Zero adapters: {condition_modulation_trainable:,}")
    print(f"    condition-latent cross-attention adapters: {condition_cross_attention_trainable:,}")
    print(f"    other transformer trainable: {other_transformer_trainable:,}")
    print(f"  frozen: {frozen_total:,}")
    print(f"    transformer base: {frozen_transformer:,}")
    print(f"    VAE: {frozen_vae:,}")
    print(f"    text encoder: {frozen_text_encoder:,}")


def run_training(config=None, vae_lora_checkpoint=None):
    config = config or TrainingConfig()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={weight_dtype}")
    print(f"model={config.model_id}")
    print(
        f"inference: steps={config.inference_num_steps}, "
        f"guidance_scale={config.inference_guidance_scale:g}; "
        f"training_sigmas={config.training_sigma_mode}"
    )
    if config.inference_guidance_scale > 1.0 and not config.prompt:
        print("inference CFG is neutral because both prompts are empty; using one model pass per step")
    data_grid_size = simulation_grid_size(config)
    cpu_generated_samples = simulations_run_on_cpu(config, device)
    sample_sim_device = torch.device("cpu") if cpu_generated_samples else device
    sample_num_workers = cpu_sim_worker_count(config) if cpu_generated_samples else 0

    print(
        f"Generating {config.pde_name} data: sim_nx={data_grid_size}, save_steps={config.sim_save_steps}, "
        f"image={config.output_image_size}x{config.output_image_size}, initial_grid={config.initial_grid_size}, "
        f"sim_batch_size={config.sim_batch_size}"
    )
    if config.pde_kind == "heat":
        print(
            "Heat forcing: "
            f"up to {HEAT_FORCING_NUM_MODES} modes, scale={HEAT_FORCING_SCALE:g}"
        )
    elif config.pde_kind == "poisson":
        print(
            "Poisson source: "
            f"{POISSON_NUM_GAUSSIAN_MODES} Gaussian modes, scale={POISSON_SOURCE_SCALE:g}"
        )
    elif config.pde_kind == "ks":
        ks_nx, ks_time_frames, ks_dt = ks_effective_discretization(
            Nx=config.ks_grid_size,
            image_size=config.output_image_size,
        )
        print(
            "Kuramoto-Sivashinsky: "
            f"L={KS_DOMAIN_LENGTH:g}, grid={ks_nx}, frames={ks_time_frames}, dt={ks_dt:g}, "
            f"burn_T={KS_BURN_T:g}, horizon_T={KS_HORIZON_T:g}, "
            f"steps_per_frame={KS_STEPS_PER_FRAME}, init_modes={KS_INITIAL_NUM_MODES}, "
            f"init_decay={KS_INITIAL_DECAY:g}, condition={config.ks_condition_encoding}, "
            f"value_bounds={KS_VALUE_BOUNDS}, cmap={KS_CMAP_NAME}"
        )
    elif config.pde_kind in {"navier_stokes", "navier_stokes_multiple"}:
        navier_final_time = (
            NAVIER_STOKES_MULTIPLE_TIMES[-1]
            if config.pde_kind == "navier_stokes_multiple"
            else config.navier_stokes_final_time
        )
        print(
            "2D unforced Navier-Stokes: "
            f"periodic square L={NAVIER_STOKES_DOMAIN_SIZE:g}, grid={config.navier_stokes_grid_size}, "
            f"T={navier_final_time:g}, CFL={NAVIER_STOKES_CFL:g}, "
            f"mixture_components={NAVIER_STOKES_MIN_COMPONENTS}..{NAVIER_STOKES_MAX_COMPONENTS}, "
            f"sigma={NAVIER_STOKES_SIGMA_MIN:g}..{NAVIER_STOKES_SIGMA_MAX:g}, "
            f"initial_speed={NAVIER_STOKES_INITIAL_SPEED_MIN:g}..{NAVIER_STOKES_INITIAL_SPEED_MAX:g}, "
            f"density={NAVIER_STOKES_DENSITY_MIN:g}..{NAVIER_STOKES_DENSITY_MAX:g}, "
            f"kinematic_viscosity={NAVIER_STOKES_VISCOSITY_MIN:g}..{NAVIER_STOKES_VISCOSITY_MAX:g}"
        )
        if config.pde_kind == "navier_stokes_multiple":
            print(
                "joint target sequence: "
                f"times={NAVIER_STOKES_MULTIPLE_TIMES}, four noisy latent grids + one shared condition grid"
            )
    elif config.pde_kind == "airfoil":
        print(
            "Airfoil flow: "
            f"points={AIRFOIL_MIN_POINTS}..{AIRFOIL_MAX_POINTS}, "
            f"stretch=({AIRFOIL_X_STRETCH:g}, {AIRFOIL_Y_STRETCH:g}), "
            f"U={AIRFOIL_FLOW_SPEED:g}, color={AIRFOIL_COLOR_MODE}"
        )
    elif config.pde_kind == "elliptic":
        print(
            "Elliptic coefficients: "
            f"fields=a20,a11,a02,a10,a01,a00,f, max_cycles={ELLIPTIC_MAX_CYCLES:g}, "
            f"a={ELLIPTIC_A_MIN:g}..{ELLIPTIC_A_MAX:g}, "
            f"reaction={ELLIPTIC_REACTION_MIN:g}..{ELLIPTIC_REACTION_MAX:g}, "
            f"eval_cache={config.elliptic_cache_dir}"
        )
    elif config.pde_kind == "elasticity":
        sigma_x_label = _fixed_or_range(
            ELASTICITY_FAR_FIELD_STRESS,
            ELASTICITY_HORIZONTAL_STRESS_MIN,
            ELASTICITY_HORIZONTAL_STRESS_MAX,
        )
        sigma_y_label = "0" if ELASTICITY_FAR_FIELD_STRESS is not None else _fixed_or_range(
            None,
            ELASTICITY_VERTICAL_STRESS_MIN,
            ELASTICITY_VERTICAL_STRESS_MAX,
        )
        print(
            "Elasticity holes: "
            f"points={ELASTICITY_MIN_POINTS}..{ELASTICITY_MAX_POINTS}, "
            f"samples_per_segment={ELASTICITY_SAMPLES_PER_SEGMENT}, handle_scale={ELASTICITY_HANDLE_SCALE:g}, "
            f"hole_box={ELASTICITY_HOLE_BOX_FRACTION:g}*grid, "
            f"lambda={_fixed_or_range(ELASTICITY_LAMBDA, ELASTICITY_LAMBDA_MIN, ELASTICITY_LAMBDA_MAX)}, "
            f"mu={_fixed_or_range(ELASTICITY_MU, ELASTICITY_MU_MIN, ELASTICITY_MU_MAX)}, "
            f"sigma_x={sigma_x_label}, "
            f"sigma_y={sigma_y_label}, "
            f"plane_stress={ELASTICITY_PLANE_STRESS}, stress_percentile={ELASTICITY_STRESS_PERCENTILE:g}"
        )
    elif config.pde_kind == "eikonal":
        print(
            "Eikonal refractive index: "
            f"components={EIKONAL_MIN_COMPONENTS}..{EIKONAL_MAX_COMPONENTS}, "
            f"sigma={EIKONAL_SIGMA_MIN:g}..{EIKONAL_SIGMA_MAX:g}, "
            f"weight_sigma={EIKONAL_WEIGHT_SIGMA:g}, "
            f"mean={EIKONAL_REFRACTIVE_INDEX_MEAN:g}, "
            f"contrast={EIKONAL_REFRACTIVE_INDEX_CONTRAST:g}, "
            f"n={EIKONAL_REFRACTIVE_INDEX_MIN:g}..{EIKONAL_REFRACTIVE_INDEX_MAX:g}, "
            f"time_vmax={EIKONAL_SOLUTION_VMAX:g}, "
            f"time_gamma={EIKONAL_SOLUTION_GAMMA:g}, "
            f"time_midpoint={EIKONAL_SOLUTION_VMAX * (0.5 ** (1.0 / EIKONAL_SOLUTION_GAMMA)):g}, "
            f"time_mid_contrast={EIKONAL_SOLUTION_MIDPOINT_CONTRAST:g}, "
            f"cmap={EIKONAL_CMAP_NAME}, time_cmap={EIKONAL_SOLUTION_CMAP_NAME}, source=center"
        )
    elif config.pde_kind == "ot":
        print(
            "Optimal transport Sinkhorn: "
            f"solve_grid={config.ot_solve_grid_size}, "
            f"components={OT_MIN_COMPONENTS}..{OT_MAX_COMPONENTS}, "
            f"sigma={OT_SIGMA_MIN:g}..{OT_SIGMA_MAX:g}, "
            f"eps={OT_EPSILON:g}, iters={OT_SINKHORN_ITERS}, "
            f"cost_strength={OT_COST_STRENGTH:g}, "
            f"cost_sigma_multiplier={OT_COST_SIGMA_MULTIPLIER:g}, "
            f"cost_path_samples={OT_COST_PATH_SAMPLES}, "
            f"logprob={OT_LOGPROB_VMIN:g}..{OT_LOGPROB_VMAX:g}, "
            f"potential={OT_POTENTIAL_VMIN:g}..{OT_POTENTIAL_VMAX:g}, "
            f"potential_cmap={OT_POTENTIAL_CMAP_NAME}"
        )
    elif config.pde_kind == "fracture":
        print(
            "Phase-field fracture: "
            f"grid={config.fracture_grid_size}, "
            f"nu={FRACTURE_NU_MIN:g}..{FRACTURE_NU_MAX:g}, "
            f"Gc={FRACTURE_GC_MIN:g}..{FRACTURE_GC_MAX:g}, "
            f"eps_low={FRACTURE_LOW_STRAIN_MIN:g}..{FRACTURE_LOW_STRAIN_MAX:g}, "
            f"eps_high={FRACTURE_HIGH_STRAIN_MIN:g}..{FRACTURE_HIGH_STRAIN_MAX:g}, "
            f"eps_biaxial={FRACTURE_BIAXIAL_STRAIN_MIN:g}..{FRACTURE_BIAXIAL_STRAIN_MAX:g}, "
            f"defects={FRACTURE_MIN_DEFECTS}..{FRACTURE_MAX_DEFECTS}, "
            f"steps={config.fracture_steps}, inner_iters={config.fracture_inner_iters}"
        )
    print(
        "training data: streaming fresh simulations on the fly "
        f"from seed {config.train_seed_offset}; no fixed training sample count"
    )
    if cpu_generated_samples:
        print(f"sample generation: CPU with {sample_num_workers} workers")
    else:
        print("sample generation: CUDA in the training process")
    eval_pair_count = config.num_eval_pairs
    if config.pde_kind == "ks":
        eval_pair_count = max(eval_pair_count, config.ks_debug_num_samples)
    print("generating fixed validation records...")
    eval_records = make_pde_records_with_workers(
        eval_pair_count,
        seed_offset=config.eval_seed_offset,
        sim_device=sample_sim_device,
        config=config,
        num_workers=sample_num_workers,
    )
    print(f"generated {len(eval_records)} fixed eval records")

    eval_seeds = {record["params"]["seed"] for record in eval_records}
    print(
        f"validation seeds fixed at {min(eval_seeds)}..{max(eval_seeds)}; "
        "streaming training skips that validation range"
    )
    train_dataset = StreamingPdePairDataset(
        config=config,
        image_size=config.train_image_size,
        seed_offset=config.train_seed_offset,
        sim_device=sample_sim_device,
        skip_seed_ranges=[(config.eval_seed_offset, config.eval_seed_offset + eval_pair_count)],
    )
    train_loader_kwargs = {}
    if sample_num_workers > 0:
        train_loader_kwargs.update(
            num_workers=sample_num_workers,
            persistent_workers=True,
            prefetch_factor=2,
        )
        if device.type == "cuda":
            train_loader_kwargs["pin_memory"] = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        **train_loader_kwargs,
    )

    model_source = resolve_pretrained_model_source(config.model_id)
    if str(model_source) != config.model_id:
        _stage(f"loading Flux pipeline from local snapshot {model_source}")
    else:
        _stage("loading Flux pipeline")
    pipe = Flux2KleinPipeline.from_pretrained(
        model_source,
        torch_dtype=weight_dtype,
        token=False,
    )
    vae_lora_checkpoint = resolve_vae_lora_checkpoint(config, vae_lora_checkpoint)
    if vae_lora_checkpoint is not None:
        if not vae_lora_checkpoint.is_file():
            raise FileNotFoundError(f"VAE LoRA checkpoint not found: {vae_lora_checkpoint}")
        _stage(f"loading VAE LoRA from {vae_lora_checkpoint}")
        pipe.vae.load_lora_adapter(
            vae_lora_checkpoint.parent,
            weight_name=vae_lora_checkpoint.name,
            prefix=None,
            adapter_name="vae_finetune",
            low_cpu_mem_usage=True,
        )
    elif config.pde_kind == "ks":
        _stage(
            "no KS VAE LoRA checkpoint found in "
            f"{Path(config.ks_vae_lora_dir).expanduser().resolve()}; using the base VAE"
        )
    _stage("pipeline loaded; moving pipeline to training device")
    pipe.to(device)
    _stage("pipeline on device; freezing base modules")
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    _stage("encoding prompt")
    with torch.no_grad():
        prompt_embeds, text_ids = pipe.encode_prompt(
            config.prompt,
            device=device,
            max_sequence_length=config.max_sequence_length,
            text_encoder_out_layers=config.text_encoder_out_layers,
        )
        if config.inference_guidance_scale > 1.0:
            negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
                "",
                device=device,
                max_sequence_length=config.max_sequence_length,
                text_encoder_out_layers=config.text_encoder_out_layers,
            )
        else:
            negative_prompt_embeds = None
            negative_text_ids = None
    prompt_embeds = prompt_embeds.detach().to(device=device, dtype=weight_dtype)
    text_ids = text_ids.detach().to(device=device)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.detach().to(
            device=device,
            dtype=weight_dtype,
        )
        negative_text_ids = negative_text_ids.detach().to(device=device)

    _stage("prompt encoded; moving text encoder to CPU")
    pipe.text_encoder.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if config.pde_kind == "ks" and config.ks_debug_num_samples > 0:
        _stage("evaluating base and fine-tuned VAE round-trip physical error")
        show_ks_vae_roundtrip_physical_error(
            pipe,
            eval_records,
            device=device,
            n=config.ks_debug_num_samples,
            seed=config.seed,
        )

    _stage("attaching LoRA adapters")
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=flux2_klein_lora_targets(pipe.transformer),
    )
    pipe.transformer.add_adapter(lora_config)

    for p in pipe.transformer.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    if config.pde_kind == "heat":
        thermal_modulation = attach_thermal_coefficient_modulation(
            pipe.transformer,
            bottleneck_dim=config.thermal_modulation_bottleneck_dim,
            log_alpha_mean=config.heat_log_diffusivity_mean,
            log_alpha_std=config.heat_log_diffusivity_std,
        )
        thermal_modulation.to(device=device)
        print(
            "thermal AdaLN conditioning: "
            f"(log(alpha) - {config.heat_log_diffusivity_mean:.6g}) / {config.heat_log_diffusivity_std:.6g}"
        )
    elif config.pde_kind == "elasticity":
        scalar_modulation = attach_scalar_parameter_modulation(
            pipe.transformer,
            bottleneck_dim=config.thermal_modulation_bottleneck_dim,
            parameter_names=config.elasticity_conditioning_names,
            parameter_transforms=config.elasticity_conditioning_transforms,
            normalization_mean=config.elasticity_conditioning_mean,
            normalization_std=config.elasticity_conditioning_std,
        )
        scalar_modulation.to(device=device)
        print(
            "elasticity AdaLN conditioning: "
            + ", ".join(
                f"{name}:{transform}"
                for name, transform in zip(
                    config.elasticity_conditioning_names,
                    config.elasticity_conditioning_transforms,
                )
            )
        )
    elif config.pde_kind == "fracture":
        scalar_modulation = attach_scalar_parameter_modulation(
            pipe.transformer,
            bottleneck_dim=config.thermal_modulation_bottleneck_dim,
            parameter_names=config.fracture_conditioning_names,
            parameter_transforms=config.fracture_conditioning_transforms,
            normalization_mean=config.fracture_conditioning_mean,
            normalization_std=config.fracture_conditioning_std,
        )
        scalar_modulation.to(device=device)
        print(
            "fracture AdaLN conditioning: "
            + ", ".join(
                f"{name}:{transform}"
                for name, transform in zip(
                    config.fracture_conditioning_names,
                    config.fracture_conditioning_transforms,
                )
            )
        )
    elif config.pde_kind in {"navier_stokes", "navier_stokes_multiple"}:
        scalar_modulation = attach_scalar_parameter_modulation(
            pipe.transformer,
            bottleneck_dim=config.thermal_modulation_bottleneck_dim,
            parameter_names=config.navier_stokes_conditioning_names,
            parameter_transforms=config.navier_stokes_conditioning_transforms,
            normalization_mean=config.navier_stokes_conditioning_mean,
            normalization_std=config.navier_stokes_conditioning_std,
        )
        scalar_modulation.to(device=device)
        print(
            "Navier-Stokes AdaLN conditioning: "
            + ", ".join(
                f"{name}:{transform}"
                for name, transform in zip(
                    config.navier_stokes_conditioning_names,
                    config.navier_stokes_conditioning_transforms,
                )
            )
        )
    else:
        print("scalar parameter conditioning: disabled")

    if config.pde_kind == "ks":
        if config.ks_condition_adapter_mode == "adaln_zero":
            condition_modulation = attach_condition_latent_adaln_zero(pipe.transformer)
            condition_modulation.to(device=device)
            print(
                "KS condition adapter: unpooled per-token Linear AdaLN-Zero "
                "on target image tokens"
            )
        elif config.ks_condition_adapter_mode == "cross_attention":
            condition_cross_attention = attach_condition_latent_cross_attention(pipe.transformer)
            condition_cross_attention.to(device=device)
            print(
                "KS condition adapter: unpooled block-end 512d/8-head cross-attention "
                "residual on all image tokens"
            )
        else:
            print("KS condition adapter: disabled")

    if config.transformer_gradient_checkpointing and hasattr(
        pipe.transformer,
        "enable_gradient_checkpointing",
    ):
        _stage("enabling gradient checkpointing")
        pipe.transformer.enable_gradient_checkpointing()
    else:
        _stage("transformer gradient checkpointing disabled")

    trainable, total = trainable_parameter_count(pipe.transformer)
    print(f"trainable transformer params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    print_parameter_report(pipe)

    optimizer = torch.optim.AdamW(
        [p for p in pipe.transformer.parameters() if p.requires_grad],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    if config.run_initial_validation and config.validation_num_images:
        print("initial validation inference before training")
        pipe.transformer.eval()
        _show_validation_debug(
            pipe,
            eval_records,
            prompt_embeds,
            text_ids,
            device,
            config,
            config.seed,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
        )

    if config.transformer_compile_regions:
        compile_stats = compile_flux2_regions(
            pipe.transformer,
            mode=config.transformer_compile_mode,
        )
        _stage(
            "regional compilation configured: "
            f"{compile_stats['dit_blocks']} DiT blocks, "
            f"{compile_stats['adaln_zero']} shared AdaLN-Zero modules, "
            f"{compile_stats['scalar_adaln_zero']} scalar AdaLN-Zero modules, "
            f"{compile_stats['condition_adaln_zero']} condition-latent AdaLN-Zero modules, "
            f"{compile_stats['condition_cross_attention']} condition cross-attention modules; "
            f"mode={config.transformer_compile_mode}; first training use will compile"
        )

    pipe.transformer.train()
    loader_iter = iter(train_loader)
    printed_schedule = False
    loss_history = []
    ema_loss = None
    ema_loss_decay = 0.99

    print("starting LoRA training; tqdm shows loss and ema")
    progress = tqdm(
        range(0, config.max_train_steps + 1),
        desc="LoRA steps",
        file=sys.stdout,
        miniters=1,
        mininterval=0.1,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    for global_step in progress:
        current_lr = _cosine_decay_lr(config.learning_rate, global_step, config.max_train_steps)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(config.grad_accum_steps):
            batch = next(loader_iter)
            loss = pde_lora_loss(
                pipe,
                batch,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                device=device,
                num_inference_steps=config.inference_num_steps,
                training_sigma_mode=config.training_sigma_mode,
                print_schedule=not printed_schedule,
            )
            printed_schedule = True
            (loss / config.grad_accum_steps).backward()
            accumulated_loss += loss.detach().item() / config.grad_accum_steps

        torch.nn.utils.clip_grad_norm_([p for p in pipe.transformer.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        loss_history.append(accumulated_loss)
        if ema_loss is None:
            ema_loss = accumulated_loss
        else:
            ema_loss = ema_loss_decay * ema_loss + (1.0 - ema_loss_decay) * accumulated_loss
        progress.set_description(f"LoRA steps loss={accumulated_loss:.4f} ema={ema_loss:.4f}")
        progress.set_postfix_str(
            f"loss={accumulated_loss:.4f} ema={ema_loss:.4f} lr={current_lr:.2e}",
            refresh=True,
        )

        if config.validate_every_n_steps and global_step > 0 and global_step % config.validate_every_n_steps == 0:
            progress.write(f"validation inference at step {global_step}")
            _show_validation_debug(
                pipe,
                eval_records,
                prompt_embeds,
                text_ids,
                device,
                config,
                config.seed + global_step,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
            )
            show_smoothed_loss(loss_history, alpha=config.loss_ema_alpha)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pipe.transformer.eval()
    if config.transformer_compile_regions:
        unwrap_flux2_regions(pipe.transformer)
    transformer_lora_layers = get_peft_model_state_dict(pipe.transformer)
    Flux2KleinPipeline.save_lora_weights(
        save_directory=config.output_dir,
        transformer_lora_layers=transformer_lora_layers,
    )
    if hasattr(pipe.transformer, "thermal_modulation"):
        torch.save(pipe.transformer.thermal_modulation.state_dict(), config.output_dir / "thermal_modulation.pt")
        torch.save(
            {
                "state_dict": pipe.transformer.thermal_modulation.state_dict(),
                "parameter_names": getattr(pipe.transformer.thermal_modulation, "parameter_names", ()),
                "parameter_transforms": getattr(pipe.transformer.thermal_modulation, "parameter_transforms", ()),
            },
            config.output_dir / "scalar_modulation.pt",
        )
    if hasattr(pipe.transformer, "condition_latent_adaln_zero"):
        torch.save(
            pipe.transformer.condition_latent_adaln_zero.state_dict(),
            config.output_dir / "condition_latent_adaln_zero.pt",
        )
    if hasattr(pipe.transformer, "condition_latent_cross_attention"):
        torch.save(
            pipe.transformer.condition_latent_cross_attention.state_dict(),
            config.output_dir / "condition_latent_cross_attention.pt",
        )
    print(f"saved LoRA to {config.output_dir.resolve()}")

    _show_validation_debug(
        pipe,
        eval_records,
        prompt_embeds,
        text_ids,
        device,
        config,
        config.seed,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_text_ids=negative_text_ids,
    )
    show_smoothed_loss(loss_history, alpha=config.loss_ema_alpha)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "pipe": pipe,
        "train_records": None,
        "train_dataset": train_dataset,
        "eval_records": eval_records,
        "loss_history": loss_history,
        "prompt_embeds": prompt_embeds,
        "text_ids": text_ids,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_text_ids": negative_text_ids,
        "device": device,
    }
