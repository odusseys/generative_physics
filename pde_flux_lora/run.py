import gc
import random
from itertools import cycle

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import TrainingConfig
from .data import PdePairDataset, assert_disjoint_record_splits, make_pde_records
from .flux_lora import flux2_klein_lora_targets, pde_lora_loss, trainable_parameter_count
from .thermal_modulation import attach_thermal_coefficient_modulation
from .visualization import show_random_inference_grid, show_smoothed_loss


def _count_parameters(parameters):
    return sum(p.numel() for p in parameters)


def print_parameter_report(pipe):
    transformer_params = list(pipe.transformer.named_parameters())
    lora_trainable = _count_parameters(p for name, p in transformer_params if p.requires_grad and "lora_" in name)
    thermal_trainable = _count_parameters(
        p for name, p in transformer_params if p.requires_grad and name.startswith("thermal_modulation.")
    )
    other_transformer_trainable = _count_parameters(
        p
        for name, p in transformer_params
        if p.requires_grad and "lora_" not in name and not name.startswith("thermal_modulation.")
    )
    frozen_transformer = _count_parameters(p for _, p in transformer_params if not p.requires_grad)
    frozen_vae = _count_parameters(p for p in pipe.vae.parameters() if not p.requires_grad)
    frozen_text_encoder = _count_parameters(p for p in pipe.text_encoder.parameters() if not p.requires_grad)

    trainable_total = lora_trainable + thermal_trainable + other_transformer_trainable
    frozen_total = frozen_transformer + frozen_vae + frozen_text_encoder
    print("parameter report:")
    print(f"  trainable: {trainable_total:,}")
    print(f"    LoRA adapters: {lora_trainable:,}")
    print(f"    thermal AdaLN modulators: {thermal_trainable:,}")
    print(f"    other transformer trainable: {other_transformer_trainable:,}")
    print(f"  frozen: {frozen_total:,}")
    print(f"    transformer base: {frozen_transformer:,}")
    print(f"    VAE: {frozen_vae:,}")
    print(f"    text encoder: {frozen_text_encoder:,}")


def run_training(config=None):
    config = config or TrainingConfig()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={weight_dtype}")
    data_grid_size = config.sim_nx
    if config.pde_kind == "poisson":
        data_grid_size = config.poisson_grid_size
    elif config.pde_kind == "fourier":
        data_grid_size = config.fourier_grid_size

    print("generating training records...")
    print(
        f"Generating {config.pde_name} data: sim_nx={data_grid_size}, save_steps={config.sim_save_steps}, "
        f"image={config.output_image_size}x{config.output_image_size}, initial_grid={config.initial_grid_size}, "
        f"sim_batch_size={config.sim_batch_size}"
    )
    if config.pde_kind == "heat":
        print(
            "Heat forcing: "
            f"up to {config.heat_forcing_num_modes} modes, scale={config.heat_forcing_scale:g}"
        )
    elif config.pde_kind == "poisson":
        print(
            "Poisson source: "
            f"{config.poisson_num_gaussian_modes} Gaussian modes, scale={config.poisson_source_scale:g}"
        )
    elif config.pde_kind == "fourier":
        print(
            "Fourier source: "
            f"{config.fourier_num_modes} random-covariance Gaussian modes, "
            f"sigma={config.fourier_gaussian_sigma_min:g}..{config.fourier_gaussian_sigma_max:g}, "
            f"scale={config.fourier_scale:g}, fft_shift={config.fourier_shift}"
        )
    train_records = make_pde_records(
        config.num_train_pairs,
        seed_offset=config.train_seed_offset,
        sim_device=device,
        config=config,
    )
    eval_records = make_pde_records(
        config.num_eval_pairs,
        seed_offset=config.eval_seed_offset,
        sim_device=device,
        config=config,
    )
    print(f"generated {len(train_records)} train records and {len(eval_records)} eval records")

    train_seeds, eval_seeds = assert_disjoint_record_splits(train_records, eval_records)
    print(
        f"train/eval split OK: {len(train_records)} train seeds "
        f"{min(train_seeds)}..{max(train_seeds)}, {len(eval_records)} val seeds "
        f"{min(eval_seeds)}..{max(eval_seeds)}"
    )
    train_loader = DataLoader(
        PdePairDataset(train_records, image_size=config.train_image_size),
        batch_size=config.train_batch_size,
        shuffle=True,
        drop_last=True,
    )

    pipe = Flux2KleinPipeline.from_pretrained(config.model_id, torch_dtype=weight_dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    with torch.no_grad():
        prompt_embeds, text_ids = pipe.encode_prompt(
            config.prompt,
            device=device,
            max_sequence_length=config.max_sequence_length,
            text_encoder_out_layers=config.text_encoder_out_layers,
        )
    prompt_embeds = prompt_embeds.detach().to(device=device, dtype=weight_dtype)
    text_ids = text_ids.detach().to(device=device)

    pipe.text_encoder.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
    else:
        print("scalar parameter conditioning: disabled")

    if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()

    trainable, total = trainable_parameter_count(pipe.transformer)
    print(f"trainable transformer params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    print_parameter_report(pipe)

    optimizer = torch.optim.AdamW(
        [p for p in pipe.transformer.parameters() if p.requires_grad],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    pipe.transformer.train()
    loader_iter = cycle(train_loader)
    printed_schedule = False
    loss_history = []

    progress = tqdm(range(0, config.max_train_steps + 1), desc="LoRA steps")
    for global_step in progress:
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
                num_inference_steps=config.distilled_num_inference_steps,
                print_schedule=not printed_schedule,
            )
            printed_schedule = True
            (loss / config.grad_accum_steps).backward()
            accumulated_loss += loss.detach().item() / config.grad_accum_steps

        torch.nn.utils.clip_grad_norm_([p for p in pipe.transformer.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        loss_history.append(accumulated_loss)
        progress.set_postfix(loss=f"{accumulated_loss:.4f}")

        if config.validate_every_n_steps and global_step % config.validate_every_n_steps == 0:
            show_random_inference_grid(
                pipe,
                eval_records,
                prompt_embeds,
                text_ids,
                device=device,
                pde_name=config.pde_name,
                train_image_size=config.train_image_size,
                num_inference_steps=config.distilled_num_inference_steps,
                n=8,
                seed=config.seed + global_step,
            )
            show_smoothed_loss(loss_history, alpha=config.loss_ema_alpha)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pipe.transformer.eval()
    transformer_lora_layers = get_peft_model_state_dict(pipe.transformer)
    Flux2KleinPipeline.save_lora_weights(
        save_directory=config.output_dir,
        transformer_lora_layers=transformer_lora_layers,
    )
    if hasattr(pipe.transformer, "thermal_modulation"):
        torch.save(pipe.transformer.thermal_modulation.state_dict(), config.output_dir / "thermal_modulation.pt")
    print(f"saved LoRA to {config.output_dir.resolve()}")

    show_random_inference_grid(
        pipe,
        eval_records,
        prompt_embeds,
        text_ids,
        device=device,
        pde_name=config.pde_name,
        train_image_size=config.train_image_size,
        num_inference_steps=config.distilled_num_inference_steps,
        n=8,
        seed=config.seed,
    )
    show_smoothed_loss(loss_history, alpha=config.loss_ema_alpha)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "pipe": pipe,
        "train_records": train_records,
        "eval_records": eval_records,
        "loss_history": loss_history,
        "prompt_embeds": prompt_embeds,
        "text_ids": text_ids,
        "device": device,
    }
