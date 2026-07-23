import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .fracture import FRACTURE_GRID_SIZE, FRACTURE_INNER_ITERS, FRACTURE_STEPS
from .ks import KS_CONDITION_ENCODING, KS_GRID_SIZE, KS_RENDER_SIZE
from .navier_stokes import (
    NAVIER_STOKES_CONDITIONING_MEAN,
    NAVIER_STOKES_CONDITIONING_NAMES,
    NAVIER_STOKES_CONDITIONING_STD,
    NAVIER_STOKES_CONDITIONING_TRANSFORMS,
    NAVIER_STOKES_FINAL_TIME,
    NAVIER_STOKES_GRID_SIZE,
)


FLUX2_KLEIN_DISTILLED_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
FLUX2_KLEIN_BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"


@dataclass
class TrainingConfig:
    model_id: str | None = None
    pde_kind: str = "heat"

    train_image_size: int = 256
    output_image_size: int = 256
    initial_grid_size: int = 256
    sim_save_steps: int = 512
    sim_nx: int = 2048
    sim_adaptive_dt: bool = False
    sim_compile_step: bool = True
    sim_num_workers: int = 16
    stream_chunk_size: int = 8
    sim_progress_update_every: int = 8

    ks_grid_size: int = KS_GRID_SIZE
    ks_condition_encoding: str = KS_CONDITION_ENCODING
    ks_debug_num_samples: int = 50
    ks_vae_lora_dir: str = "ks_vae_lora"
    ks_condition_adapter_mode: str = "adaln_zero"

    navier_stokes_grid_size: int = NAVIER_STOKES_GRID_SIZE
    navier_stokes_final_time: float = NAVIER_STOKES_FINAL_TIME
    navier_stokes_multiple_debug_samples: int = 4

    elliptic_grid_size: int = 256
    elliptic_cache_dir: str = "/home/ubuntu/datasets/elliptic"

    eikonal_grid_size: int = 256

    ot_solve_grid_size: int = 24

    poisson_grid_size: int = 256

    airfoil_grid_size: int = 256

    elasticity_grid_size: int = 256

    fracture_grid_size: int = FRACTURE_GRID_SIZE
    fracture_steps: int = FRACTURE_STEPS
    fracture_inner_iters: int = FRACTURE_INNER_ITERS

    num_eval_pairs: int = 24
    train_seed_offset: int = 10_000
    eval_seed_offset: int = 20_000
    train_batch_size: int = 2
    grad_accum_steps: int = 2
    max_train_steps: int = 10000
    validate_every_n_steps: int = 100
    validation_num_images: int = 8
    run_initial_validation: bool = True
    loss_ema_alpha: float = 0.03

    learning_rate: float = 3e-5
    lora_rank: int = 16
    lora_dropout: float = 0.0
    thermal_modulation_bottleneck_dim: int = 64
    transformer_gradient_checkpointing: bool = False
    transformer_compile_regions: bool = True
    transformer_compile_mode: str = "default"

    distilled_num_inference_steps: int = 4
    base_num_inference_steps: int = 20
    base_guidance_scale: float = 4.0
    max_sequence_length: int = 512
    text_encoder_out_layers: tuple[int, int, int] = (9, 18, 27)
    seed: int = 1234

    def __post_init__(self):
        if self.model_id is None:
            self.model_id = (
                FLUX2_KLEIN_BASE_MODEL_ID
                if self.pde_kind == "ks"
                else FLUX2_KLEIN_DISTILLED_MODEL_ID
            )
        if self.distilled_num_inference_steps < 1 or self.base_num_inference_steps < 1:
            raise ValueError("inference step counts must be positive.")
        if self.base_guidance_scale < 1.0:
            raise ValueError("base_guidance_scale must be at least 1.")
        if self.pde_kind == "ks":
            if self.ks_condition_adapter_mode not in {"adaln_zero", "cross_attention", "none"}:
                raise ValueError(
                    "ks_condition_adapter_mode must be 'adaln_zero', 'cross_attention', or 'none'."
                )
            if self.ks_condition_encoding not in {"hilbert", "y_constant"}:
                raise ValueError(
                    "ks_condition_encoding must be 'hilbert' or 'y_constant'."
                )
            if self.train_image_size == 256:
                self.train_image_size = KS_RENDER_SIZE
            if self.output_image_size == 256:
                self.output_image_size = KS_RENDER_SIZE
        if self.pde_kind in {"navier_stokes", "navier_stokes_multiple"}:
            if self.navier_stokes_grid_size < 16:
                raise ValueError("navier_stokes_grid_size must be at least 16.")
        if self.pde_kind == "navier_stokes" and self.navier_stokes_final_time <= 0.0:
            raise ValueError("navier_stokes_final_time must be positive.")

    @property
    def pde_name(self) -> str:
        if self.pde_kind == "heat":
            return "Heat"
        if self.pde_kind == "cgl":
            return "Complex Ginzburg-Landau"
        if self.pde_kind == "burgers":
            return "Burgers"
        if self.pde_kind == "poisson":
            return "Poisson"
        if self.pde_kind == "ks":
            return "Kuramoto-Sivashinsky"
        if self.pde_kind == "navier_stokes":
            return "2D Navier-Stokes"
        if self.pde_kind == "navier_stokes_multiple":
            return "2D Navier-Stokes (joint four-time output)"
        if self.pde_kind == "airfoil":
            return "Airfoil Potential Flow"
        if self.pde_kind == "elliptic":
            return "Variable-Coefficient Elliptic PDE"
        if self.pde_kind == "elasticity":
            return "Random-Hole Elasticity"
        if self.pde_kind == "eikonal":
            return "Eikonal Travel Time"
        if self.pde_kind == "ot":
            return "Optimal Transport"
        if self.pde_kind == "fracture":
            return "Phase-Field Fracture"
        raise ValueError(
            f"Unknown pde_kind={self.pde_kind!r}; expected 'heat', 'cgl', 'burgers', 'poisson', 'ks', "
            "'navier_stokes', 'navier_stokes_multiple', 'airfoil', 'elliptic', 'elasticity', 'eikonal', 'ot', "
            "or 'fracture'."
        )

    @property
    def output_dir(self) -> Path:
        if self.pde_kind == "ks" and self.model_id == FLUX2_KLEIN_BASE_MODEL_ID:
            return Path("ks_flux2_klein_base_4b_lora")
        return Path(f"{self.pde_kind}_flux2_klein_4b_lora")

    @property
    def inference_num_steps(self) -> int:
        if self.pde_kind == "ks" and self.model_id == FLUX2_KLEIN_BASE_MODEL_ID:
            return self.base_num_inference_steps
        return self.distilled_num_inference_steps

    @property
    def inference_guidance_scale(self) -> float:
        if self.pde_kind == "ks" and self.model_id == FLUX2_KLEIN_BASE_MODEL_ID:
            return self.base_guidance_scale
        return 1.0

    @property
    def training_sigma_mode(self) -> str:
        if self.pde_kind == "ks" and self.model_id == FLUX2_KLEIN_BASE_MODEL_ID:
            return "inference_aligned"
        return "distilled"

    @property
    def prompt(self) -> str:
        return ""

    @property
    def lora_alpha(self) -> int:
        return self.lora_rank

    @property
    def heat_log_diffusivity_mean(self) -> float:
        from .heat import HEAT_DIFFUSIVITY_MAX, HEAT_DIFFUSIVITY_MIN

        return 0.5 * (math.log(HEAT_DIFFUSIVITY_MIN) + math.log(HEAT_DIFFUSIVITY_MAX))

    @property
    def heat_log_diffusivity_std(self) -> float:
        from .heat import HEAT_DIFFUSIVITY_MAX, HEAT_DIFFUSIVITY_MIN

        return (math.log(HEAT_DIFFUSIVITY_MAX) - math.log(HEAT_DIFFUSIVITY_MIN)) / math.sqrt(12.0)

    @staticmethod
    def _log_uniform_mean_std(min_value: float, max_value: float) -> tuple[float, float]:
        return (
            0.5 * (math.log(min_value) + math.log(max_value)),
            (math.log(max_value) - math.log(min_value)) / math.sqrt(12.0),
        )

    @staticmethod
    def _uniform_mean_std(min_value: float, max_value: float) -> tuple[float, float]:
        return (
            0.5 * (float(min_value) + float(max_value)),
            (float(max_value) - float(min_value)) / math.sqrt(12.0),
        )

    @property
    def elasticity_conditioning_names(self) -> tuple[str, ...]:
        from .elasticity import ELASTICITY_CONDITIONING_NAMES

        return ELASTICITY_CONDITIONING_NAMES

    @property
    def elasticity_conditioning_transforms(self) -> tuple[str, ...]:
        from .elasticity import ELASTICITY_CONDITIONING_TRANSFORMS

        return ELASTICITY_CONDITIONING_TRANSFORMS

    @property
    def elasticity_conditioning_mean(self) -> tuple[float, ...]:
        from .elasticity import (
            ELASTICITY_HORIZONTAL_STRESS_MAX,
            ELASTICITY_HORIZONTAL_STRESS_MIN,
            ELASTICITY_LAMBDA_MAX,
            ELASTICITY_LAMBDA_MIN,
            ELASTICITY_MU_MAX,
            ELASTICITY_MU_MIN,
            ELASTICITY_VERTICAL_STRESS_MAX,
            ELASTICITY_VERTICAL_STRESS_MIN,
        )

        lambda_mean, _ = self._log_uniform_mean_std(ELASTICITY_LAMBDA_MIN, ELASTICITY_LAMBDA_MAX)
        mu_mean, _ = self._log_uniform_mean_std(ELASTICITY_MU_MIN, ELASTICITY_MU_MAX)
        sigma_x_mean, _ = self._uniform_mean_std(
            ELASTICITY_HORIZONTAL_STRESS_MIN,
            ELASTICITY_HORIZONTAL_STRESS_MAX,
        )
        sigma_y_mean, _ = self._uniform_mean_std(
            ELASTICITY_VERTICAL_STRESS_MIN,
            ELASTICITY_VERTICAL_STRESS_MAX,
        )
        return (lambda_mean, mu_mean, sigma_x_mean, sigma_y_mean)

    @property
    def elasticity_conditioning_std(self) -> tuple[float, ...]:
        from .elasticity import (
            ELASTICITY_HORIZONTAL_STRESS_MAX,
            ELASTICITY_HORIZONTAL_STRESS_MIN,
            ELASTICITY_LAMBDA_MAX,
            ELASTICITY_LAMBDA_MIN,
            ELASTICITY_MU_MAX,
            ELASTICITY_MU_MIN,
            ELASTICITY_VERTICAL_STRESS_MAX,
            ELASTICITY_VERTICAL_STRESS_MIN,
        )

        _, lambda_std = self._log_uniform_mean_std(ELASTICITY_LAMBDA_MIN, ELASTICITY_LAMBDA_MAX)
        _, mu_std = self._log_uniform_mean_std(ELASTICITY_MU_MIN, ELASTICITY_MU_MAX)
        _, sigma_x_std = self._uniform_mean_std(
            ELASTICITY_HORIZONTAL_STRESS_MIN,
            ELASTICITY_HORIZONTAL_STRESS_MAX,
        )
        _, sigma_y_std = self._uniform_mean_std(
            ELASTICITY_VERTICAL_STRESS_MIN,
            ELASTICITY_VERTICAL_STRESS_MAX,
        )
        return (lambda_std, mu_std, sigma_x_std, sigma_y_std)

    @property
    def fracture_conditioning_names(self) -> tuple[str, ...]:
        from .fracture import FRACTURE_CONDITIONING_NAMES

        return FRACTURE_CONDITIONING_NAMES

    @property
    def fracture_conditioning_transforms(self) -> tuple[str, ...]:
        from .fracture import FRACTURE_CONDITIONING_TRANSFORMS

        return FRACTURE_CONDITIONING_TRANSFORMS

    @property
    def fracture_conditioning_mean(self) -> tuple[float, ...]:
        from .fracture import FRACTURE_CONDITIONING_MEAN

        return FRACTURE_CONDITIONING_MEAN

    @property
    def fracture_conditioning_std(self) -> tuple[float, ...]:
        from .fracture import FRACTURE_CONDITIONING_STD

        return FRACTURE_CONDITIONING_STD

    @property
    def navier_stokes_conditioning_names(self) -> tuple[str, ...]:
        return NAVIER_STOKES_CONDITIONING_NAMES

    @property
    def navier_stokes_conditioning_transforms(self) -> tuple[str, ...]:
        return NAVIER_STOKES_CONDITIONING_TRANSFORMS

    @property
    def navier_stokes_conditioning_mean(self) -> tuple[float, ...]:
        return NAVIER_STOKES_CONDITIONING_MEAN

    @property
    def navier_stokes_conditioning_std(self) -> tuple[float, ...]:
        return NAVIER_STOKES_CONDITIONING_STD

    @property
    def sim_batch_size(self) -> int:
        if self.pde_kind == "poisson":
            return 32 if torch.cuda.is_available() else 1
        if self.pde_kind in {"navier_stokes", "navier_stokes_multiple"}:
            return 8 if torch.cuda.is_available() else 1
        if self.pde_kind in {"airfoil", "elliptic", "elasticity", "eikonal", "ot", "fracture"}:
            return 1
        return 256 if torch.cuda.is_available() else 1
