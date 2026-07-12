from dataclasses import dataclass
from pathlib import Path
import math

import torch

from .fracture import FRACTURE_GRID_SIZE, FRACTURE_INNER_ITERS, FRACTURE_STEPS
from .ks import KS_GRID_SIZE


@dataclass
class TrainingConfig:
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
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

    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_dropout: float = 0.0
    thermal_modulation_bottleneck_dim: int = 64

    distilled_num_inference_steps: int = 4
    max_sequence_length: int = 512
    text_encoder_out_layers: tuple[int, int, int] = (9, 18, 27)
    seed: int = 1234

    def __post_init__(self):
        if self.pde_kind == "ks":
            if self.train_image_size == 256:
                self.train_image_size = self.ks_grid_size
            if self.output_image_size == 256:
                self.output_image_size = self.ks_grid_size

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
            "'airfoil', 'elliptic', 'elasticity', 'eikonal', 'ot', or 'fracture'."
        )

    @property
    def output_dir(self) -> Path:
        return Path(f"{self.pde_kind}_flux2_klein_4b_lora")

    @property
    def prompt(self) -> str:
        if self.pde_kind == "heat":
            return (
                "Given the initial 1D Heat condition image, forcing image, and thermal diffusivity coefficient, "
                "generate the Heat time-evolution image."
            )
        if self.pde_kind == "poisson":
            return "Given the 2D Poisson source image, generate the zero-boundary Poisson solution image."
        if self.pde_kind == "ks":
            return (
                "Given the initial Kuramoto-Sivashinsky state image, "
                "generate the future space-time trajectory image."
            )
        if self.pde_kind == "airfoil":
            return "Given the no-flow airfoil image in uniform rightward flow, generate the potential-flow image."
        if self.pde_kind == "elliptic":
            return (
                "Given seven 2D variable-coefficient elliptic PDE images for a20, a11, a02, a10, a01, a00, "
                "and forcing f, generate the zero-boundary solution image."
            )
        if self.pde_kind == "elasticity":
            return (
                "Given a binary mask image of a random hole in an elastic plate plus material and biaxial "
                "far-field stress scalars, generate the stress field image."
            )
        if self.pde_kind == "eikonal":
            return "Given a 2D refractive index image, generate the center-source eikonal propagation time image."
        if self.pde_kind == "ot":
            return (
                "Given source distribution, target distribution, and transport cost images, "
                "generate the entropic optimal transport source potential image."
            )
        if self.pde_kind == "fracture":
            return (
                "Given an initial phase-field crack damage image plus material and biaxial strain scalars, "
                "generate the final fracture damage image."
            )
        return f"Given the initial 1D {self.pde_name} condition image, generate the {self.pde_name} time-evolution image."

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
    def sim_batch_size(self) -> int:
        if self.pde_kind == "poisson":
            return 32 if torch.cuda.is_available() else 1
        if self.pde_kind in {"airfoil", "elliptic", "elasticity", "eikonal", "ot", "fracture"}:
            return 1
        return 256 if torch.cuda.is_available() else 1
