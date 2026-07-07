from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import torch


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
    sim_progress_update_every: int = 8

    heat_t: float = 1.5
    heat_num_modes: int = 32
    heat_scale: float = 1.0
    heat_forcing_num_modes: int = 12
    heat_forcing_scale: float = 1.0
    heat_diffusivity_min: float = 1e-5
    heat_diffusivity_max: float = 1e-2

    elliptic_grid_size: int = 256
    elliptic_max_cycles: float = 5.5
    elliptic_a_min: float = 0.025
    elliptic_a_max: float = 0.075
    elliptic_mixed_rho: float = 0.12
    elliptic_first_order_scale: float = 0.02
    elliptic_reaction_min: float = 40.0
    elliptic_reaction_max: float = 120.0
    elliptic_solution_vmin: float = 0.0
    elliptic_solution_vmax: float = 1.0
    elliptic_num_workers: int = 16
    elliptic_worker_chunksize: int = 1
    elliptic_cache_dir: str = "/home/ubuntu/datasets/elliptic"

    cgl_t: float = 12.0
    cgl_domain_length: float = 128.0
    cgl_c1: float = 2.0
    cgl_c3: float = 1.2
    cgl_num_modes: int = 16
    cgl_amp_scale: float = 0.25
    cgl_phase_scale: float = np.pi
    cgl_substeps_per_frame: int = 8

    poisson_grid_size: int = 256
    poisson_num_gaussian_modes: int = 12
    poisson_source_scale: float = 1.0
    poisson_solution_vmax: float = 0.05

    fourier_grid_size: int = 256
    fourier_num_modes: int = 32
    fourier_scale: float = 1.0
    fourier_gaussian_sigma_min: float = 0.006
    fourier_gaussian_sigma_max: float = 0.12
    fourier_max_frequency: int = 32
    fourier_shift: bool = True

    airfoil_grid_size: int = 256
    airfoil_min_points: int = 5
    airfoil_max_points: int = 10
    airfoil_samples_per_segment: int = 28
    airfoil_handle_scale: float = 0.12
    airfoil_body_box_fraction: float = 0.5
    airfoil_x_stretch: float = 1.8
    airfoil_y_stretch: float = 0.55
    airfoil_flow_speed: float = 1.0
    airfoil_color_mode: str = "rgb"
    airfoil_speed_vmin: float = 0.5
    airfoil_speed_vmax: float = 2.1
    airfoil_num_workers: int = 0
    airfoil_worker_chunksize: int = 4

    num_train_pairs: int = 2048 // 2
    num_eval_pairs: int = 24
    train_seed_offset: int = 10_000
    eval_seed_offset: int = 20_000
    train_batch_size: int = 2
    grad_accum_steps: int = 2
    max_train_steps: int = 10000
    validate_every_n_steps: int = 100
    validation_num_images: int = 8
    loss_ema_alpha: float = 0.08

    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_dropout: float = 0.0
    thermal_modulation_bottleneck_dim: int = 64

    distilled_num_inference_steps: int = 4
    max_sequence_length: int = 512
    text_encoder_out_layers: tuple[int, int, int] = (9, 18, 27)
    seed: int = 1234

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
        if self.pde_kind == "fourier":
            return "Fourier Transform"
        if self.pde_kind == "airfoil":
            return "Airfoil Potential Flow"
        if self.pde_kind == "elliptic":
            return "Variable-Coefficient Elliptic PDE"
        raise ValueError(
            f"Unknown pde_kind={self.pde_kind!r}; expected 'heat', 'cgl', 'burgers', 'poisson', "
            "'fourier', 'airfoil', or 'elliptic'."
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
        if self.pde_kind == "fourier":
            return (
                f"Given a scalar function image with {self.fourier_num_modes} random-covariance Gaussian modes, "
                "generate the log magnitude of its 2D Fourier transform."
            )
        if self.pde_kind == "airfoil":
            return "Given the no-flow airfoil image in uniform rightward flow, generate the potential-flow image."
        if self.pde_kind == "elliptic":
            return (
                "Given seven 2D variable-coefficient elliptic PDE images for a20, a11, a02, a10, a01, a00, "
                "and forcing f, generate the zero-boundary solution image."
            )
        return f"Given the initial 1D {self.pde_name} condition image, generate the {self.pde_name} time-evolution image."

    @property
    def lora_alpha(self) -> int:
        return self.lora_rank

    @property
    def heat_log_diffusivity_mean(self) -> float:
        return 0.5 * (math.log(self.heat_diffusivity_min) + math.log(self.heat_diffusivity_max))

    @property
    def heat_log_diffusivity_std(self) -> float:
        return (math.log(self.heat_diffusivity_max) - math.log(self.heat_diffusivity_min)) / math.sqrt(12.0)

    @property
    def sim_batch_size(self) -> int:
        if self.pde_kind in {"poisson", "fourier"}:
            return 32 if torch.cuda.is_available() else 1
        if self.pde_kind in {"airfoil", "elliptic"}:
            return 1
        return 256 if torch.cuda.is_available() else 1
