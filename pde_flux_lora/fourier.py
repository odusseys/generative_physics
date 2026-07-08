import torch

from .gaussian_modes import anisotropic_gaussian_mode_function_batch
from .numerics import downsample_solution_torch
from .rendering import scalar_to_rgb_uint8_torch


FOURIER_NUM_MODES = 32
FOURIER_SCALE = 1.0
FOURIER_GAUSSIAN_SIGMA_MIN = 0.006
FOURIER_GAUSSIAN_SIGMA_MAX = 0.12
FOURIER_MAX_FREQUENCY = 32
FOURIER_SHIFT = True


def _gaussian_fourier_function_batch(
    seeds,
    grid_size,
    num_gaussian_modes,
    scale,
    sigma_min,
    sigma_max,
    device,
    dtype=torch.float32,
):
    if scale <= 0:
        raise ValueError("fourier_scale must be positive.")
    _, function = anisotropic_gaussian_mode_function_batch(
        seeds=seeds,
        grid_size=grid_size,
        num_gaussian_modes=num_gaussian_modes,
        device=device,
        dtype=dtype,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    return function * scale


@torch.no_grad()
def generate_fourier_image_pairs(
    seeds,
    sim_nx=256,
    output_size=256,
    num_modes=FOURIER_NUM_MODES,
    scale=FOURIER_SCALE,
    sigma_min=FOURIER_GAUSSIAN_SIGMA_MIN,
    sigma_max=FOURIER_GAUSSIAN_SIGMA_MAX,
    max_frequency=FOURIER_MAX_FREQUENCY,
    fft_shift=FOURIER_SHIFT,
    cmap_name="viridis",
    sim_device=None,
):
    del max_frequency

    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed")

    device = torch.device(sim_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    function = _gaussian_fourier_function_batch(
        seeds=seeds,
        grid_size=sim_nx,
        num_gaussian_modes=num_modes,
        scale=scale,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=device,
    )

    transformed = torch.log1p(torch.fft.fft2(function, dim=(-2, -1), norm="ortho").abs())
    if fft_shift:
        transformed = torch.fft.fftshift(transformed, dim=(-2, -1))

    function_img = downsample_solution_torch(function, output_size=output_size).float()
    transformed_img = downsample_solution_torch(transformed, output_size=output_size).float()

    batch_size = len(seeds)
    function_vmin = torch.zeros(batch_size, device=function_img.device, dtype=function_img.dtype)
    function_vmax = torch.full((batch_size,), float(scale), device=function_img.device, dtype=function_img.dtype)
    transformed_vmin = torch.zeros(batch_size, device=transformed_img.device, dtype=transformed_img.dtype)
    transformed_vmax = torch.quantile(transformed_img.flatten(1), 0.995, dim=1).clamp_min(1e-6)
    function_rgb = (
        scalar_to_rgb_uint8_torch(function_img, function_vmin, function_vmax, cmap_name=cmap_name)
        .detach()
        .cpu()
        .numpy()
    )
    transformed_rgb = (
        scalar_to_rgb_uint8_torch(transformed_img, transformed_vmin, transformed_vmax, cmap_name=cmap_name)
        .detach()
        .cpu()
        .numpy()
    )
    transformed_vmax = transformed_vmax.detach().cpu().numpy()
    del function, transformed, function_img, transformed_img

    pairs = []
    for seed, solution_image, source_image, solution_bound in zip(
        seeds,
        transformed_rgb,
        function_rgb,
        transformed_vmax,
    ):
        params = {
            "seed": seed,
            "pde": "fourier",
            "sim_nx": sim_nx,
            "output_size": output_size,
            "num_gaussian_modes": num_modes,
            "scale": scale,
            "gaussian_covariance": "random",
            "gaussian_sigma_min": float(sigma_min),
            "gaussian_sigma_max": float(sigma_max),
            "gaussian_covariance_min_eigenvalue": float(sigma_min**2),
            "gaussian_covariance_max_eigenvalue": float(sigma_max**2),
            "fft_shift": bool(fft_shift),
            "transform": "fft2_log_magnitude",
            "source_vmin": 0.0,
            "source_vmax": float(scale),
            "solution_vmin": 0.0,
            "solution_vmax": float(solution_bound),
        }
        pairs.append((solution_image, source_image, params))
    return pairs
