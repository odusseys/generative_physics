import torch

from .gaussian_modes import gaussian_mode_function_batch
from .numerics import downsample_solution_torch
from .rendering import scalar_to_rgb_uint8_torch


def _dst_i(x, dim=-1):
    n = x.shape[dim]
    zeros_shape = list(x.shape)
    zeros_shape[dim] = 1
    zeros = torch.zeros(zeros_shape, device=x.device, dtype=x.dtype)
    extended = torch.cat([zeros, x, zeros, -torch.flip(x, dims=(dim,))], dim=dim)
    transformed = torch.fft.fft(extended, dim=dim)
    return -transformed.imag.narrow(dim, 1, n)


def _idst_i(x, dim=-1):
    return _dst_i(x, dim=dim) / (2.0 * (x.shape[dim] + 1))


def _dst2_i(x):
    return _dst_i(_dst_i(x, dim=-1), dim=-2)


def _idst2_i(x):
    return _idst_i(_idst_i(x, dim=-1), dim=-2)


def _poisson_source_batch(
    seeds,
    grid_size,
    num_gaussian_modes,
    device,
    dtype=torch.float32,
    sigma_min=0.04,
    sigma_max=0.16,
):
    return gaussian_mode_function_batch(
        seeds=seeds,
        grid_size=grid_size,
        num_gaussian_modes=num_gaussian_modes,
        device=device,
        dtype=dtype,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )


@torch.no_grad()
def solve_poisson_batch_torch(
    seeds,
    grid_size=256,
    num_gaussian_modes=12,
    source_scale=1.0,
    sim_device=None,
    dtype=torch.float32,
):
    device = torch.device(sim_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if source_scale <= 0:
        raise ValueError("poisson_source_scale must be positive.")

    x, source = _poisson_source_batch(
        seeds,
        grid_size=grid_size,
        num_gaussian_modes=num_gaussian_modes,
        device=device,
        dtype=dtype,
    )
    interior_source = source[:, 1:-1, 1:-1] * source_scale
    interior_n = interior_source.shape[-1]
    h = 1.0 / (grid_size - 1)

    modes = torch.arange(1, interior_n + 1, device=device, dtype=dtype)
    eigenvalues_1d = 4.0 * torch.sin(torch.pi * modes / (2.0 * (interior_n + 1))) ** 2 / (h**2)
    eigenvalues = eigenvalues_1d[:, None] + eigenvalues_1d[None, :]

    source_hat = _dst2_i(interior_source)
    solution_hat = source_hat / eigenvalues[None]
    interior_solution = _idst2_i(solution_hat)

    solution = torch.zeros_like(source)
    solution[:, 1:-1, 1:-1] = interior_solution
    return x.detach().cpu().numpy(), x.detach().cpu().numpy(), source, solution


def generate_poisson_image_pairs(
    seeds,
    sim_nx=256,
    output_size=256,
    num_gaussian_modes=12,
    source_scale=1.0,
    solution_vmax=0.05,
    cmap_name="viridis",
    sim_device=None,
):
    if solution_vmax <= 0:
        raise ValueError("poisson_solution_vmax must be positive.")

    seeds = list(seeds)
    _, _, source, solution = solve_poisson_batch_torch(
        seeds=seeds,
        grid_size=sim_nx,
        num_gaussian_modes=num_gaussian_modes,
        source_scale=source_scale,
        sim_device=sim_device,
    )

    source_img = downsample_solution_torch(source, output_size=output_size).float()
    solution_img = downsample_solution_torch(solution, output_size=output_size).float()

    batch_size = len(seeds)
    source_vmin = torch.zeros(batch_size, device=source_img.device, dtype=source_img.dtype)
    source_vmax = torch.ones(batch_size, device=source_img.device, dtype=source_img.dtype)
    solution_vmin = torch.zeros(batch_size, device=solution_img.device, dtype=solution_img.dtype)
    solution_vmax_tensor = torch.full(
        (batch_size,), float(solution_vmax), device=solution_img.device, dtype=solution_img.dtype
    )

    source_rgb = (
        scalar_to_rgb_uint8_torch(source_img, source_vmin, source_vmax, cmap_name=cmap_name).detach().cpu().numpy()
    )
    solution_rgb = (
        scalar_to_rgb_uint8_torch(solution_img, solution_vmin, solution_vmax_tensor, cmap_name=cmap_name)
        .detach()
        .cpu()
        .numpy()
    )
    del source, solution, source_img, solution_img

    pairs = []
    for seed, solution_image, source_image in zip(seeds, solution_rgb, source_rgb):
        params = {
            "seed": seed,
            "pde": "poisson",
            "sim_nx": sim_nx,
            "output_size": output_size,
            "num_gaussian_modes": num_gaussian_modes,
            "source_scale": source_scale,
            "source_vmin": 0.0,
            "source_vmax": 1.0,
            "solution_vmin": 0.0,
            "solution_vmax": float(solution_vmax),
        }
        pairs.append((solution_image, source_image, params))
    return pairs
