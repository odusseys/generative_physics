import numpy as np
import torch


def gaussian_mode_function_batch(
    seeds,
    grid_size,
    num_gaussian_modes,
    device,
    dtype=torch.float32,
    sigma_min=0.04,
    sigma_max=0.16,
):
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")
    if num_gaussian_modes <= 0:
        raise ValueError("num_gaussian_modes must be positive.")
    if sigma_min <= 0 or sigma_max <= 0 or sigma_min > sigma_max:
        raise ValueError("Gaussian sigma bounds must be positive and ordered.")

    seeds = list(seeds)
    coords = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    function = torch.zeros((len(seeds), grid_size, grid_size), device=device, dtype=dtype)

    centers = []
    sigmas = []
    weights = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        centers.append(rng.uniform(0.0, 1.0, size=(num_gaussian_modes, 2)).astype(np.float32)[None, ...])
        sigmas.append(rng.uniform(sigma_min, sigma_max, size=num_gaussian_modes).astype(np.float32)[None, :])
        weights.append(rng.uniform(0.25, 1.0, size=num_gaussian_modes).astype(np.float32)[None, :])

    centers = torch.as_tensor(np.concatenate(centers, axis=0), device=device, dtype=dtype)
    sigmas = torch.as_tensor(np.concatenate(sigmas, axis=0), device=device, dtype=dtype)
    weights = torch.as_tensor(np.concatenate(weights, axis=0), device=device, dtype=dtype)

    for mode in range(num_gaussian_modes):
        center_x = centers[:, mode, 0, None, None]
        center_y = centers[:, mode, 1, None, None]
        sigma = sigmas[:, mode, None, None]
        radius_sq = (x[None] - center_x) ** 2 + (y[None] - center_y) ** 2
        function += weights[:, mode, None, None] * torch.exp(-0.5 * radius_sq / (sigma**2))

    function = function - function.amin(dim=(1, 2), keepdim=True)
    function = function / function.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return coords, function.clamp(0.0, 1.0)


def anisotropic_gaussian_mode_function_batch(
    seeds,
    grid_size,
    num_gaussian_modes,
    device,
    dtype=torch.float32,
    sigma_min=0.04,
    sigma_max=0.16,
):
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")
    if num_gaussian_modes <= 0:
        raise ValueError("num_gaussian_modes must be positive.")
    if sigma_min <= 0 or sigma_max <= 0 or sigma_min > sigma_max:
        raise ValueError("Gaussian sigma bounds must be positive and ordered.")

    seeds = list(seeds)
    coords = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    function = torch.zeros((len(seeds), grid_size, grid_size), device=device, dtype=dtype)

    centers = []
    sigmas = []
    angles = []
    weights = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        centers.append(rng.uniform(0.0, 1.0, size=(num_gaussian_modes, 2)).astype(np.float32)[None, ...])
        sigmas.append(rng.uniform(sigma_min, sigma_max, size=(num_gaussian_modes, 2)).astype(np.float32)[None, ...])
        angles.append(rng.uniform(0.0, np.pi, size=num_gaussian_modes).astype(np.float32)[None, :])
        weights.append(rng.uniform(0.25, 1.0, size=num_gaussian_modes).astype(np.float32)[None, :])

    centers = torch.as_tensor(np.concatenate(centers, axis=0), device=device, dtype=dtype)
    sigmas = torch.as_tensor(np.concatenate(sigmas, axis=0), device=device, dtype=dtype)
    angles = torch.as_tensor(np.concatenate(angles, axis=0), device=device, dtype=dtype)
    weights = torch.as_tensor(np.concatenate(weights, axis=0), device=device, dtype=dtype)

    for mode in range(num_gaussian_modes):
        center_x = centers[:, mode, 0, None, None]
        center_y = centers[:, mode, 1, None, None]
        sigma_x = sigmas[:, mode, 0, None, None]
        sigma_y = sigmas[:, mode, 1, None, None]
        angle = angles[:, mode, None, None]
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        dx = x[None] - center_x
        dy = y[None] - center_y
        u = cos_angle * dx + sin_angle * dy
        v = -sin_angle * dx + cos_angle * dy
        mahalanobis_sq = (u / sigma_x) ** 2 + (v / sigma_y) ** 2
        function += weights[:, mode, None, None] * torch.exp(-0.5 * mahalanobis_sq)

    function = function - function.amin(dim=(1, 2), keepdim=True)
    function = function / function.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return coords, function.clamp(0.0, 1.0)
