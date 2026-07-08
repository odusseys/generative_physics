import numpy as np
import torch
from tqdm.auto import tqdm

from .numerics import downsample_solution_torch, periodic_linear_upsample_1d
from .rendering import repeat_rgb_row_view, scalar_to_rgb_uint8_torch


HEAT_T = 1.5
HEAT_NUM_MODES = 24
HEAT_SCALE = 1.0
HEAT_FORCING_NUM_MODES = 12
HEAT_FORCING_SCALE = 1.0
HEAT_DIFFUSIVITY_MIN = 1e-5
HEAT_DIFFUSIVITY_MAX = 1e-2


def _heat_initial_and_diffusivity_batch(
    seeds,
    nx,
    num_modes,
    scale,
    diffusivity_min,
    diffusivity_max,
    device,
    dtype=torch.float32,
):
    if num_modes < 1:
        raise ValueError("heat_num_modes must be at least 1.")
    if diffusivity_min <= 0 or diffusivity_max <= 0:
        raise ValueError("Heat thermal diffusivity bounds must be positive.")
    if diffusivity_min > diffusivity_max:
        raise ValueError("heat_diffusivity_min must be <= heat_diffusivity_max.")

    seeds = list(seeds)
    x = torch.arange(nx, device=device, dtype=dtype) / nx
    k = torch.arange(1, num_modes + 1, device=device, dtype=dtype)
    denom = np.arange(1, num_modes + 1, dtype=np.float32)
    amps = []
    phases = []
    diffusivities = []
    mode_counts = []

    log_min = np.log(diffusivity_min)
    log_max = np.log(diffusivity_max)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        mode_count = int(rng.integers(1, int(num_modes) + 1))
        amp = np.zeros(int(num_modes), dtype=np.float32)
        phase = np.zeros(int(num_modes), dtype=np.float32)
        amp[:mode_count] = rng.normal(size=mode_count).astype(np.float32) / denom[:mode_count]
        phase[:mode_count] = rng.uniform(0, 2 * np.pi, size=mode_count).astype(np.float32)
        amps.append(amp[None, :])
        phases.append(phase[None, :])
        diffusivities.append(np.exp(rng.uniform(log_min, log_max)))
        mode_counts.append(mode_count)

    amps = torch.as_tensor(np.concatenate(amps, axis=0), device=device, dtype=dtype)
    phases = torch.as_tensor(np.concatenate(phases, axis=0), device=device, dtype=dtype)
    diffusivities = torch.as_tensor(diffusivities, device=device, dtype=dtype)
    mode_counts = torch.as_tensor(mode_counts, device=device, dtype=torch.long)
    angles = 2 * torch.pi * k[None, :, None] * x[None, None, :] + phases[:, :, None]
    u = (amps[:, :, None] * torch.sin(angles)).sum(dim=1)
    u = scale * u / (u.std(dim=-1, keepdim=True, unbiased=False) + 1e-12)
    return x, u, diffusivities, mode_counts


def _forcing_rng(seed, endpoint_index=0):
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([int(seed), 0xF0A, int(endpoint_index)]))


def _heat_forcing_batch(
    seeds,
    nx,
    num_modes,
    scale,
    device,
    dtype=torch.float32,
    endpoint_index=0,
):
    if num_modes < 0:
        raise ValueError("heat_forcing_num_modes must be non-negative.")
    if scale < 0:
        raise ValueError("heat_forcing_scale must be non-negative.")

    seeds = list(seeds)
    x = torch.arange(nx, device=device, dtype=dtype) / nx
    if num_modes == 0 or scale == 0:
        mode_counts = torch.zeros(len(seeds), device=device, dtype=torch.long)
        return x, torch.zeros((len(seeds), nx), device=device, dtype=dtype), mode_counts

    k = torch.arange(1, num_modes + 1, device=device, dtype=dtype)
    denom = np.arange(1, num_modes + 1, dtype=np.float32)
    amps = []
    phases = []
    mode_counts = []

    for seed in seeds:
        rng = _forcing_rng(seed, endpoint_index=endpoint_index)
        mode_count = int(rng.integers(1, num_modes + 1))
        amp = np.zeros(num_modes, dtype=np.float32)
        phase = np.zeros(num_modes, dtype=np.float32)
        amp[:mode_count] = rng.normal(size=mode_count).astype(np.float32) / denom[:mode_count]
        phase[:mode_count] = rng.uniform(0, 2 * np.pi, size=mode_count).astype(np.float32)
        amps.append(amp[None, :])
        phases.append(phase[None, :])
        mode_counts.append(mode_count)

    amps = torch.as_tensor(np.concatenate(amps, axis=0), device=device, dtype=dtype)
    phases = torch.as_tensor(np.concatenate(phases, axis=0), device=device, dtype=dtype)
    mode_counts = torch.as_tensor(mode_counts, device=device, dtype=torch.long)
    angles = 2 * torch.pi * k[None, :, None] * x[None, None, :] + phases[:, :, None]
    forcing = (amps[:, :, None] * torch.sin(angles)).sum(dim=1)
    forcing = scale * forcing / (forcing.std(dim=-1, keepdim=True, unbiased=False) + 1e-12)
    return x, forcing, mode_counts


def _interpolate_forcing_batch(forcing_start, forcing_end, num_steps):
    weights = torch.linspace(
        0.0,
        1.0,
        int(num_steps),
        device=forcing_start.device,
        dtype=forcing_start.dtype,
    )
    weights = weights[None, :, None]
    return (1.0 - weights) * forcing_start[:, None, :] + weights * forcing_end[:, None, :]


@torch.no_grad()
def solve_heat_batch_torch(
    seeds,
    nx=2048,
    T=HEAT_T,
    num_modes=HEAT_NUM_MODES,
    scale=HEAT_SCALE,
    forcing_num_modes=HEAT_FORCING_NUM_MODES,
    forcing_scale=HEAT_FORCING_SCALE,
    diffusivity_min=HEAT_DIFFUSIVITY_MIN,
    diffusivity_max=HEAT_DIFFUSIVITY_MAX,
    save_steps=512,
    initial_grid_size=None,
    device=None,
    dtype=torch.float32,
    progress=False,
    progress_desc=None,
    progress_position=1,
    progress_update_every=8,
    return_initial_grid=False,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed")

    initial_grid_size = int(initial_grid_size or nx)
    _, u_initial, diffusivities, initial_mode_counts = _heat_initial_and_diffusivity_batch(
        seeds,
        initial_grid_size,
        num_modes,
        scale,
        diffusivity_min,
        diffusivity_max,
        device=device,
        dtype=dtype,
    )
    _, forcing_start_initial, forcing_start_mode_counts = _heat_forcing_batch(
        seeds,
        initial_grid_size,
        forcing_num_modes,
        forcing_scale,
        device=device,
        dtype=dtype,
        endpoint_index=0,
    )
    _, forcing_end_initial, forcing_end_mode_counts = _heat_forcing_batch(
        seeds,
        initial_grid_size,
        forcing_num_modes,
        forcing_scale,
        device=device,
        dtype=dtype,
        endpoint_index=1,
    )
    u0 = periodic_linear_upsample_1d(u_initial, nx)
    forcing_start = periodic_linear_upsample_1d(forcing_start_initial, nx)
    forcing_end = periodic_linear_upsample_1d(forcing_end_initial, nx)
    save_t_cpu = np.linspace(0, T, save_steps, dtype=np.float64)
    save_t = torch.as_tensor(save_t_cpu, device=device, dtype=dtype)

    wavenumbers = 2 * torch.pi * torch.fft.fftfreq(nx, d=1.0 / nx, device=device).to(dtype)
    u0_hat = torch.fft.fft(u0, dim=-1)
    forcing_start_hat = torch.fft.fft(forcing_start, dim=-1)
    forcing_delta_hat = torch.fft.fft(forcing_end - forcing_start, dim=-1)
    wavenumber_sq = wavenumbers**2
    nonzero_wavenumbers = wavenumber_sq > 0
    forcing_interpolation_T = float(T) if T > 0 else None
    U = torch.empty((len(seeds), save_steps, nx), device=device, dtype=dtype)

    pbar = tqdm(
        total=save_steps,
        desc=progress_desc or "Heat saves",
        position=progress_position,
        leave=True,
        mininterval=0.5,
        disable=not progress,
    )
    pending_progress = 0
    try:
        for start in range(0, save_steps, progress_update_every):
            stop = min(start + progress_update_every, save_steps)
            t_chunk = save_t[start:stop]
            decay = torch.exp(
                -diffusivities[:, None, None] * (wavenumbers[None, None, :] ** 2) * t_chunk[None, :, None]
            )
            forced_hat = u0_hat[:, None, :] * decay
            if forcing_num_modes > 0 and forcing_scale > 0:
                forcing_start_factor = torch.empty_like(decay)
                forcing_interp_factor = torch.zeros_like(decay)
                forcing_start_factor[:, :, ~nonzero_wavenumbers] = t_chunk[None, :, None]
                if forcing_interpolation_T is not None:
                    forcing_interp_factor[:, :, ~nonzero_wavenumbers] = (
                        t_chunk[None, :, None] ** 2 / (2.0 * forcing_interpolation_T)
                    )

                damped_rates = diffusivities[:, None, None] * wavenumber_sq[None, None, nonzero_wavenumbers]
                z = damped_rates * t_chunk[None, :, None]
                forcing_start_factor[:, :, nonzero_wavenumbers] = -torch.expm1(-z) / damped_rates
                if forcing_interpolation_T is not None:
                    small_z = z.abs() < 1e-3
                    interp_integral_regular = (z - 1.0 + torch.exp(-z)) / (damped_rates**2)
                    interp_integral_series = (t_chunk[None, :, None] ** 2) * (
                        0.5 - z / 6.0 + z**2 / 24.0 - z**3 / 120.0
                    )
                    interp_integral = torch.where(small_z, interp_integral_series, interp_integral_regular)
                    forcing_interp_factor[:, :, nonzero_wavenumbers] = interp_integral / forcing_interpolation_T

                forced_hat = (
                    forced_hat
                    + forcing_start_hat[:, None, :] * forcing_start_factor
                    + forcing_delta_hat[:, None, :] * forcing_interp_factor
                )
            U[:, start:stop] = torch.fft.ifft(forced_hat, dim=-1).real
            pending_progress += stop - start
            if pending_progress >= progress_update_every or stop == save_steps:
                pbar.update(pending_progress)
                pending_progress = 0
                pbar.set_postfix(t=f"{float(t_chunk[-1]):.3f}")
    finally:
        if pending_progress:
            pbar.update(pending_progress)
        pbar.close()

    x = torch.arange(nx, device=device, dtype=dtype) / nx
    if return_initial_grid:
        return (
            x.detach().cpu().numpy(),
            save_t_cpu,
            U,
            u_initial,
            forcing_start_initial,
            forcing_end_initial,
            diffusivities,
            initial_mode_counts,
            forcing_start_mode_counts,
            forcing_end_mode_counts,
        )
    return x.detach().cpu().numpy(), save_t_cpu, U, diffusivities


def generate_heat_image_pairs(
    seeds,
    sim_nx=2048,
    output_size=512,
    initial_grid_size=None,
    T=HEAT_T,
    num_modes=HEAT_NUM_MODES,
    scale=HEAT_SCALE,
    forcing_num_modes=HEAT_FORCING_NUM_MODES,
    forcing_scale=HEAT_FORCING_SCALE,
    diffusivity_min=HEAT_DIFFUSIVITY_MIN,
    diffusivity_max=HEAT_DIFFUSIVITY_MAX,
    save_steps=512,
    cmap_name="coolwarm",
    sim_device=None,
    progress=False,
    progress_desc=None,
    progress_position=1,
    progress_update_every=8,
):
    seeds = list(seeds)
    initial_grid_size = int(initial_grid_size or output_size)
    (
        _,
        _,
        U,
        u_initial,
        forcing_start_initial,
        forcing_end_initial,
        diffusivities,
        initial_mode_counts,
        forcing_start_mode_counts,
        forcing_end_mode_counts,
    ) = solve_heat_batch_torch(
        seeds=seeds,
        nx=sim_nx,
        T=T,
        num_modes=num_modes,
        scale=scale,
        forcing_num_modes=forcing_num_modes,
        forcing_scale=forcing_scale,
        diffusivity_min=diffusivity_min,
        diffusivity_max=diffusivity_max,
        save_steps=save_steps,
        initial_grid_size=initial_grid_size,
        device=sim_device,
        progress=progress,
        progress_desc=progress_desc,
        progress_position=progress_position,
        progress_update_every=progress_update_every,
        return_initial_grid=True,
    )
    U_img = downsample_solution_torch(U, output_size=output_size)
    u0_img = periodic_linear_upsample_1d(u_initial, output_size)
    forcing_start_img = periodic_linear_upsample_1d(forcing_start_initial, output_size)
    forcing_end_img = periodic_linear_upsample_1d(forcing_end_initial, output_size)
    forcing_img = _interpolate_forcing_batch(forcing_start_img, forcing_end_img, output_size)
    U_img[:, 0] = u0_img
    solution_bound = U_img.abs().float().amax(dim=(1, 2))
    forcing_bound = forcing_img.abs().float().amax(dim=(1, 2))
    bound = torch.maximum(solution_bound, forcing_bound).clamp_min(1e-6)
    solution_rgb = scalar_to_rgb_uint8_torch(U_img.float(), -bound, bound, cmap_name=cmap_name).detach().cpu().numpy()
    initial_row_rgb = (
        scalar_to_rgb_uint8_torch(u0_img[:, None, :].float(), -bound, bound, cmap_name=cmap_name)[:, 0]
        .detach()
        .cpu()
        .numpy()
    )
    forcing_rgb = (
        scalar_to_rgb_uint8_torch(forcing_img.float(), -bound, bound, cmap_name=cmap_name).detach().cpu().numpy()
    )
    bound = bound.detach().cpu().numpy()
    diffusivities = diffusivities.detach().cpu().numpy()
    initial_mode_counts = initial_mode_counts.detach().cpu().numpy()
    forcing_start_mode_counts = forcing_start_mode_counts.detach().cpu().numpy()
    forcing_end_mode_counts = forcing_end_mode_counts.detach().cpu().numpy()
    del U, U_img, u_initial, u0_img, forcing_start_initial, forcing_end_initial, forcing_start_img, forcing_end_img
    del forcing_img

    pairs = []
    for (
        seed,
        solution_img,
        initial_row,
        forcing_solution_img,
        bound_i,
        diffusivity,
        initial_mode_count,
        forcing_start_mode_count,
        forcing_end_mode_count,
    ) in zip(
        seeds,
        solution_rgb,
        initial_row_rgb,
        forcing_rgb,
        bound,
        diffusivities,
        initial_mode_counts,
        forcing_start_mode_counts,
        forcing_end_mode_counts,
    ):
        params = {
            "seed": seed,
            "pde": "heat",
            "sim_nx": sim_nx,
            "output_size": output_size,
            "initial_grid_size": initial_grid_size,
            "T": T,
            "num_modes": num_modes,
            "initial_active_modes": int(initial_mode_count),
            "scale": scale,
            "forcing_num_modes": forcing_num_modes,
            "forcing_active_modes": int(max(forcing_start_mode_count, forcing_end_mode_count)),
            "forcing_start_active_modes": int(forcing_start_mode_count),
            "forcing_end_active_modes": int(forcing_end_mode_count),
            "forcing_scale": forcing_scale,
            "thermal_diffusivity": float(diffusivity),
            "diffusivity_min": diffusivity_min,
            "diffusivity_max": diffusivity_max,
            "vmin": -float(bound_i),
            "vmax": float(bound_i),
        }
        pairs.append(
            (
                solution_img,
                repeat_rgb_row_view(initial_row, output_size),
                forcing_solution_img,
                params,
            )
        )
    return pairs
