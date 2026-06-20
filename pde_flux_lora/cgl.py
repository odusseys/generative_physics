import numpy as np
import torch
from tqdm.auto import tqdm

from .numerics import downsample_complex_solution_torch, periodic_linear_upsample_1d
from .rendering import complex_to_rgb_uint8_torch, repeat_rgb_row_view


def _cgl_initial_batch(seeds, nx, num_modes, amp_scale, phase_scale, device, dtype=torch.float32):
    seeds = list(seeds)
    x = torch.arange(nx, device=device, dtype=dtype) / nx
    k = torch.arange(1, num_modes + 1, device=device, dtype=dtype)
    denom = np.arange(1, num_modes + 1, dtype=np.float32)
    phase_amps = []
    phase_offsets = []
    amp_amps = []
    amp_offsets = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        phase_amps.append((rng.normal(size=num_modes).astype(np.float32) / denom)[None, :])
        phase_offsets.append(rng.uniform(0, 2 * np.pi, size=num_modes).astype(np.float32)[None, :])
        amp_amps.append((rng.normal(size=num_modes).astype(np.float32) / denom)[None, :])
        amp_offsets.append(rng.uniform(0, 2 * np.pi, size=num_modes).astype(np.float32)[None, :])

    phase_amps = torch.as_tensor(np.concatenate(phase_amps, axis=0), device=device, dtype=dtype)
    phase_offsets = torch.as_tensor(np.concatenate(phase_offsets, axis=0), device=device, dtype=dtype)
    amp_amps = torch.as_tensor(np.concatenate(amp_amps, axis=0), device=device, dtype=dtype)
    amp_offsets = torch.as_tensor(np.concatenate(amp_offsets, axis=0), device=device, dtype=dtype)

    angles_phase = 2 * torch.pi * k[None, :, None] * x[None, None, :] + phase_offsets[:, :, None]
    angles_amp = 2 * torch.pi * k[None, :, None] * x[None, None, :] + amp_offsets[:, :, None]
    phase = (phase_amps[:, :, None] * torch.sin(angles_phase)).sum(dim=1)
    amp_noise = (amp_amps[:, :, None] * torch.sin(angles_amp)).sum(dim=1)
    phase = phase_scale * phase / (phase.std(dim=-1, keepdim=True, unbiased=False) + 1e-12)
    amp_noise = amp_noise / (amp_noise.std(dim=-1, keepdim=True, unbiased=False) + 1e-12)
    amp = torch.clamp(1.0 + amp_scale * amp_noise, min=0.05)
    return amp.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))


@torch.no_grad()
def solve_complex_ginzburg_landau_batch_torch(
    seeds,
    nx=2048,
    domain_length=256.0,
    T=8.0,
    c1=1.5,
    c3=-1.0,
    num_modes=16,
    amp_scale=0.25,
    phase_scale=np.pi,
    save_steps=512,
    substeps_per_frame=4,
    initial_grid_size=None,
    device=None,
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
    A_initial = _cgl_initial_batch(seeds, initial_grid_size, num_modes, amp_scale, phase_scale, device=device)
    A = periodic_linear_upsample_1d(A_initial, nx)
    U = torch.empty((len(seeds), save_steps, nx), device=device, dtype=torch.complex64)
    U[:, 0] = A

    k = 2 * torch.pi * torch.fft.fftfreq(nx, d=domain_length / nx, device=device)
    L = 1.0 - (1.0 + 1j * c1) * (k.to(torch.complex64) ** 2)
    dt = T / max(save_steps - 1, 1) / substeps_per_frame
    E = torch.exp(L * dt)
    phi = torch.where(torch.abs(L) > 1e-12, (E - 1.0) / L, torch.full_like(L, dt))

    pbar = tqdm(
        total=save_steps - 1,
        desc=progress_desc or "CGL saves",
        position=progress_position,
        leave=True,
        mininterval=0.5,
        disable=not progress,
    )
    pending_progress = 0
    try:
        for save_idx in range(1, save_steps):
            for _ in range(substeps_per_frame):
                nonlinear = -(1.0 - 1j * c3) * (A.abs() ** 2) * A
                A_hat = torch.fft.fft(A, dim=-1)
                N_hat = torch.fft.fft(nonlinear, dim=-1)
                A = torch.fft.ifft(E[None, :] * A_hat + phi[None, :] * N_hat, dim=-1)
            U[:, save_idx] = A
            pending_progress += 1
            if pending_progress >= progress_update_every or save_idx == save_steps - 1:
                pbar.update(pending_progress)
                pending_progress = 0
                pbar.set_postfix(t=f"{save_idx * T / (save_steps - 1):.2f}")
    finally:
        if pending_progress:
            pbar.update(pending_progress)
        pbar.close()

    x = torch.arange(nx, device=device, dtype=torch.float32) / nx
    save_t = np.linspace(0, T, save_steps, dtype=np.float64)
    if return_initial_grid:
        return x.detach().cpu().numpy(), save_t, U, A_initial
    return x.detach().cpu().numpy(), save_t, U


def generate_cgl_image_pairs(
    seeds,
    sim_nx=2048,
    domain_length=256.0,
    output_size=512,
    initial_grid_size=None,
    T=8.0,
    c1=1.5,
    c3=-1.0,
    num_modes=16,
    amp_scale=0.25,
    phase_scale=np.pi,
    save_steps=512,
    substeps_per_frame=4,
    sim_device=None,
    progress=False,
    progress_desc=None,
    progress_position=1,
    progress_update_every=8,
):
    seeds = list(seeds)
    initial_grid_size = int(initial_grid_size or output_size)
    _, _, U, A_initial = solve_complex_ginzburg_landau_batch_torch(
        seeds=seeds,
        nx=sim_nx,
        domain_length=domain_length,
        T=T,
        c1=c1,
        c3=c3,
        num_modes=num_modes,
        amp_scale=amp_scale,
        phase_scale=phase_scale,
        save_steps=save_steps,
        substeps_per_frame=substeps_per_frame,
        initial_grid_size=initial_grid_size,
        device=sim_device,
        progress=progress,
        progress_desc=progress_desc,
        progress_position=progress_position,
        progress_update_every=progress_update_every,
        return_initial_grid=True,
    )
    U_img = downsample_complex_solution_torch(U, output_size=output_size)
    A0_img = periodic_linear_upsample_1d(A_initial, output_size)
    U_img[:, 0] = A0_img
    amp_vmax = U_img.abs().float().amax(dim=(1, 2)).clamp_min(1e-6)
    solution_rgb = complex_to_rgb_uint8_torch(U_img, amp_vmax=amp_vmax).detach().cpu().numpy()
    initial_row_rgb = complex_to_rgb_uint8_torch(A0_img[:, None, :], amp_vmax=amp_vmax)[:, 0].detach().cpu().numpy()
    amp_vmax = amp_vmax.detach().cpu().numpy()
    del U, U_img, A_initial, A0_img

    pairs = []
    for seed, solution_img, initial_row, amp_vmax_i in zip(seeds, solution_rgb, initial_row_rgb, amp_vmax):
        params = {
            "seed": seed,
            "pde": "cgl",
            "sim_nx": sim_nx,
            "domain_length": domain_length,
            "output_size": output_size,
            "initial_grid_size": initial_grid_size,
            "T": T,
            "c1": c1,
            "c3": c3,
            "num_modes": num_modes,
            "amp_scale": amp_scale,
            "phase_scale": phase_scale,
            "substeps_per_frame": substeps_per_frame,
            "amp_vmax": float(amp_vmax_i),
        }
        pairs.append((solution_img, repeat_rgb_row_view(initial_row, output_size), params))
    return pairs
