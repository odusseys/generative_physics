import numpy as np
import torch
from tqdm.auto import tqdm

from .numerics import downsample_complex_solution_torch, periodic_linear_upsample_1d
from .rendering import complex_to_rgb_uint8_torch, repeat_rgb_row_view


CGL_T = 12.0
CGL_DOMAIN_LENGTH = 128.0
CGL_C1 = 2.0
CGL_C3 = 1.2
CGL_NUM_MODES = 8
CGL_AMP_SCALE = 0.25
CGL_PHASE_SCALE = np.pi
CGL_SUBSTEPS_PER_FRAME = 8


def _cgl_initial_batch(seeds, nx, num_modes, amp_scale, phase_scale, device, dtype=torch.float32):
    if num_modes < 2:
        raise ValueError("cgl_num_modes must be at least 2.")

    seeds = list(seeds)
    x = torch.arange(nx, device=device, dtype=dtype) / nx
    k = torch.arange(1, num_modes + 1, device=device, dtype=dtype)
    denom = np.arange(1, num_modes + 1, dtype=np.float32)
    phase_amps = []
    phase_offsets = []
    amp_amps = []
    amp_offsets = []
    mode_counts = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        mode_count = int(rng.integers(2, int(num_modes) + 1))
        phase_amp = np.zeros(int(num_modes), dtype=np.float32)
        phase_offset = np.zeros(int(num_modes), dtype=np.float32)
        amp_amp = np.zeros(int(num_modes), dtype=np.float32)
        amp_offset = np.zeros(int(num_modes), dtype=np.float32)
        phase_amp[:mode_count] = rng.normal(size=mode_count).astype(np.float32) / denom[:mode_count]
        phase_offset[:mode_count] = rng.uniform(0, 2 * np.pi, size=mode_count).astype(np.float32)
        amp_amp[:mode_count] = rng.normal(size=mode_count).astype(np.float32) / denom[:mode_count]
        amp_offset[:mode_count] = rng.uniform(0, 2 * np.pi, size=mode_count).astype(np.float32)
        phase_amps.append(phase_amp[None, :])
        phase_offsets.append(phase_offset[None, :])
        amp_amps.append(amp_amp[None, :])
        amp_offsets.append(amp_offset[None, :])
        mode_counts.append(mode_count)

    phase_amps = torch.as_tensor(np.concatenate(phase_amps, axis=0), device=device, dtype=dtype)
    phase_offsets = torch.as_tensor(np.concatenate(phase_offsets, axis=0), device=device, dtype=dtype)
    amp_amps = torch.as_tensor(np.concatenate(amp_amps, axis=0), device=device, dtype=dtype)
    amp_offsets = torch.as_tensor(np.concatenate(amp_offsets, axis=0), device=device, dtype=dtype)
    mode_counts = torch.as_tensor(mode_counts, device=device, dtype=torch.long)

    angles_phase = 2 * torch.pi * k[None, :, None] * x[None, None, :] + phase_offsets[:, :, None]
    angles_amp = 2 * torch.pi * k[None, :, None] * x[None, None, :] + amp_offsets[:, :, None]
    phase = (phase_amps[:, :, None] * torch.sin(angles_phase)).sum(dim=1)
    amp_noise = (amp_amps[:, :, None] * torch.sin(angles_amp)).sum(dim=1)
    phase = phase_scale * phase / (phase.std(dim=-1, keepdim=True, unbiased=False) + 1e-12)
    amp_noise = amp_noise / (amp_noise.std(dim=-1, keepdim=True, unbiased=False) + 1e-12)
    amp = torch.clamp(1.0 + amp_scale * amp_noise, min=0.05)
    return amp.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64)), mode_counts


@torch.no_grad()
def solve_complex_ginzburg_landau_batch_torch(
    seeds,
    nx=2048,
    domain_length=CGL_DOMAIN_LENGTH,
    T=CGL_T,
    c1=CGL_C1,
    c3=CGL_C3,
    num_modes=CGL_NUM_MODES,
    amp_scale=CGL_AMP_SCALE,
    phase_scale=CGL_PHASE_SCALE,
    save_steps=512,
    substeps_per_frame=CGL_SUBSTEPS_PER_FRAME,
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
    A_initial, mode_counts = _cgl_initial_batch(
        seeds, initial_grid_size, num_modes, amp_scale, phase_scale, device=device
    )
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
        return x.detach().cpu().numpy(), save_t, U, A_initial, mode_counts
    return x.detach().cpu().numpy(), save_t, U


def generate_cgl_image_pairs(
    seeds,
    sim_nx=2048,
    domain_length=CGL_DOMAIN_LENGTH,
    output_size=512,
    initial_grid_size=None,
    T=CGL_T,
    c1=CGL_C1,
    c3=CGL_C3,
    num_modes=CGL_NUM_MODES,
    amp_scale=CGL_AMP_SCALE,
    phase_scale=CGL_PHASE_SCALE,
    save_steps=512,
    substeps_per_frame=CGL_SUBSTEPS_PER_FRAME,
    sim_device=None,
    progress=False,
    progress_desc=None,
    progress_position=1,
    progress_update_every=8,
):
    seeds = list(seeds)
    initial_grid_size = int(initial_grid_size or output_size)
    _, _, U, A_initial, mode_counts = solve_complex_ginzburg_landau_batch_torch(
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
    mode_counts = mode_counts.detach().cpu().numpy()
    del U, U_img, A_initial, A0_img

    pairs = []
    for seed, solution_img, initial_row, amp_vmax_i, active_modes in zip(
        seeds, solution_rgb, initial_row_rgb, amp_vmax, mode_counts
    ):
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
            "active_modes": int(active_modes),
            "amp_scale": amp_scale,
            "phase_scale": phase_scale,
            "substeps_per_frame": substeps_per_frame,
            "amp_vmax": float(amp_vmax_i),
        }
        pairs.append((solution_img, repeat_rgb_row_view(initial_row, output_size), params))
    return pairs
