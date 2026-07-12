import numpy as np
import torch
from tqdm.auto import tqdm

from .numerics import downsample_solution_torch, periodic_linear_upsample_1d
from .rendering import repeat_rgb_row_view, scalar_to_rgb_uint8_torch


def solve_hard_burgers(nx=1024, T=0.8, nu=1e-4, num_modes=24, scale=1.0, seed=None, cfl=0.35, save_steps=256):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, nx, endpoint=False)
    dx = 1.0 / nx

    active_modes = int(rng.integers(1, int(num_modes) + 1))
    amps = rng.normal(size=active_modes) / np.arange(1, active_modes + 1)
    phases = rng.uniform(0, 2 * np.pi, size=active_modes)

    u = np.zeros_like(x)
    for k, a, p in zip(range(1, active_modes + 1), amps, phases):
        u += a * np.sin(2 * np.pi * k * x + p)
    u = scale * u / (np.std(u) + 1e-12)

    save_t = np.linspace(0, T, save_steps)
    U = np.empty((save_steps, nx), dtype=np.float32)
    U[0] = u
    save_idx = 1
    t = 0.0

    def pad(v, ng=3):
        return np.concatenate([v[-ng:], v, v[:ng]])

    def weno5_left(v, eps=1e-6):
        v0 = v[:-5]
        v1 = v[1:-4]
        v2 = v[2:-3]
        v3 = v[3:-2]
        v4 = v[4:-1]
        p0 = (2 * v0 - 7 * v1 + 11 * v2) / 6
        p1 = (-v1 + 5 * v2 + 2 * v3) / 6
        p2 = (2 * v2 + 5 * v3 - v4) / 6
        b0 = (13 / 12) * (v0 - 2 * v1 + v2) ** 2 + 0.25 * (v0 - 4 * v1 + 3 * v2) ** 2
        b1 = (13 / 12) * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - v3) ** 2
        b2 = (13 / 12) * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (3 * v2 - 4 * v3 + v4) ** 2
        a0 = 0.1 / (eps + b0) ** 2
        a1 = 0.6 / (eps + b1) ** 2
        a2 = 0.3 / (eps + b2) ** 2
        return (a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2)

    def weno5_right(v):
        return weno5_left(v[::-1])[::-1]

    def rhs(v):
        vp = pad(v)
        f = 0.5 * vp**2
        alpha = np.max(np.abs(vp)) + 1e-12
        flux = weno5_left(0.5 * (f + alpha * vp)) + weno5_right(0.5 * (f - alpha * vp))
        adv = -(flux[1:] - flux[:-1]) / dx
        if nu > 0:
            diff = nu * (np.roll(v, -1) - 2 * v + np.roll(v, 1)) / dx**2
            return adv + diff
        return adv

    def rk3(v, dt):
        k1 = rhs(v)
        v1 = v + dt * k1
        k2 = rhs(v1)
        v2 = 0.75 * v + 0.25 * (v1 + dt * k2)
        k3 = rhs(v2)
        return (1 / 3) * v + (2 / 3) * (v2 + dt * k3)

    while t < T - 1e-15:
        umax = max(np.max(np.abs(u)), 1e-12)
        dt_adv = cfl * dx / umax
        dt_diff = 0.25 * dx**2 / nu if nu > 0 else np.inf
        dt = min(dt_adv, dt_diff)
        if save_idx < save_steps:
            dt = min(dt, save_t[save_idx] - t)
        if t + dt > T:
            dt = T - t
        u = rk3(u, dt)
        t += dt
        while save_idx < save_steps and t >= save_t[save_idx] - 1e-14:
            U[save_idx] = u
            save_idx += 1
    return x, save_t, U


def _burgers_initial_batch(seeds, nx, num_modes, scale, device, dtype=torch.float32):
    if num_modes < 1:
        raise ValueError("burgers num_modes must be at least 1.")

    seeds = list(seeds)
    x = torch.arange(nx, device=device, dtype=dtype) / nx
    k = torch.arange(1, num_modes + 1, device=device, dtype=dtype)
    denom = np.arange(1, num_modes + 1, dtype=np.float32)
    amps = []
    phases = []
    mode_counts = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        active_modes = int(rng.integers(1, int(num_modes) + 1))
        amp = np.zeros(int(num_modes), dtype=np.float32)
        phase = np.zeros(int(num_modes), dtype=np.float32)
        amp[:active_modes] = rng.normal(size=active_modes).astype(np.float32) / denom[:active_modes]
        phase[:active_modes] = rng.uniform(0, 2 * np.pi, size=active_modes).astype(np.float32)
        amps.append(amp[None, :])
        phases.append(phase[None, :])
        mode_counts.append(active_modes)
    amps = torch.as_tensor(np.concatenate(amps, axis=0), device=device, dtype=dtype)
    phases = torch.as_tensor(np.concatenate(phases, axis=0), device=device, dtype=dtype)
    mode_counts = torch.as_tensor(mode_counts, device=device, dtype=torch.long)
    angles = 2 * torch.pi * k[None, :, None] * x[None, None, :] + phases[:, :, None]
    u = (amps[:, :, None] * torch.sin(angles)).sum(dim=1)
    return x, scale * u / (u.std(dim=-1, keepdim=True, unbiased=False) + 1e-12), mode_counts


def _weno5_left_torch(v, eps=1e-6):
    v0 = v[..., :-5]
    v1 = v[..., 1:-4]
    v2 = v[..., 2:-3]
    v3 = v[..., 3:-2]
    v4 = v[..., 4:-1]
    p0 = (2 * v0 - 7 * v1 + 11 * v2) / 6
    p1 = (-v1 + 5 * v2 + 2 * v3) / 6
    p2 = (2 * v2 + 5 * v3 - v4) / 6
    b0 = (13 / 12) * (v0 - 2 * v1 + v2) ** 2 + 0.25 * (v0 - 4 * v1 + 3 * v2) ** 2
    b1 = (13 / 12) * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - v3) ** 2
    b2 = (13 / 12) * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (3 * v2 - 4 * v3 + v4) ** 2
    a0 = 0.1 / (eps + b0) ** 2
    a1 = 0.6 / (eps + b1) ** 2
    a2 = 0.3 / (eps + b2) ** 2
    return (a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2)


def _weno5_right_torch(v):
    return torch.flip(_weno5_left_torch(torch.flip(v, dims=(-1,))), dims=(-1,))


def _burgers_rhs_torch(u, dx, nu):
    vp = torch.cat([u[..., -3:], u, u[..., :3]], dim=-1)
    f = 0.5 * vp**2
    alpha = vp.abs().amax(dim=-1, keepdim=True) + 1e-12
    flux = _weno5_left_torch(0.5 * (f + alpha * vp)) + _weno5_right_torch(0.5 * (f - alpha * vp))
    adv = -(flux[..., 1:] - flux[..., :-1]) / dx
    if nu > 0:
        diff = nu * (torch.roll(u, -1, dims=-1) - 2 * u + torch.roll(u, 1, dims=-1)) / dx**2
        return adv + diff
    return adv


def _burgers_rk3_torch(u, dt, dx, nu):
    k1 = _burgers_rhs_torch(u, dx, nu)
    u1 = u + dt * k1
    k2 = _burgers_rhs_torch(u1, dx, nu)
    u2 = 0.75 * u + 0.25 * (u1 + dt * k2)
    k3 = _burgers_rhs_torch(u2, dx, nu)
    return (1 / 3) * u + (2 / 3) * (u2 + dt * k3)


_BURGERS_RK3_COMPILED = {}


def _get_burgers_step_fn(device, dtype, use_compile=True):
    device = torch.device(device)
    if not use_compile or device.type != "cuda" or not hasattr(torch, "compile"):
        return _burgers_rk3_torch
    key = (device.type, str(dtype))
    if key not in _BURGERS_RK3_COMPILED:
        _BURGERS_RK3_COMPILED[key] = torch.compile(
            _burgers_rk3_torch,
            fullgraph=True,
            options={"triton.cudagraphs": False},
        )
    return _BURGERS_RK3_COMPILED[key]


@torch.no_grad()
def solve_hard_burgers_batch_torch(
    seeds,
    nx=2048,
    T=0.15,
    nu=3e-5,
    num_modes=24,
    scale=1.5,
    cfl=0.35,
    save_steps=256,
    initial_grid_size=None,
    device=None,
    dtype=torch.float32,
    adaptive_dt=False,
    compile_step=True,
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
    _, u_initial, mode_counts = _burgers_initial_batch(
        seeds, initial_grid_size, num_modes, scale, device=device, dtype=dtype
    )
    u = periodic_linear_upsample_1d(u_initial, nx)
    x = torch.arange(nx, device=device, dtype=dtype) / nx
    dx = 1.0 / nx
    save_t_cpu = np.linspace(0, T, save_steps, dtype=np.float64)
    U = torch.empty((len(seeds), save_steps, nx), device=device, dtype=dtype)
    U[:, 0] = u

    initial_umax = max(float(u.abs().amax().detach().cpu()), 1e-12)
    dt_adv_base = cfl * dx / initial_umax
    dt_diff = 0.25 * dx**2 / nu if nu > 0 else np.inf
    dt_base = min(dt_adv_base, dt_diff)

    t = 0.0
    save_idx = 1
    step_fn = _get_burgers_step_fn(device, dtype, use_compile=compile_step)
    dt_tensor = torch.empty((), device=device, dtype=dtype)
    pending_progress = 0
    pbar = tqdm(
        total=save_steps - 1,
        desc=progress_desc or "RK saves",
        position=progress_position,
        leave=True,
        mininterval=0.5,
        disable=not progress,
    )
    try:
        while t < T - 1e-15:
            if adaptive_dt:
                umax = max(float(u.abs().amax().detach().cpu()), 1e-12)
                dt = min(cfl * dx / umax, dt_diff)
            else:
                dt = dt_base
            if save_idx < save_steps:
                dt = min(dt, save_t_cpu[save_idx] - t)
            if t + dt > T:
                dt = T - t
            dt_tensor.fill_(dt)
            u = step_fn(u, dt_tensor, dx, nu)
            t += dt
            saved_now = 0
            while save_idx < save_steps and t >= save_t_cpu[save_idx] - 1e-14:
                U[:, save_idx] = u
                save_idx += 1
                saved_now += 1
            if saved_now:
                pending_progress += saved_now
                if pending_progress >= progress_update_every or save_idx >= save_steps:
                    pbar.update(pending_progress)
                    pending_progress = 0
                    pbar.set_postfix(t=f"{t:.3f}")
    finally:
        if pending_progress:
            pbar.update(pending_progress)
        pbar.close()

    if return_initial_grid:
        return x.detach().cpu().numpy(), save_t_cpu, U, u_initial, mode_counts
    return x.detach().cpu().numpy(), save_t_cpu, U


def generate_burgers_image_pairs(
    seeds,
    sim_nx=2048,
    output_size=512,
    initial_grid_size=None,
    T=0.15,
    nu=3e-5,
    num_modes=24,
    scale=1.5,
    save_steps=512,
    cfl=0.35,
    cmap_name="viridis",
    sim_device=None,
    adaptive_dt=False,
    compile_step=True,
    progress=False,
    progress_desc=None,
    progress_position=1,
    progress_update_every=8,
):
    seeds = list(seeds)
    initial_grid_size = int(initial_grid_size or output_size)
    _, _, U, u_initial, mode_counts = solve_hard_burgers_batch_torch(
        seeds=seeds,
        nx=sim_nx,
        T=T,
        nu=nu,
        num_modes=num_modes,
        scale=scale,
        cfl=cfl,
        save_steps=save_steps,
        initial_grid_size=initial_grid_size,
        device=sim_device,
        adaptive_dt=adaptive_dt,
        compile_step=compile_step,
        progress=progress,
        progress_desc=progress_desc,
        progress_position=progress_position,
        progress_update_every=progress_update_every,
        return_initial_grid=True,
    )
    U_img = downsample_solution_torch(U, output_size=output_size)
    u0_img = periodic_linear_upsample_1d(u_initial, output_size)
    U_img[:, 0] = u0_img
    bound = U_img.abs().float().amax(dim=(1, 2)).clamp_min(1e-6)
    solution_rgb = scalar_to_rgb_uint8_torch(U_img.float(), -bound, bound, cmap_name=cmap_name).detach().cpu().numpy()
    initial_row_rgb = (
        scalar_to_rgb_uint8_torch(u0_img[:, None, :].float(), -bound, bound, cmap_name=cmap_name)[:, 0]
        .detach()
        .cpu()
        .numpy()
    )
    bound = bound.detach().cpu().numpy()
    mode_counts = mode_counts.detach().cpu().numpy()
    del U, U_img, u_initial, u0_img

    pairs = []
    for seed, solution_img, initial_row, bound_i, active_modes in zip(
        seeds, solution_rgb, initial_row_rgb, bound, mode_counts
    ):
        params = {
            "seed": seed,
            "sim_nx": sim_nx,
            "output_size": output_size,
            "initial_grid_size": initial_grid_size,
            "T": T,
            "nu": nu,
            "num_modes": num_modes,
            "active_modes": int(active_modes),
            "scale": scale,
            "vmin": -float(bound_i),
            "vmax": float(bound_i),
        }
        pairs.append((solution_img, repeat_rgb_row_view(initial_row, output_size), params))
    return pairs


def generate_burgers_image_pair(seed=None, nx=2048, nt=512, output_size=512, **kwargs):
    return generate_burgers_image_pairs([seed], sim_nx=nx, output_size=output_size, save_steps=nt, **kwargs)[0]
