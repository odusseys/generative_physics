import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image


KS_GRID_SIZE = 512
KS_DOMAIN_LENGTH = 64.0 * np.pi
KS_DT = 1.0 / 16.0
KS_BURN_T = 50.0
KS_STEPS_PER_FRAME = 4
KS_INITIAL_NUM_MODES = 32
KS_INITIAL_DECAY = 1.0
KS_INITIAL_AMP = 1.0
KS_VALUE_BOUNDS = (-4.0, 4.0)
KS_CMAP_NAME = "inferno"
KS_REJECT_IF_CLIPPED = False
KS_MAX_TRIES = 64


def ks_operators(Nx=KS_GRID_SIZE, Lx=KS_DOMAIN_LENGTH, dt=KS_DT, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float64
    kx = torch.cat(
        [
            torch.arange(0, Nx // 2, device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=dtype),
            torch.arange(-Nx // 2 + 1, 0, device=device, dtype=dtype),
        ]
    )

    alpha = 2.0 * torch.pi * kx / Lx
    Lop = alpha**2 - alpha**4
    G = (-0.5j * alpha).to(torch.complex128)
    A_inv = (1.0 / (1.0 - 0.5 * dt * Lop)).to(torch.complex128)
    B = (1.0 + 0.5 * dt * Lop).to(torch.complex128)
    return G, A_inv, B


def random_spectral_ic(
    Nx=KS_GRID_SIZE,
    n_modes=KS_INITIAL_NUM_MODES,
    decay=KS_INITIAL_DECAY,
    amp=KS_INITIAL_AMP,
    generator=None,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float64
    j = torch.arange(Nx, device=device, dtype=dtype)
    modes = torch.arange(1, n_modes + 1, device=device, dtype=dtype)
    phase = 2.0 * torch.pi * modes[:, None] * j[None, :] / Nx
    a = torch.randn(n_modes, device=device, dtype=dtype, generator=generator) / modes**decay
    b = torch.randn(n_modes, device=device, dtype=dtype, generator=generator) / modes**decay

    u = (a[:, None] * torch.cos(phase) + b[:, None] * torch.sin(phase)).sum(dim=0)
    u = u - u.mean()
    return amp * u / u.abs().amax().clamp_min(1e-12)


def augment_ks_symmetry(u, generator=None):
    Nx = u.shape[-1]
    device = u.device
    shift = int(torch.randint(Nx, (), device=device, generator=generator).item())
    u = torch.roll(u, shifts=shift, dims=-1)

    if torch.rand((), device=device, generator=generator).item() < 0.5:
        idx = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.long),
                torch.arange(Nx - 1, 0, -1, device=device, dtype=torch.long),
            ]
        )
        u = -u[idx]

    return u - u.mean()


def ks_integrate_cnab2_torch(
    u0,
    Lx=KS_DOMAIN_LENGTH,
    dt=KS_DT,
    Nt=1020,
    nsave=KS_STEPS_PER_FRAME,
):
    u0 = u0.to(torch.float64)
    Nx = u0.numel()
    device = u0.device
    G, A_inv, B = ks_operators(Nx=Nx, Lx=Lx, dt=dt, device=device)

    uhat = torch.fft.fft(u0)
    uhat[0] = 0.0
    uhat[Nx // 2] = 0.0

    def nonlinear(uhat_i):
        u = torch.fft.ifft(uhat_i).real
        Nhat = G * torch.fft.fft(u * u)
        Nhat[0] = 0.0
        Nhat[Nx // 2] = 0.0
        return Nhat

    Nprev = nonlinear(uhat)
    Ncur = Nprev.clone()
    n_out = Nt // nsave + 1
    U = torch.empty(n_out, Nx, device=device, dtype=torch.float64)
    U[0] = torch.fft.ifft(uhat).real
    out_i = 1

    for n in range(1, Nt + 1):
        uhat = A_inv * (B * uhat + 1.5 * dt * Ncur - 0.5 * dt * Nprev)
        uhat[0] = 0.0
        uhat[Nx // 2] = 0.0
        Nprev = Ncur
        Ncur = nonlinear(uhat)

        if n % nsave == 0:
            U[out_i] = torch.fft.ifft(uhat).real
            out_i += 1

    return U


def array_to_flame_pil(U, value_bounds=KS_VALUE_BOUNDS, cmap_name=KS_CMAP_NAME):
    if torch.is_tensor(U):
        U = U.detach().cpu().numpy()

    vmin, vmax = value_bounds
    Z = np.clip((U - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = colormaps[cmap_name](Z)[..., :3]
    return Image.fromarray((255.0 * rgb).astype(np.uint8), mode="RGB")


def generate_ks_sample_no_bank(
    seed=0,
    Lx=KS_DOMAIN_LENGTH,
    Nx=KS_GRID_SIZE,
    image_size=256,
    dt=KS_DT,
    burn_T=KS_BURN_T,
    steps_per_frame=KS_STEPS_PER_FRAME,
    n_ic_modes=KS_INITIAL_NUM_MODES,
    ic_decay=KS_INITIAL_DECAY,
    ic_amp=KS_INITIAL_AMP,
    value_bounds=KS_VALUE_BOUNDS,
    cmap_name=KS_CMAP_NAME,
    reject_if_clipped=KS_REJECT_IF_CLIPPED,
    max_tries=KS_MAX_TRIES,
    return_data=False,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if Nx != image_size:
        raise ValueError("Use Nx=image_size for exact raw KS output.")

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    Nt_burn = int(round(burn_T / dt))
    Nt_horizon = (image_size - 1) * steps_per_frame
    horizon_T = Nt_horizon * dt

    for attempt in range(max_tries):
        u_seed = random_spectral_ic(
            Nx=Nx,
            n_modes=n_ic_modes,
            decay=ic_decay,
            amp=ic_amp,
            generator=generator,
            device=device,
        )

        if Nt_burn > 0:
            U_burn = ks_integrate_cnab2_torch(u_seed, Lx=Lx, dt=dt, Nt=Nt_burn, nsave=Nt_burn)
            u0 = U_burn[-1]
        else:
            u0 = u_seed

        u0 = augment_ks_symmetry(u0, generator=generator)
        U = ks_integrate_cnab2_torch(u0, Lx=Lx, dt=dt, Nt=Nt_horizon, nsave=steps_per_frame)
        mn = float(U.min().item())
        mx = float(U.max().item())
        clipped = (mn < value_bounds[0]) or (mx > value_bounds[1])

        if (not reject_if_clipped) or (not clipped):
            U_cpu = U.detach().cpu().numpy()
            u0_cpu = U_cpu[0].copy()
            init_panel = np.repeat(u0_cpu[None, :], image_size, axis=0)
            traj_img = array_to_flame_pil(U_cpu, value_bounds=value_bounds, cmap_name=cmap_name)
            init_img = array_to_flame_pil(init_panel, value_bounds=value_bounds, cmap_name=cmap_name)
            clip_mask = (U_cpu < value_bounds[0]) | (U_cpu > value_bounds[1])
            meta = {
                "seed": int(seed),
                "attempt": attempt + 1,
                "Lx": float(Lx),
                "Nx": int(Nx),
                "image_size": int(image_size),
                "dt": float(dt),
                "burn_T": float(burn_T),
                "horizon_T": float(horizon_T),
                "steps_per_frame": int(steps_per_frame),
                "value_bounds": tuple(float(x) for x in value_bounds),
                "u_min": mn,
                "u_max": mx,
                "clipped": bool(clipped),
                "clip_fraction": float(clip_mask.mean()),
                "shape": tuple(int(x) for x in U_cpu.shape),
            }
            if return_data:
                return traj_img, init_img, meta, U_cpu, u0_cpu
            return traj_img, init_img, meta

    raise RuntimeError(
        "No unclipped KS sample accepted. Increase value_bounds, set reject_if_clipped=False, or reduce horizon_T."
    )


def generate_ks_image_pairs(
    seeds,
    sim_nx=KS_GRID_SIZE,
    output_size=256,
    sim_device=None,
):
    pairs = []
    for seed in seeds:
        solution_img, initial_img, params = generate_ks_sample_no_bank(
            seed=int(seed),
            Nx=int(sim_nx),
            image_size=int(output_size),
            device=sim_device,
        )
        params = dict(params)
        params["pde"] = "ks"
        params["solution_target"] = "kuramoto_sivashinsky_trajectory"
        pairs.append((solution_img, initial_img, params))
    return pairs


generate_ks_256_sample_no_bank = generate_ks_sample_no_bank
