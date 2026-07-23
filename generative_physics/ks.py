import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


KS_GRID_SIZE = 512
KS_RENDER_SIZE = 512
KS_TIME_FRAMES = 512
KS_DOMAIN_LENGTH = 64.0 * np.pi
KS_DT = 0.01
KS_BURN_T = 50.0
KS_STEPS_PER_FRAME = 25
KS_HORIZON_T = 50.0
KS_INITIAL_NUM_MODES = 32
KS_INITIAL_DECAY = 1.0
KS_INITIAL_AMP = 1.0
KS_VALUE_BOUNDS = (-4.0, 4.0)
KS_HILBERT_ORDER = 3
KS_HILBERT_CMAP_NAME = f"hilbert{KS_HILBERT_ORDER}"
KS_CMAP_NAME = "gray"
KS_CONDITION_ENCODING = "y_constant"
KS_REJECT_IF_CLIPPED = False
KS_MAX_TRIES = 64
KS_RANDOM_SAMPLING_MAE = 0.20495


_KS_HILBERT_LUT_CACHE = {}
_KS_INFERNO_LUT = torch.from_numpy(
    colormaps["inferno"](np.linspace(0.0, 1.0, 256, dtype=np.float32))[..., :3].astype(np.float32)
)
_KS_INFERNO_LUMINANCE = (
    _KS_INFERNO_LUT * torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)
).sum(-1).numpy()


def _hilbert_d2xy(order, d):
    """Map an integer Hilbert index to integer coordinates on a 2**order grid."""
    n = 1 << order
    x = 0
    y = 0
    s = 1
    t = int(d)

    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def make_hilbert_lut(order=KS_HILBERT_ORDER, device="cpu"):
    device = torch.device(device)
    key = (int(order), str(device))
    lut = _KS_HILBERT_LUT_CACHE.get(key)
    if lut is None:
        n = 1 << int(order)
        points = [_hilbert_d2xy(order, d) for d in range(n * n)]
        lut = torch.tensor(points, dtype=torch.float32, device=device) / (n - 1)
        _KS_HILBERT_LUT_CACHE[key] = lut
    return lut


def hilbert_encode(t, lut):
    t = t.clamp(0.0, 1.0)
    u = t * (lut.shape[0] - 1)
    i0 = torch.floor(u).long()
    i1 = torch.clamp(i0 + 1, max=lut.shape[0] - 1)
    alpha = (u - i0).unsqueeze(-1)
    return (1.0 - alpha) * lut[i0] + alpha * lut[i1]


def hilbert_decode(xy, lut, chunk_size=4096):
    original_shape = xy.shape[:-1]
    xy = xy.reshape(-1, 2)
    starts = lut[:-1]
    segments = lut[1:] - starts
    segment_norm2 = (segments * segments).sum(-1).clamp_min(1e-12)
    decoded = []

    for batch in xy.split(chunk_size):
        delta = batch[:, None, :] - starts[None, :, :]
        alpha = (
            (delta * segments[None, :, :]).sum(-1) / segment_norm2[None, :]
        ).clamp(0.0, 1.0)
        projected = starts[None, :, :] + alpha[..., None] * segments[None, :, :]
        distance2 = ((batch[:, None, :] - projected) ** 2).sum(-1)
        best_segment = distance2.argmin(dim=1)
        best_alpha = alpha[torch.arange(batch.shape[0], device=batch.device), best_segment]
        decoded.append((best_segment.float() + best_alpha) / (lut.shape[0] - 1))

    return torch.cat(decoded).reshape(original_shape)


def _hilbert_order(cmap_name):
    prefix = "hilbert"
    if not str(cmap_name).startswith(prefix):
        raise ValueError(f"Unsupported KS color scale: {cmap_name!r}")
    return int(str(cmap_name)[len(prefix):])


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


def ks_integrate_cnab2_batch(
    initial_states,
    Lx=KS_DOMAIN_LENGTH,
    dt=KS_DT,
    Nt=1020,
    nsave=KS_STEPS_PER_FRAME,
):
    """Integrate a batch of KS states with the same CNAB2 discretization."""
    initial_states = torch.as_tensor(initial_states)
    if initial_states.ndim == 1:
        initial_states = initial_states.unsqueeze(0)
    if initial_states.ndim != 2:
        raise ValueError("initial_states must have shape [batch, x].")
    if Nt < 0 or nsave < 1 or Nt % nsave != 0:
        raise ValueError("Nt must be nonnegative and divisible by positive nsave.")

    initial_states = initial_states.to(torch.float64)
    batch_size, nx = initial_states.shape
    G, A_inv, B = ks_operators(Nx=nx, Lx=Lx, dt=dt, device=initial_states.device)
    uhat = torch.fft.fft(initial_states, dim=-1)
    uhat[..., 0] = 0.0
    uhat[..., nx // 2] = 0.0

    def nonlinear(current_uhat):
        values = torch.fft.ifft(current_uhat, dim=-1).real
        result = G * torch.fft.fft(values.square(), dim=-1)
        result[..., 0] = 0.0
        result[..., nx // 2] = 0.0
        return result

    nonlinear_previous = nonlinear(uhat)
    nonlinear_current = nonlinear_previous.clone()
    trajectory = torch.empty(
        (batch_size, Nt // nsave + 1, nx),
        device=initial_states.device,
        dtype=torch.float64,
    )
    trajectory[:, 0] = torch.fft.ifft(uhat, dim=-1).real
    output_index = 1
    for step in range(1, Nt + 1):
        uhat = A_inv * (
            B * uhat
            + 1.5 * dt * nonlinear_current
            - 0.5 * dt * nonlinear_previous
        )
        uhat[..., 0] = 0.0
        uhat[..., nx // 2] = 0.0
        nonlinear_previous = nonlinear_current
        nonlinear_current = nonlinear(uhat)
        if step % nsave == 0:
            trajectory[:, output_index] = torch.fft.ifft(uhat, dim=-1).real
            output_index += 1
    return trajectory


def array_to_flame_pil(U, value_bounds=KS_VALUE_BOUNDS, cmap_name=KS_CMAP_NAME):
    rgb = array_to_scalar_rgb(U, value_bounds=value_bounds, cmap_name=cmap_name)
    rgb_uint8 = torch.round(torch.from_numpy(rgb) * 255.0).clamp(0, 255).to(torch.uint8).numpy()
    return Image.fromarray(rgb_uint8, mode="RGB")


def array_to_hilbert_rgb(U, value_bounds=KS_VALUE_BOUNDS, cmap_name=KS_HILBERT_CMAP_NAME):
    """Encode scalar values as float32 Hilbert RGB without 8-bit quantization."""
    vmin, vmax = value_bounds
    values = torch.as_tensor(U, dtype=torch.float32, device="cpu")
    normalized = ((values - vmin) / (vmax - vmin)).clamp(0.0, 1.0)
    lut = make_hilbert_lut(_hilbert_order(cmap_name), device=normalized.device)
    xy = hilbert_encode(normalized, lut)
    blue = torch.zeros_like(xy[..., :1])
    return torch.cat([xy, blue], dim=-1).numpy()


def array_to_inferno_rgb(U, value_bounds=KS_VALUE_BOUNDS):
    """Encode scalar values as continuous float32 inferno RGB."""
    vmin, vmax = value_bounds
    values = torch.as_tensor(U, dtype=torch.float32, device="cpu")
    normalized = ((values - vmin) / (vmax - vmin)).clamp(0.0, 1.0)
    return hilbert_encode(normalized, _KS_INFERNO_LUT).numpy()


def array_to_scalar_rgb(U, value_bounds=KS_VALUE_BOUNDS, cmap_name=KS_CMAP_NAME):
    if cmap_name == "gray":
        vmin, vmax = value_bounds
        values = torch.as_tensor(U, dtype=torch.float32, device="cpu")
        normalized = ((values - vmin) / (vmax - vmin)).clamp(0.0, 1.0)
        return normalized[..., None].repeat(1, 1, 3).numpy()
    if cmap_name == "inferno":
        return array_to_inferno_rgb(U, value_bounds=value_bounds)
    if str(cmap_name).startswith("hilbert"):
        return array_to_hilbert_rgb(U, value_bounds=value_bounds, cmap_name=cmap_name)
    raise ValueError(f"Unsupported KS color scale: {cmap_name!r}")


def initial_condition_to_y_constant_rgb(
    u0,
    image_size,
    value_bounds=KS_VALUE_BOUNDS,
):
    """Render a grayscale 1D state identically along the image y-axis."""
    image_size = int(image_size)
    values = torch.as_tensor(u0, dtype=torch.float32, device="cpu").reshape(1, 1, -1)
    if values.shape[-1] != image_size:
        values = F.interpolate(values, size=image_size, mode="linear", align_corners=True)
    values = values[0, 0]
    vmin, vmax = value_bounds
    normalized = ((values - vmin) / (vmax - vmin)).clamp(0.0, 1.0)
    return normalized[None, :, None].repeat(image_size, 1, 3).numpy()


def encode_ks_initial_condition(
    u0,
    image_size,
    encoding=KS_CONDITION_ENCODING,
    value_bounds=KS_VALUE_BOUNDS,
    cmap_name=KS_CMAP_NAME,
):
    if encoding == "y_constant":
        return initial_condition_to_y_constant_rgb(
            u0,
            image_size=image_size,
            value_bounds=value_bounds,
        )
    if encoding == "hilbert":
        panel = np.repeat(np.asarray(u0)[None, :], int(image_size), axis=0)
        panel = resize_ks_scalar_render(panel, image_size)
        return array_to_hilbert_rgb(
            panel,
            value_bounds=value_bounds,
            cmap_name=KS_HILBERT_CMAP_NAME,
        )
    raise ValueError(
        f"Unsupported KS condition encoding {encoding!r}; expected 'hilbert' or 'y_constant'."
    )


def resize_ks_scalar_render(U, image_size):
    """Resample a fixed numerical trajectory only for square image rendering."""
    values = torch.as_tensor(U, dtype=torch.float32, device="cpu")
    if values.shape == (image_size, image_size):
        return values.numpy()
    rendered = F.interpolate(
        values[None, None],
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=True,
    )
    return rendered[0, 0].numpy()


def flame_image_to_normalized_array(image, cmap_name=KS_CMAP_NAME):
    """Decode KS RGB rendering to normalized scalar values in [0, 1]."""
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    else:
        rgb = np.asarray(image, dtype=np.float32)
        if rgb.size and float(np.nanmax(rgb)) > 1.0:
            rgb = rgb / 255.0
    rgb = np.array(rgb[..., :3], dtype=np.float32, copy=True)
    if cmap_name == "gray":
        return np.clip(rgb.mean(axis=-1), 0.0, 1.0).astype(np.float32)
    if cmap_name == "inferno":
        luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        return np.interp(
            luminance,
            _KS_INFERNO_LUMINANCE,
            np.linspace(0.0, 1.0, len(_KS_INFERNO_LUMINANCE), dtype=np.float32),
        ).astype(np.float32)
    rgb = torch.from_numpy(rgb)
    lut = make_hilbert_lut(_hilbert_order(cmap_name), device=rgb.device)
    return hilbert_decode(rgb[..., :2], lut).numpy()


def ks_effective_discretization(Nx=KS_GRID_SIZE, image_size=KS_RENDER_SIZE, steps_per_frame=KS_STEPS_PER_FRAME):
    """Choose a direct square simulation for renders above the base grid size."""
    Nx = int(Nx)
    image_size = int(image_size)
    steps_per_frame = int(steps_per_frame)
    if image_size > KS_GRID_SIZE:
        Nx = max(Nx, image_size)
        time_frames = image_size
    else:
        time_frames = KS_TIME_FRAMES
    dt = KS_HORIZON_T / ((time_frames - 1) * steps_per_frame)
    return Nx, time_frames, dt


def generate_ks_sample_no_bank(
    seed=0,
    Lx=KS_DOMAIN_LENGTH,
    Nx=KS_GRID_SIZE,
    image_size=KS_RENDER_SIZE,
    dt=KS_DT,
    burn_T=KS_BURN_T,
    steps_per_frame=KS_STEPS_PER_FRAME,
    n_ic_modes=KS_INITIAL_NUM_MODES,
    ic_decay=KS_INITIAL_DECAY,
    ic_amp=KS_INITIAL_AMP,
    value_bounds=KS_VALUE_BOUNDS,
    cmap_name=KS_CMAP_NAME,
    condition_encoding=KS_CONDITION_ENCODING,
    reject_if_clipped=KS_REJECT_IF_CLIPPED,
    max_tries=KS_MAX_TRIES,
    return_data=False,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    requested_Nx = int(Nx)
    Nx, time_frames, dt = ks_effective_discretization(
        Nx=Nx,
        image_size=image_size,
        steps_per_frame=steps_per_frame,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    Nt_burn = int(round(burn_T / dt))
    Nt_horizon = (time_frames - 1) * steps_per_frame
    horizon_T = KS_HORIZON_T

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
            rendered_U = resize_ks_scalar_render(U_cpu, image_size)
            traj_img = array_to_scalar_rgb(rendered_U, value_bounds=value_bounds, cmap_name=cmap_name)
            init_img = encode_ks_initial_condition(
                u0_cpu,
                image_size=image_size,
                encoding=condition_encoding,
                value_bounds=value_bounds,
                cmap_name=cmap_name,
            )
            clip_mask = (U_cpu < value_bounds[0]) | (U_cpu > value_bounds[1])
            meta = {
                "seed": int(seed),
                "attempt": attempt + 1,
                "Lx": float(Lx),
                "Nx": int(Nx),
                "requested_Nx": requested_Nx,
                "image_size": int(image_size),
                "time_frames": int(time_frames),
                "dt": float(dt),
                "burn_T": float(burn_T),
                "horizon_T": float(horizon_T),
                "steps_per_frame": int(steps_per_frame),
                "condition_encoding": str(condition_encoding),
                "value_bounds": tuple(float(x) for x in value_bounds),
                "u_min": mn,
                "u_max": mx,
                "clipped": bool(clipped),
                "clip_fraction": float(clip_mask.mean()),
                "shape": tuple(int(x) for x in U_cpu.shape),
                "render_shape": tuple(int(x) for x in rendered_U.shape),
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
    output_size=KS_RENDER_SIZE,
    condition_encoding=KS_CONDITION_ENCODING,
    sim_device=None,
):
    pairs = []
    for seed in seeds:
        solution_img, initial_img, params = generate_ks_sample_no_bank(
            seed=int(seed),
            Nx=int(sim_nx),
            image_size=int(output_size),
            condition_encoding=condition_encoding,
            device=sim_device,
        )
        params = dict(params)
        params["pde"] = "ks"
        params["solution_target"] = "kuramoto_sivashinsky_trajectory"
        pairs.append((solution_img, initial_img, params))
    return pairs


generate_ks_256_sample_no_bank = generate_ks_sample_no_bank
