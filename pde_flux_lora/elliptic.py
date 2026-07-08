import json
import os
from pathlib import Path

import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from PIL import Image

from .color_ramps import make_perceptual_color_ramps, scalar_to_rgb_uint16


COEFFICIENT_NAMES = ("a20", "a11", "a02", "a10", "a01", "a00", "f")
FIELD_NAMES = (*COEFFICIENT_NAMES, "u")
COLOR_RAMPS = dict(
    zip(
        FIELD_NAMES,
        make_perceptual_color_ramps(
            len(FIELD_NAMES),
            n_grid=4096,
            sweeps_deg=(0,),
            objective="independent",
            seed=0,
        ),
    )
)
DEFAULT_CACHE_DIR = Path("/home/ubuntu/datasets/elliptic")
ELLIPTIC_CACHE_VERSION = 5
ELLIPTIC_MAX_CYCLES = 5.5
# These ranges make the lower-order and mixed coefficients visible in the
# solution while keeping the reaction-coupled forcing in a common [0, 1] scale.
ELLIPTIC_A_MIN = 0.015
ELLIPTIC_A_MAX = 0.12
ELLIPTIC_MIXED_RHO = 0.35
ELLIPTIC_FIRST_ORDER_SCALE = 0.08
ELLIPTIC_REACTION_MIN = 0.5
ELLIPTIC_REACTION_MAX = 10.0
ELLIPTIC_SOLUTION_VMIN = 0.0
ELLIPTIC_SOLUTION_VMAX = 1.0


def make_frequency_grid(L):
    fx = np.fft.fftfreq(L, d=1.0 / L)
    fy = np.fft.rfftfreq(L, d=1.0 / L)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    return np.sqrt(FX**2 + FY**2)


def normalize01(x, rng):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-10:
        return np.full_like(x, rng.random())
    return (x - lo) / (hi - lo)


def bandwidth_field(L, rng, m, R, max_cycles=ELLIPTIC_MAX_CYCLES):
    if m <= 1e-12:
        return np.full((L, L), rng.random(), dtype=np.float32)
    z = rng.random((L, L), dtype=np.float32)
    Z = sfft.rfft2(z)
    cutoff = max(1e-6, m * max_cycles)
    H = np.exp(-0.5 * (R / cutoff) ** 8).astype(np.float32)
    x = sfft.irfft2(Z * H, s=(L, L)).real.astype(np.float32)
    return normalize01(x, rng).astype(np.float32)


def signed_bandwidth_field(L, rng, m, R, max_cycles=ELLIPTIC_MAX_CYCLES):
    return 2.0 * bandwidth_field(L, rng, m, R, max_cycles) - 1.0


def generate_elliptic_pde_data(
    L,
    seed,
    max_cycles=ELLIPTIC_MAX_CYCLES,
    a_min=ELLIPTIC_A_MIN,
    a_max=ELLIPTIC_A_MAX,
    mixed_rho=ELLIPTIC_MIXED_RHO,
    first_order_scale=ELLIPTIC_FIRST_ORDER_SCALE,
    reaction_min=ELLIPTIC_REACTION_MIN,
    reaction_max=ELLIPTIC_REACTION_MAX,
):
    rng = np.random.default_rng(seed)
    R = make_frequency_grid(L)

    mix = {k: float(rng.random()) for k in COEFFICIENT_NAMES}

    a20_u = bandwidth_field(L, rng, mix["a20"], R, max_cycles)
    a02_u = bandwidth_field(L, rng, mix["a02"], R, max_cycles)
    eta = signed_bandwidth_field(L, rng, mix["a11"], R, max_cycles)

    a20 = a_min + (a_max - a_min) * a20_u
    a02 = a_min + (a_max - a_min) * a02_u
    a11 = 2.0 * mixed_rho * np.sqrt(a20 * a02) * eta

    a10 = first_order_scale * signed_bandwidth_field(L, rng, mix["a10"], R, max_cycles)
    a01 = first_order_scale * signed_bandwidth_field(L, rng, mix["a01"], R, max_cycles)

    q = reaction_min + (reaction_max - reaction_min) * bandwidth_field(L, rng, mix["a00"], R, max_cycles)
    a00 = -q

    r = bandwidth_field(L, rng, mix["f"], R, max_cycles)
    f = a00 * r

    return {
        "a20": a20.astype(np.float32),
        "a11": a11.astype(np.float32),
        "a02": a02.astype(np.float32),
        "a10": a10.astype(np.float32),
        "a01": a01.astype(np.float32),
        "a00": a00.astype(np.float32),
        "f": f.astype(np.float32),
        "r": r.astype(np.float32),
        "boundary": np.zeros((L, L), dtype=np.float32),
        "mix": mix,
        "max_cycles": max_cycles,
    }


def assemble_sparse_system(data):
    L = data["f"].shape[0]
    h = 1.0 / (L - 1)
    h2 = h * h
    m = L - 2

    I, J = np.meshgrid(np.arange(1, L - 1), np.arange(1, L - 1), indexing="ij")
    rows = ((I - 1) * m + (J - 1)).ravel()
    rhs = np.array(data["f"][1:-1, 1:-1], dtype=np.float64).ravel().copy()
    boundary = data["boundary"]

    a20 = data["a20"][1:-1, 1:-1]
    a11 = data["a11"][1:-1, 1:-1]
    a02 = data["a02"][1:-1, 1:-1]
    a10 = data["a10"][1:-1, 1:-1]
    a01 = data["a01"][1:-1, 1:-1]
    a00 = data["a00"][1:-1, 1:-1]

    terms = [
        (1, 0, a20 / h2),
        (0, 0, -2 * a20 / h2),
        (-1, 0, a20 / h2),
        (0, 1, a02 / h2),
        (0, 0, -2 * a02 / h2),
        (0, -1, a02 / h2),
        (1, 1, a11 / (4 * h2)),
        (1, -1, -a11 / (4 * h2)),
        (-1, 1, -a11 / (4 * h2)),
        (-1, -1, a11 / (4 * h2)),
        (1, 0, a10 / (2 * h)),
        (-1, 0, -a10 / (2 * h)),
        (0, 1, a01 / (2 * h)),
        (0, -1, -a01 / (2 * h)),
        (0, 0, a00),
    ]

    row_chunks, col_chunks, val_chunks = [], [], []
    for di, dj, coef in terms:
        Ti, Tj = I + di, J + dj
        coef_flat = coef.astype(np.float64).ravel()
        mask = (Ti >= 1) & (Ti <= L - 2) & (Tj >= 1) & (Tj <= L - 2)
        mask_flat = mask.ravel()

        if mask_flat.any():
            row_chunks.append(rows[mask_flat])
            col_chunks.append(((Ti[mask] - 1) * m + (Tj[mask] - 1)).ravel())
            val_chunks.append(coef_flat[mask_flat])

        if (~mask_flat).any():
            rhs[rows[~mask_flat]] -= coef_flat[~mask_flat] * boundary[Ti[~mask], Tj[~mask]].ravel()

    A = sp.csr_matrix(
        (np.concatenate(val_chunks), (np.concatenate(row_chunks), np.concatenate(col_chunks))),
        shape=(m * m, m * m),
    )
    return A, rhs


def solve_elliptic_pde(data):
    L = data["f"].shape[0]
    A, rhs = assemble_sparse_system(data)
    u_int = spla.spsolve(A, rhs)
    m = L - 2
    u = data["boundary"].copy()
    u[1:-1, 1:-1] = u_int.reshape(m, m).astype(np.float32)
    return u


def ellipticity_min(data):
    a20, a11, a02 = data["a20"], data["a11"], data["a02"]
    return 0.5 * ((a20 + a02) - np.sqrt((a20 - a02) ** 2 + a11**2))


def _field_ranges(a_min, a_max, mixed_rho, first_order_scale, reaction_min, reaction_max, solution_vmin, solution_vmax):
    mixed_abs = 2.0 * mixed_rho * a_max
    return {
        "a20": (a_min, a_max),
        "a11": (-mixed_abs, mixed_abs),
        "a02": (a_min, a_max),
        "a10": (-first_order_scale, first_order_scale),
        "a01": (-first_order_scale, first_order_scale),
        "a00": (-reaction_max, -reaction_min),
        "f": (-reaction_max, 0.0),
        "u": (solution_vmin, solution_vmax),
    }


def _resize_scalar_field(array, output_size):
    array = np.asarray(array, dtype=np.float32)
    if array.shape[:2] == (output_size, output_size):
        return array
    pil = Image.fromarray(np.ascontiguousarray(array))
    return np.asarray(pil.resize((output_size, output_size), Image.Resampling.BICUBIC), dtype=np.float32)


def _cache_signature(
    sim_nx,
    output_size,
    max_cycles,
    a_min,
    a_max,
    mixed_rho,
    first_order_scale,
    reaction_min,
    reaction_max,
    solution_vmin,
    solution_vmax,
):
    return {
        "cache_version": ELLIPTIC_CACHE_VERSION,
        "field_names": list(FIELD_NAMES),
        "color_ramps": {name: COLOR_RAMPS[name].params for name in FIELD_NAMES},
        "render_dtype": "uint16",
        "sim_nx": int(sim_nx),
        "output_size": int(output_size),
        "max_cycles": float(max_cycles),
        "a_min": float(a_min),
        "a_max": float(a_max),
        "mixed_rho": float(mixed_rho),
        "first_order_scale": float(first_order_scale),
        "reaction_min": float(reaction_min),
        "reaction_max": float(reaction_max),
        "solution_vmin": float(solution_vmin),
        "solution_vmax": float(solution_vmax),
    }


def _cache_path(cache_dir, index):
    return Path(cache_dir) / f"{int(index):08d}.npz"


def _load_cached_record(cache_dir, index, signature):
    path = _cache_path(cache_dir, index)
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as payload:
            cached_signature = json.loads(str(payload["signature_json"].item()))
            if cached_signature != signature:
                return None

            condition_images = [image for image in payload["condition_images"]]
            condition_names = [str(name) for name in payload["condition_names"]]
            solution = payload["solution"]
            params = json.loads(str(payload["params_json"].item()))
    except Exception:
        return None

    return {
        "condition_images": condition_images,
        "condition_names": condition_names,
        "solution": solution,
        "params": params,
    }


def _save_cached_record(cache_dir, index, signature, record):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, index)
    tmp_path = cache_dir / f".{int(index):08d}.{os.getpid()}.tmp"
    try:
        with tmp_path.open("wb") as f:
            np.savez(
                f,
                condition_images=np.stack(record["condition_images"], axis=0).astype(np.uint16, copy=False),
                condition_names=np.asarray(record["condition_names"]),
                solution=np.asarray(record["solution"], dtype=np.uint16),
                params_json=np.asarray(json.dumps(record["params"], sort_keys=True)),
                signature_json=np.asarray(json.dumps(signature, sort_keys=True)),
            )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _generate_elliptic_record(
    seed,
    ranges,
    signature,
    sim_nx,
    output_size,
    max_cycles,
    a_min,
    a_max,
    mixed_rho,
    first_order_scale,
    reaction_min,
    reaction_max,
    solution_vmin,
    solution_vmax,
    cache_dir,
):
    seed = int(seed)
    if cache_dir is not None:
        cached = _load_cached_record(cache_dir, seed, signature)
        if cached is not None:
            return cached

    data = generate_elliptic_pde_data(
        L=sim_nx,
        seed=seed,
        max_cycles=max_cycles,
        a_min=a_min,
        a_max=a_max,
        mixed_rho=mixed_rho,
        first_order_scale=first_order_scale,
        reaction_min=reaction_min,
        reaction_max=reaction_max,
    )
    u = solve_elliptic_pde(data)

    condition_images = []
    for name in COEFFICIENT_NAMES:
        vmin, vmax = ranges[name]
        field = _resize_scalar_field(data[name], output_size)
        condition_images.append(scalar_to_rgb_uint16(field, vmin, vmax, COLOR_RAMPS[name]))

    u_vmin, u_vmax = ranges["u"]
    solution_field = _resize_scalar_field(u, output_size)
    solution_image = scalar_to_rgb_uint16(solution_field, u_vmin, u_vmax, COLOR_RAMPS["u"])

    params = {
        "seed": int(seed),
        "pde": "elliptic",
        "sim_nx": int(sim_nx),
        "output_size": int(output_size),
        "max_cycles": float(max_cycles),
        "a_min": float(a_min),
        "a_max": float(a_max),
        "mixed_rho": float(mixed_rho),
        "first_order_scale": float(first_order_scale),
        "reaction_min": float(reaction_min),
        "reaction_max": float(reaction_max),
        "solution_vmin": float(solution_vmin),
        "solution_vmax": float(solution_vmax),
        "lambda_min": float(ellipticity_min(data).min()),
    }
    params.update({f"mix_{name}": float(data["mix"][name]) for name in COEFFICIENT_NAMES})
    record = {
        "condition_images": condition_images,
        "condition_names": list(COEFFICIENT_NAMES),
        "solution": solution_image,
        "params": params,
    }
    if cache_dir is not None:
        _save_cached_record(cache_dir, seed, signature, record)
    return record


def generate_elliptic_image_pairs(
    seeds,
    sim_nx=256,
    output_size=256,
    max_cycles=ELLIPTIC_MAX_CYCLES,
    a_min=ELLIPTIC_A_MIN,
    a_max=ELLIPTIC_A_MAX,
    mixed_rho=ELLIPTIC_MIXED_RHO,
    first_order_scale=ELLIPTIC_FIRST_ORDER_SCALE,
    reaction_min=ELLIPTIC_REACTION_MIN,
    reaction_max=ELLIPTIC_REACTION_MAX,
    solution_vmin=ELLIPTIC_SOLUTION_VMIN,
    solution_vmax=ELLIPTIC_SOLUTION_VMAX,
    cache_dir=DEFAULT_CACHE_DIR,
):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if sim_nx < 3:
        raise ValueError("elliptic_grid_size must be at least 3.")
    if output_size <= 0:
        raise ValueError("output_size must be positive.")
    if a_min <= 0 or a_max <= 0 or a_min > a_max:
        raise ValueError("elliptic_a_min/a_max must be positive with min <= max.")
    if mixed_rho < 0:
        raise ValueError("elliptic_mixed_rho must be non-negative.")
    if first_order_scale < 0:
        raise ValueError("elliptic_first_order_scale must be non-negative.")
    if reaction_min <= 0 or reaction_max <= 0 or reaction_min > reaction_max:
        raise ValueError("elliptic_reaction_min/max must be positive with min <= max.")
    if solution_vmin >= solution_vmax:
        raise ValueError("elliptic_solution_vmin must be less than elliptic_solution_vmax.")

    ranges = _field_ranges(
        a_min,
        a_max,
        mixed_rho,
        first_order_scale,
        reaction_min,
        reaction_max,
        solution_vmin,
        solution_vmax,
    )
    signature = _cache_signature(
        sim_nx,
        output_size,
        max_cycles,
        a_min,
        a_max,
        mixed_rho,
        first_order_scale,
        reaction_min,
        reaction_max,
        solution_vmin,
        solution_vmax,
    )

    records = []
    for seed in seeds:
        records.append(
            _generate_elliptic_record(
                seed=seed,
                ranges=ranges,
                signature=signature,
                sim_nx=sim_nx,
                output_size=output_size,
                max_cycles=max_cycles,
                a_min=a_min,
                a_max=a_max,
                mixed_rho=mixed_rho,
                first_order_scale=first_order_scale,
                reaction_min=reaction_min,
                reaction_max=reaction_max,
                solution_vmin=solution_vmin,
                solution_vmax=solution_vmax,
                cache_dir=cache_dir,
            )
        )
    return records
