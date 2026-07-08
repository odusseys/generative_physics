from heapq import heappop, heappush

import numpy as np
from PIL import Image

from .colorization import apply_midpoint_contrast
from .rendering import array_to_pil


EIKONAL_MIN_COMPONENTS = 1
EIKONAL_MAX_COMPONENTS = 12
EIKONAL_SIGMA_MIN = 0.05
EIKONAL_SIGMA_MAX = 0.25
EIKONAL_WEIGHT_SIGMA = 0.25
EIKONAL_REFRACTIVE_INDEX_MEAN = 0.5
EIKONAL_REFRACTIVE_INDEX_CONTRAST = 0.35
EIKONAL_REFRACTIVE_INDEX_MIN = 0.05
EIKONAL_REFRACTIVE_INDEX_MAX = 1.0
EIKONAL_SOLUTION_VMAX = 0.75
EIKONAL_SOLUTION_GAMMA = 0.6
EIKONAL_SOLUTION_MIDPOINT_CONTRAST = 1.5
EIKONAL_CMAP_NAME = "viridis"
EIKONAL_SOLUTION_CMAP_NAME = "coolwarm"


def generate_anisotropic_normal_refractive_index(
    size=256,
    rng=None,
    min_components=EIKONAL_MIN_COMPONENTS,
    max_components=EIKONAL_MAX_COMPONENTS,
    sigma_min=EIKONAL_SIGMA_MIN,
    sigma_max=EIKONAL_SIGMA_MAX,
    weight_sigma=EIKONAL_WEIGHT_SIGMA,
    refractive_index_mean=EIKONAL_REFRACTIVE_INDEX_MEAN,
    refractive_index_contrast=EIKONAL_REFRACTIVE_INDEX_CONTRAST,
    refractive_index_min=EIKONAL_REFRACTIVE_INDEX_MIN,
    refractive_index_max=EIKONAL_REFRACTIVE_INDEX_MAX,
):
    rng = np.random.default_rng(rng)
    coords = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(coords, coords, indexing="xy")

    n_components = rng.integers(min_components, max_components + 1)
    field = np.zeros((size, size), dtype=np.float64)

    for _ in range(n_components):
        cx, cy = rng.uniform(0.0, 1.0, size=2)
        sx = rng.uniform(float(sigma_min), float(sigma_max))
        sy = rng.uniform(float(sigma_min), float(sigma_max))
        theta = rng.uniform(0.0, 2.0 * np.pi)

        dx = X - cx
        dy = Y - cy
        c = np.cos(theta)
        s = np.sin(theta)

        u = c * dx + s * dy
        v = -s * dx + c * dy
        component = np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
        component /= 2.0 * np.pi * sx * sy

        weight = rng.lognormal(mean=0.0, sigma=float(weight_sigma))
        field += weight * component

    field = np.maximum(field, 1e-12)
    field /= field.mean() + 1e-12
    field = np.exp(float(refractive_index_contrast) * np.log(field))
    for _ in range(8):
        field *= float(refractive_index_mean) / (field.mean() + 1e-12)
        field = np.clip(field, float(refractive_index_min), float(refractive_index_max))
    return field.astype(np.float32), int(n_components)


def solve_eikonal_fast_marching(n):
    height, width = n.shape
    h = 1.0 / (max(height, width) - 1)
    slowness = np.asarray(n, dtype=np.float64)

    T = np.full((height, width), np.inf, dtype=np.float64)
    accepted = np.zeros((height, width), dtype=bool)

    si, sj = height // 2, width // 2
    T[si, sj] = 0.0
    accepted[si, sj] = True

    heap = []

    def update(i, j):
        x_vals = []
        y_vals = []

        if i > 0 and accepted[i - 1, j]:
            x_vals.append(T[i - 1, j])
        if i + 1 < height and accepted[i + 1, j]:
            x_vals.append(T[i + 1, j])
        if j > 0 and accepted[i, j - 1]:
            y_vals.append(T[i, j - 1])
        if j + 1 < width and accepted[i, j + 1]:
            y_vals.append(T[i, j + 1])

        vals = []
        if x_vals:
            vals.append(min(x_vals))
        if y_vals:
            vals.append(min(y_vals))
        if not vals:
            return np.inf

        vals.sort()
        tau = h * slowness[i, j]
        a = vals[0]
        if len(vals) == 1:
            return a + tau

        b = vals[1]
        if b - a >= tau:
            return a + tau

        disc = 2.0 * tau * tau - (b - a) ** 2
        return 0.5 * (a + b + np.sqrt(max(disc, 0.0)))

    def push_update(i, j):
        if 0 <= i < height and 0 <= j < width and not accepted[i, j]:
            new_t = update(i, j)
            if new_t < T[i, j]:
                T[i, j] = new_t
                heappush(heap, (new_t, i, j))

    push_update(si - 1, sj)
    push_update(si + 1, sj)
    push_update(si, sj - 1)
    push_update(si, sj + 1)

    while heap:
        t, i, j = heappop(heap)
        if accepted[i, j] or t != T[i, j]:
            continue

        accepted[i, j] = True

        push_update(i - 1, j)
        push_update(i + 1, j)
        push_update(i, j - 1)
        push_update(i, j + 1)

    return T.astype(np.float32)


def _resize_rgb(rgb, output_size):
    image = Image.fromarray(np.ascontiguousarray(rgb))
    if image.size != (output_size, output_size):
        image = image.resize((output_size, output_size), Image.Resampling.BICUBIC)
    return np.asarray(image.convert("RGB")).copy()


def _colormap_midpoint_value(vmin, vmax, gamma):
    span = max(float(vmax) - float(vmin), 1e-6)
    return float(vmin) + span * (0.5 ** (1.0 / float(gamma)))


def _scalar_to_rgb(array, vmin, vmax, output_size, cmap_name, gamma=1.0, midpoint_contrast=1.0):
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if midpoint_contrast <= 0:
        raise ValueError("midpoint_contrast must be positive")
    if gamma != 1.0 or midpoint_contrast != 1.0:
        span = max(float(vmax) - float(vmin), 1e-6)
        z = np.clip((np.asarray(array, dtype=np.float32) - float(vmin)) / span, 0.0, 1.0)
        z = z ** float(gamma)
        z = apply_midpoint_contrast(z, contrast=midpoint_contrast, midpoint=0.5)
        array = float(vmin) + span * z
    rgb = np.asarray(array_to_pil(array, vmin=vmin, vmax=vmax, cmap_name=cmap_name).convert("RGB"))
    return _resize_rgb(rgb, output_size)


def generate_eikonal_image_pairs(
    seeds,
    sim_nx=256,
    output_size=256,
    min_components=EIKONAL_MIN_COMPONENTS,
    max_components=EIKONAL_MAX_COMPONENTS,
    sigma_min=EIKONAL_SIGMA_MIN,
    sigma_max=EIKONAL_SIGMA_MAX,
    weight_sigma=EIKONAL_WEIGHT_SIGMA,
    refractive_index_mean=EIKONAL_REFRACTIVE_INDEX_MEAN,
    refractive_index_contrast=EIKONAL_REFRACTIVE_INDEX_CONTRAST,
    refractive_index_min=EIKONAL_REFRACTIVE_INDEX_MIN,
    refractive_index_max=EIKONAL_REFRACTIVE_INDEX_MAX,
    solution_vmax=EIKONAL_SOLUTION_VMAX,
    solution_gamma=EIKONAL_SOLUTION_GAMMA,
    solution_midpoint_contrast=EIKONAL_SOLUTION_MIDPOINT_CONTRAST,
    cmap_name=EIKONAL_CMAP_NAME,
    solution_cmap_name=EIKONAL_SOLUTION_CMAP_NAME,
):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if sim_nx < 16:
        raise ValueError("eikonal grid size must be at least 16")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if min_components < 1 or max_components < min_components:
        raise ValueError("eikonal components must satisfy 1 <= min_components <= max_components")
    if sigma_min <= 0 or sigma_max < sigma_min:
        raise ValueError("eikonal sigmas must satisfy 0 < sigma_min <= sigma_max")
    if weight_sigma < 0:
        raise ValueError("eikonal_weight_sigma must be nonnegative")
    if refractive_index_mean <= 0:
        raise ValueError("eikonal_refractive_index_mean must be positive")
    if refractive_index_contrast < 0:
        raise ValueError("eikonal_refractive_index_contrast must be nonnegative")
    if refractive_index_min >= refractive_index_max:
        raise ValueError("eikonal refractive_index_min must be < refractive_index_max")
    if solution_vmax <= 0:
        raise ValueError("eikonal_solution_vmax must be positive")
    if solution_gamma <= 0:
        raise ValueError("eikonal_solution_gamma must be positive")
    if solution_midpoint_contrast <= 0:
        raise ValueError("eikonal_solution_midpoint_contrast must be positive")

    solution_vmax = float(solution_vmax)
    solution_midpoint_value = _colormap_midpoint_value(0.0, solution_vmax, solution_gamma)
    pairs = []
    for seed in seeds:
        refractive_index, n_components = generate_anisotropic_normal_refractive_index(
            size=sim_nx,
            rng=seed,
            min_components=min_components,
            max_components=max_components,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            weight_sigma=weight_sigma,
            refractive_index_mean=refractive_index_mean,
            refractive_index_contrast=refractive_index_contrast,
            refractive_index_min=refractive_index_min,
            refractive_index_max=refractive_index_max,
        )
        propagation_time = solve_eikonal_fast_marching(refractive_index)

        refractive_img = _scalar_to_rgb(
            refractive_index,
            vmin=refractive_index_min,
            vmax=refractive_index_max,
            output_size=output_size,
            cmap_name=cmap_name,
        )
        time_img = _scalar_to_rgb(
            propagation_time,
            vmin=0.0,
            vmax=solution_vmax,
            output_size=output_size,
            cmap_name=solution_cmap_name,
            gamma=solution_gamma,
            midpoint_contrast=solution_midpoint_contrast,
        )
        params = {
            "seed": int(seed),
            "pde": "eikonal",
            "sim_nx": int(sim_nx),
            "output_size": int(output_size),
            "min_components": int(min_components),
            "max_components": int(max_components),
            "num_components": int(n_components),
            "sigma_min": float(sigma_min),
            "sigma_max": float(sigma_max),
            "weight_sigma": float(weight_sigma),
            "refractive_index_mean_target": float(refractive_index_mean),
            "refractive_index_mean_actual": float(np.mean(refractive_index)),
            "refractive_index_contrast": float(refractive_index_contrast),
            "refractive_index_vmin": float(refractive_index_min),
            "refractive_index_vmax": float(refractive_index_max),
            "solution_vmin": 0.0,
            "solution_vmax": float(solution_vmax),
            "solution_normalization": "fixed",
            "solution_gamma": float(solution_gamma),
            "solution_midpoint_value": float(solution_midpoint_value),
            "solution_midpoint_contrast": float(solution_midpoint_contrast),
            "refractive_index_cmap": str(cmap_name),
            "solution_cmap": str(solution_cmap_name),
            "source": "center",
            "equation": "|grad T| = n",
            "solver": "first_order_fast_marching",
        }
        pairs.append((time_img, refractive_img, params))

    return pairs
