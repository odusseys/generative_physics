import numpy as np
from PIL import Image

from .rendering import array_to_pil


CONDITION_NAMES = ("source", "target", "cost")
OT_MIN_COMPONENTS = 2
OT_MAX_COMPONENTS = 6
OT_SIGMA_MIN = 0.05
OT_SIGMA_MAX = 0.18
OT_DENSITY_FLOOR = 1e-8
OT_EPSILON = 0.055
OT_SINKHORN_ITERS = 90
OT_COST_STRENGTH = 8.0
OT_COST_SIGMA_MULTIPLIER = 1.5
OT_COST_PATH_SAMPLES = 9
OT_LOGPROB_VMIN = -33.67282751800
OT_LOGPROB_VMAX = -9.152773795274
OT_SOURCE_CMAP_NAME = "viridis"
OT_TARGET_CMAP_NAME = "viridis"
OT_COST_CMAP_NAME = "magma"
OT_POTENTIAL_CMAP_NAME = "coolwarm"
OT_POTENTIAL_VMIN = -0.4
OT_POTENTIAL_VMAX = 0.4


def make_grid(n):
    x = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    return xx, yy, np.stack([xx, yy], axis=-1)


def sample_mixture_params(
    rng,
    min_components=OT_MIN_COMPONENTS,
    max_components=OT_MAX_COMPONENTS,
    sigma_min=OT_SIGMA_MIN,
    sigma_max=OT_SIGMA_MAX,
):
    k = int(rng.integers(int(min_components), int(max_components) + 1))
    weights = rng.random(k)
    weights /= weights.sum()

    params = []
    for weight in weights:
        mu = rng.uniform(0.1, 0.9, size=2)
        s1 = rng.uniform(float(sigma_min), float(sigma_max))
        s2 = rng.uniform(float(sigma_min), float(sigma_max))
        theta = rng.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s], [s, c]])
        sigma = rotation @ np.diag([s1**2, s2**2]) @ rotation.T
        params.append((float(weight), mu.astype(np.float64), np.linalg.inv(sigma)))
    return params


def eval_mixture(params, n, floor=OT_DENSITY_FLOOR):
    xx, yy, _ = make_grid(n)
    z = np.zeros((int(n), int(n)), dtype=np.float64)

    for weight, mu, sigma_inv in params:
        dx = np.stack([xx - mu[0], yy - mu[1]], axis=-1)
        q = np.einsum("...i,ij,...j->...", dx, sigma_inv, dx)
        z += weight * np.exp(-0.5 * q)

    return z + float(floor)


def normalize_density(z):
    z = np.maximum(np.asarray(z, dtype=np.float64), 0.0)
    total = z.sum()
    if total <= 0.0:
        return np.full_like(z, 1.0 / z.size)
    return z / total


def normalize01(z):
    z = np.asarray(z, dtype=np.float32)
    z = z - float(z.min())
    scale = float(z.max())
    if scale <= 0.0:
        return np.zeros_like(z, dtype=np.float32)
    return (z / scale).astype(np.float32)


def _resize_scalar_field(array, output_size):
    array = np.asarray(array, dtype=np.float32)
    if array.shape[:2] == (int(output_size), int(output_size)):
        return array
    image = Image.fromarray(np.ascontiguousarray(array))
    return np.asarray(image.resize((int(output_size), int(output_size)), Image.Resampling.BICUBIC), dtype=np.float32)


def _pairwise_squared_distances(n):
    _, _, grid = make_grid(n)
    pts = grid.reshape(-1, 2)
    diff = pts[:, None, :] - pts[None, :, :]
    return (diff**2).sum(-1)


def _bilinear_sample_unit_square(field, x, y):
    n = field.shape[0]
    x = np.clip(x, 0.0, 1.0) * (n - 1)
    y = np.clip(y, 0.0, 1.0) * (n - 1)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, n - 1)
    y1 = np.clip(y0 + 1, 0, n - 1)
    wx = x - x0
    wy = y - y0

    return (
        (1.0 - wx) * (1.0 - wy) * field[y0, x0]
        + wx * (1.0 - wy) * field[y0, x1]
        + (1.0 - wx) * wy * field[y1, x0]
        + wx * wy * field[y1, x1]
    )


def _pairwise_path_average_cost(normalized_cost, num_samples=OT_COST_PATH_SAMPLES):
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("ot_cost_path_samples must be positive.")

    n = normalized_cost.shape[0]
    _, _, grid = make_grid(n)
    pts = grid.reshape(-1, 2).astype(np.float32)
    x0 = pts[:, 0][:, None]
    y0 = pts[:, 1][:, None]
    x1 = pts[:, 0][None, :]
    y1 = pts[:, 1][None, :]

    field = np.asarray(normalized_cost, dtype=np.float32)
    avg = np.zeros((pts.shape[0], pts.shape[0]), dtype=np.float32)
    if num_samples == 1:
        t_values = np.array([0.5], dtype=np.float32)
    else:
        t_values = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
    for t in t_values:
        x = (1.0 - t) * x0 + t * x1
        y = (1.0 - t) * y0 + t * y1
        avg += _bilinear_sample_unit_square(field, x, y).astype(np.float32)
    return avg / float(num_samples)


def _normalized_pairwise_transport_cost(normalized_cost, cost_strength, cost_path_samples):
    w = np.clip(normalized_cost, 0.0, 1.0)
    c = _pairwise_squared_distances(w.shape[0])
    path_cost = _pairwise_path_average_cost(w, num_samples=cost_path_samples)
    c = c * (1.0 + float(cost_strength) * path_cost)
    return c / max(float(c.max()), 1e-12)


def _sinkhorn_source_potential_from_cost(source, target, transport_cost, epsilon, num_iters):
    a = normalize_density(source).reshape(-1)
    b = normalize_density(target).reshape(-1)
    kernel = np.exp(-np.asarray(transport_cost, dtype=np.float64) / float(epsilon))
    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(int(num_iters)):
        u = a / (kernel @ v + 1e-300)
        v = b / (kernel.T @ u + 1e-300)

    return float(epsilon) * np.log(u + 1e-300)


def sinkhorn_transport_potential(
    source,
    target,
    normalized_cost,
    epsilon=OT_EPSILON,
    num_iters=OT_SINKHORN_ITERS,
    cost_strength=OT_COST_STRENGTH,
    cost_path_samples=OT_COST_PATH_SAMPLES,
):
    if epsilon <= 0.0:
        raise ValueError("ot_epsilon must be positive.")
    if num_iters <= 0:
        raise ValueError("ot_sinkhorn_iters must be positive.")
    if cost_strength < 0.0:
        raise ValueError("ot_cost_strength must be non-negative.")

    source = normalize_density(source)
    target = normalize_density(target)
    normalized_cost = np.asarray(normalized_cost, dtype=np.float32)
    n = source.shape[0]
    if (
        source.shape != target.shape
        or source.shape != normalized_cost.shape
        or source.ndim != 2
        or source.shape[0] != source.shape[1]
    ):
        raise ValueError("source, target, and cost must be square arrays with matching shapes.")
    if normalized_cost.min() < -1e-6 or normalized_cost.max() > 1.0 + 1e-6:
        raise ValueError("normalized_cost must be scaled to [0, 1] before OT simulation.")

    c = _normalized_pairwise_transport_cost(normalized_cost, cost_strength, cost_path_samples)
    potential = _sinkhorn_source_potential_from_cost(source, target, c, epsilon, num_iters)
    potential -= potential.mean()
    return potential.reshape(n, n).astype(np.float32)


def sinkhorn_debiased_source_potential(
    source,
    target,
    normalized_cost,
    epsilon=OT_EPSILON,
    num_iters=OT_SINKHORN_ITERS,
    cost_strength=OT_COST_STRENGTH,
    cost_path_samples=OT_COST_PATH_SAMPLES,
):
    if epsilon <= 0.0:
        raise ValueError("ot_epsilon must be positive.")
    if num_iters <= 0:
        raise ValueError("ot_sinkhorn_iters must be positive.")
    if cost_strength < 0.0:
        raise ValueError("ot_cost_strength must be non-negative.")
    if cost_path_samples <= 0:
        raise ValueError("ot_cost_path_samples must be positive.")

    source = normalize_density(source)
    target = normalize_density(target)
    normalized_cost = np.asarray(normalized_cost, dtype=np.float32)
    n = source.shape[0]
    if (
        source.shape != target.shape
        or source.shape != normalized_cost.shape
        or source.ndim != 2
        or source.shape[0] != source.shape[1]
    ):
        raise ValueError("source, target, and cost must be square arrays with matching shapes.")
    if normalized_cost.min() < -1e-6 or normalized_cost.max() > 1.0 + 1e-6:
        raise ValueError("normalized_cost must be scaled to [0, 1] before OT simulation.")

    c = _normalized_pairwise_transport_cost(normalized_cost, cost_strength, cost_path_samples)
    cross_potential = _sinkhorn_source_potential_from_cost(source, target, c, epsilon, num_iters)
    self_potential = _sinkhorn_source_potential_from_cost(source, source, c, epsilon, num_iters)
    potential = cross_potential - self_potential
    potential -= potential.mean()
    return potential.reshape(n, n).astype(np.float32)


def _scalar_image(array, vmin, vmax, output_size, cmap_name):
    array = _resize_scalar_field(array, output_size)
    return np.asarray(array_to_pil(array, vmin=vmin, vmax=vmax, cmap_name=cmap_name).convert("RGB")).copy()


def _log_probability_image(probability, vmin, vmax, output_size, cmap_name):
    probability = normalize_density(probability)
    log_probability = np.log(np.maximum(probability, 1e-300))
    return _scalar_image(log_probability, vmin, vmax, output_size, cmap_name), log_probability


def _fixed_signed_image(
    array,
    output_size,
    cmap_name,
    vmin=OT_POTENTIAL_VMIN,
    vmax=OT_POTENTIAL_VMAX,
):
    if vmin >= vmax:
        raise ValueError("potential_vmin must be less than potential_vmax.")
    return _scalar_image(array, vmin, vmax, output_size, cmap_name)


def generate_ot_image_pairs(
    seeds,
    sim_nx=24,
    output_size=256,
    min_components=OT_MIN_COMPONENTS,
    max_components=OT_MAX_COMPONENTS,
    sigma_min=OT_SIGMA_MIN,
    sigma_max=OT_SIGMA_MAX,
    density_floor=OT_DENSITY_FLOOR,
    epsilon=OT_EPSILON,
    sinkhorn_iters=OT_SINKHORN_ITERS,
    cost_strength=OT_COST_STRENGTH,
    cost_sigma_multiplier=OT_COST_SIGMA_MULTIPLIER,
    cost_path_samples=OT_COST_PATH_SAMPLES,
    logprob_vmin=OT_LOGPROB_VMIN,
    logprob_vmax=OT_LOGPROB_VMAX,
    source_cmap_name=OT_SOURCE_CMAP_NAME,
    target_cmap_name=OT_TARGET_CMAP_NAME,
    cost_cmap_name=OT_COST_CMAP_NAME,
    potential_cmap_name=OT_POTENTIAL_CMAP_NAME,
    potential_vmin=OT_POTENTIAL_VMIN,
    potential_vmax=OT_POTENTIAL_VMAX,
):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if sim_nx < 4:
        raise ValueError("ot_solve_grid_size must be at least 4.")
    if output_size <= 0:
        raise ValueError("output_size must be positive.")
    if min_components < 1 or max_components < min_components:
        raise ValueError("ot components must satisfy 1 <= min_components <= max_components.")
    if sigma_min <= 0.0 or sigma_max < sigma_min:
        raise ValueError("ot sigmas must satisfy 0 < sigma_min <= sigma_max.")
    if density_floor < 0.0:
        raise ValueError("ot_density_floor must be non-negative.")
    if cost_sigma_multiplier <= 0.0:
        raise ValueError("ot_cost_sigma_multiplier must be positive.")
    if cost_path_samples <= 0:
        raise ValueError("ot_cost_path_samples must be positive.")
    if logprob_vmin >= logprob_vmax:
        raise ValueError("ot_logprob_vmin must be less than ot_logprob_vmax.")
    if potential_vmin >= potential_vmax:
        raise ValueError("ot_potential_vmin must be less than ot_potential_vmax.")

    records = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        source_params = sample_mixture_params(rng, min_components, max_components, sigma_min, sigma_max)
        target_params = sample_mixture_params(rng, min_components, max_components, sigma_min, sigma_max)
        cost_sigma_min = float(sigma_min) * float(cost_sigma_multiplier)
        cost_sigma_max = float(sigma_max) * float(cost_sigma_multiplier)
        cost_params = sample_mixture_params(rng, min_components, max_components, cost_sigma_min, cost_sigma_max)

        source_solve = normalize_density(eval_mixture(source_params, sim_nx, floor=density_floor))
        target_solve = normalize_density(eval_mixture(target_params, sim_nx, floor=density_floor))
        raw_cost_solve = eval_mixture(cost_params, sim_nx, floor=0.0)
        normalized_cost_solve = normalize01(raw_cost_solve)
        potential = sinkhorn_debiased_source_potential(
            source_solve,
            target_solve,
            normalized_cost_solve,
            epsilon=epsilon,
            num_iters=sinkhorn_iters,
            cost_strength=cost_strength,
            cost_path_samples=cost_path_samples,
        )

        source_display = normalize_density(eval_mixture(source_params, output_size, floor=density_floor))
        target_display = normalize_density(eval_mixture(target_params, output_size, floor=density_floor))
        cost_display = normalize01(eval_mixture(cost_params, output_size, floor=0.0))

        source_image, source_logprob = _log_probability_image(
            source_display,
            logprob_vmin,
            logprob_vmax,
            output_size,
            source_cmap_name,
        )
        target_image, target_logprob = _log_probability_image(
            target_display,
            logprob_vmin,
            logprob_vmax,
            output_size,
            target_cmap_name,
        )
        cost_image = _scalar_image(cost_display, 0.0, 1.0, output_size, cost_cmap_name)
        potential_image = _fixed_signed_image(
            potential,
            output_size,
            potential_cmap_name,
            vmin=potential_vmin,
            vmax=potential_vmax,
        )

        params = {
            "seed": int(seed),
            "pde": "ot",
            "sim_nx": int(sim_nx),
            "solve_grid_size": int(sim_nx),
            "output_size": int(output_size),
            "min_components": int(min_components),
            "max_components": int(max_components),
            "source_components": len(source_params),
            "target_components": len(target_params),
            "cost_components": len(cost_params),
            "sigma_min": float(sigma_min),
            "sigma_max": float(sigma_max),
            "cost_sigma_min": float(cost_sigma_min),
            "cost_sigma_max": float(cost_sigma_max),
            "cost_sigma_multiplier": float(cost_sigma_multiplier),
            "density_floor": float(density_floor),
            "epsilon": float(epsilon),
            "sinkhorn_iters": int(sinkhorn_iters),
            "cost_strength": float(cost_strength),
            "cost_mode": "straight_path_average",
            "cost_path_samples": int(cost_path_samples),
            "source_vmin": float(logprob_vmin),
            "source_vmax": float(logprob_vmax),
            "target_vmin": float(logprob_vmin),
            "target_vmax": float(logprob_vmax),
            "probability_image_transform": "log_probability",
            "probability_normalization": "sum_to_one",
            "probability_log_vmin": float(logprob_vmin),
            "probability_log_vmax": float(logprob_vmax),
            "source_logprob_min": float(source_logprob.min()),
            "source_logprob_max": float(source_logprob.max()),
            "target_logprob_min": float(target_logprob.min()),
            "target_logprob_max": float(target_logprob.max()),
            "cost_vmin": 0.0,
            "cost_vmax": 1.0,
            "cost_normalization": "per_sample_minmax_before_sinkhorn",
            "solution_vmin": float(potential_vmin),
            "solution_vmax": float(potential_vmax),
            "solution_normalization": "fixed_clip",
            "solution_target": "debiased_source_dual_potential",
            "solution_raw_min": float(potential.min()),
            "solution_raw_max": float(potential.max()),
            "source_cmap": str(source_cmap_name),
            "target_cmap": str(target_cmap_name),
            "cost_cmap": str(cost_cmap_name),
            "solution_cmap": str(potential_cmap_name),
            "equation": "debiased_entropic_optimal_transport",
            "solver": "sinkhorn",
        }
        records.append(
            {
                "initial": source_image,
                "condition_images": [source_image, target_image, cost_image],
                "condition_names": list(CONDITION_NAMES),
                "solution": potential_image,
                "params": params,
            }
        )
    return records
