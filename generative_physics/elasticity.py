import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .bezier import generate_random_bezier_curve
from .colorization import cyclic_value_colorize


ELASTICITY_STRESS_BASE_COLORS = (
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
)

ELASTICITY_RENDER_LAMBDA = 1.0
ELASTICITY_RENDER_MU = 1.0
ELASTICITY_LAMBDA = None
ELASTICITY_MU = None
ELASTICITY_LAMBDA_MIN = 0.5
ELASTICITY_LAMBDA_MAX = 3.0
ELASTICITY_MU_MIN = 0.5
ELASTICITY_MU_MAX = 3.0
ELASTICITY_MIN_POINTS = 3
ELASTICITY_MAX_POINTS = 10
ELASTICITY_HOLE_BOX_FRACTION = 100.0 / 256.0
ELASTICITY_SAMPLES_PER_SEGMENT = 16
ELASTICITY_HANDLE_SCALE = 0.12
ELASTICITY_FAR_FIELD_STRESS = None
ELASTICITY_RENDER_HORIZONTAL_STRESS = 1.0
ELASTICITY_RENDER_VERTICAL_STRESS = 0.0
ELASTICITY_HORIZONTAL_STRESS = None
ELASTICITY_VERTICAL_STRESS = None
ELASTICITY_HORIZONTAL_STRESS_MIN = 0.25
ELASTICITY_HORIZONTAL_STRESS_MAX = 1.5
ELASTICITY_VERTICAL_STRESS_MIN = 0.25
ELASTICITY_VERTICAL_STRESS_MAX = 1.5
ELASTICITY_PLANE_STRESS = True
ELASTICITY_STRESS_PERCENTILE = 99.0
ELASTICITY_STRESS_GAMMA = 0.55
ELASTICITY_CONDITIONING_NAMES = ("lambda", "mu", "sigma_x", "sigma_y")
ELASTICITY_CONDITIONING_TRANSFORMS = ("log", "log", "linear", "linear")


def polygon_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def point_in_polygon(poly, xs, ys):
    inside = np.zeros(xs.shape, dtype=bool)
    x0 = poly[:, 0]
    y0 = poly[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)

    for xa, ya, xb, yb in zip(x0, y0, x1, y1):
        crosses = (ya > ys) != (yb > ys)
        x_at_y = (xb - xa) * (ys - ya) / (yb - ya + 1e-30) + xa
        inside ^= crosses & (xs < x_at_y)

    return inside


def sample_hole_polygon(
    width,
    hole_box,
    seed,
    min_points=ELASTICITY_MIN_POINTS,
    max_points=ELASTICITY_MAX_POINTS,
    samples_per_segment=ELASTICITY_SAMPLES_PER_SEGMENT,
    handle_scale=ELASTICITY_HANDLE_SCALE,
):
    center = 0.5 * (int(width) - 1)
    min_area = 0.12 * float(hole_box) * float(hole_box)
    min_extent = 0.45 * float(hole_box)

    for k in range(100):
        _, curve = generate_random_bezier_curve(
            min_points=min_points,
            max_points=max_points,
            samples_per_segment=samples_per_segment,
            seed=int(seed) + 1009 * k,
            handle_scale=handle_scale,
        )
        if np.linalg.norm(curve[0] - curve[-1]) < 1e-12:
            curve = curve[:-1]

        scale = (0.5 * float(hole_box)) / max(float(np.abs(curve).max()), 1e-12)
        poly = curve * scale + np.array([center, center])

        if abs(polygon_area(poly)) >= min_area and (poly.max(axis=0) - poly.min(axis=0)).min() >= min_extent:
            return poly

    raise RuntimeError("could not sample a non-degenerate hole polygon")


def triangle_B(coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    two_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

    if two_area <= 0:
        coords = coords[[0, 2, 1]]
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        two_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    B = np.array(
        [
            [b1, 0.0, b2, 0.0, b3, 0.0],
            [0.0, c1, 0.0, c2, 0.0, c3],
            [c1, b1, c2, b2, c3, b3],
        ],
        dtype=np.float64,
    ) / two_area

    return B, 0.5 * two_area


def _seeded_rng(seed, stream=0):
    return np.random.default_rng(np.random.SeedSequence([int(seed), 0xE1A57, int(stream)]))


def _validate_positive_range(name, min_value, max_value):
    min_value = float(min_value)
    max_value = float(max_value)
    if min_value <= 0 or max_value <= 0:
        raise ValueError(f"{name} bounds must be positive.")
    if min_value > max_value:
        raise ValueError(f"{name}_min must be <= {name}_max.")
    return min_value, max_value


def _validate_stress_range(name, min_value, max_value):
    min_value = float(min_value)
    max_value = float(max_value)
    if min_value < 0 or max_value < 0:
        raise ValueError(f"{name} bounds must be non-negative.")
    if min_value > max_value:
        raise ValueError(f"{name}_min must be <= {name}_max.")
    return min_value, max_value


def _sample_log_uniform(rng, min_value, max_value):
    if min_value == max_value:
        return float(min_value)
    return float(np.exp(rng.uniform(np.log(min_value), np.log(max_value))))


def _sample_uniform(rng, min_value, max_value):
    if min_value == max_value:
        return float(min_value)
    return float(rng.uniform(min_value, max_value))


def render_random_hole_elasticity_high_contrast(
    lmbda=ELASTICITY_RENDER_LAMBDA,
    mu=ELASTICITY_RENDER_MU,
    width=256,
    seed=0,
    hole_box=None,
    min_points=ELASTICITY_MIN_POINTS,
    max_points=ELASTICITY_MAX_POINTS,
    samples_per_segment=ELASTICITY_SAMPLES_PER_SEGMENT,
    handle_scale=ELASTICITY_HANDLE_SCALE,
    far_field_stress=ELASTICITY_FAR_FIELD_STRESS,
    horizontal_stress=ELASTICITY_RENDER_HORIZONTAL_STRESS,
    vertical_stress=ELASTICITY_RENDER_VERTICAL_STRESS,
    plane_stress=ELASTICITY_PLANE_STRESS,
    stress_percentile=ELASTICITY_STRESS_PERCENTILE,
    stress_gamma=ELASTICITY_STRESS_GAMMA,
    stress_base_colors=ELASTICITY_STRESS_BASE_COLORS,
):
    W = int(width)
    lmbda = float(lmbda)
    mu = float(mu)
    if far_field_stress is not None:
        horizontal_stress = float(far_field_stress)
        vertical_stress = 0.0
    horizontal_stress = float(horizontal_stress)
    vertical_stress = float(vertical_stress)
    hole_box = ELASTICITY_HOLE_BOX_FRACTION * W if hole_box is None else float(hole_box)

    lam = 2.0 * lmbda * mu / (lmbda + 2.0 * mu) if plane_stress else lmbda
    poly = sample_hole_polygon(
        W,
        hole_box,
        seed,
        min_points=min_points,
        max_points=max_points,
        samples_per_segment=samples_per_segment,
        handle_scale=handle_scale,
    )

    yy, xx = np.mgrid[0:W, 0:W]
    x_phys = xx.astype(np.float64)
    y_phys = (W - 1 - yy).astype(np.float64)
    hole_pixels = point_in_polygon(poly, x_phys, y_phys)

    cy, cx = np.mgrid[0 : W - 1, 0 : W - 1]
    cell_x = cx + 0.5
    cell_y = W - 1 - (cy + 0.5)
    solid_cell = ~point_in_polygon(poly, cell_x, cell_y)

    nodes = np.arange(W * W, dtype=np.int64).reshape(W, W)
    coords = np.stack([xx.reshape(-1), (W - 1 - yy).reshape(-1)], axis=1).astype(np.float64)

    cy, cx = np.nonzero(solid_cell)
    n00 = nodes[cy, cx]
    n10 = nodes[cy, cx + 1]
    n01 = nodes[cy + 1, cx]
    n11 = nodes[cy + 1, cx + 1]

    tri1 = np.stack([n00, n11, n10], axis=1)
    tri2 = np.stack([n00, n01, n11], axis=1)

    D = np.array([[lam + 2.0 * mu, lam, 0.0], [lam, lam + 2.0 * mu, 0.0], [0.0, 0.0, mu]])

    B1, A1 = triangle_B(coords[tri1[0]])
    B2, A2 = triangle_B(coords[tri2[0]])

    ke1 = A1 * (B1.T @ D @ B1)
    ke2 = A2 * (B2.T @ D @ B2)

    def block(tris, ke):
        dofs = np.empty((tris.shape[0], 6), dtype=np.int64)
        dofs[:, 0] = 2 * tris[:, 0]
        dofs[:, 1] = 2 * tris[:, 0] + 1
        dofs[:, 2] = 2 * tris[:, 1]
        dofs[:, 3] = 2 * tris[:, 1] + 1
        dofs[:, 4] = 2 * tris[:, 2]
        dofs[:, 5] = 2 * tris[:, 2] + 1

        rows = np.repeat(dofs, 6, axis=1).reshape(-1)
        cols = np.tile(dofs, (1, 6)).reshape(-1)
        vals = np.broadcast_to(ke, (tris.shape[0], 6, 6)).reshape(-1)
        return rows, cols, vals, dofs

    r1, c1, v1, dofs1 = block(tri1, ke1)
    r2, c2, v2, dofs2 = block(tri2, ke2)

    ndof = 2 * W * W
    K = coo_matrix(
        (np.concatenate([v1, v2]), (np.concatenate([r1, r2]), np.concatenate([c1, c2]))),
        shape=(ndof, ndof),
    ).tocsr()

    stiffness_diag = lam + 2.0 * mu
    stiffness_det = stiffness_diag * stiffness_diag - lam * lam
    eps_x = (stiffness_diag * horizontal_stress - lam * vertical_stress) / stiffness_det
    eps_y = (-lam * horizontal_stress + stiffness_diag * vertical_stress) / stiffness_det
    center = 0.5 * (W - 1)

    u_bc = np.zeros((W * W, 2), dtype=np.float64)
    u_bc[:, 0] = eps_x * (coords[:, 0] - center)
    u_bc[:, 1] = eps_y * (coords[:, 1] - center)

    active_nodes = np.unique(np.concatenate([tri1.reshape(-1), tri2.reshape(-1)]))
    active_dofs = np.sort(np.concatenate([2 * active_nodes, 2 * active_nodes + 1]))

    outer = (xx == 0) | (xx == W - 1) | (yy == 0) | (yy == W - 1)
    fixed_nodes = nodes[outer].reshape(-1)
    fixed = np.sort(np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1]))
    fixed = np.intersect1d(fixed, active_dofs, assume_unique=False)

    fixed_values = u_bc.reshape(-1)[fixed]
    free = np.setdiff1d(active_dofs, fixed, assume_unique=False)

    rhs = -K[free][:, fixed] @ fixed_values

    u = np.zeros(ndof, dtype=np.float64)
    u[fixed] = fixed_values
    u[free] = spsolve(K[free][:, free], rhs)

    stress_sum = np.zeros((W * W, 3), dtype=np.float64)
    weight = np.zeros(W * W, dtype=np.float64)

    def accumulate(tris, dofs, B):
        ue = u[dofs]
        strain = ue @ B.T
        stress = strain @ D.T
        for k in range(3):
            np.add.at(stress_sum, tris[:, k], stress)
            np.add.at(weight, tris[:, k], 1.0)

    accumulate(tri1, dofs1, B1)
    accumulate(tri2, dofs2, B2)

    stress = (stress_sum / np.maximum(weight[:, None], 1.0)).reshape(W, W, 3)
    active = weight.reshape(W, W) > 0

    sxx, syy, txy = stress[..., 0], stress[..., 1], stress[..., 2]
    vm = np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * txy * txy, 0.0))
    theta = 0.5 * np.arctan2(2.0 * txy, sxx - syy)

    void = hole_pixels | (~active)
    solid = ~void
    scale = max(float(np.percentile(vm[solid], stress_percentile)), 1e-12)

    phase = np.mod(theta, np.pi) / np.pi
    rgb = cyclic_value_colorize(
        phase,
        vm,
        base_colors=stress_base_colors,
        value_vmin=0.0,
        value_vmax=scale,
        gamma=stress_gamma,
        mask=void,
        mask_color=(0.0, 0.0, 0.0),
    )
    mask = np.where(hole_pixels, 0, 255).astype(np.uint8)
    return rgb, mask


def generate_elasticity_image_pairs(
    seeds,
    sim_nx=256,
    output_size=256,
    lmbda=ELASTICITY_LAMBDA,
    mu=ELASTICITY_MU,
    lmbda_min=ELASTICITY_LAMBDA_MIN,
    lmbda_max=ELASTICITY_LAMBDA_MAX,
    mu_min=ELASTICITY_MU_MIN,
    mu_max=ELASTICITY_MU_MAX,
    min_points=ELASTICITY_MIN_POINTS,
    max_points=ELASTICITY_MAX_POINTS,
    hole_box=None,
    samples_per_segment=ELASTICITY_SAMPLES_PER_SEGMENT,
    handle_scale=ELASTICITY_HANDLE_SCALE,
    far_field_stress=ELASTICITY_FAR_FIELD_STRESS,
    horizontal_stress=ELASTICITY_HORIZONTAL_STRESS,
    vertical_stress=ELASTICITY_VERTICAL_STRESS,
    horizontal_stress_min=ELASTICITY_HORIZONTAL_STRESS_MIN,
    horizontal_stress_max=ELASTICITY_HORIZONTAL_STRESS_MAX,
    vertical_stress_min=ELASTICITY_VERTICAL_STRESS_MIN,
    vertical_stress_max=ELASTICITY_VERTICAL_STRESS_MAX,
    plane_stress=ELASTICITY_PLANE_STRESS,
    stress_percentile=ELASTICITY_STRESS_PERCENTILE,
    stress_gamma=ELASTICITY_STRESS_GAMMA,
    stress_base_colors=ELASTICITY_STRESS_BASE_COLORS,
):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if sim_nx < 16:
        raise ValueError("elasticity_grid_size must be at least 16.")
    if output_size <= 0:
        raise ValueError("output_size must be positive.")
    lmbda_min, lmbda_max = _validate_positive_range("elasticity_lambda", lmbda_min, lmbda_max)
    mu_min, mu_max = _validate_positive_range("elasticity_mu", mu_min, mu_max)
    if lmbda is not None and lmbda <= 0:
        raise ValueError("elasticity lambda must be positive.")
    if mu is not None and mu <= 0:
        raise ValueError("elasticity mu must be positive.")
    if min_points < 3 or max_points < min_points:
        raise ValueError("elasticity point count must satisfy 3 <= min_points <= max_points.")
    if samples_per_segment <= 0:
        raise ValueError("elasticity_samples_per_segment must be positive.")
    if handle_scale < 0:
        raise ValueError("elasticity_handle_scale must be non-negative.")
    horizontal_stress_min, horizontal_stress_max = _validate_stress_range(
        "elasticity_horizontal_stress",
        horizontal_stress_min,
        horizontal_stress_max,
    )
    vertical_stress_min, vertical_stress_max = _validate_stress_range(
        "elasticity_vertical_stress",
        vertical_stress_min,
        vertical_stress_max,
    )
    if far_field_stress is not None and far_field_stress <= 0:
        raise ValueError("elasticity_far_field_stress must be positive.")
    if horizontal_stress is not None and horizontal_stress < 0:
        raise ValueError("elasticity horizontal stress must be non-negative.")
    if vertical_stress is not None and vertical_stress < 0:
        raise ValueError("elasticity vertical stress must be non-negative.")
    if not (0.0 < stress_percentile <= 100.0):
        raise ValueError("elasticity_stress_percentile must be in (0, 100].")
    if stress_gamma <= 0:
        raise ValueError("elasticity_stress_gamma must be positive.")

    pairs = []
    hole_box = ELASTICITY_HOLE_BOX_FRACTION * float(sim_nx) if hole_box is None else float(hole_box)
    for seed in seeds:
        rng = _seeded_rng(seed)
        lmbda_i = float(lmbda) if lmbda is not None else _sample_log_uniform(rng, lmbda_min, lmbda_max)
        mu_i = float(mu) if mu is not None else _sample_log_uniform(rng, mu_min, mu_max)
        if far_field_stress is not None:
            horizontal_stress_i = float(far_field_stress)
            vertical_stress_i = 0.0
        else:
            horizontal_stress_i = (
                float(horizontal_stress)
                if horizontal_stress is not None
                else _sample_uniform(rng, horizontal_stress_min, horizontal_stress_max)
            )
            vertical_stress_i = (
                float(vertical_stress)
                if vertical_stress is not None
                else _sample_uniform(rng, vertical_stress_min, vertical_stress_max)
            )
        stress_img, mask_img = render_random_hole_elasticity_high_contrast(
            lmbda=lmbda_i,
            mu=mu_i,
            width=sim_nx,
            seed=seed,
            hole_box=hole_box,
            min_points=min_points,
            max_points=max_points,
            samples_per_segment=samples_per_segment,
            handle_scale=handle_scale,
            horizontal_stress=horizontal_stress_i,
            vertical_stress=vertical_stress_i,
            plane_stress=plane_stress,
            stress_percentile=stress_percentile,
            stress_gamma=stress_gamma,
            stress_base_colors=stress_base_colors,
        )
        if output_size != sim_nx:
            stress_img = np.asarray(
                Image.fromarray(np.ascontiguousarray(stress_img)).resize(
                    (output_size, output_size),
                    Image.Resampling.BICUBIC,
                )
            )
            mask_img = np.asarray(
                Image.fromarray(np.ascontiguousarray(mask_img)).resize(
                    (output_size, output_size),
                    Image.Resampling.NEAREST,
                )
            )
        params = {
            "seed": int(seed),
            "pde": "elasticity",
            "sim_nx": int(sim_nx),
            "output_size": int(output_size),
            "lambda": float(lmbda_i),
            "mu": float(mu_i),
            "lambda_min": float(lmbda_min),
            "lambda_max": float(lmbda_max),
            "mu_min": float(mu_min),
            "mu_max": float(mu_max),
            "min_points": int(min_points),
            "max_points": int(max_points),
            "hole_box": float(hole_box),
            "samples_per_segment": int(samples_per_segment),
            "handle_scale": float(handle_scale),
            "horizontal_stress": float(horizontal_stress_i),
            "vertical_stress": float(vertical_stress_i),
            "horizontal_stress_min": float(horizontal_stress_min),
            "horizontal_stress_max": float(horizontal_stress_max),
            "vertical_stress_min": float(vertical_stress_min),
            "vertical_stress_max": float(vertical_stress_max),
            "plane_stress": bool(plane_stress),
            "stress_percentile": float(stress_percentile),
            "stress_gamma": float(stress_gamma),
            "stress_base_colors": tuple(tuple(float(x) for x in color) for color in stress_base_colors),
            "conditioning_names": ELASTICITY_CONDITIONING_NAMES,
            "conditioning_values": (
                float(lmbda_i),
                float(mu_i),
                float(horizontal_stress_i),
                float(vertical_stress_i),
            ),
        }
        pairs.append((stress_img, mask_img, params))

    return pairs
