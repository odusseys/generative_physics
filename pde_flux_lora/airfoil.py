import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .bezier import generate_random_bezier_curve
from .colorization import cyclic_value_colorize


AIRFOIL_BASE_COLORS = (
    (0.05, 0.18, 0.95),
    (0.00, 0.90, 1.00),
    (0.00, 0.95, 0.25),
    (1.00, 0.90, 0.00),
    (1.00, 0.10, 0.02),
    (0.90, 0.00, 1.00),
)

AIRFOIL_MIN_POINTS = 5
AIRFOIL_MAX_POINTS = 10
AIRFOIL_SAMPLES_PER_SEGMENT = 28
AIRFOIL_HANDLE_SCALE = 0.12
AIRFOIL_BODY_BOX_FRACTION = 0.5
AIRFOIL_X_STRETCH = 1.8
AIRFOIL_Y_STRETCH = 0.55
AIRFOIL_FLOW_SPEED = 1.0
AIRFOIL_COLOR_MODE = "rgb"
AIRFOIL_COLOR_GAMMA = 0.55
AIRFOIL_COLOR_PHASE_GAIN = 5.0
AIRFOIL_COLOR_PHASE_OFFSET = 0.5
AIRFOIL_COLOR_PHASE_WRAP = False
AIRFOIL_COLOR_SOFTNESS = 0.75
AIRFOIL_SPEED_VMIN = 0.5
AIRFOIL_SPEED_VMAX = 2.1


def fit_curve_to_centered_square(
    curve,
    n=256,
    box=None,
    x_stretch=AIRFOIL_X_STRETCH,
    y_stretch=AIRFOIL_Y_STRETCH,
):
    box = float(n) * 0.5 if box is None else float(box)
    q = np.asarray(curve, dtype=float).copy()
    q = q - 0.5 * (q.min(axis=0) + q.max(axis=0))
    q[:, 0] *= x_stretch
    q[:, 1] *= y_stretch

    span = q.max(axis=0) - q.min(axis=0)
    q = q * (box / max(float(span.max()), 1e-6))
    q = q + np.array([n / 2.0, n / 2.0])
    return q


def solve_potential_flow(body_xy, n=256, U=AIRFOIL_FLOW_SPEED):
    if n < 16:
        raise ValueError("airfoil grid size must be at least 16")
    if U <= 0:
        raise ValueError("airfoil flow speed must be positive")

    grid = np.arange(n)
    X, Y = np.meshgrid(grid, grid)
    inside = Path(body_xy).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(n, n)

    outer = np.zeros((n, n), dtype=bool)
    outer[0, :] = True
    outer[-1, :] = True
    outer[:, 0] = True
    outer[:, -1] = True

    fixed = inside | outer
    psi_bc = U * Y.astype(float)
    psi_bc[inside] = U * (n / 2.0)

    unknown = ~fixed
    idx = -np.ones((n, n), dtype=int)
    idx[unknown] = np.arange(unknown.sum())

    r, c = np.nonzero(unknown)
    k = idx[r, c]
    rows = [k]
    cols = [k]
    data = [np.full(k.shape, 4.0, dtype=np.float64)]
    b = np.zeros(unknown.sum(), dtype=np.float64)

    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        fixed_neighbor = fixed[rr, cc]
        if fixed_neighbor.any():
            b[k[fixed_neighbor]] += psi_bc[rr[fixed_neighbor], cc[fixed_neighbor]]

        free_neighbor = ~fixed_neighbor
        if free_neighbor.any():
            rows.append(k[free_neighbor])
            cols.append(idx[rr[free_neighbor], cc[free_neighbor]])
            data.append(np.full(int(free_neighbor.sum()), -1.0, dtype=np.float64))

    A = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(unknown.sum(), unknown.sum()),
    ).tocsr()

    psi = psi_bc.copy()
    psi[unknown] = spsolve(A, b, permc_spec="MMD_AT_PLUS_A")

    dpsi_dy, dpsi_dx = np.gradient(psi)
    u = np.ma.array(dpsi_dy, mask=inside)
    v = np.ma.array(-dpsi_dx, mask=inside)
    speed = np.ma.sqrt(u * u + v * v)
    return u, v, speed, inside


def _render_airfoil_image(
    body_xy,
    speed,
    output_size=256,
    vmin=AIRFOIL_SPEED_VMIN,
    vmax=None,
    cmap_name="viridis",
):
    n = speed.shape[0]
    dpi = 100
    fig = plt.figure(figsize=(output_size / dpi, output_size / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")

    cmap = plt.colormaps[cmap_name].copy()
    cmap.set_bad(color="white")
    clipped_speed = np.ma.clip(speed, vmin, vmax)
    ax.imshow(
        clipped_speed,
        origin="lower",
        extent=[0, n, 0, n],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",
    )
    ax.plot(body_xy[:, 0], body_xy[:, 1], color="black", linewidth=1.2)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def _resize_and_outline_airfoil(rgb, body_xy, sim_nx, output_size, outline_width=1):
    image = Image.fromarray(np.ascontiguousarray(rgb))
    if image.size != (output_size, output_size):
        image = image.resize((output_size, output_size), Image.Resampling.BICUBIC)

    scale = float(output_size) / float(sim_nx)
    outline_xy = [(float(x) * scale, float(output_size) - float(y) * scale) for x, y in body_xy]
    draw = ImageDraw.Draw(image)
    draw.line(outline_xy, fill=(0, 0, 0), width=max(1, int(outline_width)), joint="curve")
    return np.asarray(image.convert("RGB")).copy()


def _render_airfoil_rgb_image(
    body_xy,
    u,
    v,
    speed,
    inside,
    output_size=256,
    vmin=AIRFOIL_SPEED_VMIN,
    vmax=None,
    base_colors=AIRFOIL_BASE_COLORS,
    gamma=AIRFOIL_COLOR_GAMMA,
    phase_gain=AIRFOIL_COLOR_PHASE_GAIN,
    phase_offset=AIRFOIL_COLOR_PHASE_OFFSET,
    phase_wrap=AIRFOIL_COLOR_PHASE_WRAP,
    value_softness=AIRFOIL_COLOR_SOFTNESS,
):
    sim_nx = speed.shape[0]
    vmin = float(vmin)
    speed_arr = np.asarray(np.ma.filled(speed, 0.0), dtype=np.float32)
    if vmax is None:
        vmax = max(float(np.percentile(speed_arr[~inside], 99.0)), vmin + 1e-6)
    vmax = float(vmax)
    u_arr = np.asarray(np.ma.filled(u, 0.0), dtype=np.float32)
    v_arr = np.asarray(np.ma.filled(v, 0.0), dtype=np.float32)

    phase = np.arctan2(v_arr, u_arr) / (2.0 * np.pi)
    phase = phase * float(phase_gain) + float(phase_offset)
    if phase_wrap:
        phase = np.mod(phase, 1.0)
    else:
        phase = np.clip(phase, 0.0, np.nextafter(np.float32(1.0), np.float32(0.0)))
    rgb_uint8 = cyclic_value_colorize(
        phase,
        speed_arr,
        base_colors=base_colors,
        value_vmin=vmin,
        value_vmax=vmax,
        gamma=gamma,
        value_softness=value_softness,
        mask=inside,
        mask_color=(1.0, 1.0, 1.0),
    )
    rgb_uint8 = np.flipud(rgb_uint8)
    return _resize_and_outline_airfoil(rgb_uint8, body_xy, sim_nx=sim_nx, output_size=output_size)


def generate_airfoil_image_pairs(
    seeds,
    sim_nx=256,
    output_size=256,
    min_points=AIRFOIL_MIN_POINTS,
    max_points=AIRFOIL_MAX_POINTS,
    samples_per_segment=AIRFOIL_SAMPLES_PER_SEGMENT,
    handle_scale=AIRFOIL_HANDLE_SCALE,
    body_box=None,
    x_stretch=AIRFOIL_X_STRETCH,
    y_stretch=AIRFOIL_Y_STRETCH,
    flow_speed=AIRFOIL_FLOW_SPEED,
    speed_vmin=AIRFOIL_SPEED_VMIN,
    speed_vmax=AIRFOIL_SPEED_VMAX,
    cmap_name="viridis",
    color_mode=AIRFOIL_COLOR_MODE,
    rgb_base_colors=AIRFOIL_BASE_COLORS,
    rgb_gamma=AIRFOIL_COLOR_GAMMA,
    rgb_phase_gain=AIRFOIL_COLOR_PHASE_GAIN,
    rgb_phase_offset=AIRFOIL_COLOR_PHASE_OFFSET,
    rgb_phase_wrap=AIRFOIL_COLOR_PHASE_WRAP,
    rgb_value_softness=AIRFOIL_COLOR_SOFTNESS,
):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if speed_vmin >= speed_vmax:
        raise ValueError("airfoil speed_vmin must be < airfoil speed_vmax")

    pairs = []
    body_box = float(sim_nx) * AIRFOIL_BODY_BOX_FRACTION if body_box is None else float(body_box)
    for seed in seeds:
        _, curve = generate_random_bezier_curve(
            min_points=min_points,
            max_points=max_points,
            samples_per_segment=samples_per_segment,
            seed=seed,
            handle_scale=handle_scale,
        )
        body_xy = fit_curve_to_centered_square(
            curve,
            n=sim_nx,
            box=body_box,
            x_stretch=x_stretch,
            y_stretch=y_stretch,
        )
        u, v, speed, inside = solve_potential_flow(body_xy, n=sim_nx, U=flow_speed)
        vmin = float(speed_vmin)
        vmax = float(speed_vmax)
        no_flow_speed = np.ma.array(np.full((sim_nx, sim_nx), float(flow_speed)), mask=inside)
        no_flow_u = np.ma.array(np.full((sim_nx, sim_nx), float(flow_speed)), mask=inside)
        no_flow_v = np.ma.array(np.zeros((sim_nx, sim_nx)), mask=inside)

        if color_mode == "viridis":
            no_flow_img = _render_airfoil_image(
                body_xy,
                no_flow_speed,
                output_size=output_size,
                vmin=vmin,
                vmax=vmax,
                cmap_name=cmap_name,
            )
            flow_img = _render_airfoil_image(
                body_xy,
                speed,
                output_size=output_size,
                vmin=vmin,
                vmax=vmax,
                cmap_name=cmap_name,
            )
        elif color_mode == "rgb":
            no_flow_img = _render_airfoil_rgb_image(
                body_xy,
                no_flow_u,
                no_flow_v,
                no_flow_speed,
                inside,
                output_size=output_size,
                vmin=vmin,
                vmax=vmax,
                base_colors=rgb_base_colors,
                gamma=rgb_gamma,
                phase_gain=rgb_phase_gain,
                phase_offset=rgb_phase_offset,
                phase_wrap=rgb_phase_wrap,
                value_softness=rgb_value_softness,
            )
            flow_img = _render_airfoil_rgb_image(
                body_xy,
                u,
                v,
                speed,
                inside,
                output_size=output_size,
                vmin=vmin,
                vmax=vmax,
                base_colors=rgb_base_colors,
                gamma=rgb_gamma,
                phase_gain=rgb_phase_gain,
                phase_offset=rgb_phase_offset,
                phase_wrap=rgb_phase_wrap,
                value_softness=rgb_value_softness,
            )
        else:
            raise ValueError("airfoil color_mode must be 'viridis' or 'rgb'.")
        params = {
            "seed": seed,
            "pde": "airfoil",
            "sim_nx": sim_nx,
            "output_size": output_size,
            "min_points": min_points,
            "max_points": max_points,
            "samples_per_segment": samples_per_segment,
            "handle_scale": handle_scale,
            "body_box": body_box,
            "x_stretch": x_stretch,
            "y_stretch": y_stretch,
            "flow_speed": flow_speed,
            "color_mode": color_mode,
            "speed_vmin": vmin,
            "speed_vmax": vmax,
            "speed_normalization": "fixed_clip",
        }
        if color_mode == "rgb":
            params["rgb_base_colors"] = tuple(tuple(float(x) for x in color) for color in rgb_base_colors)
            params["rgb_gamma"] = float(rgb_gamma)
            params["rgb_phase_gain"] = float(rgb_phase_gain)
            params["rgb_phase_offset"] = float(rgb_phase_offset)
            params["rgb_phase_wrap"] = bool(rgb_phase_wrap)
            params["rgb_value_softness"] = float(rgb_value_softness)
            params["rgb_channels"] = (
                "cyclic amplified flow direction color, brightness=soft_rolloff(speed,vmin,vmax)^gamma"
            )
        pairs.append((flow_img, no_flow_img, params))

    return pairs
