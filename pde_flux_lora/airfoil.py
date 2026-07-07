import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .bezier import generate_random_bezier_curve
from .colorization import cyclic_value_colorize


AIRFOIL_BASE_COLORS = (
    (0.18, 0.78, 1.00),
    (0.10, 0.30, 0.95),
    (0.64, 0.20, 0.86),
    (1.00, 0.34, 0.26),
    (1.00, 0.84, 0.22),
)


def fit_curve_to_centered_square(curve, n=256, box=None, x_stretch=1.8, y_stretch=0.55):
    box = float(n) * 0.5 if box is None else float(box)
    q = np.asarray(curve, dtype=float).copy()
    q = q - 0.5 * (q.min(axis=0) + q.max(axis=0))
    q[:, 0] *= x_stretch
    q[:, 1] *= y_stretch

    span = q.max(axis=0) - q.min(axis=0)
    q = q * (box / max(float(span.max()), 1e-6))
    q = q + np.array([n / 2.0, n / 2.0])
    return q


def solve_potential_flow(body_xy, n=256, U=1.0):
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

    rows = []
    cols = []
    data = []
    b = np.zeros(unknown.sum())

    for r, c in np.argwhere(unknown):
        k = idx[r, c]
        rows.append(k)
        cols.append(k)
        data.append(4.0)

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if fixed[rr, cc]:
                b[k] += psi_bc[rr, cc]
            else:
                rows.append(k)
                cols.append(idx[rr, cc])
                data.append(-1.0)

    A = coo_matrix((data, (rows, cols)), shape=(unknown.sum(), unknown.sum())).tocsr()

    psi = psi_bc.copy()
    psi[unknown] = spsolve(A, b)

    dpsi_dy, dpsi_dx = np.gradient(psi)
    u = np.ma.array(dpsi_dy, mask=inside)
    v = np.ma.array(-dpsi_dx, mask=inside)
    speed = np.ma.sqrt(u * u + v * v)
    return u, v, speed, inside


def _render_airfoil_image(
    body_xy,
    speed,
    output_size=256,
    vmin=0.5,
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
    vmin=0.5,
    vmax=None,
    base_colors=AIRFOIL_BASE_COLORS,
    gamma=0.55,
):
    sim_nx = speed.shape[0]
    vmin = float(vmin)
    speed_arr = np.asarray(np.ma.filled(speed, 0.0), dtype=np.float32)
    if vmax is None:
        vmax = max(float(np.percentile(speed_arr[~inside], 99.0)), vmin + 1e-6)
    vmax = float(vmax)
    u_arr = np.asarray(np.ma.filled(u, 0.0), dtype=np.float32)
    v_arr = np.asarray(np.ma.filled(v, 0.0), dtype=np.float32)

    phase = (np.arctan2(v_arr, u_arr) + 2.0 * np.pi) / (2.0 * np.pi)
    rgb_uint8 = cyclic_value_colorize(
        phase,
        speed_arr,
        base_colors=base_colors,
        value_vmin=vmin,
        value_vmax=vmax,
        gamma=gamma,
        mask=inside,
        mask_color=(1.0, 1.0, 1.0),
    )
    rgb_uint8 = np.flipud(rgb_uint8)
    return _resize_and_outline_airfoil(rgb_uint8, body_xy, sim_nx=sim_nx, output_size=output_size)


def generate_airfoil_image_pairs(
    seeds,
    sim_nx=256,
    output_size=256,
    min_points=5,
    max_points=10,
    samples_per_segment=28,
    handle_scale=0.12,
    body_box=None,
    x_stretch=1.8,
    y_stretch=0.55,
    flow_speed=1.0,
    speed_vmin=0.5,
    speed_vmax=2.1,
    cmap_name="viridis",
    color_mode="viridis",
    rgb_base_colors=AIRFOIL_BASE_COLORS,
    rgb_gamma=0.55,
):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if speed_vmin >= speed_vmax:
        raise ValueError("airfoil speed_vmin must be < airfoil speed_vmax")

    pairs = []
    body_box = float(sim_nx) * 0.5 if body_box is None else float(body_box)
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
            params["rgb_channels"] = "cyclic flow direction color, brightness=clip(speed,vmin,vmax)^gamma"
        pairs.append((flow_img, no_flow_img, params))

    return pairs


def _generate_one_airfoil_image_pair(task):
    seed, kwargs = task
    return generate_airfoil_image_pairs([seed], **kwargs)[0]


def generate_airfoil_image_pairs_parallel(
    seeds,
    num_workers=0,
    worker_chunksize=4,
    progress=False,
    progress_desc=None,
    **kwargs,
):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed")

    if num_workers is None or int(num_workers) <= 0:
        num_workers = min(os.cpu_count() or 1, 8, len(seeds))
    else:
        num_workers = min(int(num_workers), len(seeds))

    if num_workers <= 1:
        pairs = []
        iterator = seeds
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc=progress_desc or "Airfoil sims", leave=True)
        for seed in iterator:
            pairs.append(generate_airfoil_image_pairs([seed], **kwargs)[0])
        return pairs

    tasks = [(seed, kwargs) for seed in seeds]
    chunksize = max(1, int(worker_chunksize))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        iterator = executor.map(_generate_one_airfoil_image_pair, tasks, chunksize=chunksize)
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=len(seeds), desc=progress_desc or "Airfoil sims", leave=True)
        return list(iterator)
