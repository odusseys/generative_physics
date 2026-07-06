import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def evaluate_cubic_bezier_segment(p0, p1, p2, p3, samples, endpoint=False):
    t = np.linspace(0.0, 1.0, int(samples) + int(endpoint))
    if not endpoint:
        t = t[:-1]
    t = t[:, None]
    omt = 1.0 - t
    return omt**3 * p0 + 3.0 * omt**2 * t * p1 + 3.0 * omt * t**2 * p2 + t**3 * p3


def evaluate_closed_bezier_spline(anchors, samples_per_segment=24, handle_scale=0.12):
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("anchors must have shape (points, 2)")
    if anchors.shape[0] < 3:
        raise ValueError("at least three anchors are required for a closed curve")

    pieces = []
    point_count = anchors.shape[0]
    for idx in range(point_count):
        p_prev = anchors[(idx - 1) % point_count]
        p0 = anchors[idx]
        p3 = anchors[(idx + 1) % point_count]
        p_next = anchors[(idx + 2) % point_count]
        c1 = p0 + float(handle_scale) * (p3 - p_prev)
        c2 = p3 - float(handle_scale) * (p_next - p0)
        pieces.append(evaluate_cubic_bezier_segment(p0, c1, c2, p3, samples_per_segment))

    curve = np.concatenate(pieces, axis=0)
    return np.concatenate((curve, curve[:1]), axis=0)


def _orientation(a, b, c):
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a, b, c, d, eps=1e-9):
    if (
        max(float(a[0]), float(b[0])) + eps < min(float(c[0]), float(d[0]))
        or max(float(c[0]), float(d[0])) + eps < min(float(a[0]), float(b[0]))
        or max(float(a[1]), float(b[1])) + eps < min(float(c[1]), float(d[1]))
        or max(float(c[1]), float(d[1])) + eps < min(float(a[1]), float(b[1]))
    ):
        return False

    def on_segment(p, q, r):
        return (
            min(float(p[0]), float(r[0])) - eps <= float(q[0]) <= max(float(p[0]), float(r[0])) + eps
            and min(float(p[1]), float(r[1])) - eps <= float(q[1]) <= max(float(p[1]), float(r[1])) + eps
        )

    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    if abs(o1) <= eps and on_segment(a, c, b):
        return True
    if abs(o2) <= eps and on_segment(a, d, b):
        return True
    if abs(o3) <= eps and on_segment(c, a, d):
        return True
    if abs(o4) <= eps and on_segment(c, b, d):
        return True
    return False


def polyline_self_intersects(points, closed=True):
    segment_count = points.shape[0] - 1
    for i in range(segment_count):
        for j in range(i + 1, segment_count):
            if abs(i - j) <= 1:
                continue
            if closed and i == 0 and j == segment_count - 1:
                continue
            if _segments_intersect(points[i], points[i + 1], points[j], points[j + 1]):
                return True
    return False


def generate_random_bezier_curve(
    min_points=5,
    max_points=10,
    samples_per_segment=28,
    seed=None,
    handle_scale=0.12,
    max_tries=300,
):
    if min_points < 3:
        raise ValueError("min_points must be at least 3")
    if max_points < min_points:
        raise ValueError("max_points must be >= min_points")

    rng = np.random.default_rng(seed)
    two_pi = 2.0 * math.pi

    for _ in range(int(max_tries)):
        point_count = int(rng.integers(min_points, max_points + 1))
        angles = np.sort(rng.random(point_count) * two_pi)
        wrapped_angles = np.concatenate((angles, angles[:1] + two_pi))
        if np.diff(wrapped_angles).min() < 0.08:
            continue

        radii = rng.uniform(0.35, 0.95, size=point_count)
        anchors = np.stack((radii * np.cos(angles), radii * np.sin(angles)), axis=1)
        anchors = anchors - anchors.mean(axis=0, keepdims=True)

        curve = evaluate_closed_bezier_spline(
            anchors,
            samples_per_segment=samples_per_segment,
            handle_scale=handle_scale,
        )

        scale = 0.92 / max(np.abs(curve).max(), 1e-6)
        anchors = anchors * scale
        curve = curve * scale

        if np.linalg.norm(curve[0] - curve[-1]) <= 1e-6 and not polyline_self_intersects(curve):
            return anchors, curve

    raise RuntimeError("could not sample a closed non-self-intersecting Bezier curve")


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


def _render_airfoil_rgb_image(body_xy, u, v, speed, inside, output_size=256, vmin=0.5, vmax=None):
    sim_nx = speed.shape[0]
    vmin = float(vmin)
    vmax = float(vmax)
    span = max(vmax - vmin, 1e-6)

    speed_arr = np.asarray(np.ma.filled(speed, 0.0), dtype=np.float32)
    speed_magnitude = np.clip(speed_arr, vmin, vmax)
    u_arr = np.asarray(np.ma.filled(u, 0.0), dtype=np.float32)
    v_arr = np.asarray(np.ma.filled(v, 0.0), dtype=np.float32)
    speed_safe = np.maximum(speed_arr, 1e-6)
    unit_u = np.where(speed_arr > 1e-6, u_arr / speed_safe, 0.0)
    unit_v = np.where(speed_arr > 1e-6, v_arr / speed_safe, 0.0)

    rgb = np.empty((sim_nx, sim_nx, 3), dtype=np.float32)
    rgb[..., 0] = (speed_magnitude - vmin) / span
    rgb[..., 1] = np.clip(0.5 * (unit_u + 1.0), 0.0, 1.0)
    rgb[..., 2] = np.clip(0.5 * (unit_v + 1.0), 0.0, 1.0)
    rgb[inside] = 1.0

    rgb_uint8 = np.flipud((rgb * 255.0).round().astype(np.uint8))
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
            params["rgb_channels"] = "R=(clip(speed,vmin,vmax)-vmin)/(vmax-vmin), G=(unit_u+1)/2, B=(unit_v+1)/2"
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
