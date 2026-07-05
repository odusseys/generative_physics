import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
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
    vmax=None,
    cmap_name="viridis",
    stream_u=None,
    stream_v=None,
    stream_density=0.55,
    stream_linewidth=0.55,
    stream_arrowsize=0.65,
):
    n = speed.shape[0]
    dpi = 100
    fig = plt.figure(figsize=(output_size / dpi, output_size / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")

    cmap = plt.colormaps[cmap_name].copy()
    cmap.set_bad(color="white")
    ax.imshow(
        speed,
        origin="lower",
        extent=[0, n, 0, n],
        vmin=0.0,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",
    )
    if stream_u is not None and stream_v is not None:
        ax.streamplot(
            np.arange(n),
            np.arange(n),
            stream_u,
            stream_v,
            density=stream_density,
            linewidth=stream_linewidth,
            arrowsize=stream_arrowsize,
            color="black",
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
    cmap_name="viridis",
    stream_density=0.55,
    stream_linewidth=0.55,
    stream_arrowsize=0.65,
):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if output_size <= 0:
        raise ValueError("output_size must be positive")

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
        vmax = max(float(np.nanpercentile(speed.compressed(), 98.5)), float(flow_speed), 1e-6)
        no_flow_speed = np.ma.array(np.full((sim_nx, sim_nx), float(flow_speed)), mask=inside)

        no_flow_img = _render_airfoil_image(
            body_xy,
            no_flow_speed,
            output_size=output_size,
            vmax=vmax,
            cmap_name=cmap_name,
        )
        flow_img = _render_airfoil_image(
            body_xy,
            speed,
            output_size=output_size,
            vmax=vmax,
            cmap_name=cmap_name,
            stream_u=u,
            stream_v=v,
            stream_density=stream_density,
            stream_linewidth=stream_linewidth,
            stream_arrowsize=stream_arrowsize,
        )
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
            "speed_vmin": 0.0,
            "speed_vmax": vmax,
            "stream_density": stream_density,
        }
        pairs.append((flow_img, no_flow_img, params))

    return pairs
