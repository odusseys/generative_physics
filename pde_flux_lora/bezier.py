import math

import numpy as np


def evaluate_cubic_bezier_segment(p0, p1, p2, p3, samples, endpoint=False):
    t = np.linspace(0.0, 1.0, int(samples) + int(endpoint))
    if not endpoint:
        t = t[:-1]
    t = t[:, None]
    omt = 1.0 - t
    return omt**3 * p0 + 3.0 * omt**2 * t * p1 + 3.0 * omt * t**2 * p2 + t**3 * p3


def evaluate_closed_bezier_spline(anchors, samples_per_segment=24, handle_scale=0.12):
    anchors = np.asarray(anchors, dtype=float)
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
    points = np.asarray(points, dtype=float)
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
