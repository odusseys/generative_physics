#!/usr/bin/env python3
"""Generate Navier-Stokes videos and Wan VAE latents.

The simulation, video rendering, and latent helpers are inlined here so this
script is the only runtime entrypoint. The notebooks are not imported or
executed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import secrets
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torchvision.io import read_video, read_video_timestamps, write_video

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("/home/azureuser/datasets/navier_fast")
DEFAULT_WAN_REPO_ROOT = REPO_ROOT / "navier" / "Wan2.2"
DEFAULT_WAN_CHECKPOINT_DIR = Path("/home/azureuser/Wan2.2-TI2V-5B")

NAVIER_GRID_SIZE = 256
DEFAULT_VIDEO_WIDTH = 480
DEFAULT_VIDEO_HEIGHT = 240
OBSTACLE_BASE_NORMALIZED_BOX = (0.33, 0.66)
OBSTACLE_SIZE_SCALE = 0.70
OBSTACLE_LEFT_PADDING = 0.10
OBSTACLE_NORMALIZED_SIZE = (
    OBSTACLE_BASE_NORMALIZED_BOX[1] - OBSTACLE_BASE_NORMALIZED_BOX[0]
) * OBSTACLE_SIZE_SCALE
OBSTACLE_NORMALIZED_X_BOX = (
    OBSTACLE_LEFT_PADDING,
    OBSTACLE_LEFT_PADDING + OBSTACLE_NORMALIZED_SIZE,
)
OBSTACLE_NORMALIZED_Y_BOX = (
    0.5 - 0.5 * OBSTACLE_NORMALIZED_SIZE,
    0.5 + 0.5 * OBSTACLE_NORMALIZED_SIZE,
)
OBSTACLE_NORMALIZED_BOX = (OBSTACLE_NORMALIZED_X_BOX, OBSTACLE_NORMALIZED_Y_BOX)
WAN_REPO_ROOT = DEFAULT_WAN_REPO_ROOT
WAN_CHECKPOINT_DIR = DEFAULT_WAN_CHECKPOINT_DIR
WAN_TI2V_5B_VAE_CHECKPOINT = "Wan2.2_VAE.pth"
WAN_TI2V_5B_VAE_STRIDE = (4, 16, 16)
DEFAULT_WAN_EXPECTED_SIZE = (DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH)


def _navier_grid_size(n: Optional[int] = None) -> int:
    size = int(NAVIER_GRID_SIZE if n is None else n)
    if size < 32:
        raise ValueError("grid size must be at least 32")
    return size


def _video_size(
    width: Optional[int] = None, height: Optional[int] = None
) -> Tuple[int, int]:
    resolved_width = int(DEFAULT_VIDEO_WIDTH if width is None else width)
    resolved_height = int(DEFAULT_VIDEO_HEIGHT if height is None else height)
    if resolved_width < 16 or resolved_height < 16:
        raise ValueError("video width and height must be at least 16")
    if resolved_width % WAN_TI2V_5B_VAE_STRIDE[2] != 0:
        raise ValueError(
            f"video width must be a multiple of {WAN_TI2V_5B_VAE_STRIDE[2]}"
        )
    if resolved_height % WAN_TI2V_5B_VAE_STRIDE[1] != 0:
        raise ValueError(
            f"video height must be a multiple of {WAN_TI2V_5B_VAE_STRIDE[1]}"
        )
    return (resolved_height, resolved_width)


def _resolve_output_hw(
    output_size: Optional[int | Tuple[int, int]],
    default_hw: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[int, int]]:
    if output_size is None:
        if default_hw is None:
            return None
        height, width = default_hw
    elif isinstance(output_size, (tuple, list)):
        if len(output_size) != 2:
            raise ValueError("output_size tuple must be (height, width)")
        height, width = output_size
    else:
        height = width = int(output_size)
    height = int(height)
    width = int(width)
    if height < 1 or width < 1:
        raise ValueError("output height and width must be positive")
    return (height, width)


def _zero_boundary_(field: torch.Tensor) -> torch.Tensor:
    field[0, :] = 0.0
    field[-1, :] = 0.0
    field[:, 0] = 0.0
    field[:, -1] = 0.0
    return field


def _set_right_flow_boundary_(velocity: torch.Tensor) -> torch.Tensor:
    velocity[0, 0, :] = 1.0
    velocity[0, -1, :] = 1.0
    velocity[0, :, 0] = 1.0
    velocity[0, :, -1] = 1.0
    velocity[1, 0, :] = 0.0
    velocity[1, -1, :] = 0.0
    velocity[1, :, 0] = 0.0
    velocity[1, :, -1] = 0.0
    return velocity


def _enforce_flow_constraints_(
    velocity: torch.Tensor, obstacle_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    _set_right_flow_boundary_(velocity)
    if obstacle_mask is not None:
        mask = obstacle_mask.to(device=velocity.device, dtype=torch.bool)
        velocity[:, mask] = 0.0
    return velocity


def _enforce_no_slip_(velocity: torch.Tensor) -> torch.Tensor:
    _zero_boundary_(velocity[0])
    _zero_boundary_(velocity[1])
    return velocity


def evaluate_cubic_bezier_segment(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    samples: int,
    endpoint: bool = False,
) -> torch.Tensor:
    t = torch.linspace(
        0.0, 1.0, int(samples) + int(endpoint), device=p0.device, dtype=p0.dtype
    )
    if not endpoint:
        t = t[:-1]
    t = t[:, None]
    omt = 1.0 - t
    return omt**3 * p0 + 3.0 * omt**2 * t * p1 + 3.0 * omt * t**2 * p2 + t**3 * p3


def evaluate_closed_bezier_spline(
    anchors: torch.Tensor, samples_per_segment: int = 24, handle_scale: float = 0.12
) -> torch.Tensor:
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("anchors must have shape (points, 2)")
    if anchors.shape[0] < 3:
        raise ValueError("at least three anchors are required for a closed curve")
    pieces = []
    n = anchors.shape[0]
    for i in range(n):
        p_prev = anchors[(i - 1) % n]
        p0 = anchors[i]
        p3 = anchors[(i + 1) % n]
        p_next = anchors[(i + 2) % n]
        c1 = p0 + float(handle_scale) * (p3 - p_prev)
        c2 = p3 - float(handle_scale) * (p_next - p0)
        pieces.append(
            evaluate_cubic_bezier_segment(p0, c1, c2, p3, samples_per_segment)
        )
    curve = torch.cat(pieces, dim=0)
    return torch.cat((curve, curve[:1]), dim=0)


def _orientation(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    eps: float = 1e-09,
) -> bool:
    if (
        max(float(a[0]), float(b[0])) + eps < min(float(c[0]), float(d[0]))
        or max(float(c[0]), float(d[0])) + eps < min(float(a[0]), float(b[0]))
        or max(float(a[1]), float(b[1])) + eps < min(float(c[1]), float(d[1]))
        or (max(float(c[1]), float(d[1])) + eps < min(float(a[1]), float(b[1])))
    ):
        return False

    def on_segment(p, q, r):
        return (
            min(float(p[0]), float(r[0])) - eps
            <= float(q[0])
            <= max(float(p[0]), float(r[0])) + eps
            and min(float(p[1]), float(r[1])) - eps
            <= float(q[1])
            <= max(float(p[1]), float(r[1])) + eps
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


def polyline_self_intersects(points: torch.Tensor, closed: bool = True) -> bool:
    points = points.detach().cpu()
    segment_count = points.shape[0] - 1
    for i in range(segment_count):
        for j in range(i + 1, segment_count):
            if abs(i - j) <= 1:
                continue
            if closed and i == 0 and (j == segment_count - 1):
                continue
            if _segments_intersect(points[i], points[i + 1], points[j], points[j + 1]):
                return True
    return False


def generate_random_bezier_curve(
    min_points: int = 3,
    max_points: int = 10,
    samples_per_segment: int = 24,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    handle_scale: float = 0.12,
    max_tries: int = 300,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if min_points < 3:
        raise ValueError("min_points must be at least 3")
    if max_points < min_points:
        raise ValueError("max_points must be >= min_points")
    device = torch.device("cpu") if device is None else torch.device(device)
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))
    two_pi = 2.0 * math.pi
    for _ in range(int(max_tries)):
        point_count = int(
            torch.randint(
                min_points, max_points + 1, (1,), generator=generator, device=device
            ).item()
        )
        angles = torch.sort(
            torch.rand(point_count, generator=generator, device=device, dtype=dtype)
            * two_pi
        ).values
        wrapped_angles = torch.cat((angles, angles[:1] + two_pi))
        if float(torch.diff(wrapped_angles).amin()) < 0.08:
            continue
        radii = torch.empty(point_count, device=device, dtype=dtype).uniform_(
            0.35, 0.95, generator=generator
        )
        anchors = torch.stack(
            (radii * torch.cos(angles), radii * torch.sin(angles)), dim=1
        )
        anchors = anchors - anchors.mean(dim=0, keepdim=True)
        curve = evaluate_closed_bezier_spline(
            anchors, samples_per_segment=samples_per_segment, handle_scale=handle_scale
        )
        scale = 0.92 / curve.abs().amax().clamp_min(1e-06)
        anchors = anchors * scale
        curve = curve * scale
        if torch.linalg.norm(curve[0] - curve[-1]) <= 1e-06 and (
            not polyline_self_intersects(curve, closed=True)
        ):
            return (anchors, curve)
    raise RuntimeError("could not sample a closed non-self-intersecting Bezier curve")


def _validate_normalized_interval(interval: Tuple[float, float], name: str) -> None:
    lo, hi = interval
    if not 0.0 <= float(lo) < float(hi) <= 1.0:
        raise ValueError(f"{name} must be inside [0, 1] with lo < hi")


def fit_curve_to_normalized_box(
    anchors: torch.Tensor,
    curve: torch.Tensor,
    normalized_box: (
        Tuple[float, float] | Tuple[Tuple[float, float], Tuple[float, float]]
    ) = OBSTACLE_NORMALIZED_BOX,
    align_x: str = "left",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(normalized_box[0], (tuple, list)):
        x_box, y_box = normalized_box  # type: ignore[assignment]
    else:
        x_box = y_box = normalized_box  # type: ignore[assignment]
    _validate_normalized_interval(x_box, "normalized x box")
    _validate_normalized_interval(y_box, "normalized y box")

    target_x_lo = -1.0 + 2.0 * float(x_box[0])
    target_x_hi = -1.0 + 2.0 * float(x_box[1])
    target_y_lo = -1.0 + 2.0 * float(y_box[0])
    target_y_hi = -1.0 + 2.0 * float(y_box[1])
    target_size = min(target_x_hi - target_x_lo, target_y_hi - target_y_lo)
    target_y_center = 0.5 * (target_y_lo + target_y_hi)
    cmin = curve.amin(dim=0)
    cmax = curve.amax(dim=0)
    center = 0.5 * (cmin + cmax)
    span = (cmax - cmin).amax().clamp_min(1e-06)
    scale = target_size / span
    scaled_cmin = (cmin - center) * scale
    scaled_cmax = (cmax - center) * scale
    if align_x == "left":
        target_x_center = target_x_lo - scaled_cmin[0]
    elif align_x == "center":
        target_x_center = 0.5 * (target_x_lo + target_x_hi)
    else:
        raise ValueError("align_x must be 'left' or 'center'")
    target_y_center = target_y_center - 0.5 * (scaled_cmin[1] + scaled_cmax[1])
    target = torch.stack(
        (
            torch.as_tensor(target_x_center, device=curve.device, dtype=curve.dtype),
            torch.as_tensor(target_y_center, device=curve.device, dtype=curve.dtype),
        )
    )
    return ((anchors - center) * scale + target, (curve - center) * scale + target)


def fit_curve_to_middle_square(
    anchors: torch.Tensor,
    curve: torch.Tensor,
    normalized_box: Tuple[float, float] = OBSTACLE_BASE_NORMALIZED_BOX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return fit_curve_to_normalized_box(
        anchors, curve, normalized_box=normalized_box, align_x="center"
    )


def polygon_obstacle_mask(
    polygon: torch.Tensor, xx: torch.Tensor, yy: torch.Tensor, dilate_pixels: int = 1
) -> torch.Tensor:
    polygon = polygon.to(device=xx.device, dtype=xx.dtype)
    inside = torch.zeros_like(xx, dtype=torch.bool)
    x = xx
    y = yy
    eps = torch.finfo(xx.dtype).eps
    for i in range(polygon.shape[0] - 1):
        x0, y0 = polygon[i]
        x1, y1 = polygon[i + 1]
        crosses = ((y0 > y) != (y1 > y)) & (
            x < (x1 - x0) * (y - y0) / (y1 - y0 + eps) + x0
        )
        inside ^= crosses
    if dilate_pixels > 0:
        k = 2 * int(dilate_pixels) + 1
        inside = F.max_pool2d(
            inside.float()[None, None],
            kernel_size=k,
            stride=1,
            padding=int(dilate_pixels),
        )[0, 0].bool()
    return inside


def make_initial_velocity_field(
    n: Optional[int] = None,
    amplitude: float = 1.0,
    modes: int = 8,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    return_grid: bool = True,
    return_obstacle: bool = False,
):
    """Create uniform rightward flow around a random closed Bezier obstacle.

    Boundary velocity is the right-pointing unit vector everywhere on the square
    boundary. The Bezier obstacle is scaled to 70% of the previous centered
    obstacle size, left-aligned with 10% normalized left padding, vertically
    centered, and has zero velocity on its filled interior.
    """
    n = _navier_grid_size(n)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    coords = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    anchors, curve = generate_random_bezier_curve(seed=seed, device=device, dtype=dtype)
    anchors, curve = fit_curve_to_normalized_box(
        anchors, curve, normalized_box=OBSTACLE_NORMALIZED_BOX, align_x="left"
    )
    obstacle_mask = polygon_obstacle_mask(curve, xx, yy, dilate_pixels=1)
    velocity = torch.zeros((2, n, n), device=device, dtype=dtype)
    velocity[0].fill_(1.0)
    velocity[1].zero_()
    _enforce_flow_constraints_(velocity, obstacle_mask)
    obstacle = {
        "anchors": anchors,
        "curve": curve,
        "mask": obstacle_mask,
        "normalized_box": OBSTACLE_NORMALIZED_BOX,
        "left_padding": OBSTACLE_LEFT_PADDING,
        "size_scale": OBSTACLE_SIZE_SCALE,
    }
    if return_grid and return_obstacle:
        return (velocity, (xx, yy), obstacle)
    if return_grid:
        return (velocity, (xx, yy))
    if return_obstacle:
        return (velocity, obstacle)
    return velocity


def divergence(
    velocity: torch.Tensor, dx: Optional[float] = None, dy: Optional[float] = None
) -> torch.Tensor:
    """Centered finite-difference divergence for a velocity tensor shaped (2, ny, nx)."""
    if velocity.ndim != 3 or velocity.shape[0] != 2:
        raise ValueError("velocity must have shape (2, ny, nx)")
    _, ny, nx = velocity.shape
    dx = 2.0 / (nx - 1) if dx is None else float(dx)
    dy = 2.0 / (ny - 1) if dy is None else float(dy)
    u, v = (velocity[0], velocity[1])
    div = torch.zeros_like(u)
    div[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dx) + (
        v[2:, 1:-1] - v[:-2, 1:-1]
    ) / (2.0 * dy)
    return div


def stable_timestep(
    velocity: torch.Tensor,
    nu: float,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    cfl: float = 0.4,
    diffusion_safety: float = 0.2,
    max_delta_t: float = 0.005,
) -> float:
    """Pick an explicit stable step from advection and diffusion limits."""
    _, ny, nx = velocity.shape
    dx = 2.0 / (nx - 1) if dx is None else float(dx)
    dy = 2.0 / (ny - 1) if dy is None else float(dy)
    h = min(dx, dy)
    max_speed = float(velocity.square().sum(dim=0).sqrt().amax().detach().cpu())
    advective = float("inf") if max_speed < 1e-12 else cfl * h / max_speed
    diffusive = float("inf") if nu <= 0.0 else diffusion_safety * h * h / float(nu)
    return max(1e-08, min(float(max_delta_t), advective, diffusive))


def _laplacian_interior(field: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    return (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (
        dx * dx
    ) + (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dy * dy)


def _upwind_gradient(
    field: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dx: float, dy: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    center = field[1:-1, 1:-1]
    u_center = u[1:-1, 1:-1]
    v_center = v[1:-1, 1:-1]
    backward_x = (center - field[1:-1, :-2]) / dx
    forward_x = (field[1:-1, 2:] - center) / dx
    backward_y = (center - field[:-2, 1:-1]) / dy
    forward_y = (field[2:, 1:-1] - center) / dy
    grad_x = torch.where(u_center >= 0.0, backward_x, forward_x)
    grad_y = torch.where(v_center >= 0.0, backward_y, forward_y)
    return (grad_x, grad_y)


_SPECTRAL_OPERATOR_CACHE = {}
_COMPILED_STEP_CACHE = {}


def _spectral_wavenumbers(
    ny: int, nx: int, dx: float, dy: float, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (
        device.type,
        device.index,
        str(dtype),
        int(ny),
        int(nx),
        float(dx),
        float(dy),
    )
    cached = _SPECTRAL_OPERATOR_CACHE.get(key)
    if cached is not None:
        return cached
    kx = (2.0 * math.pi * torch.fft.fftfreq(nx, d=dx, device=device, dtype=dtype)).view(
        1, nx
    )
    ky = (2.0 * math.pi * torch.fft.fftfreq(ny, d=dy, device=device, dtype=dtype)).view(
        ny, 1
    )
    k2_raw = kx.square() + ky.square()
    zero_mode = k2_raw == 0.0
    k2 = torch.where(zero_mode, torch.ones_like(k2_raw), k2_raw)
    cached = (kx, ky, k2, zero_mode)
    _SPECTRAL_OPERATOR_CACHE[key] = cached
    return cached


def _project_velocity_spectral_periodic_with_operators(
    provisional: torch.Tensor,
    rho: torch.Tensor | float,
    delta_t: torch.Tensor | float,
    kx: torch.Tensor,
    ky: torch.Tensor,
    k2: torch.Tensor,
    zero_mode: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    u_hat = torch.fft.fft2(provisional[0])
    v_hat = torch.fft.fft2(provisional[1])
    velocity_dot_k = kx * u_hat + ky * v_hat
    zero_complex = torch.zeros_like(u_hat)
    correction_u = torch.where(zero_mode, zero_complex, kx * velocity_dot_k / k2)
    correction_v = torch.where(zero_mode, zero_complex, ky * velocity_dot_k / k2)
    projected_u = torch.fft.ifft2(u_hat - correction_u).real
    projected_v = torch.fft.ifft2(v_hat - correction_v).real
    pressure_hat = torch.where(
        zero_mode, zero_complex, -(rho / delta_t) * (1j * velocity_dot_k) / k2
    )
    pressure = torch.fft.ifft2(pressure_hat).real
    return (torch.stack((projected_u, projected_v), dim=0), pressure)


def _project_velocity_spectral_periodic(
    provisional: torch.Tensor,
    rho: torch.Tensor | float,
    delta_t: torch.Tensor | float,
    dx: float,
    dy: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, ny, nx = provisional.shape
    operators = _spectral_wavenumbers(
        ny, nx, dx, dy, provisional.device, provisional.dtype
    )
    return _project_velocity_spectral_periodic_with_operators(
        provisional, rho, delta_t, *operators
    )


def _pressure_poisson(
    rhs: torch.Tensor,
    dx: float,
    dy: float,
    iterations: int,
    pressure: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    p = torch.zeros_like(rhs) if pressure is None else pressure.clone()
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)
    for _ in range(iterations):
        old = p
        p = old.clone()
        p[1:-1, 1:-1] = (
            (old[1:-1, 2:] + old[1:-1, :-2]) * dy2
            + (old[2:, 1:-1] + old[:-2, 1:-1]) * dx2
            - rhs[1:-1, 1:-1] * dx2 * dy2
        ) / denom
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p = p - p.mean()
    return p


def _advection_diffusion_provisional(
    velocity: torch.Tensor,
    nu: torch.Tensor | float,
    delta_t: torch.Tensor | float,
    dx: float,
    dy: float,
) -> torch.Tensor:
    u, v = (velocity[0], velocity[1])
    u_center = u[1:-1, 1:-1]
    v_center = v[1:-1, 1:-1]
    du_dx, du_dy = _upwind_gradient(u, u, v, dx, dy)
    dv_dx, dv_dy = _upwind_gradient(v, u, v, dx, dy)
    lap_u = _laplacian_interior(u, dx, dy)
    lap_v = _laplacian_interior(v, dx, dy)
    u_star = u.clone()
    v_star = v.clone()
    u_star[1:-1, 1:-1] = u_center + delta_t * (
        nu * lap_u - (u_center * du_dx + v_center * du_dy)
    )
    v_star[1:-1, 1:-1] = v_center + delta_t * (
        nu * lap_v - (u_center * dv_dx + v_center * dv_dy)
    )
    provisional = torch.stack((u_star, v_star), dim=0)
    _set_right_flow_boundary_(provisional)
    return provisional


def _navier_stokes_step_spectral_fast(
    velocity: torch.Tensor,
    pressure: torch.Tensor,
    nu: torch.Tensor | float,
    rho: torch.Tensor | float,
    delta_t: torch.Tensor | float,
    dx: float,
    dy: float,
    kx: torch.Tensor,
    ky: torch.Tensor,
    k2: torch.Tensor,
    zero_mode: torch.Tensor,
    obstacle_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    provisional = _advection_diffusion_provisional(velocity, nu, delta_t, dx, dy)
    _enforce_flow_constraints_(provisional, obstacle_mask)
    next_velocity, pressure = _project_velocity_spectral_periodic_with_operators(
        provisional, rho, delta_t, kx, ky, k2, zero_mode
    )
    _enforce_flow_constraints_(next_velocity, obstacle_mask)
    return (next_velocity, pressure)


def _navier_stokes_step(
    velocity: torch.Tensor,
    pressure: torch.Tensor,
    nu: torch.Tensor | float,
    rho: torch.Tensor | float,
    delta_t: torch.Tensor | float,
    dx: float,
    dy: float,
    pressure_iterations: int,
    pressure_method: str = "spectral",
    obstacle_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    provisional = _advection_diffusion_provisional(velocity, nu, delta_t, dx, dy)
    _enforce_flow_constraints_(provisional, obstacle_mask)
    method = pressure_method.lower()
    if method in {"spectral", "spectral_periodic", "fft"}:
        next_velocity, pressure = _project_velocity_spectral_periodic(
            provisional, rho, delta_t, dx, dy
        )
        _enforce_flow_constraints_(next_velocity, obstacle_mask)
        return (next_velocity, pressure)
    if method != "jacobi":
        raise ValueError("pressure_method must be 'spectral' or 'jacobi'")
    rhs = rho / delta_t * divergence(provisional, dx, dy)
    pressure = _pressure_poisson(rhs, dx, dy, pressure_iterations, pressure)
    u_star, v_star = (provisional[0], provisional[1])
    next_velocity = provisional.clone()
    next_velocity[0, 1:-1, 1:-1] = u_star[1:-1, 1:-1] - delta_t / rho * (
        pressure[1:-1, 2:] - pressure[1:-1, :-2]
    ) / (2.0 * dx)
    next_velocity[1, 1:-1, 1:-1] = v_star[1:-1, 1:-1] - delta_t / rho * (
        pressure[2:, 1:-1] - pressure[:-2, 1:-1]
    ) / (2.0 * dy)
    _enforce_flow_constraints_(next_velocity, obstacle_mask)
    return (next_velocity, pressure)


def _get_compiled_provisional_update():
    if not hasattr(torch, "compile"):
        return _advection_diffusion_provisional
    cached = _COMPILED_STEP_CACHE.get("provisional")
    if cached is None:
        cached = torch.compile(
            _advection_diffusion_provisional,
            fullgraph=False,
            options={"triton.cudagraphs": False},
        )
        _COMPILED_STEP_CACHE["provisional"] = cached
    return cached


def simulate_navier_stokes(
    velocity0: torch.Tensor,
    nu: float,
    rho: float,
    T: float = 5.0,
    delta_t: Optional[float] = None,
    pressure_iterations: int = 80,
    save_every: int = 10,
    cfl: float = 0.4,
    diffusion_safety: float = 0.2,
    max_delta_t: float = 0.005,
    return_pressure: bool = False,
    pressure_method: str = "spectral",
    use_compile: bool = False,
    obstacle_mask: Optional[torch.Tensor] = None,
    initial_projection_steps: int = 0,
    initial_projection_delta_t: float = 1.0,
    initial_relaxation_steps: int = 0,
    initial_relaxation_delta_t: Optional[float] = None,
    sample_times: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Simulate 2D incompressible Navier-Stokes on [-1, 1]^2.

    Uses an explicit advection/diffusion step followed by a pressure projection.
    pressure_method='spectral' uses a fast FFT Helmholtz projection; 'jacobi'
    keeps the original iterative pressure solve. If delta_t is None, a stable
    step is recomputed from the CFL and diffusion limits. Boundary velocities
    are clamped to rightward unit flow and the optional obstacle is clamped to
    zero after each substep. The raw initial condition is always used for
    samples at t=0. Optional initial projection passes and full initial
    relaxation steps are applied before positive-time samples only.

    Returns a dict containing:
      velocity: saved velocity snapshots, shape (frames, 2, ny, nx)
      time: snapshot times
      delta_t: per-step time increments
      pressure: saved pressures, only when return_pressure=True
    """
    if velocity0.ndim != 3 or velocity0.shape[0] != 2:
        raise ValueError("velocity0 must have shape (2, ny, nx)")
    if nu < 0.0:
        raise ValueError("nu must be nonnegative")
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    if T <= 0.0:
        raise ValueError("T must be positive")
    method = pressure_method.lower()
    if method in {"spectral_periodic", "fft"}:
        method = "spectral"
    if method not in {"spectral", "jacobi"}:
        raise ValueError("pressure_method must be 'spectral' or 'jacobi'")
    if save_every < 1:
        raise ValueError("save_every must be at least 1")
    if method == "jacobi" and pressure_iterations < 1:
        raise ValueError("pressure_iterations must be at least 1")
    if initial_projection_steps < 0:
        raise ValueError("initial_projection_steps must be nonnegative")
    if initial_projection_delta_t <= 0.0:
        raise ValueError("initial_projection_delta_t must be positive")
    if initial_relaxation_steps < 0:
        raise ValueError("initial_relaxation_steps must be nonnegative")
    if initial_relaxation_delta_t is not None and initial_relaxation_delta_t <= 0.0:
        raise ValueError("initial_relaxation_delta_t must be positive or None")
    sample_times_cpu = None
    if sample_times is not None:
        sample_times_cpu = torch.as_tensor(
            sample_times, dtype=torch.float64, device="cpu"
        ).flatten()
        if sample_times_cpu.numel() == 0:
            raise ValueError("sample_times must contain at least one time")
        if float(sample_times_cpu[0]) < -1e-12:
            raise ValueError("sample_times must be nonnegative")
        if float(sample_times_cpu[-1]) > float(T) + 1e-12:
            raise ValueError("sample_times cannot extend past T")
        if sample_times_cpu.numel() > 1 and torch.any(
            sample_times_cpu[1:] < sample_times_cpu[:-1] - 1e-12
        ):
            raise ValueError("sample_times must be sorted")
        sample_times_cpu = sample_times_cpu.clamp(0.0, float(T))
        if (
            float(sample_times_cpu[0]) <= 1e-12
            and (int(initial_projection_steps) > 0 or int(initial_relaxation_steps) > 0)
        ):
            warnings.warn(
                "initial_projection_steps/initial_relaxation_steps run off-camera "
                "between the raw t=0 frame and positive-time samples. Set both to "
                "0 when frame 1 should be exactly one frame interval after frame 0."
            )
    velocity = velocity0.detach().clone()
    if not torch.is_floating_point(velocity):
        velocity = velocity.float()
    _, ny, nx = velocity.shape
    if obstacle_mask is not None:
        obstacle_mask = obstacle_mask.to(device=velocity.device, dtype=torch.bool)
        if obstacle_mask.shape != (ny, nx):
            raise ValueError(
                "obstacle_mask must have shape (ny, nx) matching velocity0"
            )
    _enforce_flow_constraints_(velocity, obstacle_mask)
    raw_initial_velocity = velocity.detach().clone()
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)
    pressure = torch.zeros((ny, nx), device=velocity.device, dtype=velocity.dtype)
    raw_initial_pressure = pressure.detach().clone()
    spectral_operators = None
    if method == "spectral":
        spectral_operators = _spectral_wavenumbers(
            ny, nx, dx, dy, velocity.device, velocity.dtype
        )
        provisional_function = (
            _get_compiled_provisional_update()
            if use_compile
            else _advection_diffusion_provisional
        )
    else:
        if use_compile:
            warnings.warn(
                "use_compile currently applies only to pressure_method='spectral'; using uncompiled Jacobi."
            )
        step_function = _navier_stokes_step
    nu_tensor = torch.as_tensor(float(nu), device=velocity.device, dtype=velocity.dtype)
    rho_tensor = torch.as_tensor(
        float(rho), device=velocity.device, dtype=velocity.dtype
    )

    def choose_step_dt(current_velocity: torch.Tensor) -> float:
        if delta_t is None:
            return stable_timestep(
                current_velocity, nu, dx, dy, cfl, diffusion_safety, max_delta_t
            )
        dt_value = float(delta_t)
        if dt_value <= 0.0:
            raise ValueError("delta_t must be positive")
        return dt_value

    def choose_relaxation_dt(current_velocity: torch.Tensor) -> float:
        if initial_relaxation_delta_t is not None:
            return float(initial_relaxation_delta_t)
        return choose_step_dt(current_velocity)

    def advance_step(
        current_velocity: torch.Tensor, current_pressure: torch.Tensor, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nonlocal use_compile, provisional_function
        if method == "spectral":
            dt_arg = (
                torch.as_tensor(
                    dt, device=current_velocity.device, dtype=current_velocity.dtype
                )
                if use_compile
                else dt
            )
            nu_arg = nu_tensor if use_compile else float(nu)
            rho_arg = rho_tensor if use_compile else float(rho)
            try:
                if use_compile:
                    provisional = provisional_function(
                        current_velocity, nu_arg, dt_arg, dx, dy
                    )
                    _enforce_flow_constraints_(provisional, obstacle_mask)
                    next_velocity, next_pressure = (
                        _project_velocity_spectral_periodic_with_operators(
                            provisional, rho_arg, dt_arg, *spectral_operators
                        )
                    )
                    _enforce_flow_constraints_(next_velocity, obstacle_mask)
                    return (next_velocity, next_pressure)
                return _navier_stokes_step_spectral_fast(
                    current_velocity,
                    current_pressure,
                    nu_arg,
                    rho_arg,
                    dt_arg,
                    dx,
                    dy,
                    *spectral_operators,
                    obstacle_mask=obstacle_mask,
                )
            except Exception as exc:
                if not use_compile:
                    raise
                warnings.warn(
                    f"torch.compile provisional update failed ({exc}); falling back to eager spectral stepping."
                )
                use_compile = False
                provisional_function = _advection_diffusion_provisional
                return _navier_stokes_step_spectral_fast(
                    current_velocity,
                    current_pressure,
                    float(nu),
                    float(rho),
                    dt,
                    dx,
                    dy,
                    *spectral_operators,
                    obstacle_mask=obstacle_mask,
                )
        return step_function(
            current_velocity,
            current_pressure,
            float(nu),
            float(rho),
            dt,
            dx,
            dy,
            pressure_iterations,
            method,
            obstacle_mask=obstacle_mask,
        )

    projection_delta_ts = []
    if int(initial_projection_steps) > 0:
        projection_operators = spectral_operators
        if projection_operators is None:
            projection_operators = _spectral_wavenumbers(
                ny, nx, dx, dy, velocity.device, velocity.dtype
            )
        projection_dt = torch.as_tensor(
            float(initial_projection_delta_t),
            device=velocity.device,
            dtype=velocity.dtype,
        )
        for _ in range(int(initial_projection_steps)):
            _enforce_flow_constraints_(velocity, obstacle_mask)
            velocity, pressure = _project_velocity_spectral_periodic_with_operators(
                velocity, rho_tensor, projection_dt, *projection_operators
            )
            _enforce_flow_constraints_(velocity, obstacle_mask)
            projection_delta_ts.append(float(initial_projection_delta_t))
    relaxation_delta_ts = []
    for _ in range(int(initial_relaxation_steps)):
        warmup_dt = choose_relaxation_dt(velocity)
        velocity, pressure = advance_step(velocity, pressure, warmup_dt)
        relaxation_delta_ts.append(warmup_dt)
    frames = []
    times = []
    delta_ts = []
    pressures = [] if return_pressure else None
    t = 0.0
    step = 0
    next_sample = 0
    if sample_times_cpu is None:
        frames.append(raw_initial_velocity.detach().clone())
        times.append(0.0)
        if return_pressure:
            pressures.append(raw_initial_pressure.detach().clone())
    else:
        while (
            next_sample < sample_times_cpu.numel()
            and float(sample_times_cpu[next_sample]) <= 1e-12
        ):
            frames.append(raw_initial_velocity.detach().clone())
            times.append(float(sample_times_cpu[next_sample]))
            if return_pressure:
                pressures.append(raw_initial_pressure.detach().clone())
            next_sample += 1
    while t < float(T) - 1e-12:
        next_target_time = float(T)
        if sample_times_cpu is not None and next_sample < sample_times_cpu.numel():
            next_target_time = min(
                next_target_time, float(sample_times_cpu[next_sample])
            )
            if next_target_time <= t + 1e-12:
                frames.append(velocity.detach().clone())
                times.append(float(sample_times_cpu[next_sample]))
                if return_pressure:
                    pressures.append(pressure.detach().clone())
                next_sample += 1
                continue
        dt = min(choose_step_dt(velocity), float(T) - t, max(next_target_time - t, 0.0))
        if dt <= 1e-12:
            break
        velocity, pressure = advance_step(velocity, pressure, dt)
        t += dt
        step += 1
        delta_ts.append(dt)
        if sample_times_cpu is None and (
            step % save_every == 0 or t >= float(T) - 1e-12
        ):
            frames.append(velocity.detach().clone())
            times.append(t)
            if return_pressure:
                pressures.append(pressure.detach().clone())
        while (
            sample_times_cpu is not None
            and next_sample < sample_times_cpu.numel()
            and float(sample_times_cpu[next_sample]) <= t + 1e-10
        ):
            frames.append(velocity.detach().clone())
            times.append(float(sample_times_cpu[next_sample]))
            if return_pressure:
                pressures.append(pressure.detach().clone())
            next_sample += 1
    while sample_times_cpu is not None and next_sample < sample_times_cpu.numel():
        frames.append(velocity.detach().clone())
        times.append(float(sample_times_cpu[next_sample]))
        if return_pressure:
            pressures.append(pressure.detach().clone())
        next_sample += 1
    result = {
        "velocity": torch.stack(frames, dim=0),
        "time": torch.tensor(times, device=velocity.device, dtype=velocity.dtype),
        "delta_t": torch.tensor(delta_ts, device=velocity.device, dtype=velocity.dtype),
        "initial_projection_delta_t": torch.tensor(
            projection_delta_ts, device=velocity.device, dtype=velocity.dtype
        ),
        "initial_relaxation_delta_t": torch.tensor(
            relaxation_delta_ts, device=velocity.device, dtype=velocity.dtype
        ),
    }
    if obstacle_mask is not None:
        result["obstacle_mask"] = obstacle_mask.detach().clone()
    if return_pressure:
        result["pressure"] = torch.stack(pressures, dim=0)
    return result


def _simulation_scalar_field(
    velocity: torch.Tensor, field: str, dx: float, dy: float
) -> torch.Tensor:
    if field == "speed":
        return velocity.square().sum(dim=1).sqrt()
    if field == "u":
        return velocity[:, 0]
    if field == "v":
        return velocity[:, 1]
    if field == "vorticity":
        u = velocity[:, 0]
        v = velocity[:, 1]
        vort = torch.zeros_like(u)
        vort[:, 1:-1, 1:-1] = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2.0 * dx) - (
            u[:, 2:, 1:-1] - u[:, :-2, 1:-1]
        ) / (2.0 * dy)
        return vort
    raise ValueError("field must be one of: speed, vorticity, u, v")


def _is_velocity_component_field(field: str) -> bool:
    return field in {"speed", "velocity", "components", "uv"}


def _sampled_quantile(
    values: torch.Tensor, q: float, max_samples: int = 1000000
) -> torch.Tensor:
    flat = values.reshape(-1)
    sample_count = min(flat.numel(), int(max_samples))
    if flat.numel() > sample_count:
        indices = torch.arange(sample_count, device=flat.device, dtype=torch.int64)
        indices = indices * (flat.numel() - 1) // max(1, sample_count - 1)
        flat = flat.index_select(0, indices)
    return torch.quantile(flat, float(q))


def _obstacle_mask_for_rgb(
    obstacle_mask: Optional[torch.Tensor],
    target_hw: Tuple[int, int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if obstacle_mask is None:
        return None
    mask = obstacle_mask.detach().to(device=device, dtype=torch.bool)
    target_hw = (int(target_hw[0]), int(target_hw[1]))
    if tuple(mask.shape) != target_hw:
        mask = F.interpolate(mask.float()[None, None], size=target_hw, mode="nearest")[
            0, 0
        ].bool()
    return mask


def _velocity_components_to_rgb(
    velocity: torch.Tensor,
    output_size: Optional[int | Tuple[int, int]] = None,
    component_bound: Optional[float] = None,
    obstacle_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Encode signed 2D velocity components as RGB without taking a norm."""
    components = velocity[:, :2].float()
    _, _, ny, nx = components.shape
    output_hw = _resolve_output_hw(output_size)
    if output_hw is not None and output_hw != (ny, nx):
        components = F.interpolate(
            components,
            size=output_hw,
            mode="bilinear",
            align_corners=False,
        )
    if component_bound is None:
        bound = _sampled_quantile(components.abs(), 0.995)
    else:
        bound = torch.as_tensor(
            float(component_bound), device=components.device, dtype=components.dtype
        )
    bound = torch.clamp(
        bound, min=torch.tensor(1e-12, device=components.device, dtype=components.dtype)
    )
    encoded = (0.5 + 0.5 * components / bound).clamp(0.0, 1.0)
    blue = torch.full_like(encoded[:, 0], 0.5)
    rgb = torch.stack((encoded[:, 0], encoded[:, 1], blue), dim=-1)
    mask = _obstacle_mask_for_rgb(obstacle_mask, rgb.shape[1:3], rgb.device)
    if mask is not None:
        rgb = rgb.masked_fill(mask[None, :, :, None], 0.0)
    return (rgb, float(bound.detach().cpu()))


def _realtime_video_frame_count(T: float, fps: int, warn: bool = True) -> int:
    if T <= 0.0:
        raise ValueError("T must be positive")
    if fps < 1:
        raise ValueError("fps must be positive")
    raw_count = float(T) * int(fps)
    frame_count = max(1, int(round(raw_count)))
    if warn and not math.isclose(raw_count, frame_count, rel_tol=0.0, abs_tol=1e-6):
        warnings.warn(
            f"T * fps is {raw_count:.6g}; using {frame_count} video frames "
            f"({frame_count / float(fps):.6g}s at {int(fps)} fps)."
        )
    return frame_count


def _realtime_video_sample_times(
    T: float, fps: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    frame_count = _realtime_video_frame_count(T, fps)
    return torch.arange(frame_count, device=device, dtype=dtype) / float(fps)


def _odd_realtime_video_sample_times(
    T: float, fps: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    sample_times = _realtime_video_sample_times(T, fps, device, dtype)
    if sample_times.numel() % 2 == 0:
        if sample_times.numel() <= 1:
            raise ValueError("cannot drop the last frame from a one-frame video")
        sample_times = sample_times[:-1]
    return sample_times


def _odd_realtime_video_frame_count(T: float, fps: int, warn: bool = True) -> int:
    frame_count = _realtime_video_frame_count(T, fps, warn=warn)
    if frame_count % 2 == 0:
        if frame_count <= 1:
            raise ValueError("cannot drop the last frame from a one-frame video")
        frame_count -= 1
    return frame_count


def _realtime_video_timing_metadata(
    T: float, fps: int, frame_count: Optional[int] = None
) -> Dict[str, str | float | int | bool]:
    if frame_count is None:
        frame_count = _realtime_video_frame_count(T, fps, warn=False)
    frame_count = int(frame_count)
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    playback_duration = frame_count / float(fps)
    raw_count = float(T) * int(fps)
    realtime_frame_count = _realtime_video_frame_count(T, fps, warn=False)
    exact_duration = math.isclose(raw_count, frame_count, rel_tol=0.0, abs_tol=1e-6)
    return {
        "fps": int(fps),
        "frames": frame_count,
        "realtime_frame_count_before_odd_trim": realtime_frame_count,
        "dropped_last_frame_for_odd_count": realtime_frame_count != frame_count,
        "requested_duration_seconds": float(T),
        "playback_duration_seconds": playback_duration,
        "last_frame_time_seconds": (frame_count - 1) / float(fps),
        "duration_error_seconds": playback_duration - float(T),
        "exact_duration_at_fps": exact_duration,
        "odd_frame_count": frame_count % 2 == 1,
        "frame_time_convention": "frame_start_times",
    }


def _verify_realtime_video_tensor(
    frames: torch.Tensor,
    T: float,
    fps: int,
    expected_frame_count: Optional[int] = None,
    label: str = "video",
) -> Dict[str, str | float | int | bool]:
    expected_frame_count = (
        _realtime_video_frame_count(T, fps, warn=False)
        if expected_frame_count is None
        else int(expected_frame_count)
    )
    if frames.ndim != 4:
        raise RuntimeError(
            f"{label} frames must have shape (frames, height, width, channels)"
        )
    actual_frame_count = int(frames.shape[0])
    if actual_frame_count != expected_frame_count:
        raise RuntimeError(
            f"{label} should contain {expected_frame_count} frames for "
            f"{float(T)}s at {int(fps)} fps, got {actual_frame_count}"
        )
    return _realtime_video_timing_metadata(T, fps, expected_frame_count)


def _verify_written_video_timing(
    video_path: str | Path,
    T: float,
    fps: int,
    expected_frame_count: Optional[int] = None,
    label: str = "video",
) -> Dict[str, str | float | int | bool]:
    expected_frame_count = (
        _realtime_video_frame_count(T, fps, warn=False)
        if expected_frame_count is None
        else int(expected_frame_count)
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        timestamps, container_fps = read_video_timestamps(
            str(video_path), pts_unit="sec"
        )
    if container_fps is None:
        raise RuntimeError(f"{label} has no readable fps metadata: {video_path}")
    actual_fps = float(container_fps)
    if not math.isclose(actual_fps, float(fps), rel_tol=0.0, abs_tol=1e-3):
        raise RuntimeError(
            f"{label} fps metadata mismatch for {video_path}: "
            f"expected {int(fps)}, got {actual_fps:g}"
        )
    actual_frame_count = len(timestamps)
    if actual_frame_count != expected_frame_count:
        raise RuntimeError(
            f"{label} frame count mismatch for {video_path}: "
            f"expected {expected_frame_count}, got {actual_frame_count}"
        )
    first_timestamp = float(timestamps[0])
    last_timestamp = float(timestamps[-1])
    expected_last_timestamp = (expected_frame_count - 1) / float(fps)
    timestamp_tolerance = max(1e-3, 0.01 / float(fps))
    if not math.isclose(first_timestamp, 0.0, rel_tol=0.0, abs_tol=timestamp_tolerance):
        raise RuntimeError(
            f"{label} first timestamp mismatch for {video_path}: "
            f"expected 0, got {first_timestamp:g}"
        )
    if not math.isclose(
        last_timestamp,
        expected_last_timestamp,
        rel_tol=0.0,
        abs_tol=timestamp_tolerance,
    ):
        raise RuntimeError(
            f"{label} last timestamp mismatch for {video_path}: "
            f"expected {expected_last_timestamp:g}, got {last_timestamp:g}"
        )
    timing = _realtime_video_timing_metadata(T, fps, expected_frame_count)
    timing.update(
        {
            "timing_verified": True,
            "container_fps": actual_fps,
            "container_frames": actual_frame_count,
            "container_first_timestamp_seconds": first_timestamp,
            "container_last_timestamp_seconds": last_timestamp,
        }
    )
    return timing


def _rightward_flow_rgb(
    component_bound: Optional[float], device: torch.device
) -> torch.Tensor:
    velocity = torch.zeros((1, 2, 1, 1), device=device, dtype=torch.float32)
    velocity[:, 0].fill_(1.0)
    rgb, _ = _velocity_components_to_rgb(velocity, component_bound=component_bound)
    return rgb[0, 0, 0]


def _obstacle_mask_to_uint8_video_frames(
    obstacle_mask: torch.Tensor,
    output_size: Optional[int | Tuple[int, int]],
    frame_count: int,
    velocity_color_bound: Optional[float],
) -> Tuple[torch.Tensor, list[int]]:
    output_hw = _resolve_output_hw(output_size)
    if output_hw is None:
        grid_size = _navier_grid_size(None)
        output_hw = (grid_size, grid_size)
    mask = _obstacle_mask_for_rgb(obstacle_mask, output_hw, obstacle_mask.device)
    if mask is None:
        raise ValueError("obstacle_mask is required for mask video rendering")
    background_rgb = _rightward_flow_rgb(velocity_color_bound, mask.device)
    rgb = (
        background_rgb.view(1, 1, 1, 3)
        .expand(int(frame_count), output_hw[0], output_hw[1], 3)
        .clone()
    )
    rgb = rgb.masked_fill(mask[None, :, :, None], 0.0)
    frames = (255.0 * rgb).round().clamp(0, 255).to(torch.uint8).cpu()
    background_rgb_uint8 = (
        (255.0 * background_rgb).round().clamp(0, 255).to(torch.uint8).cpu().tolist()
    )
    return (frames, [int(channel) for channel in background_rgb_uint8])


def _select_video_indices(
    frame_count: int, max_frames: Optional[int], device: torch.device
) -> torch.Tensor:
    if max_frames is None or frame_count <= max_frames:
        return torch.arange(frame_count, device=device)
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2 when provided")
    return (
        torch.linspace(0, frame_count - 1, max_frames, device=device)
        .round()
        .long()
        .unique()
    )


def _colorize_unit_interval(values: torch.Tensor) -> torch.Tensor:
    """Fast torch-only blue/cyan/yellow/red color map for values in [0, 1]."""
    x = values.clamp(0.0, 1.0)
    red = (1.5 - (4.0 * x - 3.0).abs()).clamp(0.0, 1.0)
    green = (1.5 - (4.0 * x - 2.0).abs()).clamp(0.0, 1.0)
    blue = (1.5 - (4.0 * x - 1.0).abs()).clamp(0.0, 1.0)
    return torch.stack((red, green, blue), dim=-1)


def _simulation_to_uint8_video_frames(
    simulation: Dict[str, torch.Tensor],
    field: str = "speed",
    output_size: Optional[int | Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
    velocity_color_bound: Optional[float] = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a simulation to RGB uint8 frames for torchvision.write_video."""
    velocity = simulation["velocity"].detach()
    obstacle_mask = simulation.get("obstacle_mask")
    if obstacle_mask is not None:
        obstacle_mask = obstacle_mask.detach().to(device=velocity.device)
    if velocity.ndim != 4 or velocity.shape[1] != 2:
        raise ValueError("simulation['velocity'] must have shape (frames, 2, ny, nx)")
    if max_frames is not None and int(max_frames) != int(velocity.shape[0]):
        warnings.warn(
            "max_frames is ignored; dataset videos use the odd realtime frame count."
        )
    frame_indices = torch.arange(velocity.shape[0], device=velocity.device)
    _, _, ny, nx = velocity.shape
    output_hw = _resolve_output_hw(output_size, default_hw=(ny, nx))
    if _is_velocity_component_field(field):
        rgb, _ = _velocity_components_to_rgb(
            velocity,
            output_size=output_hw,
            component_bound=velocity_color_bound,
            obstacle_mask=obstacle_mask,
        )
    else:
        dx = 2.0 / (nx - 1)
        dy = 2.0 / (ny - 1)
        scalar = _simulation_scalar_field(velocity, field, dx, dy)
        if output_hw != (ny, nx):
            scalar = F.interpolate(
                scalar[:, None],
                size=output_hw,
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        if field in {"vorticity", "u", "v"}:
            bound = _sampled_quantile(scalar.abs(), 0.98)
            bound = torch.clamp(
                bound, min=torch.tensor(1e-12, device=scalar.device, dtype=scalar.dtype)
            )
            normalized = 0.5 + 0.5 * scalar / bound
        else:
            vmax = _sampled_quantile(scalar, 0.99)
            vmax = torch.clamp(
                vmax, min=torch.tensor(1e-12, device=scalar.device, dtype=scalar.dtype)
            )
            normalized = scalar / vmax
        rgb = _colorize_unit_interval(normalized)
        mask = _obstacle_mask_for_rgb(obstacle_mask, rgb.shape[1:3], rgb.device)
        if mask is not None:
            rgb = rgb.masked_fill(mask[None, :, :, None], 0.0)
    frames = (255.0 * rgb).round().clamp(0, 255).to(torch.uint8).cpu()
    return (frames, frame_indices.detach().cpu())


def _make_random_sample_dir(root: Path, uid_bytes: int = 8) -> Tuple[str, Path]:
    for _ in range(1000):
        uid = secrets.token_hex(uid_bytes)
        sample_dir = root / uid
        try:
            sample_dir.mkdir(parents=True, exist_ok=False)
            return (uid, sample_dir)
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a unique sample directory under {root}")


def generate_navier_dataset(
    N: int,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    n: Optional[int] = None,
    video_width: int = DEFAULT_VIDEO_WIDTH,
    video_height: int = DEFAULT_VIDEO_HEIGHT,
    T: float = 5.0,
    nu_range: Tuple[float, float] = (3e-4, 3e-3),
    rho_range: Tuple[float, float] = (0.5, 2.0),
    modes: int = 8,
    amplitude: float = 1.0,
    fps: int = 48,
    save_every: int = 10,
    max_video_frames: Optional[int] = None,
    field: str = "speed",
    delta_t: Optional[float] = None,
    pressure_method: str = "spectral",
    pressure_iterations: int = 80,
    max_delta_t: float = 5e-3,
    use_compile: bool = True,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    video_codec: str = "libx264",
    video_options: Optional[Dict[str, str]] = None,
    velocity_color_bound: Optional[float] = None,
    initial_projection_steps: int = 0,
    initial_projection_delta_t: float = 1.0,
    initial_relaxation_steps: int = 0,
    initial_relaxation_delta_t: Optional[float] = None,
) -> list[Dict[str, str | float | int | bool]]:
    """Generate N samples with flow videos and metadata."""
    if N < 1:
        raise ValueError("N must be positive")
    n = _navier_grid_size(n)
    video_hw = _video_size(video_width, video_height)
    video_height, video_width = video_hw
    if nu_range[0] <= 0.0 or nu_range[1] <= nu_range[0]:
        raise ValueError("nu_range must be positive and increasing")
    if rho_range[0] <= 0.0 or rho_range[1] <= rho_range[0]:
        raise ValueError("rho_range must be positive and increasing")
    if delta_t is not None and delta_t <= 0.0:
        raise ValueError("delta_t must be positive or None")
    if int(initial_projection_steps) != 0 or int(initial_relaxation_steps) != 0:
        raise ValueError(
            "dataset videos do not support off-camera initial projection or "
            "relaxation; use realtime simulation steps from t=0 instead"
        )
    if max_video_frames is not None:
        warnings.warn(
            "max_video_frames is ignored; videos use the odd realtime frame count."
        )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    rng = torch.Generator(device="cpu")
    if seed is None:
        rng.seed()
    else:
        rng.manual_seed(seed)
    if video_options is None:
        video_options = {"crf": "18", "preset": "veryfast"}
    if velocity_color_bound is None:
        velocity_color_bound = float(amplitude)

    sample_times = _odd_realtime_video_sample_times(
        float(T), int(fps), torch.device("cpu"), torch.float64
    )
    video_frame_count = int(sample_times.numel())
    expected_video_timing = _realtime_video_timing_metadata(
        float(T), int(fps), video_frame_count
    )
    records = []

    from tqdm import tqdm

    for sample_index in tqdm(range(int(N)), desc="Generating samples"):
        uid, sample_dir = _make_random_sample_dir(root)
        sample_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        sample_nu = float(
            torch.empty((), dtype=torch.float64)
            .uniform_(math.log(nu_range[0]), math.log(nu_range[1]), generator=rng)
            .exp()
            .item()
        )
        sample_density = float(
            torch.empty((), dtype=torch.float64)
            .uniform_(*rho_range, generator=rng)
            .item()
        )

        velocity0, _, obstacle = make_initial_velocity_field(
            n=n,
            amplitude=amplitude,
            modes=modes,
            seed=sample_seed,
            device=device,
            dtype=dtype,
            return_obstacle=True,
        )
        simulation = simulate_navier_stokes(
            velocity0,
            nu=sample_nu,
            rho=sample_density,
            T=T,
            delta_t=delta_t,
            pressure_iterations=pressure_iterations,
            save_every=save_every,
            max_delta_t=max_delta_t,
            pressure_method=pressure_method,
            use_compile=use_compile,
            obstacle_mask=obstacle["mask"],
            initial_projection_steps=initial_projection_steps,
            initial_projection_delta_t=initial_projection_delta_t,
            initial_relaxation_steps=initial_relaxation_steps,
            initial_relaxation_delta_t=initial_relaxation_delta_t,
            sample_times=sample_times,
        )
        expected_initial_velocity = velocity0.detach().clone()
        _enforce_flow_constraints_(expected_initial_velocity, obstacle["mask"])
        if not torch.allclose(
            simulation["velocity"][0], expected_initial_velocity, rtol=0.0, atol=1e-6
        ):
            raise RuntimeError(
                "simulation first frame does not match the raw initial condition"
            )

        frames, frame_indices = _simulation_to_uint8_video_frames(
            simulation,
            field=field,
            output_size=video_hw,
            velocity_color_bound=velocity_color_bound,
        )
        video_timing = _verify_realtime_video_tensor(
            frames,
            float(T),
            int(fps),
            expected_frame_count=video_frame_count,
            label="flow video",
        )

        video_path = sample_dir / "video.mp4"
        write_video(
            str(video_path),
            frames,
            fps=fps,
            video_codec=video_codec,
            options=video_options,
        )
        video_timing = _verify_written_video_timing(
            video_path,
            float(T),
            int(fps),
            expected_frame_count=video_frame_count,
            label="flow video",
        )

        video_metadata = {
            "file": "video.mp4",
            "height": int(frames.shape[1]),
            "width": int(frames.shape[2]),
            "fps": int(fps),
            "frames": int(frames.shape[0]),
            "first_frame": "raw_initial_condition",
            "timing": video_timing,
            "field": field,
            "encoding": (
                "signed_velocity_components_rgb"
                if _is_velocity_component_field(field)
                else "scalar_colormap"
            ),
            "red": (
                "0.5 + 0.5 * u / component_bound"
                if _is_velocity_component_field(field)
                else None
            ),
            "green": (
                "0.5 + 0.5 * v / component_bound"
                if _is_velocity_component_field(field)
                else None
            ),
            "blue": "0.5" if _is_velocity_component_field(field) else None,
            "component_bound": (
                float(velocity_color_bound)
                if _is_velocity_component_field(field)
                else None
            ),
            "colormap": None if _is_velocity_component_field(field) else "torch_jet",
            "frame_indices": frame_indices.tolist(),
        }
        metadata = {
            "uid": uid,
            "sample_index": sample_index,
            "seed": sample_seed,
            "grid_size": n,
            "video_height": int(video_height),
            "video_width": int(video_width),
            "domain": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
            "nu": sample_nu,
            "rho": sample_density,
            "T": float(T),
            "fixed_delta_t": None if delta_t is None else float(delta_t),
            "pressure_method": pressure_method,
            "use_compile": bool(use_compile),
            "modes": modes,
            "amplitude": float(amplitude),
            "video": video_metadata,
            "simulation_time": simulation["time"].detach().cpu().tolist(),
            "delta_t": simulation["delta_t"].detach().cpu().tolist(),
            "initial_projection_steps": int(initial_projection_steps),
            "initial_projection_delta_t": simulation["initial_projection_delta_t"]
            .detach()
            .cpu()
            .tolist(),
            "initial_relaxation_steps": int(initial_relaxation_steps),
            "initial_relaxation_delta_t": simulation["initial_relaxation_delta_t"]
            .detach()
            .cpu()
            .tolist(),
            "raw_initial_condition": "u=1, v=0 outside obstacle; u=0, v=0 inside obstacle",
            "initial_velocity_shape": list(simulation["velocity"][0].shape),
            "initial_velocity": simulation["velocity"][0]
            .detach()
            .to("cpu", dtype=torch.float32)
            .tolist(),
        }
        json_path = sample_dir / "metadata.json"
        with json_path.open("w") as f:
            json.dump(metadata, f)

        records.append(
            {
                "uid": uid,
                "folder": str(sample_dir),
                "json": str(json_path),
                "video": str(video_path),
                "nu": sample_nu,
                "rho": sample_density,
                "frames": int(frames.shape[0]),
                "video_height": int(frames.shape[1]),
                "video_width": int(frames.shape[2]),
                "video_duration_seconds": float(
                    expected_video_timing["playback_duration_seconds"]
                ),
                "timing_verified": bool(video_timing["timing_verified"]),
            }
        )
    return records


def _load_wan2_2_vae_class(wan_repo_root: str | Path = WAN_REPO_ROOT):
    """Load Wan2.2's VAE class without importing wan/__init__.py extras."""
    module_path = Path(wan_repo_root) / "wan" / "modules" / "vae2_2.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Wan2.2 VAE module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("wan_vae2_2_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Wan2_2_VAE


def load_wan_ti2v_5b_vae(
    checkpoint_dir: str | Path = WAN_CHECKPOINT_DIR,
    wan_repo_root: str | Path = WAN_REPO_ROOT,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """Load the Wan2.2 TI2V-5B VAE from /home/azureuser/Wan2.2-TI2V-5B."""
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    checkpoint_path = Path(checkpoint_dir) / WAN_TI2V_5B_VAE_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")
    Wan2_2_VAE = _load_wan2_2_vae_class(wan_repo_root)
    vae = Wan2_2_VAE(vae_pth=str(checkpoint_path), dtype=dtype, device=device)
    return vae


def load_video_for_wan_vae(
    video_path: str | Path,
    device: Optional[str | torch.device] = None,
    expected_size: Tuple[int, int] = DEFAULT_WAN_EXPECTED_SIZE,
    resize: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Read an mp4 and return Wan-normalized video shaped (C, T, H, W)."""
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        frames, _, info = read_video(
            str(video_path), pts_unit="sec", output_format="TCHW"
        )
    if frames.numel() == 0:
        raise ValueError(f"No video frames decoded from {video_path}")
    if frames.shape[1] < 3:
        frames = frames.repeat(1, 3, 1, 1)
    frames = frames[:, :3]
    frames = (
        frames.float().div_(255.0)
        if frames.dtype == torch.uint8
        else frames.float().clamp_(0.0, 1.0)
    )
    target_h, target_w = expected_size
    if tuple(frames.shape[-2:]) != (target_h, target_w):
        if not resize:
            raise ValueError(
                f"Expected {expected_size} video frames, got {tuple(frames.shape[-2:])} for {video_path}"
            )
        frames = F.interpolate(
            frames, size=expected_size, mode="bicubic", align_corners=False
        ).clamp_(0.0, 1.0)
    video = frames.mul_(2.0).sub_(1.0).permute(1, 0, 2, 3).contiguous().to(device)
    return (video, dict(info))


@torch.no_grad()
def encode_video_to_wan_latents(
    video_path: str | Path,
    vae=None,
    checkpoint_dir: str | Path = WAN_CHECKPOINT_DIR,
    wan_repo_root: str | Path = WAN_REPO_ROOT,
    device: Optional[str | torch.device] = None,
    vae_dtype: torch.dtype = torch.float32,
    save_dtype: torch.dtype = torch.float16,
    output_path: Optional[str | Path] = None,
    overwrite: bool = False,
    expected_size: Tuple[int, int] = DEFAULT_WAN_EXPECTED_SIZE,
    resize: bool = False,
) -> Dict[str, str | int | float | Tuple[int, ...]]:
    """Encode one dataset mp4 and save latents.safetensors beside it."""
    video_path = Path(video_path)
    output_path = (
        video_path.with_name("latents.safetensors")
        if output_path is None
        else Path(output_path)
    )
    if output_path.exists() and (not overwrite):
        return {
            "status": "skipped",
            "reason": "exists",
            "video": str(video_path),
            "latents": str(output_path),
        }
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    if vae is None:
        vae = load_wan_ti2v_5b_vae(
            checkpoint_dir, wan_repo_root, device=device, dtype=vae_dtype
        )
    video, video_info = load_video_for_wan_vae(
        video_path, device=device, expected_size=expected_size, resize=resize
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*"
        )
        latents = (
            vae.encode([video])[0].detach().to("cpu", dtype=save_dtype).contiguous()
        )
    metadata = {
        "source_video": video_path.name,
        "vae": "Wan2.2 TI2V-5B VAE",
        "vae_checkpoint": str(Path(checkpoint_dir) / WAN_TI2V_5B_VAE_CHECKPOINT),
        "vae_stride": json.dumps(WAN_TI2V_5B_VAE_STRIDE),
        "normalization": "uint8 [0,255] -> [0,1] -> [-1,1]",
        "input_shape_cthw": json.dumps(tuple(video.shape)),
        "latent_shape_cthw": json.dumps(tuple(latents.shape)),
        "latent_dtype": str(latents.dtype),
        "video_info": json.dumps(video_info, sort_keys=True),
    }
    save_file({"latents": latents}, str(output_path), metadata=metadata)
    return {
        "status": "ok",
        "video": str(video_path),
        "latents": str(output_path),
        "frames": int(video.shape[1]),
        "latent_shape": tuple(latents.shape),
    }


def _torch_dtype(name: str) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {name}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Navier-Stokes dataset and Wan VAE latents.",
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--wan-repo-root", type=Path, default=DEFAULT_WAN_REPO_ROOT)
    parser.add_argument(
        "--wan-checkpoint-dir", type=Path, default=DEFAULT_WAN_CHECKPOINT_DIR
    )

    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--delta-t", type=float, default=None)
    parser.add_argument("--max-delta-t", type=float, default=5e-3)
    parser.add_argument("--nu-min", type=float, default=3e-4)
    parser.add_argument("--nu-max", type=float, default=3e-3)
    parser.add_argument("--rho-min", type=float, default=0.5)
    parser.add_argument("--rho-max", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=48)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--pressure-method", type=str, default="spectral")
    parser.add_argument("--pressure-iterations", type=int, default=80)
    parser.add_argument("--initial-projection-steps", type=int, default=0)
    parser.add_argument("--initial-projection-delta-t", type=float, default=1.0)
    parser.add_argument("--initial-relaxation-steps", type=int, default=0)
    parser.add_argument("--initial-relaxation-delta-t", type=float, default=None)
    parser.add_argument("--no-compile", dest="use_compile", action="store_false")
    parser.set_defaults(use_compile=True)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sim-dtype", type=_torch_dtype, default=torch.float32)
    parser.add_argument("--vae-dtype", type=_torch_dtype, default=torch.float32)
    parser.add_argument("--latent-save-dtype", type=_torch_dtype, default=torch.float16)
    parser.add_argument("--overwrite-latents", action="store_true")
    parser.add_argument("--resize-latents", action="store_true")
    parser.add_argument(
        "--stop-on-latent-error", dest="continue_on_latent_error", action="store_false"
    )
    parser.set_defaults(continue_on_latent_error=True)

    parser.add_argument("--records-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.grid_size < 32:
        raise ValueError("--grid-size must be at least 32")
    video_height, video_width = _video_size(args.width, args.height)
    if args.nu_min <= 0 or args.nu_max <= args.nu_min:
        raise ValueError("--nu-min/--nu-max must be positive and increasing")
    if args.rho_min <= 0 or args.rho_max <= args.rho_min:
        raise ValueError("--rho-min/--rho-max must be positive and increasing")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    global NAVIER_GRID_SIZE
    NAVIER_GRID_SIZE = int(args.grid_size)

    records = generate_navier_dataset(
        int(args.samples),
        output_root=args.output_root,
        n=int(args.grid_size),
        video_width=video_width,
        video_height=video_height,
        T=float(args.T),
        nu_range=(float(args.nu_min), float(args.nu_max)),
        rho_range=(float(args.rho_min), float(args.rho_max)),
        fps=int(args.fps),
        save_every=int(args.save_every),
        field="speed",
        delta_t=args.delta_t,
        pressure_method=args.pressure_method,
        pressure_iterations=int(args.pressure_iterations),
        max_delta_t=float(args.max_delta_t),
        use_compile=bool(args.use_compile),
        seed=args.seed,
        device=device,
        dtype=args.sim_dtype,
        velocity_color_bound=1.0,
        initial_projection_steps=int(args.initial_projection_steps),
        initial_projection_delta_t=float(args.initial_projection_delta_t),
        initial_relaxation_steps=int(args.initial_relaxation_steps),
        initial_relaxation_delta_t=args.initial_relaxation_delta_t,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    vae = load_wan_ti2v_5b_vae(
        checkpoint_dir=args.wan_checkpoint_dir,
        wan_repo_root=args.wan_repo_root,
        device=device,
        dtype=args.vae_dtype,
    )

    videos_to_encode = [
        (record, "video", Path(record["video"]), None) for record in records
    ]

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(videos_to_encode, desc="Encoding Wan VAE latents")
    except Exception:
        iterator = videos_to_encode

    latent_records = []
    for record, kind, video_path, output_path in iterator:
        try:
            latent_record = encode_video_to_wan_latents(
                video_path,
                vae=vae,
                checkpoint_dir=args.wan_checkpoint_dir,
                wan_repo_root=args.wan_repo_root,
                device=device,
                vae_dtype=args.vae_dtype,
                save_dtype=args.latent_save_dtype,
                output_path=output_path,
                overwrite=args.overwrite_latents,
                expected_size=(video_height, video_width),
                resize=args.resize_latents,
            )
        except Exception as exc:
            if not args.continue_on_latent_error:
                raise
            latent_record = {
                "status": "error",
                "video": str(video_path),
                "error": repr(exc),
            }
        latent_record["kind"] = kind
        latent_record["uid"] = record.get("uid")
        latent_records.append(latent_record)

    summary = {
        "samples_requested": int(args.samples),
        "samples_generated": len(records),
        "videos_generated": len(records),
        "latents_ok": sum(1 for item in latent_records if item.get("status") == "ok"),
        "latents_skipped": sum(
            1 for item in latent_records if item.get("status") == "skipped"
        ),
        "latents_error": sum(
            1 for item in latent_records if item.get("status") == "error"
        ),
        "video_latents_ok": sum(
            1
            for item in latent_records
            if item.get("kind") == "video" and item.get("status") == "ok"
        ),
        "output_root": str(args.output_root),
        "grid_size": int(args.grid_size),
        "video_height": int(video_height),
        "video_width": int(video_width),
        "fps": int(args.fps),
        "T": float(args.T),
        "frames_per_video": _odd_realtime_video_frame_count(
            float(args.T), int(args.fps), warn=False
        ),
        "video_timing": _realtime_video_timing_metadata(
            float(args.T),
            int(args.fps),
            _odd_realtime_video_frame_count(float(args.T), int(args.fps), warn=False),
        ),
    }
    print(json.dumps(summary, indent=2))

    if args.records_json is not None:
        args.records_json.parent.mkdir(parents=True, exist_ok=True)
        args.records_json.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "samples": records,
                    "latents": latent_records,
                },
                indent=2,
            )
        )

    return 0 if summary["latents_error"] == 0 else 1


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        raise SystemExit(main())
