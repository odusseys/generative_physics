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
DEFAULT_WITH_VORTICITY_OUTPUT_ROOT = Path(
    "/home/azureuser/datasets/navier_with_vorticity"
)
DEFAULT_WAN_REPO_ROOT = REPO_ROOT / "navier" / "Wan2.2"
DEFAULT_WAN_CHECKPOINT_DIR = Path("/home/azureuser/Wan2.2-TI2V-5B")

NAVIER_GRID_SIZE = 256
DEFAULT_VIDEO_WIDTH = 256
DEFAULT_VIDEO_HEIGHT = 128
DEFAULT_SAMPLE_COUNT = 2000
DEFAULT_BATCH_SIZE = 2
DEFAULT_DURATION_SECONDS = 2.5
DEFAULT_FPS = 64
DEFAULT_STACK_VORTICITY = True
DEFAULT_WIND_SPEED = 2.0
DEFAULT_MAX_DELTA_T = 5e-3
DEFAULT_OBSTACLE_METHOD = "penalized_spectral"
DEFAULT_MASKED_PRESSURE_MAX_ITERATIONS = 50
DEFAULT_MASKED_PRESSURE_TOLERANCE = 1e-3
DEFAULT_INITIAL_MASKED_PRESSURE_MAX_ITERATIONS = 120
DEFAULT_INITIAL_MASKED_PRESSURE_TOLERANCE = 1e-5
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
DEFAULT_WAN_EXPECTED_SIZE = (
    DEFAULT_VIDEO_HEIGHT * (2 if DEFAULT_STACK_VORTICITY else 1),
    DEFAULT_VIDEO_WIDTH,
)


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


def _set_right_flow_boundary_(
    velocity: torch.Tensor,
    boundary_u: float = 1.0,
    boundary_v: float = 0.0,
) -> torch.Tensor:
    velocity[0, 0, :] = boundary_u
    velocity[0, -1, :] = boundary_u
    velocity[0, :, 0] = boundary_u
    velocity[0, :, -1] = boundary_u
    velocity[1, 0, :] = boundary_v
    velocity[1, -1, :] = boundary_v
    velocity[1, :, 0] = boundary_v
    velocity[1, :, -1] = boundary_v
    return velocity


def _enforce_flow_constraints_(
    velocity: torch.Tensor,
    obstacle_mask: Optional[torch.Tensor] = None,
    boundary_u: float = 1.0,
    boundary_v: float = 0.0,
) -> torch.Tensor:
    _set_right_flow_boundary_(
        velocity, boundary_u=boundary_u, boundary_v=boundary_v
    )
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

    Boundary velocity is the right-pointing freestream vector everywhere on the square
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
    velocity[0].fill_(float(amplitude))
    velocity[1].zero_()
    _enforce_flow_constraints_(
        velocity,
        obstacle_mask,
        boundary_u=float(amplitude),
        boundary_v=0.0,
    )
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


def _fluid_mask(
    obstacle_mask: Optional[torch.Tensor], shape, device: torch.device
) -> torch.Tensor:
    if len(shape) < 2:
        raise ValueError("shape must include ny and nx")
    ny, nx = int(shape[-2]), int(shape[-1])
    if obstacle_mask is None:
        return torch.ones((ny, nx), device=device, dtype=torch.bool)
    mask = obstacle_mask.to(device=device, dtype=torch.bool)
    if mask.shape != (ny, nx):
        raise ValueError("obstacle_mask must have shape (ny, nx)")
    return ~mask


def _face_fluxes_from_cell_velocity(
    velocity: torch.Tensor,
    fluid_mask: torch.Tensor,
    boundary_u: float = 1.0,
    boundary_v: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if velocity.ndim != 3 or velocity.shape[0] != 2:
        raise ValueError("velocity must have shape (2, ny, nx)")
    _, ny, nx = velocity.shape
    fluid_mask = fluid_mask.to(device=velocity.device, dtype=torch.bool)
    if fluid_mask.shape != (ny, nx):
        raise ValueError("fluid_mask must have shape (ny, nx)")

    u, v = velocity[0], velocity[1]
    u_face = torch.zeros((ny, nx + 1), device=velocity.device, dtype=velocity.dtype)
    v_face = torch.zeros((ny + 1, nx), device=velocity.device, dtype=velocity.dtype)

    open_x = fluid_mask[:, :-1] & fluid_mask[:, 1:]
    open_y = fluid_mask[:-1, :] & fluid_mask[1:, :]
    u_interior = u_face[:, 1:-1]
    v_interior = v_face[1:-1, :]
    u_interior[open_x] = 0.5 * (u[:, :-1] + u[:, 1:])[open_x]
    v_interior[open_y] = 0.5 * (v[:-1, :] + v[1:, :])[open_y]

    u_face[:, 0] = boundary_u
    u_face[:, -1] = boundary_u
    v_face[0, :] = boundary_v
    v_face[-1, :] = boundary_v
    return (u_face, v_face)


def _finite_volume_divergence_from_faces(
    u_face: torch.Tensor,
    v_face: torch.Tensor,
    fluid_mask: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    ny, nx_plus_one = u_face.shape
    ny_plus_one, nx = v_face.shape
    if nx_plus_one != nx + 1 or ny_plus_one != ny + 1:
        raise ValueError("face flux shapes must be (ny, nx + 1) and (ny + 1, nx)")
    fluid_mask = fluid_mask.to(device=u_face.device, dtype=torch.bool)
    if fluid_mask.shape != (ny, nx):
        raise ValueError("fluid_mask must have shape (ny, nx)")
    div = (u_face[:, 1:] - u_face[:, :-1]) / float(dx) + (
        v_face[1:, :] - v_face[:-1, :]
    ) / float(dy)
    div = div.clone()
    div[~fluid_mask] = 0.0
    return div


def _masked_poisson_geometry(
    fluid_mask: torch.Tensor,
    dx: float,
    dy: float,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor | float]:
    fluid_mask = fluid_mask.to(dtype=torch.bool)
    dtype = torch.float32 if dtype is None else dtype
    if not bool(fluid_mask.any().detach().cpu()):
        raise ValueError("fluid_mask must contain at least one fluid cell")

    open_e = torch.zeros_like(fluid_mask)
    open_w = torch.zeros_like(fluid_mask)
    open_s = torch.zeros_like(fluid_mask)
    open_n = torch.zeros_like(fluid_mask)

    open_x = fluid_mask[:, :-1] & fluid_mask[:, 1:]
    open_y = fluid_mask[:-1, :] & fluid_mask[1:, :]
    open_e[:, :-1] = open_x
    open_w[:, 1:] = open_x
    open_s[:-1, :] = open_y
    open_n[1:, :] = open_y

    dx2 = float(dx) * float(dx)
    dy2 = float(dy) * float(dy)
    open_e_f = open_e.to(dtype=dtype)
    open_w_f = open_w.to(dtype=dtype)
    open_s_f = open_s.to(dtype=dtype)
    open_n_f = open_n.to(dtype=dtype)
    denom = (
        open_e_f / dx2
        + open_w_f / dx2
        + open_s_f / dy2
        + open_n_f / dy2
    )
    inv_diag = torch.where(
        denom > 0.0,
        1.0 / denom,
        torch.zeros_like(denom),
    )
    fluid_float = fluid_mask.to(dtype=dtype)
    fluid_count = fluid_float.sum().clamp_min(1.0)
    return {
        "fluid_mask": fluid_mask,
        "fluid_float": fluid_float,
        "fluid_count": fluid_count,
        "open_e": open_e,
        "open_w": open_w,
        "open_s": open_s,
        "open_n": open_n,
        "denom": denom,
        "inv_diag": inv_diag,
        "dx": float(dx),
        "dy": float(dy),
    }


def _zero_mean_fluid(
    field: torch.Tensor, geometry: Dict[str, torch.Tensor | float]
) -> torch.Tensor:
    fluid = geometry["fluid_float"].to(device=field.device, dtype=field.dtype)
    count = geometry["fluid_count"].to(device=field.device, dtype=field.dtype)
    field = field * fluid
    return field - fluid * (field.sum() / count)


def _masked_positive_laplacian(
    phi: torch.Tensor, geometry: Dict[str, torch.Tensor | float]
) -> torch.Tensor:
    dx = float(geometry["dx"])
    dy = float(geometry["dy"])
    dx2 = dx * dx
    dy2 = dy * dy

    open_e = geometry["open_e"].to(device=phi.device, dtype=phi.dtype)
    open_w = geometry["open_w"].to(device=phi.device, dtype=phi.dtype)
    open_s = geometry["open_s"].to(device=phi.device, dtype=phi.dtype)
    open_n = geometry["open_n"].to(device=phi.device, dtype=phi.dtype)
    denom = geometry["denom"].to(device=phi.device, dtype=phi.dtype)
    fluid = geometry["fluid_float"].to(device=phi.device, dtype=phi.dtype)

    neighbor_sum = torch.zeros_like(phi)
    neighbor_sum[:, :-1] += open_e[:, :-1] * phi[:, 1:] / dx2
    neighbor_sum[:, 1:] += open_w[:, 1:] * phi[:, :-1] / dx2
    neighbor_sum[:-1, :] += open_s[:-1, :] * phi[1:, :] / dy2
    neighbor_sum[1:, :] += open_n[1:, :] * phi[:-1, :] / dy2

    out = (denom * phi - neighbor_sum) * fluid
    return _zero_mean_fluid(out, geometry)


def _masked_poisson_pcg(
    rhs: torch.Tensor,
    geometry: Dict[str, torch.Tensor | float],
    max_iterations: int = 80,
    tolerance: float = 1e-4,
    initial_phi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if rhs.ndim != 2:
        raise ValueError("rhs must have shape (ny, nx)")
    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    inv_diag = geometry["inv_diag"].to(device=rhs.device, dtype=rhs.dtype)
    rhs = _zero_mean_fluid(rhs, geometry)
    b = _zero_mean_fluid(-rhs, geometry)

    if initial_phi is None:
        x = torch.zeros_like(rhs)
    else:
        x = _zero_mean_fluid(
            initial_phi.to(device=rhs.device, dtype=rhs.dtype), geometry
        )

    r = _zero_mean_fluid(b - _masked_positive_laplacian(x, geometry), geometry)
    z = _zero_mean_fluid(inv_diag * r, geometry)
    p = z.clone()

    rz_old = (r * z).sum()
    b_norm = torch.sqrt((b * b).sum()).clamp_min(1e-20)
    residual_limit = float(tolerance) * b_norm

    if torch.sqrt((r * r).sum()) <= residual_limit:
        return x

    for _ in range(max_iterations):
        Ap = _masked_positive_laplacian(p, geometry)
        denom = (p * Ap).sum().clamp_min(1e-20)
        alpha = rz_old / denom

        x = _zero_mean_fluid(x + alpha * p, geometry)
        r = _zero_mean_fluid(r - alpha * Ap, geometry)

        if torch.sqrt((r * r).sum()) <= residual_limit:
            break

        z = _zero_mean_fluid(inv_diag * r, geometry)
        rz_new = (r * z).sum()
        beta = rz_new / rz_old.clamp_min(1e-20)
        p = _zero_mean_fluid(z + beta * p, geometry)
        rz_old = rz_new

    return _zero_mean_fluid(x, geometry)


def _project_velocity_masked_fv(
    provisional: torch.Tensor,
    rho: torch.Tensor | float,
    delta_t: torch.Tensor | float,
    dx: float,
    dy: float,
    obstacle_mask: torch.Tensor,
    pressure_max_iterations: int,
    pressure_tolerance: float,
    pressure: Optional[torch.Tensor] = None,
    geometry: Optional[Dict[str, torch.Tensor | float]] = None,
    boundary_u: float = 1.0,
    boundary_v: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if provisional.ndim != 3 or provisional.shape[0] != 2:
        raise ValueError("provisional must have shape (2, ny, nx)")
    fluid_mask = _fluid_mask(obstacle_mask, provisional.shape[-2:], provisional.device)
    if not bool(fluid_mask.any().detach().cpu()):
        raise ValueError("obstacle_mask leaves no fluid cells")
    if geometry is None:
        geometry = _masked_poisson_geometry(fluid_mask, dx, dy, dtype=provisional.dtype)
    else:
        fluid_mask = geometry["fluid_mask"].to(device=provisional.device, dtype=torch.bool)

    u_face, v_face = _face_fluxes_from_cell_velocity(
        provisional,
        fluid_mask,
        boundary_u=boundary_u,
        boundary_v=boundary_v,
    )
    div = _finite_volume_divergence_from_faces(u_face, v_face, fluid_mask, dx, dy)
    rhs = _zero_mean_fluid(div.clone(), geometry)

    rho_tensor = torch.as_tensor(rho, device=provisional.device, dtype=provisional.dtype)
    dt_tensor = torch.as_tensor(
        delta_t, device=provisional.device, dtype=provisional.dtype
    )
    initial_phi = None
    if pressure is not None:
        initial_phi = pressure.to(device=provisional.device, dtype=provisional.dtype)
        initial_phi = initial_phi * dt_tensor / rho_tensor
    phi = _masked_poisson_pcg(
        rhs,
        geometry,
        max_iterations=pressure_max_iterations,
        tolerance=pressure_tolerance,
        initial_phi=initial_phi,
    )

    u_face_new = u_face.clone()
    v_face_new = v_face.clone()
    open_x = fluid_mask[:, :-1] & fluid_mask[:, 1:]
    open_y = fluid_mask[:-1, :] & fluid_mask[1:, :]
    u_interior = u_face_new[:, 1:-1]
    v_interior = v_face_new[1:-1, :]
    u_interior[open_x] -= (phi[:, 1:] - phi[:, :-1])[open_x] / float(dx)
    v_interior[open_y] -= (phi[1:, :] - phi[:-1, :])[open_y] / float(dy)

    u_face_new[:, 0] = boundary_u
    u_face_new[:, -1] = boundary_u
    v_face_new[0, :] = boundary_v
    v_face_new[-1, :] = boundary_v

    u_new = 0.5 * (u_face_new[:, :-1] + u_face_new[:, 1:])
    v_new = 0.5 * (v_face_new[:-1, :] + v_face_new[1:, :])
    next_velocity = torch.stack((u_new, v_new), dim=0)
    _enforce_flow_constraints_(
        next_velocity,
        obstacle_mask,
        boundary_u=boundary_u,
        boundary_v=boundary_v,
    )
    pressure = rho_tensor / dt_tensor * phi
    return (next_velocity, pressure)


def _apply_brinkman_obstacle_damping_(
    velocity: torch.Tensor,
    obstacle_mask: Optional[torch.Tensor],
    delta_t: torch.Tensor | float,
    eta: float = 1e-3,
) -> torch.Tensor:
    if obstacle_mask is None:
        return velocity
    if eta <= 0.0:
        raise ValueError("obstacle_penalty_eta must be positive")
    mask = obstacle_mask.to(device=velocity.device, dtype=torch.bool)
    dt_tensor = torch.as_tensor(delta_t, device=velocity.device, dtype=velocity.dtype)
    damping = torch.exp(-dt_tensor / float(eta))
    velocity[:, mask] *= damping
    return velocity


def masked_divergence(
    velocity: torch.Tensor,
    obstacle_mask: Optional[torch.Tensor],
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    boundary_u: float = 1.0,
    boundary_v: float = 0.0,
    geometry: Optional[Dict[str, torch.Tensor | float]] = None,
) -> torch.Tensor:
    if velocity.ndim != 3 or velocity.shape[0] != 2:
        raise ValueError("velocity must have shape (2, ny, nx)")
    _, ny, nx = velocity.shape
    dx = 2.0 / (nx - 1) if dx is None else float(dx)
    dy = 2.0 / (ny - 1) if dy is None else float(dy)
    if geometry is None:
        fluid_mask = _fluid_mask(obstacle_mask, (ny, nx), velocity.device)
    else:
        fluid_mask = geometry["fluid_mask"].to(device=velocity.device, dtype=torch.bool)
    u_face, v_face = _face_fluxes_from_cell_velocity(
        velocity, fluid_mask, boundary_u=boundary_u, boundary_v=boundary_v
    )
    return _finite_volume_divergence_from_faces(u_face, v_face, fluid_mask, dx, dy)


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
    boundary_u: float = 1.0,
    boundary_v: float = 0.0,
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
    _set_right_flow_boundary_(
        provisional, boundary_u=boundary_u, boundary_v=boundary_v
    )
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
    masked_pressure_iterations: int = 80,
    masked_pressure_max_iterations: Optional[int] = None,
    masked_pressure_tolerance: float = 1e-4,
    masked_poisson_geometry: Optional[Dict[str, torch.Tensor | float]] = None,
    boundary_u: float = 1.0,
    boundary_v: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    provisional = _advection_diffusion_provisional(
        velocity,
        nu,
        delta_t,
        dx,
        dy,
        boundary_u=boundary_u,
        boundary_v=boundary_v,
    )
    _enforce_flow_constraints_(
        provisional,
        obstacle_mask,
        boundary_u=boundary_u,
        boundary_v=boundary_v,
    )
    if obstacle_mask is not None:
        if masked_pressure_max_iterations is None:
            masked_pressure_max_iterations = int(masked_pressure_iterations)
        return _project_velocity_masked_fv(
            provisional,
            rho,
            delta_t,
            dx,
            dy,
            obstacle_mask,
            int(masked_pressure_max_iterations),
            float(masked_pressure_tolerance),
            pressure=pressure,
            geometry=masked_poisson_geometry,
            boundary_u=boundary_u,
            boundary_v=boundary_v,
        )
    next_velocity, pressure = _project_velocity_spectral_periodic_with_operators(
        provisional, rho, delta_t, kx, ky, k2, zero_mode
    )
    _enforce_flow_constraints_(
        next_velocity,
        obstacle_mask,
        boundary_u=boundary_u,
        boundary_v=boundary_v,
    )
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
    masked_pressure_iterations: Optional[int] = None,
    masked_pressure_max_iterations: int = 80,
    masked_pressure_tolerance: float = 1e-4,
    masked_poisson_geometry: Optional[Dict[str, torch.Tensor | float]] = None,
    boundary_u: float = 1.0,
    boundary_v: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    provisional = _advection_diffusion_provisional(
        velocity,
        nu,
        delta_t,
        dx,
        dy,
        boundary_u=boundary_u,
        boundary_v=boundary_v,
    )
    _enforce_flow_constraints_(
        provisional,
        obstacle_mask,
        boundary_u=boundary_u,
        boundary_v=boundary_v,
    )
    if obstacle_mask is not None:
        if masked_pressure_iterations is not None:
            masked_pressure_max_iterations = int(masked_pressure_iterations)
        return _project_velocity_masked_fv(
            provisional,
            rho,
            delta_t,
            dx,
            dy,
            obstacle_mask,
            int(masked_pressure_max_iterations),
            float(masked_pressure_tolerance),
            pressure=pressure,
            geometry=masked_poisson_geometry,
            boundary_u=boundary_u,
            boundary_v=boundary_v,
        )
    method = pressure_method.lower()
    if method in {"spectral", "spectral_periodic", "fft"}:
        next_velocity, pressure = _project_velocity_spectral_periodic(
            provisional, rho, delta_t, dx, dy
        )
        _enforce_flow_constraints_(
            next_velocity,
            obstacle_mask,
            boundary_u=boundary_u,
            boundary_v=boundary_v,
        )
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
    _enforce_flow_constraints_(
        next_velocity,
        obstacle_mask,
        boundary_u=boundary_u,
        boundary_v=boundary_v,
    )
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
    T: float = DEFAULT_DURATION_SECONDS,
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
    masked_pressure_iterations: Optional[int] = None,
    masked_pressure_max_iterations: int = 80,
    masked_pressure_tolerance: float = 1e-4,
    initial_masked_pressure_max_iterations: int = 150,
    initial_masked_pressure_tolerance: float = 1e-5,
    obstacle_method: str = "masked_pcg",
    obstacle_penalty_eta: float = 1e-3,
    project_initial: bool = True,
    boundary_u: float = 1.0,
    boundary_v: float = 0.0,
    initial_projection_steps: int = 0,
    initial_projection_delta_t: float = 1.0,
    initial_relaxation_steps: int = 0,
    initial_relaxation_delta_t: Optional[float] = None,
    sample_times: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Simulate 2D incompressible Navier-Stokes on [-1, 1]^2.

    Uses an explicit advection/diffusion step followed by a pressure projection.
    pressure_method='spectral' uses a fast FFT Helmholtz projection for
    obstacle-free simulations; obstacle simulations use masked finite-volume PCG
    projection by default. If delta_t is None, a stable step is recomputed from
    the CFL and diffusion limits. Boundary velocities are clamped to the
    configured freestream. By default, samples at t=0 use a projected initial
    field. Optional initial projection passes and full initial relaxation steps
    are applied before positive-time samples only.

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
    obstacle_method = obstacle_method.lower()
    if obstacle_method not in {"masked_pcg", "penalized_spectral"}:
        raise ValueError("obstacle_method must be 'masked_pcg' or 'penalized_spectral'")
    if save_every < 1:
        raise ValueError("save_every must be at least 1")
    if method == "jacobi" and pressure_iterations < 1:
        raise ValueError("pressure_iterations must be at least 1")
    if masked_pressure_iterations is not None:
        masked_pressure_max_iterations = int(masked_pressure_iterations)
    masked_pressure_max_iterations = int(masked_pressure_max_iterations)
    initial_masked_pressure_max_iterations = int(initial_masked_pressure_max_iterations)
    if masked_pressure_max_iterations < 1:
        raise ValueError("masked_pressure_max_iterations must be positive")
    if initial_masked_pressure_max_iterations < 1:
        raise ValueError("initial_masked_pressure_max_iterations must be positive")
    if masked_pressure_tolerance <= 0.0:
        raise ValueError("masked_pressure_tolerance must be positive")
    if initial_masked_pressure_tolerance <= 0.0:
        raise ValueError("initial_masked_pressure_tolerance must be positive")
    if obstacle_penalty_eta <= 0.0:
        raise ValueError("obstacle_penalty_eta must be positive")
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
                "between the saved t=0 frame and positive-time samples. Set both "
                "to 0 when frame 1 should be exactly one frame interval after frame 0."
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
    if obstacle_mask is not None and obstacle_method == "penalized_spectral":
        _set_right_flow_boundary_(
            velocity, boundary_u=boundary_u, boundary_v=boundary_v
        )
    else:
        _enforce_flow_constraints_(
            velocity,
            obstacle_mask,
            boundary_u=boundary_u,
            boundary_v=boundary_v,
        )
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)
    pressure = torch.zeros((ny, nx), device=velocity.device, dtype=velocity.dtype)
    nu_tensor = torch.as_tensor(float(nu), device=velocity.device, dtype=velocity.dtype)
    rho_tensor = torch.as_tensor(
        float(rho), device=velocity.device, dtype=velocity.dtype
    )
    masked_poisson_geometry = None
    if obstacle_mask is not None and obstacle_method == "masked_pcg":
        masked_poisson_geometry = _masked_poisson_geometry(
            ~obstacle_mask, dx, dy, dtype=velocity.dtype
        )
    if project_initial and obstacle_mask is not None and obstacle_method == "masked_pcg":
        velocity, pressure = _project_velocity_masked_fv(
            velocity,
            rho_tensor,
            torch.as_tensor(1.0, device=velocity.device, dtype=velocity.dtype),
            dx,
            dy,
            obstacle_mask,
            initial_masked_pressure_max_iterations,
            float(initial_masked_pressure_tolerance),
            pressure=pressure,
            geometry=masked_poisson_geometry,
            boundary_u=boundary_u,
            boundary_v=boundary_v,
        )
    elif (
        project_initial
        and obstacle_mask is not None
        and obstacle_method == "penalized_spectral"
    ):
        projection_dt = torch.as_tensor(
            1.0, device=velocity.device, dtype=velocity.dtype
        )
        _apply_brinkman_obstacle_damping_(
            velocity, obstacle_mask, projection_dt, eta=obstacle_penalty_eta
        )
        velocity, pressure = _project_velocity_spectral_periodic(
            velocity,
            rho_tensor,
            projection_dt,
            dx,
            dy,
        )
        _set_right_flow_boundary_(
            velocity, boundary_u=boundary_u, boundary_v=boundary_v
        )
    elif project_initial and obstacle_mask is None:
        velocity, pressure = _project_velocity_spectral_periodic(
            velocity,
            rho_tensor,
            torch.as_tensor(1.0, device=velocity.device, dtype=velocity.dtype),
            dx,
            dy,
        )
        _enforce_flow_constraints_(
            velocity,
            obstacle_mask,
            boundary_u=boundary_u,
            boundary_v=boundary_v,
        )
    raw_initial_velocity = velocity.detach().clone()
    raw_initial_pressure = pressure.detach().clone()
    spectral_operators = None
    provisional_function = _advection_diffusion_provisional
    step_function = _navier_stokes_step
    if method == "spectral" or (
        obstacle_mask is not None and obstacle_method == "penalized_spectral"
    ):
        if obstacle_mask is None or obstacle_method == "penalized_spectral":
            spectral_operators = _spectral_wavenumbers(
                ny, nx, dx, dy, velocity.device, velocity.dtype
            )
    if method == "spectral":
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
        if obstacle_mask is not None:
            compiled_provisional = (
                method == "spectral"
                and use_compile
            )
            dt_arg = (
                torch.as_tensor(
                    dt, device=current_velocity.device, dtype=current_velocity.dtype
                )
                if compiled_provisional
                else dt
            )
            nu_arg = nu_tensor if compiled_provisional else float(nu)
            try:
                provisional = provisional_function(
                    current_velocity,
                    nu_arg,
                    dt_arg,
                    dx,
                    dy,
                    boundary_u,
                    boundary_v,
                )
            except Exception as exc:
                if not compiled_provisional:
                    raise
                warnings.warn(
                    f"torch.compile provisional update failed ({exc}); falling back to eager masked stepping."
                )
                use_compile = False
                provisional_function = _advection_diffusion_provisional
                provisional = provisional_function(
                    current_velocity,
                    float(nu),
                    dt,
                    dx,
                    dy,
                    boundary_u,
                    boundary_v,
                )
            _enforce_flow_constraints_(
                provisional,
                obstacle_mask if obstacle_method == "masked_pcg" else None,
                boundary_u=boundary_u,
                boundary_v=boundary_v,
            )
            if obstacle_method == "penalized_spectral":
                dt_tensor = torch.as_tensor(
                    dt, device=current_velocity.device, dtype=current_velocity.dtype
                )
                _apply_brinkman_obstacle_damping_(
                    provisional,
                    obstacle_mask,
                    dt_tensor,
                    eta=obstacle_penalty_eta,
                )
                next_velocity, next_pressure = (
                    _project_velocity_spectral_periodic_with_operators(
                        provisional, rho_tensor, dt_tensor, *spectral_operators
                    )
                )
                _set_right_flow_boundary_(
                    next_velocity, boundary_u=boundary_u, boundary_v=boundary_v
                )
                return (next_velocity, next_pressure)

            return _project_velocity_masked_fv(
                provisional,
                rho_tensor,
                torch.as_tensor(
                    dt, device=current_velocity.device, dtype=current_velocity.dtype
                ),
                dx,
                dy,
                obstacle_mask,
                masked_pressure_max_iterations,
                float(masked_pressure_tolerance),
                pressure=current_pressure,
                geometry=masked_poisson_geometry,
                boundary_u=boundary_u,
                boundary_v=boundary_v,
            )

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
                        current_velocity,
                        nu_arg,
                        dt_arg,
                        dx,
                        dy,
                        boundary_u,
                        boundary_v,
                    )
                    _enforce_flow_constraints_(
                        provisional,
                        obstacle_mask,
                        boundary_u=boundary_u,
                        boundary_v=boundary_v,
                    )
                    next_velocity, next_pressure = (
                        _project_velocity_spectral_periodic_with_operators(
                            provisional, rho_arg, dt_arg, *spectral_operators
                        )
                    )
                    _enforce_flow_constraints_(
                        next_velocity,
                        obstacle_mask,
                        boundary_u=boundary_u,
                        boundary_v=boundary_v,
                    )
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
                    boundary_u=boundary_u,
                    boundary_v=boundary_v,
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
                    boundary_u=boundary_u,
                    boundary_v=boundary_v,
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
            masked_pressure_iterations=masked_pressure_iterations,
            masked_pressure_max_iterations=masked_pressure_max_iterations,
            masked_pressure_tolerance=float(masked_pressure_tolerance),
            masked_poisson_geometry=masked_poisson_geometry,
            boundary_u=boundary_u,
            boundary_v=boundary_v,
        )

    projection_delta_ts = []
    if int(initial_projection_steps) > 0:
        projection_dt = torch.as_tensor(
            float(initial_projection_delta_t),
            device=velocity.device,
            dtype=velocity.dtype,
        )
        projection_operators = spectral_operators
        if obstacle_mask is None and projection_operators is None:
            projection_operators = _spectral_wavenumbers(
                ny, nx, dx, dy, velocity.device, velocity.dtype
            )
        for _ in range(int(initial_projection_steps)):
            _enforce_flow_constraints_(
                velocity,
                obstacle_mask if obstacle_method == "masked_pcg" else None,
                boundary_u=boundary_u,
                boundary_v=boundary_v,
            )
            if obstacle_mask is not None and obstacle_method == "masked_pcg":
                velocity, pressure = _project_velocity_masked_fv(
                    velocity,
                    rho_tensor,
                    projection_dt,
                    dx,
                    dy,
                    obstacle_mask,
                    initial_masked_pressure_max_iterations,
                    float(initial_masked_pressure_tolerance),
                    pressure=pressure,
                    geometry=masked_poisson_geometry,
                    boundary_u=boundary_u,
                    boundary_v=boundary_v,
                )
            elif obstacle_mask is not None and obstacle_method == "penalized_spectral":
                _apply_brinkman_obstacle_damping_(
                    velocity,
                    obstacle_mask,
                    projection_dt,
                    eta=obstacle_penalty_eta,
                )
                velocity, pressure = _project_velocity_spectral_periodic_with_operators(
                    velocity, rho_tensor, projection_dt, *projection_operators
                )
                _set_right_flow_boundary_(
                    velocity, boundary_u=boundary_u, boundary_v=boundary_v
                )
            else:
                velocity, pressure = _project_velocity_spectral_periodic_with_operators(
                    velocity, rho_tensor, projection_dt, *projection_operators
                )
                _enforce_flow_constraints_(
                    velocity,
                    obstacle_mask,
                    boundary_u=boundary_u,
                    boundary_v=boundary_v,
                )
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


def _set_right_flow_boundary_batched_(
    velocity: torch.Tensor,
    boundary_u: float = 1.0,
    boundary_v: float = 0.0,
) -> torch.Tensor:
    velocity[:, 0, 0, :] = boundary_u
    velocity[:, 0, -1, :] = boundary_u
    velocity[:, 0, :, 0] = boundary_u
    velocity[:, 0, :, -1] = boundary_u
    velocity[:, 1, 0, :] = boundary_v
    velocity[:, 1, -1, :] = boundary_v
    velocity[:, 1, :, 0] = boundary_v
    velocity[:, 1, :, -1] = boundary_v
    return velocity


def _batched_laplacian_interior(
    field: torch.Tensor, dx: float, dy: float
) -> torch.Tensor:
    return (
        field[:, 1:-1, 2:]
        - 2.0 * field[:, 1:-1, 1:-1]
        + field[:, 1:-1, :-2]
    ) / (dx * dx) + (
        field[:, 2:, 1:-1]
        - 2.0 * field[:, 1:-1, 1:-1]
        + field[:, :-2, 1:-1]
    ) / (dy * dy)


def _batched_upwind_gradient(
    field: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dx: float, dy: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    center = field[:, 1:-1, 1:-1]
    u_center = u[:, 1:-1, 1:-1]
    v_center = v[:, 1:-1, 1:-1]
    backward_x = (center - field[:, 1:-1, :-2]) / dx
    forward_x = (field[:, 1:-1, 2:] - center) / dx
    backward_y = (center - field[:, :-2, 1:-1]) / dy
    forward_y = (field[:, 2:, 1:-1] - center) / dy
    grad_x = torch.where(u_center >= 0.0, backward_x, forward_x)
    grad_y = torch.where(v_center >= 0.0, backward_y, forward_y)
    return (grad_x, grad_y)


def _advection_diffusion_provisional_batched(
    velocity: torch.Tensor,
    nu: torch.Tensor | float,
    delta_t: torch.Tensor | float,
    dx: float,
    dy: float,
    boundary_u: float = 1.0,
    boundary_v: float = 0.0,
) -> torch.Tensor:
    u = velocity[:, 0]
    v = velocity[:, 1]
    u_center = u[:, 1:-1, 1:-1]
    v_center = v[:, 1:-1, 1:-1]
    du_dx, du_dy = _batched_upwind_gradient(u, u, v, dx, dy)
    dv_dx, dv_dy = _batched_upwind_gradient(v, u, v, dx, dy)
    lap_u = _batched_laplacian_interior(u, dx, dy)
    lap_v = _batched_laplacian_interior(v, dx, dy)
    nu_tensor = torch.as_tensor(nu, device=velocity.device, dtype=velocity.dtype)
    if nu_tensor.ndim == 0:
        nu_tensor = nu_tensor.view(1, 1, 1)
    else:
        nu_tensor = nu_tensor.reshape(-1, 1, 1)
    dt_tensor = torch.as_tensor(delta_t, device=velocity.device, dtype=velocity.dtype)
    u_star = u.clone()
    v_star = v.clone()
    u_star[:, 1:-1, 1:-1] = u_center + dt_tensor * (
        nu_tensor * lap_u - (u_center * du_dx + v_center * du_dy)
    )
    v_star[:, 1:-1, 1:-1] = v_center + dt_tensor * (
        nu_tensor * lap_v - (u_center * dv_dx + v_center * dv_dy)
    )
    provisional = torch.stack((u_star, v_star), dim=1)
    _set_right_flow_boundary_batched_(
        provisional, boundary_u=boundary_u, boundary_v=boundary_v
    )
    return provisional


def _project_velocity_spectral_periodic_batched_with_operators(
    provisional: torch.Tensor,
    rho: torch.Tensor | float,
    delta_t: torch.Tensor | float,
    kx: torch.Tensor,
    ky: torch.Tensor,
    k2: torch.Tensor,
    zero_mode: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    u_hat = torch.fft.fft2(provisional[:, 0], dim=(-2, -1))
    v_hat = torch.fft.fft2(provisional[:, 1], dim=(-2, -1))
    velocity_dot_k = kx * u_hat + ky * v_hat
    zero_complex = torch.zeros_like(u_hat)
    correction_u = torch.where(zero_mode, zero_complex, kx * velocity_dot_k / k2)
    correction_v = torch.where(zero_mode, zero_complex, ky * velocity_dot_k / k2)
    projected_u = torch.fft.ifft2(u_hat - correction_u, dim=(-2, -1)).real
    projected_v = torch.fft.ifft2(v_hat - correction_v, dim=(-2, -1)).real
    rho_tensor = torch.as_tensor(rho, device=provisional.device, dtype=provisional.dtype)
    if rho_tensor.ndim == 0:
        rho_tensor = rho_tensor.view(1, 1, 1)
    else:
        rho_tensor = rho_tensor.reshape(-1, 1, 1)
    dt_tensor = torch.as_tensor(delta_t, device=provisional.device, dtype=provisional.dtype)
    pressure_hat = torch.where(
        zero_mode,
        zero_complex,
        -(rho_tensor / dt_tensor) * (1j * velocity_dot_k) / k2,
    )
    pressure = torch.fft.ifft2(pressure_hat, dim=(-2, -1)).real
    return (torch.stack((projected_u, projected_v), dim=1), pressure)


def _apply_brinkman_obstacle_damping_batched_(
    velocity: torch.Tensor,
    obstacle_mask: Optional[torch.Tensor],
    delta_t: torch.Tensor | float,
    eta: float = 1e-3,
) -> torch.Tensor:
    if obstacle_mask is None:
        return velocity
    if eta <= 0.0:
        raise ValueError("obstacle_penalty_eta must be positive")
    mask = obstacle_mask.to(device=velocity.device, dtype=torch.bool)
    dt_tensor = torch.as_tensor(delta_t, device=velocity.device, dtype=velocity.dtype)
    damping = torch.exp(-dt_tensor / float(eta))
    factor = torch.ones(
        (velocity.shape[0], 1, velocity.shape[-2], velocity.shape[-1]),
        device=velocity.device,
        dtype=velocity.dtype,
    )
    factor = factor.masked_fill(mask[:, None], damping)
    velocity.mul_(factor)
    return velocity


def _stable_timestep_batched(
    velocity: torch.Tensor,
    nu: torch.Tensor | float,
    dx: float,
    dy: float,
    cfl: float,
    diffusion_safety: float,
    max_delta_t: float,
) -> float:
    h = min(float(dx), float(dy))
    max_speed = float(velocity.square().sum(dim=1).sqrt().amax().detach().cpu())
    nu_tensor = torch.as_tensor(nu, device=velocity.device, dtype=velocity.dtype)
    max_nu = float(nu_tensor.amax().detach().cpu())
    advective = float("inf") if max_speed < 1e-12 else float(cfl) * h / max_speed
    diffusive = (
        float("inf")
        if max_nu <= 0.0
        else float(diffusion_safety) * h * h / max_nu
    )
    return max(1e-08, min(float(max_delta_t), advective, diffusive))


def simulate_navier_stokes_penalized_spectral_batch(
    velocity0: torch.Tensor,
    nu: torch.Tensor | float,
    rho: torch.Tensor | float,
    T: float = DEFAULT_DURATION_SECONDS,
    delta_t: Optional[float] = None,
    save_every: int = 10,
    cfl: float = 0.4,
    diffusion_safety: float = 0.2,
    max_delta_t: float = DEFAULT_MAX_DELTA_T,
    obstacle_mask: Optional[torch.Tensor] = None,
    obstacle_penalty_eta: float = 1e-3,
    project_initial: bool = True,
    boundary_u: float = 1.0,
    boundary_v: float = 0.0,
    sample_times: Optional[torch.Tensor] = None,
) -> list[Dict[str, torch.Tensor]]:
    """Batch the fast obstacle approximation over velocity0 shaped (B, 2, ny, nx)."""
    if velocity0.ndim != 4 or velocity0.shape[1] != 2:
        raise ValueError("velocity0 must have shape (batch, 2, ny, nx)")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if save_every < 1:
        raise ValueError("save_every must be at least 1")
    velocity = velocity0.detach().clone()
    if not torch.is_floating_point(velocity):
        velocity = velocity.float()
    batch, _, ny, nx = velocity.shape
    if obstacle_mask is not None:
        obstacle_mask = obstacle_mask.to(device=velocity.device, dtype=torch.bool)
        if obstacle_mask.shape != (batch, ny, nx):
            raise ValueError(
                "obstacle_mask must have shape (batch, ny, nx) matching velocity0"
            )
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
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)
    nu_tensor = torch.as_tensor(nu, device=velocity.device, dtype=velocity.dtype)
    rho_tensor = torch.as_tensor(rho, device=velocity.device, dtype=velocity.dtype)
    if nu_tensor.ndim > 0 and int(nu_tensor.numel()) != batch:
        raise ValueError("batched nu must be scalar or have one value per sample")
    if rho_tensor.ndim > 0 and int(rho_tensor.numel()) != batch:
        raise ValueError("batched rho must be scalar or have one value per sample")
    if obstacle_penalty_eta <= 0.0:
        raise ValueError("obstacle_penalty_eta must be positive")
    _set_right_flow_boundary_batched_(
        velocity, boundary_u=boundary_u, boundary_v=boundary_v
    )
    pressure = torch.zeros((batch, ny, nx), device=velocity.device, dtype=velocity.dtype)
    operators = _spectral_wavenumbers(ny, nx, dx, dy, velocity.device, velocity.dtype)
    if project_initial:
        projection_dt = torch.as_tensor(1.0, device=velocity.device, dtype=velocity.dtype)
        _apply_brinkman_obstacle_damping_batched_(
            velocity, obstacle_mask, projection_dt, eta=obstacle_penalty_eta
        )
        velocity, pressure = _project_velocity_spectral_periodic_batched_with_operators(
            velocity, rho_tensor, projection_dt, *operators
        )
        _set_right_flow_boundary_batched_(
            velocity, boundary_u=boundary_u, boundary_v=boundary_v
        )
    raw_initial_velocity = velocity.detach().clone()
    raw_initial_pressure = pressure.detach().clone()

    def choose_step_dt(current_velocity: torch.Tensor) -> float:
        if delta_t is None:
            return _stable_timestep_batched(
                current_velocity,
                nu_tensor,
                dx,
                dy,
                cfl,
                diffusion_safety,
                max_delta_t,
            )
        dt_value = float(delta_t)
        if dt_value <= 0.0:
            raise ValueError("delta_t must be positive")
        return dt_value

    def advance_step(
        current_velocity: torch.Tensor, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt_tensor = torch.as_tensor(dt, device=current_velocity.device, dtype=current_velocity.dtype)
        provisional = _advection_diffusion_provisional_batched(
            current_velocity,
            nu_tensor,
            dt_tensor,
            dx,
            dy,
            boundary_u=boundary_u,
            boundary_v=boundary_v,
        )
        _apply_brinkman_obstacle_damping_batched_(
            provisional, obstacle_mask, dt_tensor, eta=obstacle_penalty_eta
        )
        next_velocity, next_pressure = (
            _project_velocity_spectral_periodic_batched_with_operators(
                provisional, rho_tensor, dt_tensor, *operators
            )
        )
        _set_right_flow_boundary_batched_(
            next_velocity, boundary_u=boundary_u, boundary_v=boundary_v
        )
        return (next_velocity, next_pressure)

    frames = []
    pressures = []
    times = []
    delta_ts = []
    t = 0.0
    step = 0
    next_sample = 0
    if sample_times_cpu is None:
        frames.append(raw_initial_velocity.detach().clone())
        pressures.append(raw_initial_pressure.detach().clone())
        times.append(0.0)
    else:
        while (
            next_sample < sample_times_cpu.numel()
            and float(sample_times_cpu[next_sample]) <= 1e-12
        ):
            frames.append(raw_initial_velocity.detach().clone())
            pressures.append(raw_initial_pressure.detach().clone())
            times.append(float(sample_times_cpu[next_sample]))
            next_sample += 1
    while t < float(T) - 1e-12:
        next_target_time = float(T)
        if sample_times_cpu is not None and next_sample < sample_times_cpu.numel():
            next_target_time = min(
                next_target_time, float(sample_times_cpu[next_sample])
            )
            if next_target_time <= t + 1e-12:
                frames.append(velocity.detach().clone())
                pressures.append(pressure.detach().clone())
                times.append(float(sample_times_cpu[next_sample]))
                next_sample += 1
                continue
        dt = min(choose_step_dt(velocity), float(T) - t, max(next_target_time - t, 0.0))
        if dt <= 1e-12:
            break
        velocity, pressure = advance_step(velocity, dt)
        t += dt
        step += 1
        delta_ts.append(dt)
        if sample_times_cpu is None and (
            step % save_every == 0 or t >= float(T) - 1e-12
        ):
            frames.append(velocity.detach().clone())
            pressures.append(pressure.detach().clone())
            times.append(t)
        while (
            sample_times_cpu is not None
            and next_sample < sample_times_cpu.numel()
            and float(sample_times_cpu[next_sample]) <= t + 1e-10
        ):
            frames.append(velocity.detach().clone())
            pressures.append(pressure.detach().clone())
            times.append(float(sample_times_cpu[next_sample]))
            next_sample += 1
    while sample_times_cpu is not None and next_sample < sample_times_cpu.numel():
        frames.append(velocity.detach().clone())
        pressures.append(pressure.detach().clone())
        times.append(float(sample_times_cpu[next_sample]))
        next_sample += 1

    velocity_frames = torch.stack(frames, dim=1)
    pressure_frames = torch.stack(pressures, dim=1)
    time_tensor = torch.tensor(times, device=velocity.device, dtype=velocity.dtype)
    delta_t_tensor = torch.tensor(delta_ts, device=velocity.device, dtype=velocity.dtype)
    empty_steps = torch.empty(0, device=velocity.device, dtype=velocity.dtype)
    results = []
    for item_index in range(batch):
        result = {
            "velocity": velocity_frames[item_index],
            "time": time_tensor,
            "delta_t": delta_t_tensor,
            "initial_projection_delta_t": empty_steps,
            "initial_relaxation_delta_t": empty_steps,
            "pressure": pressure_frames[item_index],
        }
        if obstacle_mask is not None:
            result["obstacle_mask"] = obstacle_mask[item_index].detach().clone()
        results.append(result)
    return results


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
    return _realtime_video_sample_times(T, fps, device, dtype)


def _prepend_initial_condition_for_odd_count(T: float, fps: int) -> bool:
    return _realtime_video_frame_count(T, fps, warn=False) % 2 == 0


def _prepended_initial_realtime_video_frame_count(
    T: float, fps: int, warn: bool = True
) -> int:
    frame_count = _realtime_video_frame_count(T, fps, warn=warn)
    return frame_count + int(frame_count % 2 == 0)


def _odd_realtime_video_frame_count(T: float, fps: int, warn: bool = True) -> int:
    return _prepended_initial_realtime_video_frame_count(T, fps, warn=warn)


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
    prepended_initial_condition = frame_count == realtime_frame_count + 1
    dropped_last_frame = frame_count == realtime_frame_count - 1
    exact_duration = math.isclose(raw_count, frame_count, rel_tol=0.0, abs_tol=1e-6)
    return {
        "fps": int(fps),
        "frames": frame_count,
        "realtime_frame_count": realtime_frame_count,
        "realtime_frame_count_before_odd_trim": realtime_frame_count,
        "prepended_initial_condition_frame": prepended_initial_condition,
        "dropped_last_frame_for_odd_count": dropped_last_frame,
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


def _read_video_timestamps_pyav(
    video_path: str | Path,
) -> Tuple[list[float], Optional[float]]:
    try:
        import av
    except Exception as exc:
        raise RuntimeError("PyAV is not available") from exc

    timestamps = []
    container_fps = None
    with av.open(str(video_path)) as container:
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        if not video_streams:
            raise RuntimeError(f"no video stream found: {video_path}")
        stream = video_streams[0]
        if stream.average_rate is not None:
            container_fps = float(stream.average_rate)
        elif stream.base_rate is not None:
            container_fps = float(stream.base_rate)
        for frame in container.decode(stream):
            if frame.time is not None:
                timestamps.append(float(frame.time))
            elif frame.pts is not None and stream.time_base is not None:
                timestamps.append(float(frame.pts * stream.time_base))
    return (timestamps, container_fps)


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
    timestamp_errors = []
    try:
        timestamps, container_fps = _read_video_timestamps_pyav(video_path)
    except Exception as exc:
        timestamp_errors.append(f"PyAV: {exc!r}")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="torchvision"
                )
                timestamps, container_fps = read_video_timestamps(
                    str(video_path), pts_unit="sec"
                )
        except Exception as fallback_exc:
            timestamp_errors.append(f"torchvision: {fallback_exc!r}")
            timing = _realtime_video_timing_metadata(T, fps, expected_frame_count)
            timing.update(
                {
                    "timing_verified": False,
                    "container_fps": None,
                    "container_frames": None,
                    "container_first_timestamp_seconds": None,
                    "container_last_timestamp_seconds": None,
                    "timing_verification_error": "; ".join(timestamp_errors),
                }
            )
            warnings.warn(
                f"Could not verify written {label} timing for {video_path}: "
                f"{timing['timing_verification_error']}"
            )
            return timing
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
    component_bound: Optional[float], device: torch.device, wind_speed: float = 1.0
) -> torch.Tensor:
    velocity = torch.zeros((1, 2, 1, 1), device=device, dtype=torch.float32)
    velocity[:, 0].fill_(float(wind_speed))
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
    background_wind_speed = (
        1.0 if velocity_color_bound is None else float(velocity_color_bound)
    )
    background_rgb = _rightward_flow_rgb(
        velocity_color_bound, mask.device, wind_speed=background_wind_speed
    )
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
    prepend_initial_condition: bool = False,
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
            "max_frames is ignored; dataset videos use the realtime frame count "
            "plus an optional prepended initial-condition frame."
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
            if field == "speed" and velocity_color_bound is not None:
                vmax = torch.as_tensor(
                    float(velocity_color_bound),
                    device=scalar.device,
                    dtype=scalar.dtype,
                )
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
    if prepend_initial_condition:
        frames = torch.cat((frames[:1].clone(), frames), dim=0)
        frame_indices = torch.cat(
            (
                torch.full((1,), -1, device=frame_indices.device, dtype=frame_indices.dtype),
                frame_indices,
            ),
            dim=0,
        )
    return (frames, frame_indices.detach().cpu())


def _simulation_to_stacked_uint8_video_frames(
    simulation: Dict[str, torch.Tensor],
    field: str = "speed",
    output_size: Optional[int | Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
    velocity_color_bound: Optional[float] = 1.0,
    prepend_initial_condition: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render flow over vorticity as one vertically stacked RGB video."""
    flow_frames, frame_indices = _simulation_to_uint8_video_frames(
        simulation,
        field=field,
        output_size=output_size,
        max_frames=max_frames,
        velocity_color_bound=velocity_color_bound,
        prepend_initial_condition=prepend_initial_condition,
    )
    vorticity_frames, vorticity_indices = _simulation_to_uint8_video_frames(
        simulation,
        field="vorticity",
        output_size=output_size,
        max_frames=max_frames,
        velocity_color_bound=velocity_color_bound,
        prepend_initial_condition=prepend_initial_condition,
    )
    if not torch.equal(frame_indices, vorticity_indices):
        raise RuntimeError("flow and vorticity videos sampled different frame indices")
    if flow_frames.shape != vorticity_frames.shape:
        raise RuntimeError(
            "flow and vorticity videos must have the same shape before stacking"
        )
    return (torch.cat((flow_frames, vorticity_frames), dim=1), frame_indices)


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
    output_root: Optional[str | Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n: Optional[int] = None,
    video_width: int = DEFAULT_VIDEO_WIDTH,
    video_height: int = DEFAULT_VIDEO_HEIGHT,
    T: float = DEFAULT_DURATION_SECONDS,
    nu_range: Tuple[float, float] = (3e-4, 3e-3),
    rho_range: Tuple[float, float] = (0.5, 2.0),
    modes: int = 8,
    amplitude: float = DEFAULT_WIND_SPEED,
    fps: int = DEFAULT_FPS,
    save_every: int = 10,
    max_video_frames: Optional[int] = None,
    field: str = "speed",
    stack_vorticity: bool = DEFAULT_STACK_VORTICITY,
    delta_t: Optional[float] = None,
    pressure_method: str = "spectral",
    pressure_iterations: int = 80,
    masked_pressure_iterations: Optional[int] = None,
    masked_pressure_max_iterations: int = DEFAULT_MASKED_PRESSURE_MAX_ITERATIONS,
    masked_pressure_tolerance: float = DEFAULT_MASKED_PRESSURE_TOLERANCE,
    initial_masked_pressure_max_iterations: int = (
        DEFAULT_INITIAL_MASKED_PRESSURE_MAX_ITERATIONS
    ),
    initial_masked_pressure_tolerance: float = (
        DEFAULT_INITIAL_MASKED_PRESSURE_TOLERANCE
    ),
    obstacle_method: str = DEFAULT_OBSTACLE_METHOD,
    obstacle_penalty_eta: float = 1e-3,
    max_delta_t: float = DEFAULT_MAX_DELTA_T,
    use_compile: bool = True,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    video_codec: str = "libx264",
    video_options: Optional[Dict[str, str]] = None,
    verify_written_videos: bool = False,
    velocity_color_bound: Optional[float] = None,
    initial_projection_steps: int = 0,
    initial_projection_delta_t: float = 1.0,
    initial_relaxation_steps: int = 0,
    initial_relaxation_delta_t: Optional[float] = None,
    encode_latents: bool = False,
    vae=None,
    wan_checkpoint_dir: str | Path = WAN_CHECKPOINT_DIR,
    wan_repo_root: str | Path = WAN_REPO_ROOT,
    vae_dtype: torch.dtype = torch.float32,
    latent_save_dtype: torch.dtype = torch.float16,
    overwrite_latents: bool = False,
    resize_latents: bool = False,
    continue_on_latent_error: bool = True,
    latent_records: Optional[list[Dict]] = None,
) -> list[Dict[str, str | float | int | bool]]:
    """Generate N samples with flow videos, optional vorticity stacks, and metadata."""
    if N < 1:
        raise ValueError("N must be positive")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    n = _navier_grid_size(n)
    video_hw = _video_size(video_width, video_height)
    video_height, video_width = video_hw
    if nu_range[0] <= 0.0 or nu_range[1] <= nu_range[0]:
        raise ValueError("nu_range must be positive and increasing")
    if rho_range[0] <= 0.0 or rho_range[1] <= rho_range[0]:
        raise ValueError("rho_range must be positive and increasing")
    if delta_t is not None and delta_t <= 0.0:
        raise ValueError("delta_t must be positive or None")
    obstacle_method = obstacle_method.lower()
    if obstacle_method not in {"masked_pcg", "penalized_spectral"}:
        raise ValueError("obstacle_method must be 'masked_pcg' or 'penalized_spectral'")
    if masked_pressure_iterations is not None:
        masked_pressure_max_iterations = int(masked_pressure_iterations)
    masked_pressure_max_iterations = int(masked_pressure_max_iterations)
    initial_masked_pressure_max_iterations = int(initial_masked_pressure_max_iterations)
    if masked_pressure_max_iterations < 1:
        raise ValueError("masked_pressure_max_iterations must be positive")
    if initial_masked_pressure_max_iterations < 1:
        raise ValueError("initial_masked_pressure_max_iterations must be positive")
    if masked_pressure_tolerance <= 0.0:
        raise ValueError("masked_pressure_tolerance must be positive")
    if initial_masked_pressure_tolerance <= 0.0:
        raise ValueError("initial_masked_pressure_tolerance must be positive")
    if obstacle_penalty_eta <= 0.0:
        raise ValueError("obstacle_penalty_eta must be positive")
    if int(initial_projection_steps) != 0 or int(initial_relaxation_steps) != 0:
        raise ValueError(
            "dataset videos do not support off-camera initial projection or "
            "relaxation; use realtime simulation steps from t=0 instead"
        )
    if max_video_frames is not None:
        warnings.warn(
            "max_video_frames is ignored; videos use the realtime frame count "
            "plus an optional prepended initial-condition frame."
        )

    if output_root is None:
        output_root = (
            DEFAULT_WITH_VORTICITY_OUTPUT_ROOT
            if stack_vorticity
            else DEFAULT_OUTPUT_ROOT
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
    if latent_records is None:
        latent_records = []
    if encode_latents and vae is None:
        vae = load_wan_ti2v_5b_vae(
            checkpoint_dir=wan_checkpoint_dir,
            wan_repo_root=wan_repo_root,
            device=device,
            dtype=vae_dtype,
        )

    sample_times = _realtime_video_sample_times(
        float(T), int(fps), torch.device("cpu"), torch.float64
    )
    prepend_initial_condition = _prepend_initial_condition_for_odd_count(
        float(T), int(fps)
    )
    video_frame_count = int(sample_times.numel()) + int(prepend_initial_condition)
    expected_video_timing = _realtime_video_timing_metadata(
        float(T), int(fps), video_frame_count
    )
    records = []

    from tqdm import tqdm

    with tqdm(total=int(N), desc="Generating samples") as progress:
        next_sample_index = 0
        while next_sample_index < int(N):
            current_batch_size = min(batch_size, int(N) - next_sample_index)
            batch_specs = []
            for batch_offset in range(current_batch_size):
                sample_index = next_sample_index + batch_offset
                uid, sample_dir = _make_random_sample_dir(root)
                sample_seed = int(
                    torch.randint(0, 2**31 - 1, (1,), generator=rng).item()
                )
                sample_nu = float(
                    torch.empty((), dtype=torch.float64)
                    .uniform_(
                        math.log(nu_range[0]), math.log(nu_range[1]), generator=rng
                    )
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
                batch_specs.append(
                    {
                        "index": sample_index,
                        "uid": uid,
                        "sample_dir": sample_dir,
                        "seed": sample_seed,
                        "nu": sample_nu,
                        "rho": sample_density,
                        "velocity0": velocity0,
                        "obstacle": obstacle,
                    }
                )

            if obstacle_method == "penalized_spectral" and len(batch_specs) > 1:
                simulations = simulate_navier_stokes_penalized_spectral_batch(
                    torch.stack([spec["velocity0"] for spec in batch_specs], dim=0),
                    nu=torch.tensor(
                        [spec["nu"] for spec in batch_specs],
                        device=device,
                        dtype=dtype,
                    ),
                    rho=torch.tensor(
                        [spec["rho"] for spec in batch_specs],
                        device=device,
                        dtype=dtype,
                    ),
                    T=T,
                    delta_t=delta_t,
                    save_every=save_every,
                    max_delta_t=max_delta_t,
                    obstacle_mask=torch.stack(
                        [spec["obstacle"]["mask"] for spec in batch_specs], dim=0
                    ),
                    obstacle_penalty_eta=float(obstacle_penalty_eta),
                    project_initial=True,
                    boundary_u=float(amplitude),
                    boundary_v=0.0,
                    sample_times=sample_times,
                )
            else:
                simulations = [
                    simulate_navier_stokes(
                        spec["velocity0"],
                        nu=spec["nu"],
                        rho=spec["rho"],
                        T=T,
                        delta_t=delta_t,
                        pressure_iterations=pressure_iterations,
                        save_every=save_every,
                        max_delta_t=max_delta_t,
                        pressure_method=pressure_method,
                        use_compile=use_compile,
                        obstacle_mask=spec["obstacle"]["mask"],
                        masked_pressure_iterations=masked_pressure_iterations,
                        masked_pressure_max_iterations=masked_pressure_max_iterations,
                        masked_pressure_tolerance=float(masked_pressure_tolerance),
                        initial_masked_pressure_max_iterations=initial_masked_pressure_max_iterations,
                        initial_masked_pressure_tolerance=float(
                            initial_masked_pressure_tolerance
                        ),
                        obstacle_method=obstacle_method,
                        obstacle_penalty_eta=float(obstacle_penalty_eta),
                        project_initial=True,
                        boundary_u=float(amplitude),
                        boundary_v=0.0,
                        initial_projection_steps=initial_projection_steps,
                        initial_projection_delta_t=initial_projection_delta_t,
                        initial_relaxation_steps=initial_relaxation_steps,
                        initial_relaxation_delta_t=initial_relaxation_delta_t,
                        sample_times=sample_times,
                    )
                    for spec in batch_specs
                ]

            for spec, simulation in zip(batch_specs, simulations):
                obstacle = spec["obstacle"]
                if not torch.isfinite(simulation["velocity"][0]).all():
                    raise RuntimeError("simulation first frame contains non-finite values")

                if obstacle_method == "masked_pcg" and not torch.allclose(
                    simulation["velocity"][0][:, obstacle["mask"]],
                    torch.zeros_like(simulation["velocity"][0][:, obstacle["mask"]]),
                    rtol=0.0,
                    atol=1e-6,
                ):
                    raise RuntimeError(
                        "simulation first frame violates obstacle no-slip mask"
                    )

                video_label = (
                    "flow/vorticity stacked video"
                    if stack_vorticity
                    else "flow video"
                )
                if stack_vorticity:
                    frames, _ = _simulation_to_stacked_uint8_video_frames(
                        simulation,
                        field=field,
                        output_size=video_hw,
                        velocity_color_bound=velocity_color_bound,
                        prepend_initial_condition=prepend_initial_condition,
                    )
                else:
                    frames, _ = _simulation_to_uint8_video_frames(
                        simulation,
                        field=field,
                        output_size=video_hw,
                        velocity_color_bound=velocity_color_bound,
                        prepend_initial_condition=prepend_initial_condition,
                    )
                video_timing = _verify_realtime_video_tensor(
                    frames,
                    float(T),
                    int(fps),
                    expected_frame_count=video_frame_count,
                    label=video_label,
                )

                sample_dir = spec["sample_dir"]
                video_path = sample_dir / "video.mp4"
                write_video(
                    str(video_path),
                    frames,
                    fps=fps,
                    video_codec=video_codec,
                    options=video_options,
                )
                if verify_written_videos:
                    video_timing = _verify_written_video_timing(
                        video_path,
                        float(T),
                        int(fps),
                        expected_frame_count=video_frame_count,
                        label=video_label,
                    )
                    video_timing["timing_verification_scope"] = "written_video"
                    video_timing["written_video_timing_verified"] = bool(
                        video_timing["timing_verified"]
                    )
                else:
                    video_timing = dict(video_timing)
                    video_timing.update(
                        {
                            "timing_verified": True,
                            "timing_verification_scope": "rendered_tensor",
                            "written_video_timing_verified": False,
                        }
                    )

                latent_record = None
                if encode_latents:
                    try:
                        latent_record = encode_frames_to_wan_latents(
                            frames,
                            video_path,
                            vae=vae,
                            checkpoint_dir=wan_checkpoint_dir,
                            wan_repo_root=wan_repo_root,
                            device=device,
                            vae_dtype=vae_dtype,
                            save_dtype=latent_save_dtype,
                            overwrite=overwrite_latents,
                            expected_size=(int(frames.shape[1]), int(frames.shape[2])),
                            resize=resize_latents,
                        )
                    except Exception as exc:
                        if not continue_on_latent_error:
                            raise
                        latent_record = {
                            "status": "error",
                            "video": str(video_path),
                            "error": repr(exc),
                        }
                    latent_record["kind"] = "video"
                    latent_record["video_layout"] = (
                        "flow_over_vorticity" if stack_vorticity else "single"
                    )
                    latent_record["uid"] = spec["uid"]
                    latent_records.append(latent_record)

                metadata = {
                    "nu": spec["nu"],
                    "rho": spec["rho"],
                    "field": field,
                    "stack_vorticity": bool(stack_vorticity),
                    "video_layout": (
                        "flow_over_vorticity" if stack_vorticity else "single"
                    ),
                    "panel_height": int(video_height),
                    "panel_width": int(video_width),
                    "video_height": int(frames.shape[1]),
                    "video_width": int(frames.shape[2]),
                }
                json_path = sample_dir / "metadata.json"
                with json_path.open("w") as f:
                    json.dump(metadata, f)

                record = {
                    "uid": spec["uid"],
                    "folder": str(sample_dir),
                    "json": str(json_path),
                    "video": str(video_path),
                    "nu": spec["nu"],
                    "rho": spec["rho"],
                    "frames": int(frames.shape[0]),
                    "video_height": int(frames.shape[1]),
                    "video_width": int(frames.shape[2]),
                    "panel_height": int(video_height),
                    "panel_width": int(video_width),
                    "stack_vorticity": bool(stack_vorticity),
                    "video_layout": (
                        "flow_over_vorticity" if stack_vorticity else "single"
                    ),
                    "video_duration_seconds": float(
                        expected_video_timing["playback_duration_seconds"]
                    ),
                    "timing_verified": bool(video_timing["timing_verified"]),
                    "timing_verification_scope": str(
                        video_timing["timing_verification_scope"]
                    ),
                    "written_video_timing_verified": bool(
                        video_timing["written_video_timing_verified"]
                    ),
                }
                if latent_record is not None:
                    if "latents" in latent_record:
                        record["latents"] = latent_record["latents"]
                    record["latent_status"] = latent_record.get("status")
                records.append(record)

            next_sample_index += current_batch_size
            progress.update(current_batch_size)
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


def load_uint8_frames_for_wan_vae(
    frames: torch.Tensor,
    device: Optional[str | torch.device] = None,
    expected_size: Tuple[int, int] = DEFAULT_WAN_EXPECTED_SIZE,
    resize: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Return Wan-normalized video shaped (C, T, H, W) from uint8 video frames."""
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    if frames.ndim != 4:
        raise ValueError("frames must have shape (T, H, W, C) or (T, C, H, W)")
    if frames.shape[-1] <= 4:
        frames = frames.permute(0, 3, 1, 2)
    elif frames.shape[1] <= 4:
        frames = frames
    else:
        raise ValueError("could not infer channel dimension for video frames")
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
                f"Expected {expected_size} video frames, got {tuple(frames.shape[-2:])}"
            )
        frames = F.interpolate(
            frames, size=expected_size, mode="bicubic", align_corners=False
        ).clamp_(0.0, 1.0)
    video = frames.mul_(2.0).sub_(1.0).permute(1, 0, 2, 3).contiguous().to(device)
    info = {
        "source": "in_memory_frames",
        "decoded_video": False,
        "frames": int(video.shape[1]),
        "height": int(video.shape[2]),
        "width": int(video.shape[3]),
    }
    return (video, info)


def _save_wan_latents(
    video: torch.Tensor,
    video_info: Dict,
    output_path: Path,
    source_video_name: str,
    checkpoint_dir: str | Path,
    save_dtype: torch.dtype,
    vae,
) -> Dict[str, str | int | float | Tuple[int, ...]]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*"
        )
        latents = (
            vae.encode([video])[0].detach().to("cpu", dtype=save_dtype).contiguous()
        )
    metadata = {
        "source_video": source_video_name,
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
        "latents": str(output_path),
        "frames": int(video.shape[1]),
        "latent_shape": tuple(latents.shape),
    }


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
    record = _save_wan_latents(
        video,
        video_info,
        output_path,
        video_path.name,
        checkpoint_dir,
        save_dtype,
        vae,
    )
    record["video"] = str(video_path)
    return record


@torch.no_grad()
def encode_frames_to_wan_latents(
    frames: torch.Tensor,
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
    """Encode rendered dataset frames and save latents.safetensors beside the mp4."""
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
    video, video_info = load_uint8_frames_for_wan_vae(
        frames, device=device, expected_size=expected_size, resize=resize
    )
    record = _save_wan_latents(
        video,
        video_info,
        output_path,
        video_path.name,
        checkpoint_dir,
        save_dtype,
        vae,
    )
    record["video"] = str(video_path)
    return record


def self_test_masked_projection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    velocity0, _, obstacle = make_initial_velocity_field(
        n=64,
        amplitude=1.0,
        seed=0,
        device=device,
        return_obstacle=True,
    )
    simulation = simulate_navier_stokes(
        velocity0,
        nu=1e-3,
        rho=1.0,
        T=0.05,
        delta_t=None,
        max_delta_t=1e-3,
        pressure_method="spectral",
        pressure_iterations=80,
        masked_pressure_max_iterations=80,
        masked_pressure_tolerance=1e-4,
        initial_masked_pressure_max_iterations=150,
        initial_masked_pressure_tolerance=1e-5,
        obstacle_method="masked_pcg",
        obstacle_mask=obstacle["mask"],
        sample_times=torch.linspace(0.0, 0.05, 6, device="cpu", dtype=torch.float64),
        project_initial=True,
        boundary_u=1.0,
        boundary_v=0.0,
    )

    fluid = ~obstacle["mask"]
    initial_div = masked_divergence(
        simulation["velocity"][0],
        obstacle["mask"],
        boundary_u=1.0,
        boundary_v=0.0,
    )
    final_div = masked_divergence(
        simulation["velocity"][-1],
        obstacle["mask"],
        boundary_u=1.0,
        boundary_v=0.0,
    )

    print(
        "finite values:",
        bool(torch.isfinite(simulation["velocity"]).all().detach().cpu()),
    )
    print(
        "initial masked divergence linf:",
        float(initial_div[fluid].abs().amax().detach().cpu()),
    )
    print(
        "final masked divergence linf:",
        float(final_div[fluid].abs().amax().detach().cpu()),
    )
    print(
        "initial max speed:",
        float(
            simulation["velocity"][0]
            .square()
            .sum(dim=0)
            .sqrt()[fluid]
            .amax()
            .detach()
            .cpu()
        ),
    )
    print(
        "final max speed:",
        float(
            simulation["velocity"][-1]
            .square()
            .sum(dim=0)
            .sqrt()[fluid]
            .amax()
            .detach()
            .cpu()
        ),
    )


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
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--grid-size", type=int, default=NAVIER_GRID_SIZE)
    parser.add_argument("--width", type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    parser.add_argument(
        "--with-vorticity",
        "--vorticity-stack",
        dest="stack_vorticity",
        action="store_true",
        help="write flow over vorticity as the dataset video",
    )
    parser.add_argument(
        "--no-vorticity",
        "--no-vorticity-stack",
        dest="stack_vorticity",
        action="store_false",
        help="write only the original single flow video",
    )
    parser.set_defaults(stack_vorticity=DEFAULT_STACK_VORTICITY)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--wan-repo-root", type=Path, default=DEFAULT_WAN_REPO_ROOT)
    parser.add_argument(
        "--wan-checkpoint-dir", type=Path, default=DEFAULT_WAN_CHECKPOINT_DIR
    )

    parser.add_argument("--T", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--delta-t", type=float, default=None)
    parser.add_argument("--max-delta-t", type=float, default=DEFAULT_MAX_DELTA_T)
    parser.add_argument("--nu-min", type=float, default=3e-4)
    parser.add_argument("--nu-max", type=float, default=3e-3)
    parser.add_argument("--rho-min", type=float, default=0.5)
    parser.add_argument("--rho-max", type=float, default=2.0)
    parser.add_argument("--amplitude", type=float, default=DEFAULT_WIND_SPEED)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--pressure-method", type=str, default="spectral")
    parser.add_argument("--pressure-iterations", type=int, default=80)
    parser.add_argument("--masked-pressure-iterations", type=int, default=None)
    parser.add_argument(
        "--masked-pressure-max-iterations",
        type=int,
        default=DEFAULT_MASKED_PRESSURE_MAX_ITERATIONS,
    )
    parser.add_argument(
        "--masked-pressure-tolerance",
        type=float,
        default=DEFAULT_MASKED_PRESSURE_TOLERANCE,
    )
    parser.add_argument(
        "--initial-masked-pressure-max-iterations",
        type=int,
        default=DEFAULT_INITIAL_MASKED_PRESSURE_MAX_ITERATIONS,
    )
    parser.add_argument(
        "--initial-masked-pressure-tolerance",
        type=float,
        default=DEFAULT_INITIAL_MASKED_PRESSURE_TOLERANCE,
    )
    parser.add_argument(
        "--obstacle-method",
        type=str,
        default=DEFAULT_OBSTACLE_METHOD,
        choices=("masked_pcg", "penalized_spectral"),
    )
    parser.add_argument("--obstacle-penalty-eta", type=float, default=1e-3)
    parser.add_argument("--initial-projection-steps", type=int, default=0)
    parser.add_argument("--initial-projection-delta-t", type=float, default=1.0)
    parser.add_argument("--initial-relaxation-steps", type=int, default=0)
    parser.add_argument("--initial-relaxation-delta-t", type=float, default=None)
    parser.add_argument("--self-test-masked-projection", action="store_true")
    parser.add_argument(
        "--verify-written-videos",
        action="store_true",
        help="decode each written mp4 to verify container fps/timestamps",
    )
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
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
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

    if args.self_test_masked_projection:
        self_test_masked_projection()
        return 0

    output_root = args.output_root
    if output_root is None:
        output_root = (
            DEFAULT_WITH_VORTICITY_OUTPUT_ROOT
            if args.stack_vorticity
            else DEFAULT_OUTPUT_ROOT
        )
    output_video_height = video_height * (2 if args.stack_vorticity else 1)

    latent_records = []
    records = generate_navier_dataset(
        int(args.samples),
        output_root=output_root,
        batch_size=int(args.batch_size),
        n=int(args.grid_size),
        video_width=video_width,
        video_height=video_height,
        T=float(args.T),
        nu_range=(float(args.nu_min), float(args.nu_max)),
        rho_range=(float(args.rho_min), float(args.rho_max)),
        amplitude=float(args.amplitude),
        fps=int(args.fps),
        save_every=int(args.save_every),
        field="speed",
        stack_vorticity=bool(args.stack_vorticity),
        delta_t=args.delta_t,
        pressure_method=args.pressure_method,
        pressure_iterations=int(args.pressure_iterations),
        masked_pressure_iterations=args.masked_pressure_iterations,
        masked_pressure_max_iterations=int(args.masked_pressure_max_iterations),
        masked_pressure_tolerance=float(args.masked_pressure_tolerance),
        initial_masked_pressure_max_iterations=int(
            args.initial_masked_pressure_max_iterations
        ),
        initial_masked_pressure_tolerance=float(
            args.initial_masked_pressure_tolerance
        ),
        obstacle_method=args.obstacle_method,
        obstacle_penalty_eta=float(args.obstacle_penalty_eta),
        max_delta_t=float(args.max_delta_t),
        use_compile=bool(args.use_compile),
        seed=args.seed,
        device=device,
        dtype=args.sim_dtype,
        verify_written_videos=bool(args.verify_written_videos),
        velocity_color_bound=float(args.amplitude),
        initial_projection_steps=int(args.initial_projection_steps),
        initial_projection_delta_t=float(args.initial_projection_delta_t),
        initial_relaxation_steps=int(args.initial_relaxation_steps),
        initial_relaxation_delta_t=args.initial_relaxation_delta_t,
        encode_latents=True,
        wan_checkpoint_dir=args.wan_checkpoint_dir,
        wan_repo_root=args.wan_repo_root,
        vae_dtype=args.vae_dtype,
        latent_save_dtype=args.latent_save_dtype,
        overwrite_latents=args.overwrite_latents,
        resize_latents=args.resize_latents,
        continue_on_latent_error=bool(args.continue_on_latent_error),
        latent_records=latent_records,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = {
        "samples_requested": int(args.samples),
        "batch_size": int(args.batch_size),
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
        "output_root": str(output_root),
        "grid_size": int(args.grid_size),
        "stack_vorticity": bool(args.stack_vorticity),
        "video_layout": (
            "flow_over_vorticity" if args.stack_vorticity else "single"
        ),
        "panel_height": int(video_height),
        "panel_width": int(video_width),
        "video_height": int(output_video_height),
        "video_width": int(video_width),
        "verify_written_videos": bool(args.verify_written_videos),
        "fps": int(args.fps),
        "T": float(args.T),
        "frames_per_video": _prepended_initial_realtime_video_frame_count(
            float(args.T), int(args.fps), warn=False
        ),
        "video_timing": _realtime_video_timing_metadata(
            float(args.T),
            int(args.fps),
            _prepended_initial_realtime_video_frame_count(
                float(args.T), int(args.fps), warn=False
            ),
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
