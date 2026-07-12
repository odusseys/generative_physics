"""Temporally coherent BAPPS-style video distortions.

The transforms in this module preserve video shape and use parameters that are
constant or smoothly varying over time. They are intended for offline dataset
generation and qualitative inspection, not as differentiable training layers.
The registry covers BAPPS's photometric, corruption, spatial, temporal, and
compression families, plus a representative composition of atomic transforms.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

import av
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import HTML, Video, display
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

__all__ = [
    "AUGMENTATIONS",
    "AugmentationConfig",
    "PACKAGE_ROOT",
    "VideoClip",
    "assert_valid_augmentation",
    "discover_sample_paths",
    "distortion_diagnostics",
    "encode_mp4",
    "generator_for",
    "load_video",
    "render_sample",
    "run_synthetic_contract_tests",
]


@dataclass
class AugmentationConfig:
    """Runtime settings for previewing or exporting distorted video pairs."""

    preview_max_edge: Optional[int] = 512
    severity: float = 0.65
    seed: int = 2026
    encode_crf: int = 25
    encode_preset: str = "veryfast"
    save_outputs: bool = False
    output_dir: Optional[Path] = None
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        self.device = self.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be in [0,1]")
        if self.preview_max_edge is not None and self.preview_max_edge < 32:
            raise ValueError("preview_max_edge must be at least 32 or None")
        if not 0 <= self.encode_crf <= 51:
            raise ValueError("encode_crf must be in [0,51]")
        if self.save_outputs and self.output_dir is None:
            raise ValueError("save_outputs=True requires output_dir")


@dataclass
class VideoClip:
    path: Path
    frames: torch.Tensor  # [T,3,H,W], float in [0,1]
    fps: float


Augmentation = Callable[[torch.Tensor, float, torch.Generator], torch.Tensor]


def discover_sample_paths(sample_dir: Path) -> list[Path]:
    paths = sorted(Path(sample_dir).glob("*.mp4"))
    if not paths:
        raise FileNotFoundError(f"No MP4 samples found in {sample_dir}")
    return paths


def even_preview_size(
    height: int,
    width: int,
    max_edge: Optional[int],
) -> tuple[int, int]:
    if max_edge is None or max(height, width) <= max_edge:
        return height - height % 2, width - width % 2
    scale = max_edge / max(height, width)
    new_h = max(2, int(round(height * scale)))
    new_w = max(2, int(round(width * scale)))
    return new_h - new_h % 2, new_w - new_w % 2


@torch.inference_mode()
def load_video(
    path: Path,
    max_edge: Optional[int],
    device: torch.device,
) -> VideoClip:
    """Decode an MP4 and optionally downsize it frame-by-frame."""

    path = Path(path)
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 24.0)
        target_size = even_preview_size(
            stream.height,
            stream.width,
            max_edge,
        )
        for frame in container.decode(video=0):
            tensor = (
                torch.from_numpy(frame.to_ndarray(format="rgb24").copy())
                .permute(2, 0, 1)
                .float()
                .div_(255.0)
            )
            if tuple(tensor.shape[-2:]) != target_size:
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=target_size,
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )[0]
            frames.append(tensor.clamp_(0, 1))
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return VideoClip(
        path=path,
        frames=torch.stack(frames).to(device),
        fps=fps,
    )


@torch.inference_mode()
def encode_mp4(
    frames: torch.Tensor,
    fps: float,
    crf: int = 25,
    preset: str = "veryfast",
) -> bytes:
    """Encode [T,3,H,W] RGB floats into an in-memory H.264 MP4."""

    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"Expected [T,3,H,W], got {tuple(frames.shape)}")
    array = (
        frames.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    height, width = array.shape[1:3]
    if height % 2 or width % 2:
        raise ValueError("H.264 preview dimensions must be even")

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream(
            "libx264",
            rate=Fraction(fps).limit_denominator(1000),
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": preset}
        for rgb in array:
            video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def display_video_grid(
    title: str,
    encoded: dict[str, bytes],
    diagnostics: dict[str, dict[str, float]],
) -> None:
    """Display original and distorted clips in a responsive notebook grid."""

    tiles = []
    for name, payload in encoded.items():
        video_html = Video(
            data=payload,
            embed=True,
            mimetype="video/mp4",
            html_attributes=(
                "controls autoplay loop muted playsinline preload=metadata"
            ),
        )._repr_html_()
        metric = diagnostics.get(name)
        caption = (
            name
            if metric is None
            else (
                f"{name} · MAE {metric['mae']:.3f} · "
                f"residual Δt {metric['residual_dt']:.3f}"
            )
        )
        tiles.append(
            "<figure class=video-tile>"
            f"<figcaption>{caption}</figcaption>{video_html}</figure>"
        )

    styles = """
    <style>
      .video-grid {
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
        gap:14px;
        align-items:start;
      }
      .video-tile {
        margin:0;
        padding:10px;
        border:1px solid #d0d7de;
        border-radius:10px;
        background:#fafbfc;
      }
      .video-tile figcaption {
        font:600 12px/1.4 system-ui;
        margin-bottom:7px;
        color:#30363d;
      }
      .video-tile video {
        width:100%;
        height:auto;
        display:block;
        border-radius:6px;
        background:#111;
      }
    </style>
    """
    display(
        HTML(
            styles
            + f"<h3>{title}</h3><div class=video-grid>"
            + "".join(tiles)
            + "</div>"
        )
    )


def save_encoded_previews(
    sample_name: str,
    encoded: dict[str, bytes],
    output_dir: Path,
) -> None:
    sample_dir = Path(output_dir) / Path(sample_name).stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in encoded.items():
        (sample_dir / f"{name}.mp4").write_bytes(payload)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8]
    return int.from_bytes(digest, "big") % (2**63 - 1)


def generator_for(
    sample: str,
    augmentation: str,
    base_seed: int,
) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(
        stable_seed(base_seed, sample, augmentation)
    )


def smooth_signal(
    frame_count: int,
    generator: torch.Generator,
    knots: int = 5,
) -> torch.Tensor:
    """Sample a low-frequency, normalized 1-D temporal control signal."""

    knot_count = min(max(2, knots), frame_count)
    values = torch.randn(1, 1, knot_count, generator=generator)
    signal = F.interpolate(
        values,
        size=frame_count,
        mode="linear",
        align_corners=True,
    )[0, 0]
    signal = signal - signal.mean()
    return signal / signal.abs().amax().clamp_min(1e-6)


def gaussian_kernel2d(
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=device,
        dtype=dtype,
    )
    kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(3, 1, -1, -1).contiguous()


def assert_valid_augmentation(
    reference: torch.Tensor,
    augmented: torch.Tensor,
    name: str,
) -> None:
    if augmented.shape != reference.shape:
        raise ValueError(
            f"{name} changed video shape from {tuple(reference.shape)} "
            f"to {tuple(augmented.shape)}"
        )
    if not torch.isfinite(augmented).all():
        raise ValueError(f"{name} produced NaN or Inf")
    if float(augmented.min()) < -1e-5 or float(augmented.max()) > 1 + 1e-5:
        raise ValueError(f"{name} left the [0,1] range")


def distortion_diagnostics(
    reference: torch.Tensor,
    augmented: torch.Tensor,
) -> dict[str, float]:
    """Report magnitude and frame-to-frame change of the distortion residual."""

    residual = augmented.float() - reference.float()
    residual_dt = residual[1:] - residual[:-1]
    return {
        "mae": float(residual.abs().mean()),
        "residual_dt": (
            float(residual_dt.abs().mean()) if len(residual_dt) else 0.0
        ),
    }


@torch.inference_mode()
def photometric_drift(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    count = len(frames)
    brightness = (
        smooth_signal(count, generator, 5)
        .to(frames.device)
        .view(-1, 1, 1, 1)
        * (0.20 * severity)
    )
    contrast = 1 + (
        smooth_signal(count, generator, 4)
        .to(frames.device)
        .view(-1, 1, 1, 1)
        * (0.45 * severity)
    )
    saturation = 1 + (
        smooth_signal(count, generator, 6)
        .to(frames.device)
        .view(-1, 1, 1, 1)
        * (0.58 * severity)
    )
    adjusted = (frames - 0.5) * contrast + 0.5 + brightness
    luma = (
        adjusted[:, 0:1] * 0.2126
        + adjusted[:, 1:2] * 0.7152
        + adjusted[:, 2:3] * 0.0722
    )
    return (luma + (adjusted - luma) * saturation).clamp(0, 1)


@torch.inference_mode()
def coherent_camera_drift(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    count = len(frames)
    dx = smooth_signal(count, generator, 5) * (0.09 * severity)
    dy = smooth_signal(count, generator, 6) * (0.09 * severity)
    angle = (
        smooth_signal(count, generator, 4)
        * math.radians(7.0 * severity)
    )
    zoom = 1 + smooth_signal(count, generator, 5) * (0.075 * severity)
    cosine = torch.cos(angle) / zoom
    sine = torch.sin(angle) / zoom
    theta = torch.zeros(count, 2, 3)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 0, 2] = dx
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    theta[:, 1, 2] = dy
    theta = theta.to(device=frames.device, dtype=frames.dtype)
    grid = F.affine_grid(theta, size=frames.shape, align_corners=False)
    return F.grid_sample(
        frames,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    ).clamp(0, 1)


@torch.inference_mode()
def blur_and_resample(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    del generator  # Constant through time by design.
    height, width = frames.shape[-2:]
    scale = max(0.35, 1.0 - 0.62 * severity)
    small_h = max(4, int(round(height * scale)))
    small_w = max(4, int(round(width * scale)))
    degraded = F.interpolate(
        frames,
        size=(small_h, small_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    degraded = F.interpolate(
        degraded,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    sigma = 0.35 + 1.9 * severity
    kernel = gaussian_kernel2d(
        sigma,
        frames.device,
        frames.dtype,
    )
    radius = kernel.shape[-1] // 2
    padded = F.pad(
        degraded,
        (radius, radius, radius, radius),
        mode="reflect",
    )
    return F.conv2d(padded, kernel, groups=3).clamp(0, 1)


@torch.inference_mode()
def temporal_ghosting(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    count = len(frames)
    previous = torch.cat([frames[:1], frames[:-1]], dim=0)
    previous_2 = (
        torch.cat([frames[:1], frames[:1], frames[:-2]], dim=0)
        if count > 1
        else frames
    )
    modulation = (
        smooth_signal(count, generator, 5)
        .mul(0.5)
        .add(0.5)
        .to(frames.device)
        .view(-1, 1, 1, 1)
    )
    alpha = (severity * (0.30 + 0.48 * modulation)).clamp(max=0.80)
    history = 0.72 * previous + 0.28 * previous_2
    return ((1 - alpha) * frames + alpha * history).clamp(0, 1)


@torch.inference_mode()
def correlated_noise_and_quantization(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    count, _, height, width = frames.shape
    low_h = max(4, height // 2)
    low_w = max(4, width // 2)
    rho = 0.94
    luma_state = torch.randn(1, low_h, low_w, generator=generator)
    chroma_state = torch.randn(3, low_h, low_w, generator=generator)
    noise = []
    for _ in range(count):
        luma_innovation = torch.randn(
            1,
            low_h,
            low_w,
            generator=generator,
        )
        chroma_innovation = torch.randn(
            3,
            low_h,
            low_w,
            generator=generator,
        )
        innovation_scale = math.sqrt(1 - rho**2)
        luma_state = rho * luma_state + innovation_scale * luma_innovation
        chroma_state = (
            rho * chroma_state + innovation_scale * chroma_innovation
        )
        noise.append(luma_state.expand(3, -1, -1) + 0.20 * chroma_state)
    noise_tensor = torch.stack(noise).to(
        device=frames.device,
        dtype=frames.dtype,
    )
    noise_tensor = F.interpolate(
        noise_tensor,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    noise_tensor = noise_tensor / noise_tensor.std().clamp_min(1e-6)
    noisy = (frames + noise_tensor * (0.065 * severity)).clamp(0, 1)
    levels = max(16, int(round(78 - 56 * severity)))
    return torch.round(noisy * levels) / levels


def _random_sign(generator: torch.Generator) -> float:
    return 1.0 if float(torch.rand((), generator=generator)) >= 0.5 else -1.0


@torch.inference_mode()
def color_temperature_and_gamma(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Smooth BAPPS-style color/lightness shift with a coherent direction."""

    count = len(frames)
    temperature = (
        0.34
        * severity
        * _random_sign(generator)
        * (0.70 + 0.30 * smooth_signal(count, generator, 5))
    )
    tint = (
        0.16
        * severity
        * _random_sign(generator)
        * (0.75 + 0.25 * smooth_signal(count, generator, 4))
    )
    gamma_log = (
        0.56
        * severity
        * _random_sign(generator)
        * (0.70 + 0.30 * smooth_signal(count, generator, 6))
    )
    gains = torch.stack(
        [1 + temperature, 1 + tint, 1 - temperature],
        dim=1,
    ).to(device=frames.device, dtype=frames.dtype)
    gamma = gamma_log.exp().to(
        device=frames.device,
        dtype=frames.dtype,
    ).view(-1, 1, 1, 1)
    shifted = frames * gains.view(count, 3, 1, 1)
    return shifted.clamp(1e-6, 1).pow(gamma).clamp(0, 1)


@torch.inference_mode()
def chromatic_aberration(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Shift red and blue oppositely using slowly drifting offsets."""

    count, _, height, width = frames.shape
    x_pixels = (
        (3.0 + 12.0 * severity)
        * _random_sign(generator)
        * (0.68 + 0.32 * smooth_signal(count, generator, 5))
    )
    y_pixels = (
        (2.0 + 8.0 * severity)
        * _random_sign(generator)
        * (0.68 + 0.32 * smooth_signal(count, generator, 6))
    )
    identity = torch.zeros(count, 2, 3)
    identity[:, 0, 0] = 1
    identity[:, 1, 1] = 1
    channels = []
    for channel, direction in enumerate((-1.0, 0.0, 1.0)):
        theta = identity.clone()
        theta[:, 0, 2] = direction * 2.0 * x_pixels / max(width, 1)
        theta[:, 1, 2] = direction * 2.0 * y_pixels / max(height, 1)
        theta = theta.to(device=frames.device, dtype=frames.dtype)
        grid = F.affine_grid(
            theta,
            size=(count, 1, height, width),
            align_corners=False,
        )
        channels.append(
            F.grid_sample(
                frames[:, channel : channel + 1],
                grid,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=False,
            )
        )
    return torch.cat(channels, dim=1).clamp(0, 1)


@torch.inference_mode()
def coherent_wave_warp(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply a BAPPS-style nonlinear warp whose phase drifts slowly in time."""

    count, _, height, width = frames.shape
    identity = torch.zeros(
        count,
        2,
        3,
        device=frames.device,
        dtype=frames.dtype,
    )
    identity[:, 0, 0] = 1
    identity[:, 1, 1] = 1
    grid = F.affine_grid(identity, frames.shape, align_corners=False)
    phase = (
        smooth_signal(count, generator, 5)
        .to(device=frames.device, dtype=frames.dtype)
        .view(-1, 1, 1)
        * 0.55
    )
    cycles = 1.35 + 0.8 * float(torch.rand((), generator=generator))
    amplitude = 3.0 + 15.0 * severity
    x_wave = torch.sin(math.pi * cycles * (grid[..., 1] + 1) + phase)
    y_wave = torch.sin(
        math.pi * (cycles * 0.72) * (grid[..., 0] + 1) - phase
    )
    warped_grid = grid.clone()
    warped_grid[..., 0] += x_wave * (2.0 * amplitude / max(width, 1))
    warped_grid[..., 1] += y_wave * (
        2.0 * amplitude * 0.55 / max(height, 1)
    )
    return F.grid_sample(
        frames,
        warped_grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    ).clamp(0, 1)


@torch.inference_mode()
def checkerboard_artifact(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Overlay a fixed spatial checker pattern with smooth temporal strength."""

    count, _, height, width = frames.shape
    block = max(4, int(round(18 - 10 * severity)))
    yy = torch.arange(height, device=frames.device).view(height, 1)
    xx = torch.arange(width, device=frames.device).view(1, width)
    checker = (((yy // block + xx // block) % 2) * 2 - 1).to(frames.dtype)
    color = frames.new_tensor([1.0, 0.55, -0.65]).view(1, 3, 1, 1)
    envelope = (
        smooth_signal(count, generator, 5)
        .mul(0.5)
        .add(0.5)
        .to(device=frames.device, dtype=frames.dtype)
        .view(-1, 1, 1, 1)
    )
    amplitude = severity * (0.08 + 0.09 * envelope)
    pattern = color * checker.view(1, 1, height, width)
    return (frames + amplitude * pattern).clamp(0, 1)


@torch.inference_mode()
def color_removal_and_vignette(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Combine coherent color removal with a strong, stable radial falloff."""

    count, _, height, width = frames.shape
    luma = (
        frames[:, 0:1] * 0.2126
        + frames[:, 1:2] * 0.7152
        + frames[:, 2:3] * 0.0722
    )
    envelope = (
        smooth_signal(count, generator, 4)
        .mul(0.15)
        .add(0.85)
        .to(device=frames.device, dtype=frames.dtype)
        .view(-1, 1, 1, 1)
    )
    saturation = (1.0 - 0.82 * severity * envelope).clamp(min=0.05)
    desaturated = luma + (frames - luma) * saturation

    y = torch.linspace(-1, 1, height, device=frames.device, dtype=frames.dtype)
    x = torch.linspace(-1, 1, width, device=frames.device, dtype=frames.dtype)
    radius = torch.sqrt(y[:, None].square() + x[None, :].square()).clamp(max=1)
    vignette = (1 - 0.72 * severity * radius.pow(1.7)).view(
        1,
        1,
        height,
        width,
    )
    return (desaturated * vignette).clamp(0, 1)


@torch.inference_mode()
def jpeg_compression(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Round-trip every frame through JPEG at one quality for the whole clip."""

    del generator  # Matching quality across frames is the temporal constraint.
    quality = max(7, int(round(72 - 62 * severity)))
    rgb_frames = (
        frames.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    decoded = []
    for rgb in rgb_frames:
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=2,
            optimize=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as image:
            decoded.append(
                torch.from_numpy(np.asarray(image.convert("RGB")).copy())
                .permute(2, 0, 1)
                .float()
                .div_(255)
            )
    return torch.stack(decoded).to(device=frames.device, dtype=frames.dtype)


@torch.inference_mode()
def local_exposure_flicker(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create deliberate flicker inside several slowly moving soft regions."""

    count, _, height, width = frames.shape
    y = torch.linspace(-1, 1, height, device=frames.device, dtype=frames.dtype)
    x = torch.linspace(-1, 1, width, device=frames.device, dtype=frames.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    time = torch.linspace(0, 1, count, device=frames.device, dtype=frames.dtype)
    effect = torch.zeros_like(frames)
    blob_count = 3
    for _ in range(blob_count):
        center_x = (
            float(torch.rand((), generator=generator)) * 1.2
            - 0.6
            + 0.20
            * smooth_signal(count, generator, 5).to(frames.device)
        )
        center_y = (
            float(torch.rand((), generator=generator)) * 1.2
            - 0.6
            + 0.20
            * smooth_signal(count, generator, 6).to(frames.device)
        )
        sigma = 0.14 + 0.14 * float(torch.rand((), generator=generator))
        mask = torch.exp(
            -(
                (xx[None] - center_x[:, None, None]).square()
                + (yy[None] - center_y[:, None, None]).square()
            )
            / (2 * sigma**2)
        )
        frequency = 3.0 + 6.0 * float(torch.rand((), generator=generator))
        phase = 2 * math.pi * float(torch.rand((), generator=generator))
        carrier = torch.sin(2 * math.pi * frequency * time + phase)
        color = frames.new_tensor(
            [
                0.85 + 0.30 * float(torch.rand((), generator=generator)),
                0.75 + 0.30 * float(torch.rand((), generator=generator)),
                0.85 + 0.30 * float(torch.rand((), generator=generator)),
            ]
        )
        effect += (
            mask[:, None]
            * carrier[:, None, None, None]
            * color.view(1, 3, 1, 1)
        )
    effect = effect / math.sqrt(blob_count)
    delta = effect * (0.52 * severity)
    return (frames * (1 + delta) + 0.10 * delta).clamp(0, 1)


@torch.inference_mode()
def local_elastic_warp(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Distort moving local regions while keeping their trajectories smooth."""

    count, _, height, width = frames.shape
    identity = torch.zeros(
        count,
        2,
        3,
        device=frames.device,
        dtype=frames.dtype,
    )
    identity[:, 0, 0] = 1
    identity[:, 1, 1] = 1
    grid = F.affine_grid(identity, frames.shape, align_corners=False)
    dx_field = torch.zeros(count, height, width, device=frames.device)
    dy_field = torch.zeros_like(dx_field)
    amplitude = 5.0 + 26.0 * severity
    for _ in range(2):
        center_x = (
            float(torch.rand((), generator=generator)) * 1.2
            - 0.6
            + 0.22
            * smooth_signal(count, generator, 5).to(frames.device)
        )
        center_y = (
            float(torch.rand((), generator=generator)) * 1.2
            - 0.6
            + 0.22
            * smooth_signal(count, generator, 6).to(frames.device)
        )
        sigma = 0.16 + 0.16 * float(torch.rand((), generator=generator))
        mask = torch.exp(
            -(
                (grid[..., 0] - center_x[:, None, None]).square()
                + (grid[..., 1] - center_y[:, None, None]).square()
            )
            / (2 * sigma**2)
        )
        dx_control = (
            _random_sign(generator)
            * (0.62 + 0.38 * smooth_signal(count, generator, 5))
            .to(frames.device)
            .view(-1, 1, 1)
        )
        dy_control = (
            _random_sign(generator)
            * (0.62 + 0.38 * smooth_signal(count, generator, 6))
            .to(frames.device)
            .view(-1, 1, 1)
        )
        dx_field += amplitude * mask * dx_control
        dy_field += amplitude * mask * dy_control
    warped_grid = grid.clone()
    warped_grid[..., 0] += dx_field.to(frames.dtype) * (
        2.0 / max(width, 1)
    )
    warped_grid[..., 1] += dy_field.to(frames.dtype) * (
        2.0 / max(height, 1)
    )
    return F.grid_sample(
        frames,
        warped_grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    ).clamp(0, 1)


@torch.inference_mode()
def local_block_glitch(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Toggle displaced, color-biased local blocks in short temporal bursts."""

    count, _, height, width = frames.shape
    y = torch.linspace(-1, 1, height, device=frames.device, dtype=frames.dtype)
    x = torch.linspace(-1, 1, width, device=frames.device, dtype=frames.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    time = torch.linspace(0, 1, count, device=frames.device, dtype=frames.dtype)
    output = frames.clone()
    for _ in range(3):
        center_x = float(torch.rand((), generator=generator)) * 1.4 - 0.7
        center_y = float(torch.rand((), generator=generator)) * 1.4 - 0.7
        half_width = 0.10 + 0.18 * float(torch.rand((), generator=generator))
        half_height = 0.06 + 0.14 * float(torch.rand((), generator=generator))
        edge = 70.0
        spatial_mask = (
            torch.sigmoid(edge * (xx - center_x + half_width))
            * torch.sigmoid(edge * (center_x + half_width - xx))
            * torch.sigmoid(edge * (yy - center_y + half_height))
            * torch.sigmoid(edge * (center_y + half_height - yy))
        )
        frequency = 2.5 + 4.5 * float(torch.rand((), generator=generator))
        phase = 2 * math.pi * float(torch.rand((), generator=generator))
        burst = (
            torch.sin(2 * math.pi * frequency * time + phase) > 0.15
        ).to(frames.dtype)
        mask = spatial_mask[None, None] * burst[:, None, None, None]
        max_shift = max(2, int(round(5 + 22 * severity)))
        shift_x = int(
            _random_sign(generator)
            * (max_shift // 2 + float(torch.rand((), generator=generator)) * max_shift / 2)
        )
        shift_y = int(
            _random_sign(generator)
            * (2 + float(torch.rand((), generator=generator)) * max_shift / 3)
        )
        displaced = torch.roll(output, shifts=(shift_y, shift_x), dims=(-2, -1))
        gains = frames.new_tensor([1.30, 0.70, 1.12]).view(1, 3, 1, 1)
        displaced = (displaced * gains).clamp(0, 1)
        output = output * (1 - mask) + displaced * mask
    return output.clamp(0, 1)


@torch.inference_mode()
def composed_traditional_distortion(
    frames: torch.Tensor,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Representative BAPPS-style sequence of multiple atomic distortions."""

    output = photometric_drift(frames, severity * 0.82, generator)
    output = chromatic_aberration(output, severity * 0.72, generator)
    return correlated_noise_and_quantization(
        output,
        severity * 0.68,
        generator,
    )


AUGMENTATIONS: dict[str, Augmentation] = {
    "photometric_drift": photometric_drift,
    "temperature_gamma": color_temperature_and_gamma,
    "camera_drift": coherent_camera_drift,
    "wave_warp": coherent_wave_warp,
    "chromatic_aberration": chromatic_aberration,
    "blur_resample": blur_and_resample,
    "temporal_ghosting": temporal_ghosting,
    "local_exposure_flicker": local_exposure_flicker,
    "local_elastic_warp": local_elastic_warp,
    "local_block_glitch": local_block_glitch,
    "correlated_noise": correlated_noise_and_quantization,
    "checkerboard": checkerboard_artifact,
    "color_removal_vignette": color_removal_and_vignette,
    "jpeg_compression": jpeg_compression,
    "composed_traditional": composed_traditional_distortion,
}


@torch.inference_mode()
def run_synthetic_contract_tests(
    device: torch.device,
    seed: int = 2026,
) -> None:
    """Check every transform's shape/range/finiteness and input immutability."""

    reference = (
        torch.linspace(0, 1, 12 * 3 * 48 * 64)
        .reshape(12, 3, 48, 64)
        .to(device)
    )
    reference_copy = reference.clone()
    for name, augmentation in AUGMENTATIONS.items():
        output = augmentation(
            reference,
            0.7,
            generator_for("synthetic", name, seed),
        )
        assert_valid_augmentation(reference, output, name)
    if not torch.equal(reference, reference_copy):
        raise AssertionError("An augmentation mutated its input")
    seed_a = generator_for("x", "y", seed).initial_seed()
    seed_b = generator_for("x", "y", seed).initial_seed()
    if seed_a != seed_b:
        raise AssertionError("Augmentation seeds are not deterministic")


@torch.inference_mode()
def render_sample(
    path: Path,
    config: AugmentationConfig,
    *,
    display_inline: bool = True,
) -> dict[str, dict[str, float]]:
    """Generate every registered preview for one sample and optionally display."""

    clip = load_video(
        path,
        config.preview_max_edge,
        config.device,
    )
    encoded = {
        "original": encode_mp4(
            clip.frames,
            clip.fps,
            config.encode_crf,
            config.encode_preset,
        )
    }
    diagnostics: dict[str, dict[str, float]] = {}
    for name, augmentation in AUGMENTATIONS.items():
        generator = generator_for(path.name, name, config.seed)
        augmented = augmentation(
            clip.frames,
            config.severity,
            generator,
        )
        assert_valid_augmentation(clip.frames, augmented, name)
        diagnostics[name] = distortion_diagnostics(
            clip.frames,
            augmented,
        )
        encoded[name] = encode_mp4(
            augmented,
            clip.fps,
            config.encode_crf,
            config.encode_preset,
        )
        del augmented

    if config.save_outputs:
        save_encoded_previews(
            path.name,
            encoded,
            config.output_dir,
        )
    if display_inline:
        frame_count, _, height, width = clip.frames.shape
        title = (
            f"{path.name} · {frame_count} frames · {width}×{height} · "
            f"{clip.fps:g} fps · severity {config.severity:.2f}"
        )
        display_video_grid(title, encoded, diagnostics)
    return diagnostics
