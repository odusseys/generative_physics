"""Training utilities for a Wan2.2 video latent perceptual metric.

This module contains the reusable implementation used by the featurizer
notebook: manifest validation, leakage-safe splits, Wan VAE latent caching,
the framewise latent VGG/LPIPS model, and training.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG16_BN_Weights, vgg16_bn
from tqdm.auto import tqdm

__all__ = [
    "FeaturizerConfig",
    "PACKAGE_ROOT",
    "PairRecord",
    "SynchronizedLatentAugment",
    "VideoPairLatentDataset",
    "WanLatentVGG16BN",
    "WanVideoLatentLPIPS",
    "build_latent_cache",
    "build_metric",
    "ensembled_distance",
    "evaluate",
    "make_loader",
    "normalize_target",
    "plot_training_diagnostics",
    "read_manifest",
    "resolve_device",
    "run_contract_tests",
    "run_training_pipeline",
    "safe_torch_load",
    "seed_everything",
    "split_records",
    "train",
    "validate_checkpoint_config",
    "validate_config",
]


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FeaturizerConfig:
    """Configuration for latent caching and pair-score calibration."""

    repo_root: Path = PACKAGE_ROOT
    checkpoint_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "WAN22_CHECKPOINT_DIR",
                "/home/azureuser/Wan2.2-TI2V-5B",
            )
        )
    )
    manifest_path: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "wan-lpips" / "pairs.jsonl"
    )
    video_root: Optional[Path] = None
    cache_dir: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "wan-lpips" / "latent_cache"
    )
    output_dir: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "wan-lpips" / "outputs"
    )
    resume_from: Optional[Path] = None
    trunk_checkpoint: Optional[Path] = None
    device: Optional[str] = None

    clip_frames: int = 17
    resize_hw: tuple[int, int] = (256, 256)
    cache_dtype: str = "float16"
    allow_unverified_latents: bool = False

    score_kind: str = "similarity"
    score_min: float = 0.0
    score_max: float = 1.0
    val_fraction: float = 0.10
    split_seed: int = 1234

    remove_first_pools: int = 4
    initialize_from_imagenet: bool = True
    allow_random_trunk: bool = False
    train_trunk: bool = True
    head_dropout: float = 0.5
    use_motion_branch: bool = False
    motion_weight: float = 0.25

    calibration_augment: bool = False
    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 10
    fixed_lr_epochs: int = 5
    head_lr: float = 1e-4
    trunk_lr: float = 1e-5
    beta1: float = 0.5
    weight_decay: float = 0.0
    huber_delta: float = 0.10
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    amp_dtype: str = "bfloat16"
    seed: int = 1234


def resolve_device(config: FeaturizerConfig) -> torch.device:
    return torch.device(
        config.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )


def amp_dtype(config: FeaturizerConfig) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[config.amp_dtype]


def validate_config(config: FeaturizerConfig) -> None:
    if config.clip_frames < 1 or (config.clip_frames - 1) % 4 != 0:
        raise ValueError(
            "clip_frames must equal 4k+1; Wan otherwise drops trailing frames"
        )
    if any(size < 16 or size % 16 for size in config.resize_hw):
        raise ValueError(
            "resize_hw dimensions must be positive multiples of VAE stride 16"
        )
    if config.score_kind not in {"similarity", "distance"}:
        raise ValueError("score_kind must be similarity or distance")
    if not config.score_max > config.score_min:
        raise ValueError("score_max must be greater than score_min")
    if not 0.0 < config.val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    if not 0 <= config.remove_first_pools <= 5:
        raise ValueError("remove_first_pools must be in [0,5]")
    remaining_pools = 5 - config.remove_first_pools
    latent_h = config.resize_hw[0] // 16
    latent_w = config.resize_hw[1] // 16
    if min(latent_h, latent_w) < 2**remaining_pools:
        raise ValueError(
            f"resize_hw is too small for {remaining_pools} remaining VGG pools"
        )
    if (
        config.batch_size < 1
        or config.num_workers < 0
        or config.epochs < 1
    ):
        raise ValueError(
            "batch_size/epochs must be positive and num_workers nonnegative"
        )
    if not 0 <= config.fixed_lr_epochs <= config.epochs:
        raise ValueError("fixed_lr_epochs must be between zero and epochs")
    if (
        config.head_lr <= 0
        or config.trunk_lr <= 0
        or config.weight_decay < 0
    ):
        raise ValueError(
            "learning rates must be positive and weight_decay nonnegative"
        )
    if not 0 <= config.beta1 < 1 or not 0 <= config.head_dropout < 1:
        raise ValueError("beta1 and head_dropout must be in [0,1)")
    if (
        config.huber_delta <= 0
        or config.grad_accum_steps < 1
        or config.grad_clip_norm <= 0
    ):
        raise ValueError(
            "Huber delta, accumulation steps, and gradient clip must be positive"
        )
    if config.cache_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("Unknown cache_dtype")
    if config.amp_dtype not in {"bfloat16", "float32"}:
        raise ValueError(
            "amp_dtype must be bfloat16 or float32; FP16 needs a GradScaler"
        )
    if config.motion_weight < 0 or (
        config.use_motion_branch and config.clip_frames < 5
    ):
        raise ValueError(
            "motion_weight must be nonnegative and motion mode needs five frames"
        )
    if not config.train_trunk and config.trunk_checkpoint is None:
        raise ValueError(
            "A frozen trunk requires a Wan-latent trunk_checkpoint"
        )
    if (
        config.trunk_checkpoint is None
        and not config.initialize_from_imagenet
        and not config.allow_random_trunk
    ):
        raise ValueError(
            "Without a trunk checkpoint, enable ImageNet initialization "
            "or explicitly allow a random trunk"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True


@dataclass(frozen=True)
class PairRecord:
    video_a: Optional[Path]
    video_b: Optional[Path]
    latent_a: Optional[Path]
    latent_b: Optional[Path]
    score: float
    target_distance: float
    split: Optional[str]
    group_id: Optional[str]
    row_id: str

    def content_keys(self) -> tuple[str, str]:
        return str(self.latent_a or self.video_a), str(
            self.latent_b or self.video_b
        )

    def content_aliases(self) -> tuple[str, ...]:
        paths = (
            self.video_a,
            self.latent_a,
            self.video_b,
            self.latent_b,
        )
        return tuple(
            dict.fromkeys(str(path) for path in paths if path is not None)
        )


def resolve_optional_path(
    value: Any,
    base: Path,
) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def normalize_target(
    score: float,
    config: FeaturizerConfig,
) -> float:
    if not math.isfinite(score):
        raise ValueError(f"Non-finite perceptual score: {score}")
    tolerance = 1e-8
    if (
        score < config.score_min - tolerance
        or score > config.score_max + tolerance
    ):
        raise ValueError(
            f"Score {score} is outside "
            f"[{config.score_min}, {config.score_max}]"
        )
    normalized = (
        min(max(score, config.score_min), config.score_max)
        - config.score_min
    ) / (config.score_max - config.score_min)
    return (
        1.0 - normalized
        if config.score_kind == "similarity"
        else normalized
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    row.setdefault("_row_id", f"line-{line_number}")
                    rows.append(row)
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for index, row in enumerate(rows):
            row.setdefault("_row_id", f"row-{index + 2}")
        return rows
    raise ValueError("Manifest must be .jsonl or .csv")


def read_manifest(config: FeaturizerConfig) -> list[PairRecord]:
    base = (
        config.video_root.expanduser().resolve()
        if config.video_root
        else config.manifest_path.parent.resolve()
    )
    records = []
    for index, row in enumerate(_read_rows(config.manifest_path)):
        score = float(row["score"])
        video_a = resolve_optional_path(row.get("video_a"), base)
        video_b = resolve_optional_path(row.get("video_b"), base)
        latent_a = resolve_optional_path(row.get("latent_a"), base)
        latent_b = resolve_optional_path(row.get("latent_b"), base)
        if (video_a is None and latent_a is None) or (
            video_b is None and latent_b is None
        ):
            raise ValueError(
                f"Row {index} needs a video or cached latent for each side"
            )
        for path in (latent_a, latent_b):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
        split_value = str(row.get("split", "")).strip().lower() or None
        if split_value in {"validation", "valid", "dev"}:
            split_value = "val"
        if split_value not in {None, "train", "val"}:
            raise ValueError(f"Unknown split {split_value!r}")
        records.append(
            PairRecord(
                video_a=video_a,
                video_b=video_b,
                latent_a=latent_a,
                latent_b=latent_b,
                score=score,
                target_distance=normalize_target(score, config),
                split=split_value,
                group_id=str(row.get("group_id", "")).strip() or None,
                row_id=str(row.get("_row_id", index)),
            )
        )
    if not records:
        raise ValueError("Manifest is empty")
    row_ids = [record.row_id for record in records]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("row_id values must be unique")
    pair_scores: dict[tuple[str, str], float] = {}
    for record in records:
        a, b = record.content_keys()
        if a == b and record.target_distance > 1e-6:
            raise ValueError(
                f"Identity pair {record.row_id} has nonzero target distance"
            )
        key = tuple(sorted((a, b)))
        if key in pair_scores and not math.isclose(
            pair_scores[key],
            record.target_distance,
            abs_tol=1e-6,
        ):
            warnings.warn(
                f"Duplicate/reversed pair has conflicting scores: "
                f"{record.row_id}"
            )
        pair_scores[key] = record.target_distance
    if len({round(record.target_distance, 8) for record in records}) < 2:
        warnings.warn(
            "All target distances are constant; calibration is not identifiable"
        )
    return records


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def _deterministic_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8]
    return int.from_bytes(digest, "big") / float(2**64)


def split_records(
    records: list[PairRecord],
    config: FeaturizerConfig,
) -> tuple[list[PairRecord], list[PairRecord]]:
    has_explicit = [record.split is not None for record in records]
    if any(has_explicit) and not all(has_explicit):
        raise ValueError("Either provide split for every row or for none")
    if all(has_explicit):
        train_records = [
            record for record in records if record.split == "train"
        ]
        val_records = [
            record for record in records if record.split == "val"
        ]
    else:
        disjoint_set = _DisjointSet()
        for record in records:
            aliases = record.content_aliases()
            for alias in aliases[1:]:
                disjoint_set.union(aliases[0], alias)
        keys = [
            record.group_id
            or disjoint_set.find(record.content_keys()[0])
            for record in records
        ]
        unique_keys = sorted(set(keys))
        if len(unique_keys) < 2:
            raise ValueError(
                "Automatic content-disjoint splitting needs at least two "
                "groups/components; provide explicit splits"
            )
        val_keys = {
            key
            for key in unique_keys
            if _deterministic_fraction(key, config.split_seed)
            < config.val_fraction
        }
        if not val_keys:
            val_keys.add(
                min(
                    unique_keys,
                    key=lambda key: _deterministic_fraction(
                        key,
                        config.split_seed,
                    ),
                )
            )
        if val_keys == set(unique_keys):
            val_keys.remove(
                max(
                    unique_keys,
                    key=lambda key: _deterministic_fraction(
                        key,
                        config.split_seed,
                    ),
                )
            )
        train_records = [
            record
            for record, key in zip(records, keys)
            if key not in val_keys
        ]
        val_records = [
            record
            for record, key in zip(records, keys)
            if key in val_keys
        ]
    if not train_records or not val_records:
        raise ValueError(
            "Both train and val must be nonempty; got "
            f"{len(train_records)} and {len(val_records)}"
        )
    train_content = {
        key
        for record in train_records
        for key in record.content_aliases()
    }
    val_content = {
        key
        for record in val_records
        for key in record.content_aliases()
    }
    overlap = train_content & val_content
    if overlap:
        raise ValueError(
            f"Content leakage between train and val: {sorted(overlap)[:3]}"
        )
    train_groups = {
        record.group_id
        for record in train_records
        if record.group_id is not None
    }
    val_groups = {
        record.group_id
        for record in val_records
        if record.group_id is not None
    }
    group_overlap = train_groups & val_groups
    if group_overlap:
        raise ValueError(
            f"Group leakage between train and val: "
            f"{sorted(group_overlap)[:3]}"
        )
    return train_records, val_records


def _load_wan_vae_class(config: FeaturizerConfig):
    module_path = (
        config.repo_root
        / "navier"
        / "Wan2.2"
        / "wan"
        / "modules"
        / "vae2_2.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    if importlib.util.find_spec("einops") is None:
        raise RuntimeError("The local Wan VAE requires einops")
    name = "wan_vae2_2_featurizer_local"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Wan2_2_VAE


def _load_wan_vae(
    config: FeaturizerConfig,
    device: torch.device,
):
    if device.type != "cuda":
        warnings.warn(
            "The 2.8 GB Wan VAE is intended to be cached on CUDA"
        )
    checkpoint = config.checkpoint_dir / "Wan2.2_VAE.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Wan VAE checkpoint not found: {checkpoint}"
        )
    vae_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    Wan2_2_VAE = _load_wan_vae_class(config)
    return Wan2_2_VAE(
        vae_pth=str(checkpoint),
        dtype=vae_dtype,
        device=device,
    )


def _decode_video_rgb(
    path: Path,
    output_frames: int,
) -> torch.Tensor:
    if importlib.util.find_spec("av") is None:
        raise RuntimeError("PyAV is required to decode raw videos")
    import av

    with av.open(str(path)) as container:
        total_frames = sum(1 for _ in container.decode(video=0))
    if total_frames <= 0:
        raise ValueError(f"No frames decoded from {path}")
    selected = (
        torch.linspace(0, total_frames - 1, steps=output_frames)
        .round()
        .long()
        .tolist()
    )
    frames, cursor = [], 0
    with av.open(str(path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            while (
                cursor < len(selected)
                and selected[cursor] == frame_index
            ):
                frames.append(
                    torch.from_numpy(
                        frame.to_ndarray(format="rgb24").copy()
                    )
                )
                cursor += 1
            if cursor == len(selected):
                break
    if len(frames) != output_frames:
        raise ValueError(
            f"Decoder yielded {len(frames)} selected frames for {path}; "
            f"expected {output_frames}"
        )
    return torch.stack(frames)


def _sample_and_resize_video(
    frames: torch.Tensor,
    config: FeaturizerConfig,
) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError(
            f"Expected [T,H,W,C] RGB frames, got {tuple(frames.shape)}"
        )
    if frames.shape[0] != config.clip_frames:
        raise ValueError(
            f"Decoder returned {frames.shape[0]} frames, "
            f"expected {config.clip_frames}"
        )
    video = (
        frames[..., :3]
        .permute(0, 3, 1, 2)
        .float()
        .div_(255.0)
    )
    target_h, target_w = config.resize_hw
    source_h, source_w = video.shape[-2:]
    scale = max(target_h / source_h, target_w / source_w)
    resized_h = max(target_h, int(math.ceil(source_h * scale)))
    resized_w = max(target_w, int(math.ceil(source_w * scale)))
    video = F.interpolate(
        video,
        size=(resized_h, resized_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    top = (resized_h - target_h) // 2
    left = (resized_w - target_w) // 2
    video = video[
        :,
        :,
        top : top + target_h,
        left : left + target_w,
    ].clamp_(0, 1)
    return (
        video.mul_(2)
        .sub_(1)
        .permute(1, 0, 2, 3)
        .contiguous()
    )


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_path_for(
    video_path: Path,
    config: FeaturizerConfig,
) -> Path:
    checkpoint = config.checkpoint_dir / "Wan2.2_VAE.pth"
    payload = {
        "source": _file_signature(video_path),
        "vae": _file_signature(checkpoint),
        "clip_frames": config.clip_frames,
        "resize_hw": list(config.resize_hw),
        "normalization": "rgb_uint8_to_minus1_plus1",
        "format_version": 1,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:20]
    return config.cache_dir / f"{video_path.stem}-{digest}.safetensors"


def validate_latent(
    latent: torch.Tensor,
    config: FeaturizerConfig,
    source: str = "latent",
) -> None:
    expected = (
        48,
        1 + (config.clip_frames - 1) // 4,
        config.resize_hw[0] // 16,
        config.resize_hw[1] // 16,
    )
    if tuple(latent.shape) != expected:
        raise ValueError(
            f"{source} has shape {tuple(latent.shape)}, expected {expected}"
        )
    if not torch.isfinite(latent).all():
        raise ValueError(f"{source} contains NaN or Inf")


def _validate_latent_metadata(
    metadata: dict[str, str],
    path: Path,
    config: FeaturizerConfig,
) -> None:
    if not metadata:
        if config.allow_unverified_latents:
            warnings.warn(f"Using latent without provenance metadata: {path}")
            return
        raise ValueError(
            f"{path} has no metadata; enable allow_unverified_latents "
            "only for standardized Wan2.2 VAE output"
        )
    vae_name = metadata.get("vae", "")
    try:
        stride = json.loads(metadata.get("vae_stride", ""))
    except Exception:
        stride = None
    standardized = (
        metadata.get("already_channel_standardized", "").lower() == "true"
    )
    known_local_cache = (
        "Wan2.2 TI2V-5B VAE" in vae_name
        and stride == [4, 16, 16]
    )
    if not (standardized or known_local_cache):
        message = (
            f"{path} is not verified as standardized Wan2.2 VAE output"
        )
        if config.allow_unverified_latents:
            warnings.warn(message)
        else:
            raise ValueError(message)
    if (
        "input_frames" in metadata
        and int(metadata["input_frames"]) != config.clip_frames
    ):
        raise ValueError(f"{path} was cached with a different frame count")
    if (
        "resize_hw" in metadata
        and tuple(json.loads(metadata["resize_hw"]))
        != tuple(config.resize_hw)
    ):
        raise ValueError(f"{path} was cached with a different resize_hw")
    if "input_shape_cthw" in metadata:
        shape = tuple(json.loads(metadata["input_shape_cthw"]))
        if (shape[1], shape[2], shape[3]) != (
            config.clip_frames,
            *config.resize_hw,
        ):
            raise ValueError(
                f"{path} input preprocessing shape is incompatible: {shape}"
            )


def load_latent_tensor(
    path: Path,
    config: FeaturizerConfig,
) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        key = (
            "latents"
            if "latents" in keys
            else "latent"
            if "latent" in keys
            else None
        )
        if key is None:
            raise KeyError(f"{path} must contain latents or latent")
        metadata = handle.metadata() or {}
        latent = handle.get_tensor(key).float()
    _validate_latent_metadata(metadata, path, config)
    validate_latent(latent, config, str(path))
    return latent


def _unique_uncached_videos(
    records: Iterable[PairRecord],
    config: FeaturizerConfig,
) -> list[Path]:
    videos = set()
    for record in records:
        if record.latent_a is None and record.video_a is not None:
            videos.add(record.video_a)
        if record.latent_b is None and record.video_b is not None:
            videos.add(record.video_b)
    missing = [path for path in videos if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source videos, e.g. {missing[:3]}")
    return sorted(
        path
        for path in videos
        if not cache_path_for(path, config).is_file()
    )


@torch.inference_mode()
def build_latent_cache(
    records: list[PairRecord],
    config: FeaturizerConfig,
) -> dict[str, int]:
    validate_config(config)
    device = resolve_device(config)
    pending = _unique_uncached_videos(records, config)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    if not pending:
        present = {
            path
            for record in records
            for path in (record.video_a, record.video_b)
            if path
        }
        return {"created": 0, "already_present": len(present)}
    vae = _load_wan_vae(config, device)
    cache_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[config.cache_dtype]
    created = 0
    try:
        for video_path in tqdm(pending, desc="Caching Wan latents"):
            video = _sample_and_resize_video(
                _decode_video_rgb(video_path, config.clip_frames),
                config,
            ).to(device)
            latent = vae.encode([video])[0].detach().cpu()
            validate_latent(latent, config, str(video_path))
            destination = cache_path_for(video_path, config)
            temporary = destination.with_suffix(
                destination.suffix + ".tmp"
            )
            metadata = {
                "source_video": str(video_path),
                "vae": "Wan2.2 TI2V-5B VAE",
                "vae_stride": "[4,16,16]",
                "input_frames": str(config.clip_frames),
                "resize_hw": json.dumps(config.resize_hw),
                "normalization": "[-1,1]",
                "already_channel_standardized": "true",
                "vae_checkpoint": str(
                    config.checkpoint_dir / "Wan2.2_VAE.pth"
                ),
            }
            save_file(
                {"latents": latent.to(cache_dtype).contiguous()},
                str(temporary),
                metadata=metadata,
            )
            os.replace(temporary, destination)
            created += 1
    finally:
        del vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"created": created, "already_present": 0}


def latent_path_for(
    record: PairRecord,
    side: str,
    config: FeaturizerConfig,
) -> Path:
    if side not in {"a", "b"}:
        raise ValueError("side must be a or b")
    explicit = record.latent_a if side == "a" else record.latent_b
    video = record.video_a if side == "a" else record.video_b
    if explicit is not None:
        return explicit
    if video is None:
        raise ValueError(
            f"Record {record.row_id} has no source for side {side}"
        )
    path = cache_path_for(video, config)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing cached latent {path}; run build_latent_cache first"
        )
    return path


class VideoPairLatentDataset(Dataset):
    def __init__(
        self,
        records: list[PairRecord],
        config: FeaturizerConfig,
    ):
        self.records = records
        self.config = config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "latent_a": load_latent_tensor(
                latent_path_for(record, "a", self.config),
                self.config,
            ),
            "latent_b": load_latent_tensor(
                latent_path_for(record, "b", self.config),
                self.config,
            ),
            "target": torch.tensor(
                record.target_distance,
                dtype=torch.float32,
            ),
            "score": torch.tensor(record.score, dtype=torch.float32),
            "row_id": record.row_id,
        }


class SynchronizedLatentAugment(nn.Module):
    """Apply one spatial transform consistently to both latent videos."""

    def __init__(
        self,
        flip_p: float = 0.5,
        affine_p: float = 0.8,
        cutout_p: float = 0.5,
    ):
        super().__init__()
        self.flip_p = flip_p
        self.affine_p = affine_p
        self.cutout_p = cutout_p

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if a.shape != b.shape or a.ndim != 5:
            raise ValueError(
                "Paired latents must have equal [B,C,T,H,W] shapes"
            )
        batch, channels, frames, height, width = a.shape
        a, b = a.float(), b.float()

        flip = (
            torch.rand(batch, 1, 1, 1, 1, device=a.device)
            < self.flip_p
        )
        a = torch.where(flip, a.flip(-1), a)
        b = torch.where(flip, b.flip(-1), b)

        active = torch.rand(batch, device=a.device) < self.affine_p
        angle = (
            torch.empty(batch, device=a.device)
            .uniform_(-12, 12)
            .mul_(math.pi / 180)
        )
        scale = torch.empty(batch, device=a.device).uniform_(0.9, 1.1)
        tx = torch.empty(batch, device=a.device).uniform_(-0.10, 0.10)
        ty = torch.empty(batch, device=a.device).uniform_(-0.10, 0.10)
        angle = torch.where(active, angle, torch.zeros_like(angle))
        scale = torch.where(active, scale, torch.ones_like(scale))
        tx = torch.where(active, tx, torch.zeros_like(tx))
        ty = torch.where(active, ty, torch.zeros_like(ty))
        cosine = torch.cos(angle) * scale
        sine = torch.sin(angle) * scale
        theta = torch.zeros(
            batch,
            2,
            3,
            device=a.device,
            dtype=torch.float32,
        )
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = -sine
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sine
        theta[:, 1, 1] = cosine
        theta[:, 1, 2] = ty

        joined = torch.cat([a, b], dim=0)
        flat = (
            joined.permute(0, 2, 1, 3, 4)
            .reshape(2 * batch * frames, channels, height, width)
        )
        frame_theta = torch.cat([theta, theta], dim=0).repeat_interleave(
            frames,
            dim=0,
        )
        grid = F.affine_grid(
            frame_theta,
            flat.shape,
            align_corners=False,
        )
        flat = F.grid_sample(
            flat,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )
        joined = (
            flat.reshape(
                2 * batch,
                frames,
                channels,
                height,
                width,
            )
            .permute(0, 2, 1, 3, 4)
        )
        a, b = joined[:batch], joined[batch:]

        mask = torch.ones(
            batch,
            1,
            1,
            height,
            width,
            device=a.device,
            dtype=torch.float32,
        )
        for index in range(batch):
            if torch.rand((), device=a.device) < self.cutout_p:
                cut_h = max(1, height // 2)
                cut_w = max(1, width // 2)
                top = int(
                    torch.randint(
                        0,
                        height - cut_h + 1,
                        (),
                        device=a.device,
                    )
                )
                left = int(
                    torch.randint(
                        0,
                        width - cut_w + 1,
                        (),
                        device=a.device,
                    )
                )
                mask[
                    index,
                    :,
                    :,
                    top : top + cut_h,
                    left : left + cut_w,
                ] = 0
        return a * mask, b * mask


def make_loader(
    records: list[PairRecord],
    config: FeaturizerConfig,
    training: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(
        config.seed + int(training)
    )
    return DataLoader(
        VideoPairLatentDataset(records, config),
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=config.num_workers,
        pin_memory=resolve_device(config).type == "cuda",
        persistent_workers=config.num_workers > 0,
        drop_last=False,
        generator=generator,
    )


class WanLatentVGG16BN(nn.Module):
    """VGG16-BN adapted to framewise 48-channel Wan latents."""

    channels = (64, 128, 256, 512, 512)
    slice_ranges = (
        (0, 7),
        (7, 14),
        (14, 24),
        (24, 34),
        (34, 44),
    )

    def __init__(
        self,
        in_channels: int = 48,
        remove_first_pools: int = 4,
        imagenet_init: bool = True,
        allow_random: bool = False,
    ):
        super().__init__()
        try:
            weights = (
                VGG16_BN_Weights.IMAGENET1K_V1
                if imagenet_init
                else None
            )
            backbone = vgg16_bn(weights=weights)
        except Exception as error:
            if not allow_random:
                raise RuntimeError(
                    "Could not load ImageNet VGG16-BN weights; "
                    "cache them or allow a random trunk"
                ) from error
            warnings.warn("Using a randomly initialized latent VGG trunk")
            backbone = vgg16_bn(weights=None)

        old_conv = backbone.features[0]
        first_conv = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if imagenet_init:
            with torch.no_grad():
                channel_index = (
                    torch.arange(in_channels) % old_conv.in_channels
                )
                expanded = old_conv.weight[:, channel_index] * math.sqrt(
                    old_conv.in_channels / in_channels
                )
                first_conv.weight.copy_(expanded)
                first_conv.bias.copy_(old_conv.bias)
        backbone.features[0] = first_conv
        pool_indices = [
            index
            for index, layer in enumerate(backbone.features)
            if isinstance(layer, nn.MaxPool2d)
        ]
        for index in pool_indices[:remove_first_pools]:
            backbone.features[index] = nn.Identity()

        features = backbone.features
        self.slices = nn.ModuleList(
            [
                nn.Sequential(*features[start:end])
                for start, end in self.slice_ranges
            ]
        )

    def forward(
        self,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if latent.ndim != 5 or latent.shape[1] != 48:
            raise ValueError(
                f"Expected [B,48,T,H,W], got {tuple(latent.shape)}"
            )
        batch, channels, frames, height, width = latent.shape
        hidden = (
            latent.permute(0, 2, 1, 3, 4)
            .reshape(batch * frames, channels, height, width)
        )
        outputs = []
        for block in self.slices:
            hidden = block(hidden)
            _, feature_channels, feature_h, feature_w = hidden.shape
            output = (
                hidden.reshape(
                    batch,
                    frames,
                    feature_channels,
                    feature_h,
                    feature_w,
                )
                .permute(0, 2, 1, 3, 4)
            )
            outputs.append(output)
        return tuple(outputs)


class WanVideoLatentLPIPS(nn.Module):
    """Five-level LPIPS distance over framewise Wan latent features."""

    def __init__(
        self,
        trunk: WanLatentVGG16BN,
        dropout: float = 0.5,
        use_motion_branch: bool = False,
        motion_weight: float = 0.25,
    ):
        super().__init__()
        self.trunk = trunk
        self.use_motion_branch = use_motion_branch
        self.motion_weight = float(motion_weight)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Conv3d(
                        channels,
                        1,
                        kernel_size=1,
                        bias=False,
                    ),
                )
                for channels in trunk.channels
            ]
        )
        self.reset_calibration()

    def reset_calibration(self) -> None:
        with torch.no_grad():
            for head in self.heads:
                head[-1].weight.fill_(1.0 / len(self.heads))

    @staticmethod
    def normalize_feature(
        feature: torch.Tensor,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        return feature * torch.rsqrt(
            feature.square().sum(dim=1, keepdim=True) + eps
        )

    def branch_distance(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = a.shape[0]
        features = self.trunk(torch.cat([a, b], dim=0))
        contributions = []
        for feature, head in zip(features, self.heads):
            fa, fb = feature[:batch], feature[batch:]
            difference = (
                self.normalize_feature(fa) - self.normalize_feature(fb)
            ).square()
            contributions.append(
                head(difference).mean(dim=(1, 2, 3, 4))
            )
        per_layer = torch.stack(contributions, dim=1)
        return per_layer.sum(dim=1), per_layer

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        return_per_layer: bool = False,
    ):
        if a.shape != b.shape:
            raise ValueError(
                f"Pair shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}"
            )
        distance, per_layer = self.branch_distance(a, b)
        if self.use_motion_branch:
            if a.shape[2] < 2:
                raise ValueError(
                    "Motion branch requires at least two latent frames"
                )
            motion_distance, motion_layers = self.branch_distance(
                a[:, :, 1:] - a[:, :, :-1],
                b[:, :, 1:] - b[:, :, :-1],
            )
            distance = distance + self.motion_weight * motion_distance
            per_layer = (
                per_layer + self.motion_weight * motion_layers
            )
        return (
            (distance, per_layer)
            if return_per_layer
            else distance
        )

    @torch.no_grad()
    def clamp_calibration_weights(self) -> None:
        for head in self.heads:
            head[-1].weight.clamp_(min=0)


def _extract_trunk_state(
    payload: dict[str, Any],
) -> dict[str, torch.Tensor]:
    state = payload.get(
        "metric",
        payload.get("state_dict", payload),
    )
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    if any(key.startswith("trunk.") for key in state):
        state = {
            key.removeprefix("trunk."): value
            for key, value in state.items()
            if key.startswith("trunk.")
        }
    if any(
        key.startswith(("features.", "model.features."))
        for key in state
    ):
        converted = {}
        for key, value in state.items():
            feature_key = key.removeprefix("model.")
            if not feature_key.startswith("features."):
                continue
            _, index_text, suffix = feature_key.split(".", 2)
            global_index = int(index_text)
            for slice_index, (start, end) in enumerate(
                WanLatentVGG16BN.slice_ranges
            ):
                if start <= global_index < end:
                    converted[
                        f"slices.{slice_index}."
                        f"{global_index - start}.{suffix}"
                    ] = value
                    break
        return converted
    return state


def safe_torch_load(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_metric(
    config: FeaturizerConfig,
    device: Optional[torch.device] = None,
) -> WanVideoLatentLPIPS:
    validate_config(config)
    device = device or resolve_device(config)
    if config.train_trunk and config.trunk_checkpoint is None:
        warnings.warn(
            "No Wan-latent classification checkpoint was supplied: "
            "joint pair-score trunk tuning is an LPIPS-inspired fallback"
        )
    if config.use_motion_branch and config.train_trunk:
        warnings.warn(
            "The optional motion branch shares BatchNorm statistics with "
            "appearance features; freezing a pretrained trunk is safer"
        )
    trunk = WanLatentVGG16BN(
        in_channels=48,
        remove_first_pools=config.remove_first_pools,
        imagenet_init=(
            config.initialize_from_imagenet
            and config.trunk_checkpoint is None
        ),
        allow_random=config.allow_random_trunk,
    )
    if config.trunk_checkpoint is not None:
        incompatible = trunk.load_state_dict(
            _extract_trunk_state(
                safe_torch_load(config.trunk_checkpoint)
            ),
            strict=False,
        )
        if (
            incompatible.missing_keys
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                f"Incompatible trunk checkpoint: {incompatible}"
            )
    metric = WanVideoLatentLPIPS(
        trunk,
        config.head_dropout,
        config.use_motion_branch,
        config.motion_weight,
    )
    trunk.requires_grad_(config.train_trunk)
    return metric.to(device)


def ensembled_distance(
    metric: WanVideoLatentLPIPS,
    a: torch.Tensor,
    b: torch.Tensor,
    augment: SynchronizedLatentAugment,
    repeats: int = 4,
) -> torch.Tensor:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    return torch.stack(
        [metric(*augment(a, b)) for _ in range(repeats)]
    ).mean(dim=0)


def run_contract_tests() -> None:
    test_config = FeaturizerConfig(
        initialize_from_imagenet=False,
        allow_random_trunk=True,
        train_trunk=True,
        clip_frames=5,
        resize_hw=(128, 128),
        head_dropout=0.0,
        num_workers=0,
        device="cpu",
    )
    model = build_metric(test_config).eval()
    a = torch.randn(1, 48, 2, 8, 8)
    b = torch.randn_like(a)
    with torch.no_grad():
        distance_ab, layers = model(
            a,
            b,
            return_per_layer=True,
        )
        distance_ba = model(b, a)
        identity = model(a, a)
    assert distance_ab.shape == (1,)
    assert layers.shape == (1, 5)
    assert torch.isfinite(distance_ab).all()
    assert (distance_ab >= 0).all()
    assert torch.allclose(
        distance_ab,
        distance_ba,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        identity,
        torch.zeros_like(identity),
        atol=1e-7,
    )
    augment = SynchronizedLatentAugment()
    augmented_a, augmented_b = augment(a, a.clone())
    assert augmented_a.shape == a.shape
    assert torch.equal(augmented_a, augmented_b)
    assert normalize_target(
        test_config.score_max,
        test_config,
    ) == 0.0
    assert normalize_target(
        test_config.score_min,
        test_config,
    ) == 1.0
    del model


def _autocast_context(
    device: torch.device,
    dtype: torch.dtype,
):
    enabled = device.type == "cuda" and dtype != torch.float32
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=enabled,
    )


def _regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    return F.huber_loss(
        prediction.float(),
        target.float(),
        delta=delta,
    )


def _correlation(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    x = x.double().flatten()
    y = y.double().flatten()
    if (
        x.numel() < 2
        or x.std() == 0
        or y.std() == 0
    ):
        return float("nan")
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


def _rank_values(values: torch.Tensor) -> torch.Tensor:
    values = values.double().flatten()
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _ordering_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_pairs: int = 100_000,
    seed: int = 1234,
) -> float:
    count = prediction.numel()
    if count < 2:
        return float("nan")
    total_pairs = count * (count - 1) // 2
    if total_pairs <= max_pairs:
        left, right = torch.triu_indices(
            count,
            count,
            offset=1,
        )
    else:
        generator = torch.Generator().manual_seed(seed)
        left = torch.randint(
            count,
            (max_pairs,),
            generator=generator,
        )
        right = torch.randint(
            count - 1,
            (max_pairs,),
            generator=generator,
        )
        right = right + (right >= left)
    pred_sign = torch.sign(
        prediction[left] - prediction[right]
    )
    target_sign = torch.sign(target[left] - target[right])
    tied = (target_sign == 0) | (pred_sign == 0)
    credit = torch.where(
        tied,
        torch.full_like(pred_sign, 0.5),
        (pred_sign == target_sign).float(),
    )
    return float(credit.mean())


def _summarize_predictions(
    prediction: torch.Tensor,
    target: torch.Tensor,
    seed: int,
) -> dict[str, float]:
    prediction = prediction.float()
    target = target.float()
    return {
        "mae": float((prediction - target).abs().mean()),
        "rmse": float(
            (prediction - target).square().mean().sqrt()
        ),
        "pearson": _correlation(prediction, target),
        "spearman": _correlation(
            _rank_values(prediction),
            _rank_values(target),
        ),
        "ordering_accuracy": _ordering_accuracy(
            prediction,
            target,
            seed=seed,
        ),
    }


@torch.no_grad()
def evaluate(
    metric: WanVideoLatentLPIPS,
    loader: DataLoader,
    config: FeaturizerConfig,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    device = resolve_device(config)
    dtype = amp_dtype(config)
    metric.eval()
    predictions, targets = [], []
    for batch in tqdm(loader, desc="validation", leave=False):
        a = batch["latent_a"].to(device, non_blocking=True)
        b = batch["latent_b"].to(device, non_blocking=True)
        target = batch["target"].to(
            device,
            non_blocking=True,
        )
        with _autocast_context(device, dtype):
            prediction = metric(a, b)
        predictions.append(prediction.float().cpu())
        targets.append(target.float().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    metrics = _summarize_predictions(
        prediction,
        target,
        seed=config.seed,
    )
    metrics["loss"] = float(
        _regression_loss(
            prediction,
            target,
            config.huber_delta,
        )
    )
    return metrics, prediction, target


def _optimizer_and_scheduler(
    metric: WanVideoLatentLPIPS,
    config: FeaturizerConfig,
):
    groups = [
        {
            "params": metric.heads.parameters(),
            "lr": config.head_lr,
            "name": "heads",
        }
    ]
    if config.train_trunk:
        groups.append(
            {
                "params": metric.trunk.parameters(),
                "lr": config.trunk_lr,
                "name": "trunk",
            }
        )
    optimizer = torch.optim.Adam(
        groups,
        betas=(config.beta1, 0.999),
        weight_decay=config.weight_decay,
    )

    def schedule(epoch: int) -> float:
        if epoch < config.fixed_lr_epochs:
            return 1.0
        decay_epochs = max(
            1,
            config.epochs - config.fixed_lr_epochs,
        )
        return max(
            0.0,
            (config.epochs - epoch) / decay_epochs,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        schedule,
    )
    return optimizer, scheduler


def _jsonable_config(
    config: FeaturizerConfig,
) -> dict[str, Any]:
    result = {}
    for key, value in asdict(config).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


_RESUME_CONFIG_KEYS = (
    "clip_frames",
    "resize_hw",
    "score_kind",
    "score_min",
    "score_max",
    "remove_first_pools",
    "train_trunk",
    "head_dropout",
    "use_motion_branch",
    "motion_weight",
    "calibration_augment",
    "batch_size",
    "amp_dtype",
    "huber_delta",
)


def validate_checkpoint_config(
    payload: dict[str, Any],
    config: FeaturizerConfig,
) -> None:
    saved = payload.get("config")
    if not saved:
        warnings.warn(
            "Resume checkpoint has no saved config to validate"
        )
        return
    current = _jsonable_config(config)
    mismatches = {
        key: (saved.get(key), current.get(key))
        for key in _RESUME_CONFIG_KEYS
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        raise ValueError(
            f"Resume configuration is incompatible: {mismatches}"
        )


def _capture_training_rng(
    loader: DataLoader,
) -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
        "loader": (
            loader.generator.get_state()
            if loader.generator is not None
            else None
        ),
    }


def _restore_training_rng(
    state: Optional[dict[str, Any]],
    loader: DataLoader,
) -> None:
    if not state:
        warnings.warn(
            "Resume checkpoint has no RNG state; continuation is not exact"
        )
        return
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(
            [item.cpu() for item in state["torch_cuda"]]
        )
    if (
        loader.generator is not None
        and state.get("loader") is not None
    ):
        loader.generator.set_state(state["loader"].cpu())


def _save_checkpoint(
    path: Path,
    metric: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_loss: float,
    history: list[dict[str, Any]],
    config: FeaturizerConfig,
    rng_state: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric": metric.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "history": history,
        "config": _jsonable_config(config),
        "rng_state": rng_state,
        "score_semantics": {
            "kind": config.score_kind,
            "min": config.score_min,
            "max": config.score_max,
        },
        "method": (
            "Wan2.2 framewise video LatentLPIPS; no latent L1 term"
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(
    metric: WanVideoLatentLPIPS,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: FeaturizerConfig,
) -> list[dict[str, Any]]:
    validate_config(config)
    device = resolve_device(config)
    dtype = amp_dtype(config)
    optimizer, scheduler = _optimizer_and_scheduler(
        metric,
        config,
    )
    augment = (
        SynchronizedLatentAugment().to(device)
        if config.calibration_augment
        else None
    )
    start_epoch = 0
    best_loss = float("inf")
    history = []
    if config.resume_from is not None:
        payload = safe_torch_load(config.resume_from, "cpu")
        validate_checkpoint_config(payload, config)
        metric.load_state_dict(payload["metric"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best_loss = float(payload["best_loss"])
        history = list(payload.get("history", []))
        _restore_training_rng(
            payload.get("rng_state"),
            train_loader,
        )
        best_path = config.output_dir / "best.pt"
        if not best_path.is_file():
            baseline, _, _ = evaluate(
                metric,
                val_loader,
                config,
            )
            best_loss = baseline["loss"]
            _save_checkpoint(
                best_path,
                metric,
                optimizer,
                scheduler,
                start_epoch - 1,
                best_loss,
                history,
                config,
                _capture_training_rng(train_loader),
            )

    for epoch in range(start_epoch, config.epochs):
        metric.train()
        if not config.train_trunk:
            metric.trunk.eval()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_examples = 0
        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"epoch {epoch + 1}/{config.epochs}",
        )
        for step, batch in progress:
            a = batch["latent_a"].to(
                device,
                non_blocking=True,
            )
            b = batch["latent_b"].to(
                device,
                non_blocking=True,
            )
            target = batch["target"].to(
                device,
                non_blocking=True,
            )
            if augment is not None:
                a, b = augment(a, b)
            with _autocast_context(device, dtype):
                prediction = metric(a, b)
                loss = _regression_loss(
                    prediction,
                    target,
                    config.huber_delta,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}, step {step}"
                )
            group_start = (
                step // config.grad_accum_steps
            ) * config.grad_accum_steps
            group_size = min(
                config.grad_accum_steps,
                len(train_loader) - group_start,
            )
            (loss / group_size).backward()
            should_step = (
                (step + 1) % config.grad_accum_steps == 0
                or step + 1 == len(train_loader)
            )
            if should_step:
                trainable = [
                    parameter
                    for parameter in metric.parameters()
                    if parameter.requires_grad
                ]
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable,
                    config.grad_clip_norm,
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradients at epoch "
                        f"{epoch + 1}, step {step}"
                    )
                optimizer.step()
                metric.clamp_calibration_weights()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * target.numel()
            total_examples += target.numel()
            progress.set_postfix(
                loss=total_loss / total_examples
            )

        validation, _, _ = evaluate(
            metric,
            val_loader,
            config,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / total_examples,
            **{
                f"val_{key}": value
                for key, value in validation.items()
            },
        }
        history.append(row)
        improved = validation["loss"] < best_loss
        best_loss = min(best_loss, validation["loss"])
        scheduler.step()
        rng_state = _capture_training_rng(train_loader)
        _save_checkpoint(
            config.output_dir / "latest.pt",
            metric,
            optimizer,
            scheduler,
            epoch,
            best_loss,
            history,
            config,
            rng_state,
        )
        if improved:
            _save_checkpoint(
                config.output_dir / "best.pt",
                metric,
                optimizer,
                scheduler,
                epoch,
                best_loss,
                history,
                config,
                rng_state,
            )
        print(json.dumps(row, indent=2, allow_nan=True))
    return history


def run_training_pipeline(
    config: FeaturizerConfig,
) -> dict[str, Any]:
    """Run cache, loaders, training, and best-checkpoint evaluation."""

    validate_config(config)
    seed_everything(config.seed)
    records = read_manifest(config)
    train_records, val_records = split_records(records, config)
    cache_stats = build_latent_cache(records, config)
    train_loader = make_loader(
        train_records,
        config,
        training=True,
    )
    val_loader = make_loader(
        val_records,
        config,
        training=False,
    )
    metric = build_metric(config)
    history = train(
        metric,
        train_loader,
        val_loader,
        config,
    )
    device = resolve_device(config)
    best_payload = safe_torch_load(
        config.output_dir / "best.pt",
        device,
    )
    validate_checkpoint_config(best_payload, config)
    metric.load_state_dict(best_payload["metric"])
    final_metrics, val_prediction, val_target = evaluate(
        metric,
        val_loader,
        config,
    )
    return {
        "records": records,
        "train_records": train_records,
        "val_records": val_records,
        "cache_stats": cache_stats,
        "metric": metric,
        "history": history,
        "metrics": final_metrics,
        "val_prediction": val_prediction,
        "val_target": val_target,
    }


def plot_training_diagnostics(
    state: dict[str, Any],
) -> None:
    if not state:
        print("Run training first.")
        return
    import matplotlib.pyplot as plt

    history = state["history"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        [row["epoch"] for row in history],
        [row["train_loss"] for row in history],
        label="train",
    )
    axes[0].plot(
        [row["epoch"] for row in history],
        [row["val_loss"] for row in history],
        label="val",
    )
    axes[0].set(
        xlabel="epoch",
        ylabel="Huber loss",
        title="Calibration loss",
    )
    axes[0].legend()

    target = state["val_target"].numpy()
    prediction = state["val_prediction"].numpy()
    axes[1].scatter(target, prediction, alpha=0.6)
    low = min(target.min(), prediction.min())
    high = max(target.max(), prediction.max())
    axes[1].plot(
        [low, high],
        [low, high],
        "--",
        color="black",
    )
    axes[1].set(
        xlabel="target dissimilarity",
        ylabel="predicted distance",
        title="Held-out pairs",
    )
    figure.tight_layout()
    plt.show()

    metric = state["metric"]
    layer_means = [
        float(head[-1].weight.mean().detach().cpu())
        for head in metric.heads
    ]
    print(
        "mean nonnegative calibration weight per layer:",
        layer_means,
    )
