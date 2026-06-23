#!/usr/bin/env python3
"""Recompute Wan latents from existing Navier videos with resized VAE input.

This does not regenerate or rewrite any videos. It reads each existing
``video.mp4``, resizes decoded frames in memory to a patch-compatible VAE input
size, and rewrites only ``latents.safetensors`` beside the video.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Optional

import torch

from generate_navier_dataset_with_latents_new import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_WAN_CHECKPOINT_DIR,
    DEFAULT_WAN_REPO_ROOT,
    WAN_TI2V_5B_VAE_STRIDE,
    encode_video_to_wan_latents,
    load_wan_ti2v_5b_vae,
)

DEFAULT_RESIZED_HEIGHT = 256
DEFAULT_RESIZED_WIDTH = 480
DEFAULT_VIDEO_NAME = "video.mp4"
DEFAULT_LATENTS_NAME = "latents.safetensors"
WAN_PATCH_SIZE = (1, 2, 2)


def _dtype_from_string(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    raise argparse.ArgumentTypeError(
        "dtype must be one of float32, bfloat16, or float16"
    )


def _validate_target_size(height: int, width: int) -> None:
    if height < 16 or width < 16:
        raise ValueError("--height and --width must be at least 16")
    stride_t, stride_h, stride_w = WAN_TI2V_5B_VAE_STRIDE
    del stride_t
    if height % stride_h != 0:
        raise ValueError(f"--height must be divisible by Wan VAE stride {stride_h}")
    if width % stride_w != 0:
        raise ValueError(f"--width must be divisible by Wan VAE stride {stride_w}")

    latent_h = height // stride_h
    latent_w = width // stride_w
    _, patch_h, patch_w = WAN_PATCH_SIZE
    if latent_h % patch_h != 0 or latent_w % patch_w != 0:
        raise ValueError(
            "target VAE size would produce latents incompatible with "
            f"Wan patch size {WAN_PATCH_SIZE}: latent spatial size is "
            f"({latent_h}, {latent_w})"
        )


def _discover_videos(
    dataset_root: Path, video_name: str, limit: Optional[int] = None
) -> list[Path]:
    videos = sorted(dataset_root.glob(f"*/{video_name}"))
    if limit is not None:
        videos = videos[: int(limit)]
    return videos


def _replace_latents_atomically(record: dict, temp_path: Path, output_path: Path) -> dict:
    os.replace(temp_path, output_path)
    record = dict(record)
    record["latents"] = str(output_path)
    return record


def recompute_resized_latents(
    dataset_root: Path,
    height: int,
    width: int,
    *,
    video_name: str = DEFAULT_VIDEO_NAME,
    latents_name: str = DEFAULT_LATENTS_NAME,
    checkpoint_dir: Path = DEFAULT_WAN_CHECKPOINT_DIR,
    wan_repo_root: Path = DEFAULT_WAN_REPO_ROOT,
    device: Optional[torch.device] = None,
    vae_dtype: torch.dtype = torch.float32,
    latent_save_dtype: torch.dtype = torch.float16,
    overwrite: bool = True,
    limit: Optional[int] = None,
    dry_run: bool = False,
    continue_on_error: bool = True,
) -> list[dict]:
    """Resize existing videos in memory and recompute only their latent files."""
    dataset_root = Path(dataset_root)
    _validate_target_size(int(height), int(width))
    videos = _discover_videos(dataset_root, video_name=video_name, limit=limit)
    if not videos:
        raise FileNotFoundError(f"No {video_name!r} files found under {dataset_root}")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )

    vae = None
    if not dry_run:
        vae = load_wan_ti2v_5b_vae(
            checkpoint_dir=checkpoint_dir,
            wan_repo_root=wan_repo_root,
            device=device,
            dtype=vae_dtype,
        )

    records: list[dict] = []
    from tqdm import tqdm

    with tqdm(total=len(videos), desc="Recomputing resized latents") as progress:
        for video_path in videos:
            output_path = video_path.with_name(latents_name)
            uid = video_path.parent.name
            if output_path.exists() and not overwrite:
                records.append(
                    {
                        "status": "skipped",
                        "reason": "exists",
                        "uid": uid,
                        "video": str(video_path),
                        "latents": str(output_path),
                    }
                )
                progress.update(1)
                continue

            if dry_run:
                records.append(
                    {
                        "status": "dry_run",
                        "uid": uid,
                        "video": str(video_path),
                        "latents": str(output_path),
                        "target_size_hw": [int(height), int(width)],
                    }
                )
                progress.update(1)
                continue

            temp_path = output_path.with_name(f".{output_path.name}.tmp")
            try:
                record = encode_video_to_wan_latents(
                    video_path,
                    vae=vae,
                    checkpoint_dir=checkpoint_dir,
                    wan_repo_root=wan_repo_root,
                    device=device,
                    vae_dtype=vae_dtype,
                    save_dtype=latent_save_dtype,
                    output_path=temp_path,
                    overwrite=True,
                    expected_size=(int(height), int(width)),
                    resize=True,
                )
                record = _replace_latents_atomically(record, temp_path, output_path)
                record["uid"] = uid
                record["target_size_hw"] = [int(height), int(width)]
                records.append(record)
            except Exception as exc:
                if temp_path.exists():
                    temp_path.unlink()
                if not continue_on_error:
                    raise
                records.append(
                    {
                        "status": "error",
                        "uid": uid,
                        "video": str(video_path),
                        "latents": str(output_path),
                        "error": repr(exc),
                    }
                )
            finally:
                progress.update(1)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute only Wan latents from existing Navier videos after "
            "in-memory resize to a patch-compatible VAE input size."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--height", type=int, default=DEFAULT_RESIZED_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_RESIZED_WIDTH)
    parser.add_argument("--video-name", type=str, default=DEFAULT_VIDEO_NAME)
    parser.add_argument("--latents-name", type=str, default=DEFAULT_LATENTS_NAME)
    parser.add_argument("--wan-checkpoint-dir", type=Path, default=DEFAULT_WAN_CHECKPOINT_DIR)
    parser.add_argument("--wan-repo-root", type=Path, default=DEFAULT_WAN_REPO_ROOT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vae-dtype", type=_dtype_from_string, default=torch.float32)
    parser.add_argument(
        "--latent-save-dtype", type=_dtype_from_string, default=torch.float16
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Skip samples that already have a latents file.",
    )
    parser.set_defaults(overwrite=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-on-error",
        action="store_false",
        dest="continue_on_error",
        help="Stop on the first failed latent encode.",
    )
    parser.set_defaults(continue_on_error=True)
    parser.add_argument("--records-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    records = recompute_resized_latents(
        dataset_root=args.dataset_root,
        height=int(args.height),
        width=int(args.width),
        video_name=args.video_name,
        latents_name=args.latents_name,
        checkpoint_dir=args.wan_checkpoint_dir,
        wan_repo_root=args.wan_repo_root,
        device=None if args.device is None else torch.device(args.device),
        vae_dtype=args.vae_dtype,
        latent_save_dtype=args.latent_save_dtype,
        overwrite=bool(args.overwrite),
        limit=args.limit,
        dry_run=bool(args.dry_run),
        continue_on_error=bool(args.continue_on_error),
    )

    summary = {
        "dataset_root": str(args.dataset_root),
        "target_size_hw": [int(args.height), int(args.width)],
        "videos_seen": len(records),
        "latents_ok": sum(1 for item in records if item.get("status") == "ok"),
        "latents_skipped": sum(
            1 for item in records if item.get("status") == "skipped"
        ),
        "latents_dry_run": sum(
            1 for item in records if item.get("status") == "dry_run"
        ),
        "latents_error": sum(1 for item in records if item.get("status") == "error"),
    }
    print(json.dumps(summary, indent=2))

    if args.records_json is not None:
        args.records_json.parent.mkdir(parents=True, exist_ok=True)
        args.records_json.write_text(
            json.dumps({"summary": summary, "records": records}, indent=2)
        )

    return 0 if summary["latents_error"] == 0 else 1


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        raise SystemExit(main())
