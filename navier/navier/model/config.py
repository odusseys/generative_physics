# This file is executed by navier/autoreg_windowed_lora.py.
# Keep shared globals in that compatibility module namespace.

from __future__ import annotations

import base64
import gc
import importlib.util
import io
import json
import math
import random
import re
import sys
import time
import types
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import av
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import HTML, display
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm.auto import trange

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask as create_flex_block_mask,
    )
    from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention
except Exception:
    create_flex_block_mask = None
    torch_flex_attention = None


CODE_DIR = Path(__file__).resolve().parent
LATENT_ROOT = Path("/home/azureuser/datasets/navier_fast")
WAN_REPO_ROOT = Path("/home/azureuser/physics/navier/Wan2.2")
WAN_CHECKPOINT_DIR = Path("/home/azureuser/Wan2.2-TI2V-5B")
OUTPUT_DIR = Path("/home/azureuser/physics/navier/autoreg_lora_outputs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARAM_DTYPE = torch.bfloat16
TRAIN_DTYPE = torch.bfloat16

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 1e-5
LR_WARMUP_STEPS = 100
MAX_STEPS = 40000
CLIP_GRAD_NORM = 1.0
LOSS_EMA_BETA = 0.95
TRANSFORMER_DROPPED_LAYERS = ()
TRAIN_RANDOM_LAYER_DROP = False
GRADIENT_CHECKPOINT_DISABLE_BLOCKS = "all"
TRAIN_INIT_LOGS = True
REGIONAL_COMPILE = False
REGIONAL_COMPILE_BACKEND = "inductor"
REGIONAL_COMPILE_MODE = "reduce-overhead"
REGIONAL_COMPILE_FULLGRAPH = False
REGIONAL_COMPILE_DYNAMIC = False
REGIONAL_COMPILE_CAPTURE_SCALAR_OUTPUTS = True
USE_FLEX_ATTENTION = True
FLEX_ATTENTION_BLOCK_SIZE = 128
FLEX_ATTENTION_DYNAMIC = True
FLEX_ATTENTION_RECOMPILE_LIMIT = 64
FIRST_FRAME_CONDITIONING_DIM = 128
WAN_NUM_TRAIN_TIMESTEPS = 1000
WAN_SAMPLE_SHIFT = 5.0
EVAL_INFERENCE_STEPS = 20

WINDOW_LEFT_FRAMES = 4
WINDOW_RIGHT_FRAMES = 0
MIN_LATENT_FRAME_COUNT = 36
LATENT_FRAME_COUNT = 18
PROGRESSIVE_SEQUENCE_LENGTH_START = None
PROGRESSIVE_SEQUENCE_LENGTH_STEP_INTERVAL = 500
HOLDOUT_COUNT = 100
EVAL_EVERY = 200
EVAL_FPS = 12
EVAL_ERROR_VMAX = 0.45
VIDEO_OUTPUT_SIZE = 256
SAVE_EVERY = None

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "ffn.0",
    "ffn.2",
]
PATCH_EMBEDDING_LORA_R = 16
PATCH_EMBEDDING_LORA_ALPHA = 16
TIMESTEP_ADALN_LORA_R = 16
TIMESTEP_ADALN_LORA_ALPHA = 16

SSM_DECAY_INIT = 0.999
SSM_INPUT_INIT = 1e-3
SSM_OUTPUT_INIT = 1.0
SSM_SKIP_INIT = 0.0
SSM_RHO_INIT = 1e-3
SSM_FREEZE_RHO_ZERO = False
SSM_QUERY_SCALE = None
SSM_LAYER_COUNT = 2
SSM_RECURRENT_DIM = 1024
SSM_STATE_MIXER_EXPANSION = 4
SSM_USE_CONVOLUTIONS = False

REQUIRE_RHO = True
RANDOM_SEED = 1234

CONFIG_KEYS = (
    "LATENT_ROOT",
    "WAN_REPO_ROOT",
    "WAN_CHECKPOINT_DIR",
    "OUTPUT_DIR",
    "DEVICE",
    "PARAM_DTYPE",
    "TRAIN_DTYPE",
    "BATCH_SIZE",
    "GRAD_ACCUM_STEPS",
    "LEARNING_RATE",
    "LR_WARMUP_STEPS",
    "MAX_STEPS",
    "CLIP_GRAD_NORM",
    "LOSS_EMA_BETA",
    "TRANSFORMER_DROPPED_LAYERS",
    "TRAIN_RANDOM_LAYER_DROP",
    "GRADIENT_CHECKPOINT_DISABLE_BLOCKS",
    "TRAIN_INIT_LOGS",
    "REGIONAL_COMPILE",
    "REGIONAL_COMPILE_BACKEND",
    "REGIONAL_COMPILE_MODE",
    "REGIONAL_COMPILE_FULLGRAPH",
    "REGIONAL_COMPILE_DYNAMIC",
    "REGIONAL_COMPILE_CAPTURE_SCALAR_OUTPUTS",
    "USE_FLEX_ATTENTION",
    "FLEX_ATTENTION_BLOCK_SIZE",
    "FLEX_ATTENTION_DYNAMIC",
    "FLEX_ATTENTION_RECOMPILE_LIMIT",
    "FIRST_FRAME_CONDITIONING_DIM",
    "WAN_NUM_TRAIN_TIMESTEPS",
    "WAN_SAMPLE_SHIFT",
    "EVAL_INFERENCE_STEPS",
    "WINDOW_LEFT_FRAMES",
    "WINDOW_RIGHT_FRAMES",
    "MIN_LATENT_FRAME_COUNT",
    "LATENT_FRAME_COUNT",
    "PROGRESSIVE_SEQUENCE_LENGTH_START",
    "PROGRESSIVE_SEQUENCE_LENGTH_STEP_INTERVAL",
    "HOLDOUT_COUNT",
    "EVAL_EVERY",
    "EVAL_FPS",
    "EVAL_ERROR_VMAX",
    "VIDEO_OUTPUT_SIZE",
    "SAVE_EVERY",
    "LORA_R",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_TARGET_MODULES",
    "PATCH_EMBEDDING_LORA_R",
    "PATCH_EMBEDDING_LORA_ALPHA",
    "TIMESTEP_ADALN_LORA_R",
    "TIMESTEP_ADALN_LORA_ALPHA",
    "SSM_DECAY_INIT",
    "SSM_INPUT_INIT",
    "SSM_OUTPUT_INIT",
    "SSM_SKIP_INIT",
    "SSM_RHO_INIT",
    "SSM_FREEZE_RHO_ZERO",
    "SSM_QUERY_SCALE",
    "SSM_LAYER_COUNT",
    "SSM_RECURRENT_DIM",
    "SSM_STATE_MIXER_EXPANSION",
    "SSM_USE_CONVOLUTIONS",
    "REQUIRE_RHO",
    "RANDOM_SEED",
)

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cuda.matmul.allow_tf32 = True
_REGIONAL_COMPILE_ACTIVE = False
NO_LAYER_DROP_BUCKET = -1


def get_config():
    return {key: globals()[key] for key in CONFIG_KEYS}


def configure(**overrides):
    unknown = sorted(set(overrides) - set(CONFIG_KEYS))
    if unknown:
        raise KeyError(f"Unknown config keys: {unknown}")
    for key, value in overrides.items():
        globals()[key] = value
    _validate_causal_window()
    _validate_progressive_sequence_length()
    _validate_flow_matching_config()
    if "RANDOM_SEED" in overrides:
        random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
    return get_config()


def _validate_causal_window() -> None:
    if WINDOW_RIGHT_FRAMES is None:
        raise ValueError("WINDOW_RIGHT_FRAMES must be 0 for causal windowing")
    if int(WINDOW_RIGHT_FRAMES) != 0:
        raise ValueError(
            "WINDOW_RIGHT_FRAMES must be 0; causal windowed attention is shared by all modes"
        )


def causal_window() -> tuple[int, int]:
    _validate_causal_window()
    return int(WINDOW_LEFT_FRAMES), 0


def _validate_progressive_sequence_length() -> None:
    minimum_frames = 2
    if LATENT_FRAME_COUNT is not None and int(LATENT_FRAME_COUNT) < minimum_frames:
        raise ValueError(
            f"LATENT_FRAME_COUNT must be at least {minimum_frames} or None"
        )
    if PROGRESSIVE_SEQUENCE_LENGTH_START is None:
        return
    start = int(PROGRESSIVE_SEQUENCE_LENGTH_START)
    if start < minimum_frames:
        raise ValueError(
            f"PROGRESSIVE_SEQUENCE_LENGTH_START must be at least {minimum_frames}"
        )
    if LATENT_FRAME_COUNT is None:
        raise ValueError(
            "PROGRESSIVE_SEQUENCE_LENGTH_START requires LATENT_FRAME_COUNT as the max length"
        )
    interval = int(PROGRESSIVE_SEQUENCE_LENGTH_STEP_INTERVAL)
    if interval < 1:
        raise ValueError("PROGRESSIVE_SEQUENCE_LENGTH_STEP_INTERVAL must be positive")


def _validate_flow_matching_config() -> None:
    if int(WAN_NUM_TRAIN_TIMESTEPS) < 1:
        raise ValueError("WAN_NUM_TRAIN_TIMESTEPS must be positive")
    if float(WAN_SAMPLE_SHIFT) <= 0.0:
        raise ValueError("WAN_SAMPLE_SHIFT must be positive")
    if int(EVAL_INFERENCE_STEPS) < 1:
        raise ValueError("EVAL_INFERENCE_STEPS must be positive")
    if int(TIMESTEP_ADALN_LORA_R) < 0:
        raise ValueError("TIMESTEP_ADALN_LORA_R must be non-negative")


def active_latent_frame_count(step: Optional[int] = None) -> Optional[int]:
    _validate_progressive_sequence_length()
    if LATENT_FRAME_COUNT is None:
        return None
    max_frame_count = int(LATENT_FRAME_COUNT)
    if PROGRESSIVE_SEQUENCE_LENGTH_START is None:
        return max_frame_count
    start = int(PROGRESSIVE_SEQUENCE_LENGTH_START)
    interval = int(PROGRESSIVE_SEQUENCE_LENGTH_STEP_INTERVAL)
    step = 1 if step is None else max(1, int(step))
    increments = (step - 1) // interval
    return min(max_frame_count, start + increments)


def _init_log(message: str) -> None:
    if TRAIN_INIT_LOGS:
        print(f"[train init] {message}", flush=True)


@contextmanager
def init_stage(label: str):
    if not TRAIN_INIT_LOGS:
        yield
        return
    start_time = time.perf_counter()
    print(f"[train init] {label} ...", flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"[train init] {label} done in {elapsed:.1f}s", flush=True)


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_wan_package(wan_repo_root: Path = WAN_REPO_ROOT):
    wan_root = Path(wan_repo_root) / "wan"
    modules_root = wan_root / "modules"
    utils_root = wan_root / "utils"

    wan_pkg = sys.modules.get("wan") or types.ModuleType("wan")
    wan_pkg.__path__ = [str(wan_root)]
    modules_pkg = sys.modules.get("wan.modules") or types.ModuleType("wan.modules")
    modules_pkg.__path__ = [str(modules_root)]
    utils_pkg = sys.modules.get("wan.utils") or types.ModuleType("wan.utils")
    utils_pkg.__path__ = [str(utils_root)]

    sys.modules["wan"] = wan_pkg
    sys.modules["wan.modules"] = modules_pkg
    sys.modules["wan.utils"] = utils_pkg
    return wan_root, modules_root, utils_root
