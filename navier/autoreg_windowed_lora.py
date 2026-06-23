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
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Optional

import av
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
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
TEXT_EMBED_CACHE_PATH = CODE_DIR / "wan_t5_prompt_embeddings.safetensors"
POSITIVE_PROMPT = "aerodynamic flow around an object"
NEGATIVE_PROMPT = "bad quality"

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
VELOCITY_L2_LOSS_WEIGHT = 1.0
GRADIENT_CHECKPOINT_DISABLE_BLOCKS = "all"
TRAINING_MODE = "flow_matching"
TRAIN_INIT_LOGS = True
REGIONAL_COMPILE = False
REGIONAL_COMPILE_BACKEND = "inductor"
REGIONAL_COMPILE_MODE = "reduce-overhead"
REGIONAL_COMPILE_FULLGRAPH = False
REGIONAL_COMPILE_DYNAMIC = False
REGIONAL_COMPILE_CAPTURE_SCALAR_OUTPUTS = True
USE_FLEX_ATTENTION = True
FLEX_ATTENTION_BLOCK_SIZE = 64
FIRST_FRAME_CONDITIONING_DIM = 64
ERROR_RECYCLING_WARMUP_STEPS = 200
ERROR_RECYCLING_PROB = 0.5
ERROR_RECYCLING_SCALE = 1.0
ERROR_RECYCLING_TIMESTEP_BINS = 50
ERROR_RECYCLING_BANK_SIZE = 8

WINDOW_LEFT_FRAMES = 4
WINDOW_RIGHT_FRAMES = 0
MIN_LATENT_FRAME_COUNT = 36
LATENT_FRAME_COUNT = 18
HOLDOUT_COUNT = 100
EVAL_EVERY = 200
EVAL_INFERENCE_STEPS = 25
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

DELTA_ATTENTION_BETA_INIT = 1e-3
DELTA_ATTENTION_LAMBDA_INIT = 0.999
DELTA_ATTENTION_RHO_INIT = 0.0
DELTA_ATTENTION_FREEZE_RHO_ZERO = False
DELTA_ATTENTION_KEY_SCALE = None

SSM_DECAY_INIT = 0.999
SSM_INPUT_INIT = 1e-3
SSM_OUTPUT_INIT = 1.0
SSM_SKIP_INIT = 0.0
SSM_RHO_INIT = 0.0
SSM_FREEZE_RHO_ZERO = False
SSM_QUERY_SCALE = None
SSM_RECURRENT_DIM = 1024

WAN_NUM_TRAIN_TIMESTEPS = 1000
WAN_SAMPLE_SHIFT = 5.0
WAN_SAMPLE_GUIDE_SCALE = 1.0
WAN_SAMPLE_SOLVER = "unipc"
REQUIRE_RHO = True
RANDOM_SEED = 1234

CONFIG_KEYS = (
    "LATENT_ROOT",
    "WAN_REPO_ROOT",
    "WAN_CHECKPOINT_DIR",
    "OUTPUT_DIR",
    "TEXT_EMBED_CACHE_PATH",
    "POSITIVE_PROMPT",
    "NEGATIVE_PROMPT",
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
    "VELOCITY_L2_LOSS_WEIGHT",
    "GRADIENT_CHECKPOINT_DISABLE_BLOCKS",
    "TRAINING_MODE",
    "TRAIN_INIT_LOGS",
    "REGIONAL_COMPILE",
    "REGIONAL_COMPILE_BACKEND",
    "REGIONAL_COMPILE_MODE",
    "REGIONAL_COMPILE_FULLGRAPH",
    "REGIONAL_COMPILE_DYNAMIC",
    "REGIONAL_COMPILE_CAPTURE_SCALAR_OUTPUTS",
    "USE_FLEX_ATTENTION",
    "FLEX_ATTENTION_BLOCK_SIZE",
    "FIRST_FRAME_CONDITIONING_DIM",
    "ERROR_RECYCLING_WARMUP_STEPS",
    "ERROR_RECYCLING_PROB",
    "ERROR_RECYCLING_SCALE",
    "ERROR_RECYCLING_TIMESTEP_BINS",
    "ERROR_RECYCLING_BANK_SIZE",
    "WINDOW_LEFT_FRAMES",
    "WINDOW_RIGHT_FRAMES",
    "MIN_LATENT_FRAME_COUNT",
    "LATENT_FRAME_COUNT",
    "HOLDOUT_COUNT",
    "EVAL_EVERY",
    "EVAL_INFERENCE_STEPS",
    "EVAL_FPS",
    "EVAL_ERROR_VMAX",
    "VIDEO_OUTPUT_SIZE",
    "SAVE_EVERY",
    "LORA_R",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_TARGET_MODULES",
    "DELTA_ATTENTION_BETA_INIT",
    "DELTA_ATTENTION_LAMBDA_INIT",
    "DELTA_ATTENTION_RHO_INIT",
    "DELTA_ATTENTION_FREEZE_RHO_ZERO",
    "DELTA_ATTENTION_KEY_SCALE",
    "SSM_DECAY_INIT",
    "SSM_INPUT_INIT",
    "SSM_OUTPUT_INIT",
    "SSM_SKIP_INIT",
    "SSM_RHO_INIT",
    "SSM_FREEZE_RHO_ZERO",
    "SSM_QUERY_SCALE",
    "SSM_RECURRENT_DIM",
    "WAN_NUM_TRAIN_TIMESTEPS",
    "WAN_SAMPLE_SHIFT",
    "WAN_SAMPLE_GUIDE_SCALE",
    "WAN_SAMPLE_SOLVER",
    "REQUIRE_RHO",
    "RANDOM_SEED",
)

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cuda.matmul.allow_tf32 = True
_REGIONAL_COMPILE_ACTIVE = False


def get_config():
    return {key: globals()[key] for key in CONFIG_KEYS}


def configure(**overrides):
    unknown = sorted(set(overrides) - set(CONFIG_KEYS))
    if unknown:
        raise KeyError(f"Unknown config keys: {unknown}")
    for key, value in overrides.items():
        globals()[key] = value
    _validate_training_mode()
    _validate_causal_window()
    if "RANDOM_SEED" in overrides:
        random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
    return get_config()


def _validate_training_mode() -> None:
    valid_modes = {"flow_matching", "delta_distill", "ssm", "ssm_distill"}
    if TRAINING_MODE not in valid_modes:
        raise ValueError(f"TRAINING_MODE must be one of {sorted(valid_modes)}")


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


def _padding_mask(
    batch_size: int,
    q_len: int,
    k_len: int,
    q_lens: Optional[torch.Tensor],
    k_lens: Optional[torch.Tensor],
    device: torch.device,
):
    if q_lens is None and k_lens is None:
        return None
    if q_lens is None:
        q_lens = torch.full((batch_size,), q_len, dtype=torch.long, device=device)
    else:
        q_lens = q_lens.to(device=device, dtype=torch.long)
    if k_lens is None:
        k_lens = torch.full((batch_size,), k_len, dtype=torch.long, device=device)
    else:
        k_lens = k_lens.to(device=device, dtype=torch.long)

    q_idx = torch.arange(q_len, device=device)
    k_idx = torch.arange(k_len, device=device)
    q_valid = q_idx.unsqueeze(0) < q_lens.unsqueeze(1)
    k_valid = k_idx.unsqueeze(0) < k_lens.unsqueeze(1)
    mask = q_valid.unsqueeze(2) & k_valid.unsqueeze(1)

    # Padded query rows are not used later, but giving them valid keys avoids
    # backend-specific all-masked-row behavior.
    return torch.where(q_valid.unsqueeze(2), mask, k_valid.unsqueeze(1))


def torch_sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """Wan-compatible attention using PyTorch SDPA.

    The local environment does not provide flash_attn. This fallback supports the
    same tensor layout Wan uses: [B, L, heads, dim].
    """
    del window_size, deterministic, version
    if q_scale is not None:
        q = q * q_scale

    batch_size, q_len, k_len = q.size(0), q.size(1), k.size(1)
    out_dtype = q.dtype
    attn_mask = _padding_mask(batch_size, q_len, k_len, q_lens, k_lens, q.device)
    if attn_mask is not None:
        attn_mask = attn_mask[:, None]

    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    return out.transpose(1, 2).contiguous().to(out_dtype)


def frame_window_mask(
    grid_sizes: torch.Tensor,
    seq_len: int,
    left_frames: int,
    right_frames: int,
    *,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
):
    """Build a causal block-frame mask.

    A query token in latent frame f can attend to frames [f-left_frames, f].
    The jump is exactly spatial_tokens, so the window never cuts through a frame
    boundary.
    """
    left_frames = int(left_frames)
    right_frames = int(right_frames)
    if left_frames < 0 or right_frames < 0:
        raise ValueError("left_frames and right_frames must be >= 0")
    if right_frames != 0:
        raise ValueError("right_frames must be 0 for causal windowed attention")

    device = grid_sizes.device
    batch_size = int(grid_sizes.shape[0])
    q_idx = torch.arange(seq_len, device=device)
    k_idx = torch.arange(seq_len, device=device)
    masks = []

    for batch_index, (frame_count, grid_h, grid_w) in enumerate(grid_sizes.tolist()):
        spatial_tokens = int(grid_h) * int(grid_w)
        valid_tokens = int(frame_count) * spatial_tokens
        if spatial_tokens <= 0:
            raise ValueError(f"Invalid spatial token count from grid {grid_h}x{grid_w}")
        if valid_tokens > seq_len:
            raise ValueError(
                f"Grid has {valid_tokens} tokens, but seq_len is {seq_len}"
            )

        if torch_compiler_is_compiling():
            q_valid_len = valid_tokens
            k_valid_len = valid_tokens
        else:
            q_valid_len = (
                int(q_lens[batch_index].item()) if q_lens is not None else valid_tokens
            )
            k_valid_len = (
                int(k_lens[batch_index].item()) if k_lens is not None else valid_tokens
            )
        q_valid = q_idx < q_valid_len
        k_valid = k_idx < k_valid_len

        q_frame = torch.div(q_idx, spatial_tokens, rounding_mode="floor")
        k_frame = torch.div(k_idx, spatial_tokens, rounding_mode="floor")
        frame_mask = (k_frame.unsqueeze(0) >= q_frame.unsqueeze(1) - left_frames) & (
            k_frame.unsqueeze(0) <= q_frame.unsqueeze(1) + right_frames
        )
        mask = q_valid.unsqueeze(1) & k_valid.unsqueeze(0) & frame_mask
        mask = torch.where(q_valid.unsqueeze(1), mask, k_valid.unsqueeze(0))
        masks.append(mask)

    if len(masks) != batch_size:
        raise RuntimeError("Internal error while building frame window mask")
    return torch.stack(masks, dim=0)


_FLEX_BLOCK_MASK_CACHE = {}
_COMPILED_FLEX_ATTENTION = (
    torch.compile(torch_flex_attention, dynamic=False)
    if torch_flex_attention is not None
    else None
)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def flex_attention_available() -> bool:
    return (
        USE_FLEX_ATTENTION
        and create_flex_block_mask is not None
        and _COMPILED_FLEX_ATTENTION is not None
    )


def _flex_block_size(spatial_tokens: int) -> int:
    spatial_tokens = int(spatial_tokens)
    min_block_size = 16
    if FLEX_ATTENTION_BLOCK_SIZE is not None:
        block_size = int(FLEX_ATTENTION_BLOCK_SIZE)
        while block_size > spatial_tokens and block_size > 1:
            block_size //= 2
    elif spatial_tokens >= 64:
        block_size = 64
    elif _is_power_of_two(int(spatial_tokens)):
        block_size = int(spatial_tokens)
    else:
        block_size = 32
    if not _is_power_of_two(block_size):
        raise ValueError(
            f"FLEX_ATTENTION_BLOCK_SIZE must be a power of two, got {block_size}"
        )
    while spatial_tokens % block_size != 0 and block_size > min_block_size:
        block_size //= 2
    if block_size < min_block_size:
        raise ValueError(
            f"could not choose a flex block size >= {min_block_size} "
            f"for {spatial_tokens} spatial tokens"
        )
    return block_size


def frame_window_flex_block_mask(
    *,
    seq_len: int,
    spatial_tokens: int,
    left_frames: int,
    right_frames: int,
    device: torch.device,
):
    block_size = _flex_block_size(spatial_tokens)
    key = (
        str(device),
        int(seq_len),
        int(spatial_tokens),
        int(left_frames),
        int(right_frames),
        int(block_size),
    )
    cached = _FLEX_BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    spatial_tokens = int(spatial_tokens)
    left_frames = int(left_frames)
    right_frames = int(right_frames)
    if right_frames != 0:
        raise ValueError("right_frames must be 0 for causal windowed attention")

    def mask_mod(batch, head, q_idx, kv_idx):
        del batch, head
        q_frame = q_idx // spatial_tokens
        kv_frame = kv_idx // spatial_tokens
        return (kv_frame >= q_frame - left_frames) & (
            kv_frame <= q_frame + right_frames
        )

    block_mask = create_flex_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=int(seq_len),
        KV_LEN=int(seq_len),
        device=device,
        BLOCK_SIZE=block_size,
    )
    _FLEX_BLOCK_MASK_CACHE[key] = block_mask
    return block_mask


def can_use_frame_window_flex_attention(
    q, grid_sizes: torch.Tensor, q_lens, k_lens
) -> tuple[bool, Optional[int]]:
    if not flex_attention_available() or q.device.type != "cuda":
        return False, None
    if q_lens is not None and not torch.all(q_lens.to(device=q.device) == q.size(1)):
        return False, None
    if k_lens is not None and not torch.all(k_lens.to(device=q.device) == q.size(1)):
        return False, None

    spatial_tokens = []
    for frame_count, grid_h, grid_w in grid_sizes.tolist():
        tokens = int(grid_h) * int(grid_w)
        if int(frame_count) * tokens != q.size(1):
            return False, None
        spatial_tokens.append(tokens)
    if not spatial_tokens or any(
        tokens != spatial_tokens[0] for tokens in spatial_tokens
    ):
        return False, None
    return True, spatial_tokens[0]


def frame_window_flex_attention(
    q,
    k,
    v,
    *,
    grid_sizes: torch.Tensor,
    left_frames: int,
    right_frames: int,
    softmax_scale=None,
    q_scale=None,
    dtype=torch.bfloat16,
    spatial_tokens: int,
):
    out_dtype = q.dtype
    if q_scale is not None:
        q = q * q_scale
    block_mask = frame_window_flex_block_mask(
        seq_len=q.size(1),
        spatial_tokens=spatial_tokens,
        left_frames=int(left_frames),
        right_frames=int(right_frames),
        device=q.device,
    )
    block_size = _flex_block_size(spatial_tokens)
    rows_guaranteed_safe = int(spatial_tokens) % int(block_size) == 0
    q = q.transpose(1, 2).to(dtype).contiguous()
    k = k.transpose(1, 2).to(dtype).contiguous()
    v = v.transpose(1, 2).to(dtype).contiguous()
    out = _COMPILED_FLEX_ATTENTION(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=softmax_scale,
        kernel_options={
            "BLOCK_M": block_size,
            "BLOCK_N": block_size,
            "ROWS_GUARANTEED_SAFE": rows_guaranteed_safe,
            "BLOCKS_ARE_CONTIGUOUS": False,
        },
    )
    return out.transpose(1, 2).contiguous().to(out_dtype)


def frame_window_attention(
    q,
    k,
    v,
    *,
    grid_sizes: torch.Tensor,
    left_frames: int,
    right_frames: int,
    freqs: Optional[torch.Tensor] = None,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    dtype=torch.bfloat16,
):
    out_dtype = q.dtype
    if int(right_frames) != 0:
        raise ValueError("right_frames must be 0 for causal windowed attention")
    if q_scale is not None:
        q = q * q_scale

    if q.size(1) != k.size(1):
        raise ValueError(
            "Frame-window attention expects equal query/key sequence lengths"
        )

    # Flex attention's backward kernel can exceed Triton/Inductor limits for
    # Wan's 128-dim heads here; dense SDPA is fine at the patchified L=704 size.
    flex_grad_unsupported = torch.is_grad_enabled() and (
        q.requires_grad or k.requires_grad or v.requires_grad
    )
    use_flex, spatial_tokens = (
        (False, None)
        if flex_grad_unsupported
        else can_use_frame_window_flex_attention(q, grid_sizes, q_lens, k_lens)
    )
    if use_flex:
        return frame_window_flex_attention(
            q,
            k,
            v,
            grid_sizes=grid_sizes,
            left_frames=left_frames,
            right_frames=right_frames,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            dtype=dtype,
            spatial_tokens=spatial_tokens,
        )

    attn_mask = frame_window_mask(
        grid_sizes.to(device=q.device),
        q.size(1),
        int(left_frames),
        int(right_frames),
        q_lens=q_lens,
        k_lens=k_lens,
    )[:, None]
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=softmax_scale,
    )
    return out.transpose(1, 2).contiguous().to(out_dtype)


class DeltaAttentionMemory(nn.Module):
    """Causal delta-rule fast-weight memory for one Wan self-attention layer."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        beta_init: Optional[float] = None,
        lambda_init: Optional[float] = None,
        rho_init: Optional[float] = None,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dim = self.num_heads * self.head_dim
        beta_init = DELTA_ATTENTION_BETA_INIT if beta_init is None else beta_init
        lambda_init = (
            DELTA_ATTENTION_LAMBDA_INIT if lambda_init is None else lambda_init
        )
        rho_init = DELTA_ATTENTION_RHO_INIT if rho_init is None else rho_init
        if DELTA_ATTENTION_FREEZE_RHO_ZERO:
            rho_init = 0.0
        beta_init = min(max(float(beta_init), 1e-6), 1.0 - 1e-6)
        lambda_init = min(max(float(lambda_init), 1e-6), 1.0 - 1e-6)
        self.beta_logit = nn.Parameter(
            torch.full((self.num_heads,), math.log(beta_init / (1.0 - beta_init)))
        )
        self.lambda_logit = nn.Parameter(
            torch.full((self.num_heads,), math.log(lambda_init / (1.0 - lambda_init)))
        )
        self.rho = nn.Parameter(
            torch.tensor(float(rho_init)),
            requires_grad=not bool(DELTA_ATTENTION_FREEZE_RHO_ZERO),
        )
        self.output_norm = nn.LayerNorm(self.dim, elementwise_affine=False)

    def _qk_scale(self) -> float:
        if DELTA_ATTENTION_KEY_SCALE is not None:
            return float(DELTA_ATTENTION_KEY_SCALE)
        return float(self.head_dim) ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        grid_sizes: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have matching shapes")

        batch_size, seq_len, num_heads, head_dim = q.shape
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError(
                f"DeltaAttentionMemory expected heads/head_dim {(self.num_heads, self.head_dim)}, "
                f"got {(num_heads, head_dim)}"
            )

        q_float = q.float() * self._qk_scale()
        k_float = k.float() * self._qk_scale()
        v_float = v.float()
        beta = torch.sigmoid(self.beta_logit).to(device=q.device).view(num_heads, 1, 1)
        lambda_ = (
            torch.sigmoid(self.lambda_logit).to(device=q.device).view(num_heads, 1, 1)
        )
        output = q_float.new_zeros(batch_size, seq_len, num_heads, head_dim)

        for batch_index, (frame_count, grid_h, grid_w) in enumerate(
            grid_sizes.tolist()
        ):
            spatial_tokens = int(grid_h) * int(grid_w)
            valid_tokens = int(frame_count) * spatial_tokens
            if seq_lens is not None and not torch_compiler_is_compiling():
                valid_tokens = min(valid_tokens, int(seq_lens[batch_index].item()))
            if spatial_tokens <= 0 or valid_tokens <= 0:
                continue
            if valid_tokens > seq_len:
                raise ValueError(
                    f"Grid has {valid_tokens} valid tokens, but seq_len is {seq_len}"
                )

            valid_frames = valid_tokens // spatial_tokens
            memory = q_float.new_zeros(num_heads, head_dim, head_dim)
            sample_output = []
            for frame_index in range(valid_frames):
                start = frame_index * spatial_tokens
                end = start + spatial_tokens
                q_frame = q_float[batch_index, start:end]
                k_frame = k_float[batch_index, start:end]
                v_frame = v_float[batch_index, start:end]

                predicted_v = torch.einsum("hde,shd->she", memory, k_frame)
                delta_v = v_frame - predicted_v
                update = torch.einsum("shd,she->hde", k_frame, delta_v)
                update = update / max(1, spatial_tokens)
                memory = lambda_ * memory + beta * update
                frame_output = torch.einsum("hde,shd->she", memory, q_frame)
                sample_output.append(frame_output)

            if sample_output:
                output[batch_index, : valid_frames * spatial_tokens] = torch.cat(
                    sample_output, dim=0
                )

        output = output.flatten(2)
        output = self.output_norm(output)
        output = output.view(batch_size, seq_len, num_heads, head_dim)
        return (self.rho.to(device=q.device, dtype=output.dtype) * output).to(
            dtype=q.dtype
        )


class CausalResidualConv3dBlock(nn.Module):
    """Depthwise causal temporal Conv3d residual block for reduced token grids."""

    def __init__(self, channels: int, kernel_size: tuple[int, int, int] = (3, 3, 3)):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = tuple(int(value) for value in kernel_size)
        if len(self.kernel_size) != 3:
            raise ValueError("kernel_size must be a 3-tuple")
        if any(value < 1 for value in self.kernel_size):
            raise ValueError(f"kernel_size entries must be positive: {kernel_size}")
        self.depthwise = nn.Conv3d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            groups=self.channels,
        )
        self.pointwise = nn.Conv3d(self.channels, self.channels, kernel_size=1)
        self.activation = nn.SiLU()
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        temporal, height, width = self.kernel_size
        x_padded = F.pad(
            x.to(dtype=self.depthwise.weight.dtype),
            (
                width // 2,
                width - 1 - width // 2,
                height // 2,
                height - 1 - height // 2,
                temporal - 1,
                0,
            ),
        )
        residual = self.pointwise(self.activation(self.depthwise(x_padded)))
        return x + residual.to(dtype=input_dtype)


class StateSpaceMemory(nn.Module):
    """Frame-level diagonal SSM memory for one Wan self-attention layer."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        decay_init: Optional[float] = None,
        input_init: Optional[float] = None,
        output_init: Optional[float] = None,
        skip_init: Optional[float] = None,
        rho_init: Optional[float] = None,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dim = self.num_heads * self.head_dim
        self.recurrent_dim = int(SSM_RECURRENT_DIM)
        if self.recurrent_dim < 1:
            raise ValueError("SSM_RECURRENT_DIM must be positive")
        decay_init = SSM_DECAY_INIT if decay_init is None else decay_init
        input_init = SSM_INPUT_INIT if input_init is None else input_init
        output_init = SSM_OUTPUT_INIT if output_init is None else output_init
        skip_init = SSM_SKIP_INIT if skip_init is None else skip_init
        rho_init = SSM_RHO_INIT if rho_init is None else rho_init
        if SSM_FREEZE_RHO_ZERO:
            rho_init = 0.0
        decay_init = min(max(float(decay_init), 1e-6), 1.0 - 1e-6)
        input_init = min(max(float(input_init), 1e-6), 1.0 - 1e-6)
        self.query_projection = nn.Linear(self.dim, self.recurrent_dim)
        self.input_projection = nn.Linear(self.dim, self.recurrent_dim)
        self.pre_ssm_conv = CausalResidualConv3dBlock(self.recurrent_dim)
        self.post_ssm_conv = CausalResidualConv3dBlock(self.recurrent_dim)
        self.output_projection = nn.Linear(self.recurrent_dim, self.dim)
        self.decay_logit = nn.Parameter(
            torch.full(
                (self.recurrent_dim,),
                math.log(decay_init / (1.0 - decay_init)),
            )
        )
        self.input_logit = nn.Parameter(
            torch.full(
                (self.recurrent_dim,),
                math.log(input_init / (1.0 - input_init)),
            )
        )
        self.output_gain = nn.Parameter(
            torch.full((self.recurrent_dim,), float(output_init))
        )
        self.skip_gain = nn.Parameter(
            torch.full((self.recurrent_dim,), float(skip_init))
        )
        self.rho = nn.Parameter(
            torch.tensor(float(rho_init)),
            requires_grad=not bool(SSM_FREEZE_RHO_ZERO),
        )
        self.output_norm = nn.LayerNorm(self.recurrent_dim, elementwise_affine=False)

    def _query_scale(self) -> float:
        if SSM_QUERY_SCALE is not None:
            return float(SSM_QUERY_SCALE)
        return float(self.recurrent_dim) ** -0.5

    def _shared_full_grid_shape(
        self,
        grid_sizes: torch.Tensor,
        seq_lens: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
    ) -> Optional[tuple[int, int, int]]:
        grids = [
            (int(frame_count), int(grid_h), int(grid_w))
            for frame_count, grid_h, grid_w in grid_sizes.tolist()
        ]
        if len(grids) != int(batch_size) or not grids:
            return None
        first = grids[0]
        if any(grid != first for grid in grids):
            return None
        frame_count, grid_h, grid_w = first
        if frame_count * grid_h * grid_w != int(seq_len):
            return None
        if seq_lens is not None and not torch_compiler_is_compiling():
            if not bool(torch.all(seq_lens.to(device=grid_sizes.device) == int(seq_len))):
                return None
        return first

    def _apply_grid_conv(
        self,
        x: torch.Tensor,
        conv: CausalResidualConv3dBlock,
        *,
        grid_sizes: torch.Tensor,
        seq_lens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        shared_shape = self._shared_full_grid_shape(
            grid_sizes, seq_lens, batch_size=batch_size, seq_len=seq_len
        )
        if shared_shape is not None:
            frame_count, grid_h, grid_w = shared_shape
            grid = (
                x.reshape(batch_size, frame_count, grid_h, grid_w, channels)
                .permute(0, 4, 1, 2, 3)
                .contiguous()
            )
            grid = conv(grid)
            return (
                grid.permute(0, 2, 3, 4, 1)
                .reshape(batch_size, seq_len, channels)
                .contiguous()
            )

        output = x.clone()
        for batch_index, (frame_count, grid_h, grid_w) in enumerate(
            grid_sizes.tolist()
        ):
            spatial_tokens = int(grid_h) * int(grid_w)
            valid_tokens = int(frame_count) * spatial_tokens
            if seq_lens is not None and not torch_compiler_is_compiling():
                valid_tokens = min(valid_tokens, int(seq_lens[batch_index].item()))
            if spatial_tokens <= 0 or valid_tokens <= 0:
                continue
            if valid_tokens > seq_len:
                raise ValueError(
                    f"Grid has {valid_tokens} valid tokens, but seq_len is {seq_len}"
                )
            valid_frames = valid_tokens // spatial_tokens
            if valid_frames <= 0:
                continue
            valid_tokens = valid_frames * spatial_tokens
            grid = (
                x[batch_index, :valid_tokens]
                .view(valid_frames, int(grid_h), int(grid_w), channels)
                .permute(3, 0, 1, 2)
                .unsqueeze(0)
                .contiguous()
            )
            grid = conv(grid)
            output[batch_index, :valid_tokens] = (
                grid.squeeze(0)
                .permute(1, 2, 3, 0)
                .reshape(valid_tokens, channels)
            )
        return output

    def _forward_shared_grid(
        self,
        q_float: torch.Tensor,
        v_float: torch.Tensor,
        *,
        grid_shape: tuple[int, int, int],
        decay: torch.Tensor,
        input_gain: torch.Tensor,
        output_gain: torch.Tensor,
        skip_gain: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, channels = q_float.shape
        frame_count, grid_h, grid_w = grid_shape
        spatial_tokens = int(grid_h) * int(grid_w)
        q_frames = q_float.reshape(batch_size, frame_count, spatial_tokens, channels)
        v_frames = v_float.reshape(batch_size, frame_count, spatial_tokens, channels)
        frame_inputs = v_frames.mean(dim=2)
        state = q_float.new_zeros(batch_size, channels)
        output_frames = []
        for frame_index in range(int(frame_count)):
            frame_input = frame_inputs[:, frame_index]
            state = decay * state + input_gain * frame_input
            read = output_gain * state + skip_gain * frame_input
            output_frames.append(q_frames[:, frame_index] * read.unsqueeze(1))
        return torch.stack(output_frames, dim=1).reshape_as(q_float)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        grid_sizes: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del k
        batch_size, seq_len, num_heads, head_dim = q.shape
        if v.shape != q.shape:
            raise ValueError("q and v must have matching shapes")
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError(
                f"StateSpaceMemory expected heads/head_dim {(self.num_heads, self.head_dim)}, "
                f"got {(num_heads, head_dim)}"
            )

        q_flat = q.flatten(2).to(dtype=self.query_projection.weight.dtype)
        v_flat = v.flatten(2).to(dtype=self.input_projection.weight.dtype)
        q_float = self.query_projection(q_flat).float() * self._query_scale()
        v_float = self.input_projection(v_flat).float()
        v_float = self._apply_grid_conv(
            v_float,
            self.pre_ssm_conv,
            grid_sizes=grid_sizes,
            seq_lens=seq_lens,
        )
        decay = torch.sigmoid(self.decay_logit).to(device=q.device)
        input_gain = torch.sigmoid(self.input_logit).to(device=q.device)
        output_gain = self.output_gain.to(device=q.device, dtype=q_float.dtype)
        skip_gain = self.skip_gain.to(device=q.device, dtype=q_float.dtype)
        shared_shape = self._shared_full_grid_shape(
            grid_sizes, seq_lens, batch_size=batch_size, seq_len=seq_len
        )
        if shared_shape is not None:
            output = self._forward_shared_grid(
                q_float,
                v_float,
                grid_shape=shared_shape,
                decay=decay,
                input_gain=input_gain,
                output_gain=output_gain,
                skip_gain=skip_gain,
            )
        else:
            output = q_float.new_zeros(batch_size, seq_len, self.recurrent_dim)
            for batch_index, (frame_count, grid_h, grid_w) in enumerate(
                grid_sizes.tolist()
            ):
                spatial_tokens = int(grid_h) * int(grid_w)
                valid_tokens = int(frame_count) * spatial_tokens
                if seq_lens is not None and not torch_compiler_is_compiling():
                    valid_tokens = min(valid_tokens, int(seq_lens[batch_index].item()))
                if spatial_tokens <= 0 or valid_tokens <= 0:
                    continue
                if valid_tokens > seq_len:
                    raise ValueError(
                        f"Grid has {valid_tokens} valid tokens, but seq_len is {seq_len}"
                    )

                valid_frames = valid_tokens // spatial_tokens
                state = q_float.new_zeros(self.recurrent_dim)
                sample_output = []
                for frame_index in range(valid_frames):
                    start = frame_index * spatial_tokens
                    end = start + spatial_tokens
                    q_frame = q_float[batch_index, start:end]
                    v_frame = v_float[batch_index, start:end]
                    frame_input = v_frame.mean(dim=0)
                    state = decay * state + input_gain * frame_input
                    read = output_gain * state + skip_gain * frame_input
                    sample_output.append(q_frame * read.unsqueeze(0))

                if sample_output:
                    output[batch_index, : valid_frames * spatial_tokens] = torch.cat(
                        sample_output, dim=0
                    )

        output = self.output_norm(output)
        output = self._apply_grid_conv(
            output,
            self.post_ssm_conv,
            grid_sizes=grid_sizes,
            seq_lens=seq_lens,
        )
        output = self.output_projection(
            output.to(dtype=self.output_projection.weight.dtype)
        )
        output = output.view(batch_size, seq_len, num_heads, head_dim)
        return (self.rho.to(device=q.device, dtype=output.dtype) * output).to(
            dtype=q.dtype
        )


def load_wan_model_module(wan_repo_root: Path = WAN_REPO_ROOT):
    """Load Wan modules without importing wan/__init__.py optional extras."""
    _, modules_root, _ = _ensure_wan_package(wan_repo_root)
    _load_module("wan.modules.attention", modules_root / "attention.py")
    module = _load_module("wan.modules.model", modules_root / "model.py")
    module.flash_attention = torch_sdpa_attention
    install_frame_window_self_attention(module)
    return module


def install_frame_window_self_attention(wan_model_module):
    rope_apply = wan_model_module.rope_apply

    def self_attention_forward(self, x, seq_lens, grid_sizes, freqs):
        batch_size, seq_len = x.shape[:2]
        num_heads, head_dim = self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(batch_size, seq_len, num_heads, head_dim)
        k = self.norm_k(self.k(x)).view(batch_size, seq_len, num_heads, head_dim)
        v = self.v(x).view(batch_size, seq_len, num_heads, head_dim)

        frame_window = getattr(self, "frame_window", None)
        delta_enabled = bool(getattr(self, "delta_attention_enabled", False))
        ssm_enabled = bool(getattr(self, "ssm_attention_enabled", False))
        if delta_enabled and ssm_enabled:
            raise RuntimeError("delta and SSM attention cannot be enabled together")
        if frame_window is None and not delta_enabled and not ssm_enabled:
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)
            x_out = wan_model_module.flash_attention(q=q, k=k, v=v, k_lens=seq_lens)
        else:
            # With the frame-0 anchor removed, RoPE attention is translation-invariant
            # over the bounded local window because Q/K share the same global offset.
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)
            if frame_window is None:
                left_frames, right_frames = causal_window()
            else:
                left_frames, right_frames = frame_window
            x_out = frame_window_attention(
                q=q,
                k=k,
                v=v,
                grid_sizes=grid_sizes,
                left_frames=int(left_frames),
                right_frames=int(right_frames),
                freqs=freqs,
                k_lens=seq_lens,
                dtype=TRAIN_DTYPE,
            )
            if delta_enabled:
                delta_memory = getattr(self, "delta_memory", None)
                if delta_memory is None:
                    raise RuntimeError(
                        "delta_attention_enabled=True but this attention layer has no delta_memory module"
                    )
                x_out = x_out + delta_memory(
                    q=q,
                    k=k,
                    v=v,
                    grid_sizes=grid_sizes,
                    seq_lens=seq_lens,
                )
            if ssm_enabled:
                ssm_memory = getattr(self, "ssm_memory", None)
                if ssm_memory is None:
                    raise RuntimeError(
                        "ssm_attention_enabled=True but this attention layer has no ssm_memory module"
                    )
                x_out = x_out + ssm_memory(
                    q=q,
                    k=k,
                    v=v,
                    grid_sizes=grid_sizes,
                    seq_lens=seq_lens,
                )

        x_out = x_out.flatten(2)
        return self.o(x_out)

    wan_model_module.WanSelfAttention.forward = self_attention_forward


def load_wan_scheduler_modules(wan_repo_root: Path = WAN_REPO_ROOT):
    _, _, utils_root = _ensure_wan_package(wan_repo_root)
    fm_solvers = _load_module("wan.utils.fm_solvers", utils_root / "fm_solvers.py")
    fm_solvers_unipc = _load_module(
        "wan.utils.fm_solvers_unipc", utils_root / "fm_solvers_unipc.py"
    )
    return fm_solvers, fm_solvers_unipc


def load_wan_vae_class(wan_repo_root: Path = WAN_REPO_ROOT):
    _, modules_root, _ = _ensure_wan_package(wan_repo_root)
    return _load_module("wan.modules.vae2_2", modules_root / "vae2_2.py").Wan2_2_VAE


def load_wan_t5_encoder_class(wan_repo_root: Path = WAN_REPO_ROOT):
    _, modules_root, _ = _ensure_wan_package(wan_repo_root)
    return _load_module("wan.modules.t5", modules_root / "t5.py").T5EncoderModel


wan_model_module = load_wan_model_module()
WanModel = wan_model_module.WanModel
rope_params = wan_model_module.rope_params
sinusoidal_embedding_1d = wan_model_module.sinusoidal_embedding_1d

wan_fm_solvers_module, wan_fm_solvers_unipc_module = load_wan_scheduler_modules()
FlowDPMSolverMultistepScheduler = wan_fm_solvers_module.FlowDPMSolverMultistepScheduler
FlowUniPCMultistepScheduler = wan_fm_solvers_unipc_module.FlowUniPCMultistepScheduler
get_sampling_sigmas = wan_fm_solvers_module.get_sampling_sigmas
retrieve_timesteps = wan_fm_solvers_module.retrieve_timesteps


def load_wan_transformer(
    checkpoint_dir: Path = WAN_CHECKPOINT_DIR,
    device: torch.device = DEVICE,
    dtype: torch.dtype = PARAM_DTYPE,
) -> WanModel:
    checkpoint_dir = Path(checkpoint_dir)
    try:
        model = WanModel.from_pretrained(
            str(checkpoint_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        warnings.warn(
            f"Diffusers from_pretrained failed ({exc}); falling back to manual shard load."
        )
        config = json.loads((checkpoint_dir / "config.json").read_text())
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)
        model = WanModel(**config)
        state = {}
        for shard in sorted(
            checkpoint_dir.glob("diffusion_pytorch_model-*.safetensors")
        ):
            state.update(load_file(str(shard)))
        model.load_state_dict(state, strict=True)
        del state

    model.eval().requires_grad_(False)
    return model.to(device=device, dtype=dtype)


def attach_physics_adaln_forward(model: WanModel) -> WanModel:
    """Add physics AdaLN and first-frame token conditioning."""
    dim = model.dim
    model.first_frame_conditioner = nn.Sequential(
        nn.Linear(dim, int(FIRST_FRAME_CONDITIONING_DIM)),
        nn.SiLU(),
        nn.Linear(int(FIRST_FRAME_CONDITIONING_DIM), dim),
    ).to(
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )
    nn.init.zeros_(model.first_frame_conditioner[-1].weight)
    nn.init.zeros_(model.first_frame_conditioner[-1].bias)

    model.physics_adaln = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(2, 32),
                nn.SiLU(),
                nn.Linear(32, 6 * dim),
            )
            for _ in model.blocks
        ]
    ).to(device=next(model.parameters()).device, dtype=torch.float32)
    for adaln in model.physics_adaln:
        nn.init.zeros_(adaln[-1].weight)
        nn.init.zeros_(adaln[-1].bias)

    def ensure_rope_capacity(self, grid_sizes: torch.Tensor, device: torch.device):
        max_position = int(grid_sizes.max().item())
        if max_position <= int(self.freqs.size(0)):
            return
        head_dim = int(self.dim) // int(self.num_heads)
        self.freqs = torch.cat(
            [
                rope_params(max_position, head_dim - 4 * (head_dim // 6)),
                rope_params(max_position, 2 * (head_dim // 6)),
                rope_params(max_position, 2 * (head_dim // 6)),
            ],
            dim=1,
        ).to(device)

    def forward_with_physics(
        self, x, t, context, seq_len, y=None, physics=None, first_frame=None
    ):
        if self.model_type == "i2v":
            assert y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        def latent_list(value, name: str):
            if isinstance(value, torch.Tensor):
                if value.dim() == 5:
                    return [u for u in value]
                if value.dim() == 4:
                    return [value]
                raise ValueError(
                    f"{name} tensor must have shape [B,C,F,H,W] or [C,F,H,W], got {tuple(value.shape)}"
                )
            return list(value)

        x = latent_list(x, "x")
        if first_frame is None:
            raise ValueError("first_frame is required for first-frame conditioning")
        first_frame = latent_list(first_frame, "first_frame")
        if len(first_frame) != len(x):
            raise ValueError(
                f"first_frame batch has {len(first_frame)} items but x has {len(x)}"
            )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x]
        )
        ensure_rope_capacity(self, grid_sizes, device)
        x = [u.flatten(2).transpose(1, 2) for u in x]
        conditioned = []
        for item_index, (tokens, condition_latent) in enumerate(zip(x, first_frame)):
            condition_tokens = self.patch_embedding(
                condition_latent.to(
                    device=device, dtype=self.patch_embedding.weight.dtype
                ).unsqueeze(0)
            )
            if int(condition_tokens.shape[2]) != 1:
                raise ValueError(
                    "first_frame must contain exactly one latent frame after patchification"
                )
            condition_tokens = condition_tokens.flatten(2).transpose(1, 2)
            frame_count, grid_h, grid_w = grid_sizes[item_index].tolist()
            spatial_tokens = int(grid_h) * int(grid_w)
            if condition_tokens.size(1) != spatial_tokens:
                raise ValueError(
                    f"first_frame produced {condition_tokens.size(1)} spatial tokens, expected {spatial_tokens}"
                )
            condition_tokens = self.first_frame_conditioner(condition_tokens)
            condition_tokens = (
                condition_tokens.view(1, 1, spatial_tokens, dim)
                .expand(1, int(frame_count), spatial_tokens, dim)
                .reshape(1, int(frame_count) * spatial_tokens, dim)
            )
            conditioned.append(tokens + condition_tokens.to(dtype=tokens.dtype))
        x = conditioned
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast(
            "cuda", dtype=torch.float32, enabled=DEVICE.type == "cuda"
        ):
            batch_tokens = t.size(0)
            t_flat = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t_flat)
                .unflatten(0, (batch_tokens, seq_len))
                .float()
                .to(device)
            )
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            physics_e0 = None
            if physics is not None:
                physics = physics.to(device=device, dtype=torch.float32)
                if physics.shape != (batch_tokens, 2):
                    raise ValueError(
                        f"physics must have shape ({batch_tokens}, 2), got {tuple(physics.shape)}"
                    )
                physics_e0 = torch.stack(
                    [adaln(physics) for adaln in self.physics_adaln], dim=1
                )
                physics_e0 = physics_e0.view(
                    batch_tokens, len(self.blocks), 1, 6, self.dim
                )

        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )
        for block_index, block in enumerate(self.blocks):
            kwargs["e"] = e0 if physics_e0 is None else e0 + physics_e0[:, block_index]
            x = block(x, **kwargs)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    model.forward = MethodType(forward_with_physics, model)
    return model


def add_lora_to_wan(model: WanModel):
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    peft_model = get_peft_model(model, config)
    base = peft_model.get_base_model()
    if hasattr(base, "physics_adaln"):
        for parameter in base.physics_adaln.parameters():
            parameter.requires_grad_(True)
    if hasattr(base, "first_frame_conditioner"):
        for parameter in base.first_frame_conditioner.parameters():
            parameter.requires_grad_(True)
    return peft_model


def base_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def unwrap_compiled_module(module):
    return getattr(module, "_orig_mod", module)


def torch_compiler_is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if is_compiling is not None:
        return bool(is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    is_compiling = getattr(dynamo, "is_compiling", None)
    return bool(is_compiling()) if is_compiling is not None else False


def configure_regional_compile_runtime() -> None:
    if not REGIONAL_COMPILE:
        return
    if not REGIONAL_COMPILE_CAPTURE_SCALAR_OUTPUTS:
        return
    try:
        import torch._dynamo

        torch._dynamo.config.capture_scalar_outputs = True
    except Exception as exc:
        warnings.warn(f"Could not set Dynamo capture_scalar_outputs: {exc}")


def compiler_mark_step_begin() -> None:
    if not REGIONAL_COMPILE or not _REGIONAL_COMPILE_ACTIVE or DEVICE.type != "cuda":
        return
    compiler = getattr(torch, "compiler", None)
    mark_step = getattr(compiler, "cudagraph_mark_step_begin", None)
    if mark_step is not None:
        mark_step()


def regional_compile_kwargs():
    kwargs = {
        "fullgraph": bool(REGIONAL_COMPILE_FULLGRAPH),
        "dynamic": bool(REGIONAL_COMPILE_DYNAMIC),
    }
    if REGIONAL_COMPILE_BACKEND is not None:
        kwargs["backend"] = REGIONAL_COMPILE_BACKEND
    if REGIONAL_COMPILE_MODE is not None:
        kwargs["mode"] = REGIONAL_COMPILE_MODE
    return kwargs


def remove_compiled_repeated_blocks(model) -> bool:
    base = base_model(model)
    blocks = getattr(base, "blocks", None)
    if blocks is None:
        return False

    original_blocks = getattr(blocks, "_orig_mod", None)
    if original_blocks is not None:
        base.blocks = original_blocks
        return True

    changed = False
    for block_index, block in enumerate(list(blocks)):
        original_block = getattr(block, "_orig_mod", None)
        if original_block is not None:
            blocks[block_index] = original_block
            changed = True
    return changed


def regional_compile_skip_reasons(model) -> list[str]:
    reasons = []
    base = base_model(model)
    if hasattr(base, "blocks"):
        attn_modules = [
            getattr(unwrap_compiled_module(block), "self_attn", None)
            for block in base.blocks
        ]
        if any(attn is not None and hasattr(attn, "ssm_memory") for attn in attn_modules):
            reasons.append("SSM recurrent attention is attached")
        if any(attn is not None and hasattr(attn, "delta_memory") for attn in attn_modules):
            reasons.append("delta recurrent attention is attached")
    if flex_attention_available():
        reasons.append("flex attention already compiles the sparse window kernel")
    return reasons


def apply_regional_compile(model, *, label: str = "model"):
    global _REGIONAL_COMPILE_ACTIVE
    _REGIONAL_COMPILE_ACTIVE = False
    if not REGIONAL_COMPILE:
        return model

    skip_reasons = regional_compile_skip_reasons(model)
    if skip_reasons:
        removed = remove_compiled_repeated_blocks(model)
        suffix = "; removed existing compiled repeated blocks" if removed else ""
        _init_log(
            f"skipping regional compile for {label}: "
            + "; ".join(skip_reasons)
            + "; compiling the outer Wan blocks can exceed Triton resource limits"
            + suffix
        )
        return model

    if not hasattr(torch, "compile"):
        raise RuntimeError("REGIONAL_COMPILE=True requires torch.compile")

    configure_regional_compile_runtime()

    try:
        from accelerate.utils import compile_regions, has_compiled_regions
    except Exception as exc:
        raise RuntimeError(
            "REGIONAL_COMPILE=True requires accelerate.utils.compile_regions"
        ) from exc

    base = base_model(model)
    if not hasattr(base, "blocks"):
        raise AttributeError(f"{label} has no repeated `blocks` module to compile")
    if has_compiled_regions(base.blocks):
        _init_log(f"{label} repeated blocks already regionally compiled")
        _REGIONAL_COMPILE_ACTIVE = True
        return model

    kwargs = regional_compile_kwargs()
    _init_log(f"regionally compiling {label} repeated blocks with {kwargs}")
    base.blocks = compile_regions(base.blocks, **kwargs)
    _REGIONAL_COMPILE_ACTIVE = True
    _init_log(f"regionally compiled {len(base.blocks)} {label} blocks")
    return model


@contextmanager
def temporarily_uncompiled_repeated_blocks(model):
    base = base_model(model)
    blocks = getattr(base, "blocks", None)
    if blocks is None:
        yield
        return

    original_blocks = getattr(blocks, "_orig_mod", None)
    if original_blocks is not None:
        compiled_blocks = blocks
        base.blocks = original_blocks
        try:
            yield
        finally:
            base.blocks = compiled_blocks
        return

    replacements = []
    try:
        for index, block in enumerate(blocks):
            original_block = getattr(block, "_orig_mod", None)
            if original_block is not None:
                replacements.append((index, block))
                blocks[index] = original_block
        yield
    finally:
        for index, compiled_block in replacements:
            blocks[index] = compiled_block


def lora_adapter_parameters(model):
    return [
        parameter for name, parameter in model.named_parameters() if "lora_" in name
    ]


def physics_conditioning_parameters(model):
    base = base_model(model)
    if not hasattr(base, "physics_adaln"):
        return []
    return list(base.physics_adaln.parameters())


def first_frame_conditioning_parameters(model):
    base = base_model(model)
    if not hasattr(base, "first_frame_conditioner"):
        return []
    return list(base.first_frame_conditioner.parameters())


def enable_lora_adapter_training(model) -> None:
    for parameter in lora_adapter_parameters(model):
        parameter.requires_grad_(True)

    for parameter in physics_conditioning_parameters(model):
        parameter.requires_grad_(True)

    for parameter in first_frame_conditioning_parameters(model):
        parameter.requires_grad_(True)


def attach_delta_attention(model):
    base = base_model(model)
    device = next(base.parameters()).device
    for block in base.blocks:
        block = unwrap_compiled_module(block)
        attn = block.self_attn
        if not hasattr(attn, "delta_memory"):
            attn.delta_memory = DeltaAttentionMemory(attn.num_heads, attn.head_dim).to(
                device=device
            )
        attn.delta_attention_enabled = False
    return model


def attach_ssm_attention(model):
    base = base_model(model)
    device = next(base.parameters()).device
    for block in base.blocks:
        block = unwrap_compiled_module(block)
        attn = block.self_attn
        if not hasattr(attn, "ssm_memory"):
            attn.ssm_memory = StateSpaceMemory(attn.num_heads, attn.head_dim).to(
                device=device
            )
        attn.ssm_attention_enabled = False
    return model


def set_delta_attention_enabled(model, enabled: bool):
    base = base_model(model)
    previous = []
    for block in base.blocks:
        block = unwrap_compiled_module(block)
        attn = block.self_attn
        previous.append((attn, bool(getattr(attn, "delta_attention_enabled", False))))
        attn.delta_attention_enabled = bool(enabled)
    return previous


def set_ssm_attention_enabled(model, enabled: bool):
    base = base_model(model)
    previous = []
    for block in base.blocks:
        block = unwrap_compiled_module(block)
        attn = block.self_attn
        previous.append((attn, bool(getattr(attn, "ssm_attention_enabled", False))))
        attn.ssm_attention_enabled = bool(enabled)
    return previous


@contextmanager
def delta_attention_enabled(model, enabled: bool = True):
    previous = set_delta_attention_enabled(model, enabled)
    try:
        yield
    finally:
        for attn, old_value in previous:
            attn.delta_attention_enabled = old_value


@contextmanager
def ssm_attention_enabled(model, enabled: bool = True):
    previous = set_ssm_attention_enabled(model, enabled)
    try:
        yield
    finally:
        for attn, old_value in previous:
            attn.ssm_attention_enabled = old_value


def freeze_parameters(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def delta_attention_parameters(model):
    params = []
    base = base_model(model)
    for block in base.blocks:
        block = unwrap_compiled_module(block)
        delta_memory = getattr(block.self_attn, "delta_memory", None)
        if delta_memory is not None:
            for name, parameter in delta_memory.named_parameters():
                if DELTA_ATTENTION_FREEZE_RHO_ZERO and name == "rho":
                    parameter.data.zero_()
                    parameter.requires_grad_(False)
                    continue
                params.append(parameter)
    return params


def ssm_attention_parameters(model):
    params = []
    base = base_model(model)
    for block in base.blocks:
        block = unwrap_compiled_module(block)
        ssm_memory = getattr(block.self_attn, "ssm_memory", None)
        if ssm_memory is not None:
            for name, parameter in ssm_memory.named_parameters():
                if SSM_FREEZE_RHO_ZERO and name == "rho":
                    parameter.data.zero_()
                    parameter.requires_grad_(False)
                    continue
                params.append(parameter)
    return params


def has_delta_attention(model) -> bool:
    base = base_model(model)
    return any(
        hasattr(unwrap_compiled_module(block).self_attn, "delta_memory")
        for block in base.blocks
    )


def has_ssm_attention(model) -> bool:
    base = base_model(model)
    return any(
        hasattr(unwrap_compiled_module(block).self_attn, "ssm_memory")
        for block in base.blocks
    )


def recurrent_attention_kind(model) -> Optional[str]:
    has_delta = has_delta_attention(model)
    has_ssm = has_ssm_attention(model)
    if has_delta and has_ssm:
        raise RuntimeError("Model has both delta and SSM memories attached")
    if has_delta:
        return "delta"
    if has_ssm:
        return "ssm"
    return None


def set_recurrent_attention_enabled(model, kind: str, enabled: bool):
    if kind == "delta":
        return set_delta_attention_enabled(model, enabled)
    if kind == "ssm":
        return set_ssm_attention_enabled(model, enabled)
    raise ValueError(f"Unknown recurrent attention kind: {kind!r}")


def enable_delta_attention_training(model, *, train_lora: bool = False) -> None:
    freeze_parameters(model)
    for parameter in delta_attention_parameters(model):
        parameter.requires_grad_(True)
    if train_lora:
        enable_lora_adapter_training(model)


def enable_ssm_attention_training(model, *, train_lora: bool = False) -> None:
    freeze_parameters(model)
    for parameter in ssm_attention_parameters(model):
        parameter.requires_grad_(True)
    if train_lora:
        enable_lora_adapter_training(model)


def _disabled_gradient_checkpoint_blocks(total_blocks: int) -> set[int]:
    setting = GRADIENT_CHECKPOINT_DISABLE_BLOCKS
    if setting is None:
        return set(range(total_blocks))

    if isinstance(setting, str):
        normalized = setting.strip().lower()
        if normalized == "all":
            return set(range(total_blocks))
        if normalized == "none":
            return set()
        try:
            setting = int(normalized)
        except ValueError as exc:
            raise ValueError(
                "GRADIENT_CHECKPOINT_DISABLE_BLOCKS must be 'all', 'none', an int, or a sequence of block indices"
            ) from exc

    if isinstance(setting, int):
        n_disabled = max(0, min(total_blocks, setting))
        return set(range(n_disabled))

    if isinstance(setting, (list, tuple, set)):
        disabled = {int(index) for index in setting}
        invalid = sorted(
            index for index in disabled if index < 0 or index >= total_blocks
        )
        if invalid:
            raise ValueError(
                f"Invalid gradient-checkpoint-disabled block indices for {total_blocks} blocks: {invalid}"
            )
        return disabled

    raise TypeError(
        "GRADIENT_CHECKPOINT_DISABLE_BLOCKS must be 'all', 'none', an int, or a sequence of block indices"
    )


def _wan_block_forward_with_optional_checkpoint(
    self,
    x,
    e,
    seq_lens,
    grid_sizes,
    freqs,
    context,
    context_lens,
):
    original_forward = self._original_forward_no_checkpoint
    if not (self.training and getattr(self, "gradient_checkpointing", False)):
        return original_forward(
            x,
            e=e,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context,
            context_lens=context_lens,
        )

    def run(activation):
        return original_forward(
            activation,
            e=e,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context,
            context_lens=context_lens,
        )

    return torch_checkpoint.checkpoint(run, x, use_reentrant=False)


def configure_gradient_checkpointing(model):
    base = base_model(model)
    total_blocks = len(base.blocks)
    disabled = _disabled_gradient_checkpoint_blocks(total_blocks)
    for block_index, block in enumerate(base.blocks):
        block = unwrap_compiled_module(block)
        if not hasattr(block, "_original_forward_no_checkpoint"):
            block._original_forward_no_checkpoint = block.forward
        checkpoint_enabled = block_index not in disabled
        block.gradient_checkpointing = checkpoint_enabled
        if checkpoint_enabled:
            block.forward = MethodType(
                _wan_block_forward_with_optional_checkpoint, block
            )
        else:
            block.forward = block._original_forward_no_checkpoint

    enabled_count = total_blocks - len(disabled)
    if len(disabled) == total_blocks:
        disabled_label = "all"
    elif not disabled:
        disabled_label = "none"
    else:
        disabled_label = sorted(disabled)
    print(
        f"gradient checkpointing enabled on {enabled_count}/{total_blocks} blocks; "
        f"disabled blocks: {disabled_label}"
    )
    return {
        "total_blocks": total_blocks,
        "enabled_blocks": enabled_count,
        "disabled_blocks": sorted(disabled),
    }


def require_lora_adapter_state(model, enabled: bool, label: str = "inference") -> None:
    if not hasattr(model, "get_model_status"):
        raise TypeError(f"{label} expected a PEFT model with LoRA adapters")

    status = model.get_model_status()
    if status.enabled == "irregular":
        raise RuntimeError(
            f"{label} has an irregular adapter state; some LoRA layers are enabled and others are disabled"
        )

    actual = bool(status.enabled)
    if actual != bool(enabled):
        expected = "enabled" if enabled else "disabled"
        got = "enabled" if actual else "disabled"
        raise RuntimeError(
            f"{label} expected LoRA adapters {expected}, but they are {got}"
        )


def set_frame_window(model, left_frames: Optional[int], right_frames: Optional[int]):
    base = base_model(model)
    previous = []
    if (left_frames is None) != (right_frames is None):
        raise ValueError(
            "left_frames and right_frames must both be None or both be integers"
        )
    if right_frames is not None and int(right_frames) != 0:
        raise ValueError("right_frames must be 0 for causal windowed attention")
    for block in base.blocks:
        block = unwrap_compiled_module(block)
        attn = block.self_attn
        previous.append((attn, getattr(attn, "frame_window", None)))
        attn.frame_window = (
            None if left_frames is None else (int(left_frames), int(right_frames))
        )
    return previous


@contextmanager
def attention_window(model, left_frames: Optional[int], right_frames: Optional[int]):
    previous = set_frame_window(model, left_frames, right_frames)
    try:
        yield
    finally:
        for attn, old_value in previous:
            attn.frame_window = old_value


@dataclass(frozen=True)
class LatentSample:
    latents_path: Path
    metadata_path: Path
    nu: float
    rho: float
    latent_frame_count: int

    @property
    def name(self) -> str:
        if self.latents_path.name == "latents.safetensors":
            return self.latents_path.parent.name
        return self.latents_path.stem


def _metadata_number_from_prefix(
    metadata_path: Path,
    key: str,
    *,
    chunk_size: int = 8192,
    max_bytes: int = 262144,
) -> Optional[float]:
    pattern = re.compile(
        rb'"' + re.escape(key.encode("utf-8")) + rb'"\s*:\s*([-+0-9.eE]+)'
    )
    buffer = b""
    with Path(metadata_path).open("rb") as handle:
        while len(buffer) < max_bytes:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            match = pattern.search(buffer)
            if match:
                return float(match.group(1))
    return None


def read_metadata_physics_fast(
    metadata_path: Path, require_rho: bool = REQUIRE_RHO
) -> Optional[tuple[float, float]]:
    nu = _metadata_number_from_prefix(metadata_path, "nu")
    if nu is None:
        return None
    rho = _metadata_number_from_prefix(metadata_path, "rho")
    if rho is None:
        if require_rho:
            return None
        rho = 1.0
    return float(nu), float(rho)


def read_latent_frame_count(latents_path: Path) -> int:
    with safe_open(str(latents_path), framework="pt", device="cpu") as handle:
        if "latents" in handle.keys():
            key = "latents"
        else:
            keys = list(handle.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"{latents_path} must contain a single tensor or a `latents` tensor; got keys={keys}"
                )
            key = keys[0]
        shape = tuple(handle.get_slice(key).get_shape())
    if len(shape) < 2:
        raise ValueError(
            f"{latents_path} tensor {key!r} must have a frame dimension; got shape={shape}"
        )
    return int(shape[1])


def _discovery_log(message: str) -> None:
    if TRAIN_INIT_LOGS:
        print(f"[sample discovery] {message}", flush=True)


def discover_latent_samples(
    latent_root: Path = LATENT_ROOT, require_rho: bool = REQUIRE_RHO
):
    samples = []
    latents_paths = sorted(Path(latent_root).glob("*/latents.safetensors"))
    _discovery_log(
        f"checking {len(latents_paths):,} latent files in {latent_root} "
        f"(min latent frames={MIN_LATENT_FRAME_COUNT})"
    )
    start_time = time.perf_counter()
    skipped_metadata = 0
    skipped_short = 0
    skipped_bad_latents = 0
    for index, latents_path in enumerate(latents_paths, start=1):
        checked_count = index - 1
        if TRAIN_INIT_LOGS and checked_count > 0 and checked_count % 500 == 0:
            elapsed = time.perf_counter() - start_time
            _discovery_log(
                f"checked {checked_count:,}/{len(latents_paths):,}; kept {len(samples):,}; elapsed {elapsed:.1f}s"
            )
        try:
            latent_frame_count = read_latent_frame_count(latents_path)
        except Exception:
            skipped_bad_latents += 1
            continue
        if (
            MIN_LATENT_FRAME_COUNT is not None
            and latent_frame_count < int(MIN_LATENT_FRAME_COUNT)
        ):
            skipped_short += 1
            continue
        metadata_path = latents_path.parent / "metadata.json"
        if not metadata_path.exists():
            skipped_metadata += 1
            continue
        physics = read_metadata_physics_fast(
            metadata_path, require_rho=bool(require_rho)
        )
        if physics is None:
            skipped_metadata += 1
            continue
        nu, rho = physics
        samples.append(
            LatentSample(
                latents_path=latents_path,
                metadata_path=metadata_path,
                nu=nu,
                rho=rho,
                latent_frame_count=latent_frame_count,
            )
        )
    _discovery_log(
        f"done in {time.perf_counter() - start_time:.1f}s; kept {len(samples):,}; "
        f"skipped_short={skipped_short:,}; skipped_bad_latents={skipped_bad_latents:,}; "
        f"skipped_metadata={skipped_metadata:,}"
    )
    if len(samples) <= HOLDOUT_COUNT:
        raise ValueError(
            f"Need more than {HOLDOUT_COUNT} latent files with rho/nu metadata, found {len(samples)} in {latent_root}"
        )
    return samples


def make_splits(samples, holdout_count: int = HOLDOUT_COUNT, seed: int = RANDOM_SEED):
    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[holdout_count:], samples[:holdout_count]


def physics_stats(samples):
    vals = torch.tensor(
        [[math.log(sample.nu), math.log(sample.rho)] for sample in samples],
        dtype=torch.float32,
    )
    mean = vals.mean(dim=0)
    std = vals.std(dim=0).clamp_min(1e-6)
    return mean, std


def sample_physics(
    sample: LatentSample, physics_mean: torch.Tensor, physics_std: torch.Tensor
) -> torch.Tensor:
    physics = torch.tensor(
        [math.log(sample.nu), math.log(sample.rho)], dtype=torch.float32
    )
    return (physics - physics_mean.float()) / physics_std.float()


def truncate_latent_frames(
    latents: torch.Tensor, frame_count: Optional[int] = None
) -> torch.Tensor:
    frame_count = LATENT_FRAME_COUNT if frame_count is None else frame_count
    if frame_count is None:
        return latents
    frame_count = int(frame_count)
    if frame_count < 1:
        raise ValueError("LATENT_FRAME_COUNT must be positive or None")
    return latents[:, :frame_count].contiguous()


def load_sample_latents(sample: LatentSample) -> torch.Tensor:
    return load_file(str(sample.latents_path))["latents"].float()


def stack_training_latents(latents) -> torch.Tensor:
    if isinstance(latents, torch.Tensor):
        return truncate_latent_frames(latents)
    return torch.stack([truncate_latent_frames(latent) for latent in latents], dim=0)


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, samples, physics_mean, physics_std):
        self.samples = list(samples)
        self.physics_mean = physics_mean.float()
        self.physics_std = physics_std.float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        latents = load_sample_latents(sample)
        physics = sample_physics(sample, self.physics_mean, self.physics_std)
        return {"latents": latents, "physics": physics, "sample_index": index}


def collate_batch(items):
    return {
        "latents": [item["latents"] for item in items],
        "physics": torch.stack([item["physics"] for item in items], dim=0),
        "sample_index": torch.tensor(
            [item["sample_index"] for item in items], dtype=torch.long
        ),
    }


def latent_patch_grid(
    latents: torch.Tensor,
    patch_size=(1, 2, 2),
    *,
    require_one_frame_tokens: bool = False,
):
    _, _, frames, height, width = latents.shape
    patch_f, patch_h, patch_w = patch_size
    patch_f, patch_h, patch_w = int(patch_f), int(patch_h), int(patch_w)
    if require_one_frame_tokens and patch_f != 1:
        raise ValueError(
            f"Frame-window attention expects temporal patch_size[0] == 1 so each token belongs to one latent frame; got {patch_size}"
        )
    if frames % patch_f != 0 or height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(
            f"Latent shape {tuple(latents.shape)} is not divisible by Wan patch_size {patch_size}"
        )
    return frames // patch_f, height // patch_h, width // patch_w


def latent_seq_len(latents: torch.Tensor, patch_size=(1, 2, 2)) -> int:
    grid_f, grid_h, grid_w = latent_patch_grid(latents, patch_size)
    return grid_f * grid_h * grid_w


def spatial_tokens_per_latent_frame(latents: torch.Tensor, patch_size=(1, 2, 2)) -> int:
    _, grid_h, grid_w = latent_patch_grid(
        latents, patch_size, require_one_frame_tokens=True
    )
    return grid_h * grid_w


def assert_frame_window_alignment(latents: torch.Tensor, model) -> int:
    _validate_causal_window()
    spatial_tokens = spatial_tokens_per_latent_frame(
        latents, base_model(model).patch_size
    )
    left_frames, right_frames = causal_window()
    token_window = (left_frames + 1 + right_frames) * spatial_tokens
    if token_window % spatial_tokens != 0:
        raise AssertionError("Token window must be divisible by spatial token count")
    return spatial_tokens


def _text_cache_metadata():
    return {
        "positive_prompt": str(POSITIVE_PROMPT),
        "negative_prompt": str(NEGATIVE_PROMPT),
        "checkpoint_path": str(
            Path(WAN_CHECKPOINT_DIR) / "models_t5_umt5-xxl-enc-bf16.pth"
        ),
        "tokenizer_path": str(Path(WAN_CHECKPOINT_DIR) / "google/umt5-xxl"),
        "text_len": "512",
    }


def load_cached_prompt_contexts(cache_path: Path = TEXT_EMBED_CACHE_PATH):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    expected_metadata = _text_cache_metadata()
    with safe_open(str(cache_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            return None
        if not {"positive", "negative"}.issubset(set(handle.keys())):
            return None
        return {
            "positive": handle.get_tensor("positive").contiguous(),
            "negative": handle.get_tensor("negative").contiguous(),
        }


@torch.no_grad()
def compute_prompt_contexts(device: torch.device = DEVICE):
    T5EncoderModel = load_wan_t5_encoder_class()
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=str(
            Path(WAN_CHECKPOINT_DIR) / "models_t5_umt5-xxl-enc-bf16.pth"
        ),
        tokenizer_path=str(Path(WAN_CHECKPOINT_DIR) / "google/umt5-xxl"),
    )
    text_encoder.model.to(device)
    contexts = text_encoder([POSITIVE_PROMPT, NEGATIVE_PROMPT], device)
    contexts = {
        "positive": contexts[0].detach().to("cpu", dtype=torch.bfloat16).contiguous(),
        "negative": contexts[1].detach().to("cpu", dtype=torch.bfloat16).contiguous(),
    }
    text_encoder.model.cpu()
    del text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return contexts


def load_or_compute_prompt_contexts(
    cache_path: Path = TEXT_EMBED_CACHE_PATH, device: torch.device = DEVICE
):
    cache_path = Path(cache_path)
    contexts = load_cached_prompt_contexts(cache_path)
    if contexts is not None:
        print(f"Loaded cached T5 prompt embeddings from {cache_path}")
        return contexts

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Computing T5 prompt embeddings and caching them at {cache_path}")
    contexts = compute_prompt_contexts(device=device)
    save_file(contexts, str(cache_path), metadata=_text_cache_metadata())
    return contexts


def prepare_prompt_contexts_for_device(
    contexts, device: torch.device = DEVICE, dtype: torch.dtype = TRAIN_DTYPE
):
    return {
        key: value.to(device=device, dtype=dtype).contiguous()
        for key, value in contexts.items()
    }


def prompt_context(batch_size: int, contexts, key: str = "positive"):
    context = contexts[key]
    return [context for _ in range(batch_size)]


def make_wan_scheduler(
    steps: int = EVAL_INFERENCE_STEPS,
    solver: str = WAN_SAMPLE_SOLVER,
    shift: float = WAN_SAMPLE_SHIFT,
    device: torch.device = DEVICE,
):
    solver = str(solver).lower()
    if solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=WAN_NUM_TRAIN_TIMESTEPS,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(steps, device=device, shift=shift)
        return scheduler

    if solver in {"dpm++", "dpm"}:
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=WAN_NUM_TRAIN_TIMESTEPS,
            shift=1,
            use_dynamic_shifting=False,
        )
        sampling_sigmas = get_sampling_sigmas(steps, shift)
        retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)
        return scheduler

    raise NotImplementedError(f"Unsupported Wan sample solver: {solver}")


def wan_inference_schedule(
    steps: int = EVAL_INFERENCE_STEPS,
    solver: str = WAN_SAMPLE_SOLVER,
    shift: float = WAN_SAMPLE_SHIFT,
    device: torch.device = DEVICE,
):
    scheduler = make_wan_scheduler(
        steps=steps, solver=solver, shift=shift, device=device
    )
    sigmas = scheduler.sigmas[:-1].to(device=device, dtype=torch.float32)
    timesteps = scheduler.timesteps.to(device=device)
    return sigmas, timesteps


def shift_scheduler_sigmas(
    raw_sigmas: torch.Tensor, shift: float = WAN_SAMPLE_SHIFT
) -> torch.Tensor:
    shift = float(shift)
    return shift * raw_sigmas / (1.0 + (shift - 1.0) * raw_sigmas)


def sample_wan_training_schedule(batch_size: int, device: torch.device = DEVICE):
    raw_sigmas = torch.rand(batch_size, device=device, dtype=torch.float32)
    sigmas = shift_scheduler_sigmas(raw_sigmas, shift=WAN_SAMPLE_SHIFT).clamp(
        1e-5, 1.0 - 1e-5
    )
    timesteps = sigmas * float(WAN_NUM_TRAIN_TIMESTEPS)
    return sigmas, timesteps


def sample_discrete_wan_inference_training_schedule(
    batch_size: int, device: torch.device = DEVICE
):
    """Old behavior: uniformly choose from the eval/inference schedule."""
    sigmas, timesteps = wan_inference_schedule(device=device)
    indices = torch.randint(0, sigmas.numel(), (batch_size,), device=device)
    return sigmas[indices].clamp(1e-5, 1.0 - 1e-5), timesteps[indices]


def apply_first_frame_condition(x: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x[:, :, 0:1] = clean[:, :, 0:1]
    return x


def first_frame_condition_timesteps(
    latents: torch.Tensor, timesteps: torch.Tensor, patch_size
) -> torch.Tensor:
    """Wan TI2V-style token timesteps: first latent frame is conditioning at t=0."""
    batch_size = latents.shape[0]
    grid_f, grid_h, grid_w = latent_patch_grid(
        latents, patch_size, require_one_frame_tokens=True
    )
    token_timesteps = timesteps.to(device=latents.device).view(batch_size, 1, 1, 1)
    token_timesteps = token_timesteps.expand(batch_size, grid_f, grid_h, grid_w).clone()
    token_timesteps[:, 0] = 0
    return token_timesteps.flatten(1)


class ErrorReplayBank:
    def __init__(
        self,
        bins: int = ERROR_RECYCLING_TIMESTEP_BINS,
        max_per_bin: int = ERROR_RECYCLING_BANK_SIZE,
    ):
        self.bins = int(bins)
        self.max_per_bin = int(max_per_bin)
        if self.bins < 1:
            raise ValueError("ERROR_RECYCLING_TIMESTEP_BINS must be positive")
        if self.max_per_bin < 1:
            raise ValueError("ERROR_RECYCLING_BANK_SIZE must be positive")
        self.latent_errors = [[] for _ in range(self.bins)]
        self.noise_errors = [[] for _ in range(self.bins)]

    def bin_indices(self, sigmas: torch.Tensor) -> list[int]:
        indices = (
            (sigmas.detach().float().cpu() * self.bins)
            .floor()
            .long()
            .clamp(0, self.bins - 1)
        )
        return [int(index) for index in indices.tolist()]

    def _append(self, bank, bin_index: int, value: torch.Tensor) -> None:
        entries = bank[int(bin_index)]
        entries.append(
            value.detach().to(device="cpu", dtype=torch.float16).contiguous()
        )
        if len(entries) > self.max_per_bin:
            del entries[0 : len(entries) - self.max_per_bin]

    def add(
        self,
        sigmas: torch.Tensor,
        latent_errors: torch.Tensor,
        noise_errors: torch.Tensor,
    ) -> None:
        latent_errors = latent_errors.detach().float().cpu()
        noise_errors = noise_errors.detach().float().cpu()
        latent_errors[:, :, 0:1] = 0.0
        noise_errors[:, :, 0:1] = 0.0
        for item_index, bin_index in enumerate(self.bin_indices(sigmas)):
            self._append(self.latent_errors, bin_index, latent_errors[item_index])
            self._append(self.noise_errors, bin_index, noise_errors[item_index])

    def _nearest_nonempty_bin(self, bank, bin_index: int) -> Optional[int]:
        if bank[bin_index]:
            return bin_index
        for offset in range(1, self.bins):
            left = bin_index - offset
            right = bin_index + offset
            if left >= 0 and bank[left]:
                return left
            if right < self.bins and bank[right]:
                return right
        return None

    def sample(
        self, sigmas: torch.Tensor, shape, *, device: torch.device, dtype: torch.dtype
    ):
        latent = torch.zeros(shape, device=device, dtype=dtype)
        noise = torch.zeros(shape, device=device, dtype=dtype)
        used = 0
        probability = float(ERROR_RECYCLING_PROB)
        scale = float(ERROR_RECYCLING_SCALE)
        for item_index, bin_index in enumerate(self.bin_indices(sigmas)):
            if random.random() >= probability:
                continue
            latent_bin = self._nearest_nonempty_bin(self.latent_errors, bin_index)
            noise_bin = self._nearest_nonempty_bin(self.noise_errors, bin_index)
            if latent_bin is None or noise_bin is None:
                continue
            latent_error = random.choice(self.latent_errors[latent_bin])
            noise_error = random.choice(self.noise_errors[noise_bin])
            if tuple(latent_error.shape) != tuple(shape[1:]) or tuple(
                noise_error.shape
            ) != tuple(shape[1:]):
                continue
            latent[item_index] = latent_error.to(device=device, dtype=dtype) * scale
            noise[item_index] = noise_error.to(device=device, dtype=dtype) * scale
            used += 1
        return latent, noise, used

    def __len__(self) -> int:
        return sum(len(entries) for entries in self.latent_errors)


def build_flow_matching_inputs(batch, model):
    clean = stack_training_latents(batch["latents"]).to(
        device=DEVICE, dtype=TRAIN_DTYPE
    )
    physics = batch["physics"].to(device=DEVICE, dtype=torch.float32)
    assert_frame_window_alignment(clean, model)

    batch_size = clean.shape[0]
    noise = torch.randn_like(clean)
    sigmas, timesteps = sample_wan_training_schedule(batch_size, device=DEVICE)
    view_shape = (batch_size,) + (1,) * (clean.ndim - 1)
    target = noise - clean
    target[:, :, 0:1] = 0.0
    return clean, physics, noise, sigmas, timesteps, view_shape, target


def velocity_frame_mse(pred_velocity: torch.Tensor, target_velocity: torch.Tensor):
    if pred_velocity.shape != target_velocity.shape:
        raise ValueError(
            f"pred_velocity and target_velocity shapes must match, got "
            f"{tuple(pred_velocity.shape)} and {tuple(target_velocity.shape)}"
        )
    if pred_velocity.ndim != 5:
        raise ValueError(
            f"Expected velocity tensors with shape [B, C, F, H, W], got {tuple(pred_velocity.shape)}"
        )
    return (pred_velocity.float() - target_velocity.float()).pow(2).mean(
        dim=(0, 1, 3, 4)
    )


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return
    bad_count = int((~torch.isfinite(tensor)).sum().detach().cpu())
    raise FloatingPointError(f"{name} contains {bad_count} non-finite values")


def windowed_model_velocity(
    model,
    x_t: torch.Tensor,
    timestep_tokens: torch.Tensor,
    contexts,
    seq_len: int,
    physics: torch.Tensor,
    first_frame: torch.Tensor,
):
    batch_size = x_t.shape[0]
    left_frames, right_frames = causal_window()
    with attention_window(model, left_frames, right_frames):
        with torch.amp.autocast(
            "cuda", dtype=TRAIN_DTYPE, enabled=DEVICE.type == "cuda"
        ):
            pred = model(
                [x_t[i] for i in range(batch_size)],
                timestep_tokens,
                context=contexts,
                seq_len=seq_len,
                physics=physics,
                first_frame=[first_frame[i] for i in range(batch_size)],
            )
            return torch.stack(pred, dim=0)


def model_velocity_with_attention(
    model,
    x_t: torch.Tensor,
    timestep_tokens: torch.Tensor,
    contexts,
    seq_len: int,
    physics: torch.Tensor,
    *,
    window_left_frames: Optional[int],
    window_right_frames: Optional[int],
    first_frame: torch.Tensor,
    delta_enabled: bool = False,
    ssm_enabled: bool = False,
):
    batch_size = x_t.shape[0]
    with delta_attention_enabled(model, delta_enabled):
        with ssm_attention_enabled(model, ssm_enabled):
            with attention_window(model, window_left_frames, window_right_frames):
                with torch.amp.autocast(
                    "cuda", dtype=TRAIN_DTYPE, enabled=DEVICE.type == "cuda"
                ):
                    pred = model(
                        [x_t[i] for i in range(batch_size)],
                        timestep_tokens,
                        context=contexts,
                        seq_len=seq_len,
                        physics=physics,
                        first_frame=[first_frame[i] for i in range(batch_size)],
                    )
                    return torch.stack(pred, dim=0)


def flow_matching_batch_error_recycling(
    batch, model, text_contexts, error_bank: ErrorReplayBank, step: int
):
    clean, physics, noise, sigmas, timesteps, view_shape, target = (
        build_flow_matching_inputs(batch, model)
    )
    batch_size = clean.shape[0]
    base = base_model(model)
    seq_len = latent_seq_len(clean, base.patch_size)
    timestep_tokens = first_frame_condition_timesteps(clean, timesteps, base.patch_size)
    contexts = prompt_context(batch_size, text_contexts, key="positive")

    latent_error = torch.zeros_like(clean)
    noise_error = torch.zeros_like(noise)
    recycled_count = 0
    if int(step) > int(ERROR_RECYCLING_WARMUP_STEPS):
        latent_error, noise_error, recycled_count = error_bank.sample(
            sigmas, clean.shape, device=DEVICE, dtype=TRAIN_DTYPE
        )

    recycled_clean = clean + latent_error
    recycled_noise = noise + noise_error
    x_t = (1.0 - sigmas.view(view_shape)) * recycled_clean + sigmas.view(
        view_shape
    ) * recycled_noise
    x_t = apply_first_frame_condition(x_t, clean)

    pred = windowed_model_velocity(
        model,
        x_t,
        timestep_tokens,
        contexts,
        seq_len,
        physics,
        clean[:, :, 0:1],
    )
    assert_finite_tensor("flow-matching prediction", pred)
    pred_velocity = pred[:, :, 1:].float()
    target_velocity = target[:, :, 1:].float()
    window_loss = F.mse_loss(pred_velocity, target_velocity)
    assert_finite_tensor("flow-matching loss", window_loss)
    frame_loss = velocity_frame_mse(pred_velocity, target_velocity).detach()

    with torch.no_grad():
        sigma_view = sigmas.view(view_shape)
        pred_float = pred.detach().float()
        x_float = x_t.detach().float()
        clean_float = clean.detach().float()
        noise_float = noise.detach().float()
        latent_endpoint_error = x_float - sigma_view * pred_float - clean_float
        noise_endpoint_error = x_float + (1.0 - sigma_view) * pred_float - noise_float
        assert_finite_tensor(
            "flow-matching latent endpoint error", latent_endpoint_error
        )
        assert_finite_tensor("flow-matching noise endpoint error", noise_endpoint_error)
        error_bank.add(sigmas, latent_endpoint_error, noise_endpoint_error)
        latent_error_norm = latent_endpoint_error[:, :, 1:].pow(2).mean()
        noise_error_norm = noise_endpoint_error[:, :, 1:].pow(2).mean()

    return window_loss, {
        "window": window_loss.detach(),
        "latent_err": latent_error_norm.detach(),
        "noise_err": noise_error_norm.detach(),
        "replay_used": torch.tensor(float(recycled_count) / max(1, batch_size)),
        "bank_fill": torch.tensor(float(len(error_bank))),
        "frame_loss": frame_loss,
        "total": window_loss.detach(),
    }


def flow_matching_batch(
    batch,
    model,
    text_contexts,
    *,
    error_bank: Optional[ErrorReplayBank] = None,
    step: int = 0,
):
    if error_bank is None:
        raise ValueError("flow_matching_batch requires an ErrorReplayBank")
    return flow_matching_batch_error_recycling(
        batch, model, text_contexts, error_bank, step
    )


def recurrent_flow_matching_batch(
    batch,
    student,
    text_contexts,
    *,
    recurrent_kind: str,
    error_bank: Optional[ErrorReplayBank] = None,
    step: int = 0,
):
    if recurrent_kind not in {"delta", "ssm"}:
        raise ValueError(f"Unknown recurrent kind: {recurrent_kind!r}")
    clean, physics, noise, sigmas, timesteps, view_shape, target = (
        build_flow_matching_inputs(batch, student)
    )
    batch_size = clean.shape[0]

    latent_error = torch.zeros_like(clean)
    noise_error = torch.zeros_like(noise)
    recycled_count = 0
    if error_bank is None:
        raise ValueError("recurrent_flow_matching_batch requires an ErrorReplayBank")
    if int(step) > int(ERROR_RECYCLING_WARMUP_STEPS):
        latent_error, noise_error, recycled_count = error_bank.sample(
            sigmas, clean.shape, device=DEVICE, dtype=TRAIN_DTYPE
        )

    recycled_clean = clean + latent_error
    recycled_noise = noise + noise_error
    x_t = (1.0 - sigmas.view(view_shape)) * recycled_clean + sigmas.view(
        view_shape
    ) * recycled_noise
    x_t = apply_first_frame_condition(x_t, clean)

    base = base_model(student)
    seq_len = latent_seq_len(clean, base.patch_size)
    timestep_tokens = first_frame_condition_timesteps(clean, timesteps, base.patch_size)
    contexts = prompt_context(batch_size, text_contexts, key="positive")
    left_frames, right_frames = causal_window()

    student_pred = model_velocity_with_attention(
        student,
        x_t,
        timestep_tokens,
        contexts,
        seq_len,
        physics,
        window_left_frames=left_frames,
        window_right_frames=right_frames,
        first_frame=clean[:, :, 0:1],
        delta_enabled=recurrent_kind == "delta",
        ssm_enabled=recurrent_kind == "ssm",
    )
    assert_finite_tensor("recurrent flow-matching prediction", student_pred)

    student_velocity = student_pred[:, :, 1:].float()
    target_velocity = target[:, :, 1:].float()
    window_loss = F.mse_loss(student_velocity, target_velocity)
    assert_finite_tensor("recurrent flow-matching loss", window_loss)
    frame_loss = velocity_frame_mse(student_velocity, target_velocity).detach()

    components = {
        "window": window_loss.detach(),
        "frame_loss": frame_loss,
        "total": window_loss.detach(),
    }
    with torch.no_grad():
        sigma_view = sigmas.view(view_shape)
        pred_float = student_pred.detach().float()
        x_float = x_t.detach().float()
        clean_float = clean.detach().float()
        noise_float = noise.detach().float()
        latent_endpoint_error = x_float - sigma_view * pred_float - clean_float
        noise_endpoint_error = x_float + (1.0 - sigma_view) * pred_float - noise_float
        assert_finite_tensor(
            "recurrent flow-matching latent endpoint error", latent_endpoint_error
        )
        assert_finite_tensor(
            "recurrent flow-matching noise endpoint error", noise_endpoint_error
        )
        error_bank.add(sigmas, latent_endpoint_error, noise_endpoint_error)
        components.update(
            {
                "latent_err": latent_endpoint_error[:, :, 1:].pow(2).mean(),
                "noise_err": noise_endpoint_error[:, :, 1:].pow(2).mean(),
                "replay_used": torch.tensor(float(recycled_count) / max(1, batch_size)),
                "bank_fill": torch.tensor(float(len(error_bank))),
            }
        )
    return window_loss, components


def delta_flow_matching_batch(
    batch,
    student,
    text_contexts,
    *,
    error_bank: Optional[ErrorReplayBank] = None,
    step: int = 0,
):
    return recurrent_flow_matching_batch(
        batch,
        student,
        text_contexts,
        recurrent_kind="delta",
        error_bank=error_bank,
        step=step,
    )


def ssm_flow_matching_batch(
    batch,
    student,
    text_contexts,
    *,
    error_bank: Optional[ErrorReplayBank] = None,
    step: int = 0,
):
    return recurrent_flow_matching_batch(
        batch,
        student,
        text_contexts,
        recurrent_kind="ssm",
        error_bank=error_bank,
        step=step,
    )


def _synchronize_if_cuda(device: torch.device = DEVICE) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_latent_shape(
    latent_template=None, sample: Optional[LatentSample] = None
):
    if latent_template is None:
        if sample is None:
            raise ValueError("Pass latent_template or sample to infer C/H/W")
        latent_template = load_sample_latents(sample)
    if not isinstance(latent_template, torch.Tensor):
        latent_template = torch.as_tensor(latent_template)
    if latent_template.dim() == 5:
        latent_template = latent_template[0]
    if latent_template.dim() != 4:
        raise ValueError(
            "latent_template must have shape [C, F, H, W] or [B, C, F, H, W]"
        )
    channels, _, height, width = latent_template.shape
    return int(channels), int(height), int(width)


def benchmark_forward_frame_counts(
    model,
    text_contexts,
    *,
    frame_counts=range(5, 41),
    latent_template=None,
    sample: Optional[LatentSample] = None,
    batch_size: int = 1,
    warmup: int = 2,
    repeats: int = 5,
    sigma: float = 0.5,
    physics: Optional[torch.Tensor] = None,
    delta_enabled: Optional[bool] = None,
    ssm_enabled: Optional[bool] = None,
    window_left_frames: Optional[int] = None,
    window_right_frames: Optional[int] = None,
    requires_grad: bool = False,
    train_mode: bool = False,
    seed: int = RANDOM_SEED,
):
    """Benchmark one model forward over different latent frame counts.

    The warmup passes are reported separately because torch.compile/regional
    compilation can make the first run for each sequence length much slower.
    """
    if text_contexts is None:
        raise ValueError("text_contexts are required for Wan forward benchmarking")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if int(warmup) < 0 or int(repeats) < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")

    base = base_model(model)
    channels, height, width = _benchmark_latent_shape(
        latent_template=latent_template, sample=sample
    )
    frame_counts = [int(frame_count) for frame_count in frame_counts]
    if any(frame_count < 1 for frame_count in frame_counts):
        raise ValueError("frame_counts must all be positive")

    old_training = model.training
    model.train(bool(train_mode))
    left_frames = (
        WINDOW_LEFT_FRAMES if window_left_frames is None else window_left_frames
    )
    right_frames = 0 if window_right_frames is None else window_right_frames
    use_delta = (
        has_delta_attention(model) if delta_enabled is None else bool(delta_enabled)
    )
    use_ssm = has_ssm_attention(model) if ssm_enabled is None else bool(ssm_enabled)
    if use_delta and use_ssm:
        raise ValueError("delta_enabled and ssm_enabled cannot both be true")

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(int(seed))
    contexts = prompt_context(int(batch_size), text_contexts, key="positive")
    grad_context = torch.enable_grad if requires_grad else torch.no_grad
    results = []

    try:
        for frame_count in frame_counts:
            clean = torch.randn(
                int(batch_size),
                channels,
                frame_count,
                height,
                width,
                device=DEVICE,
                dtype=TRAIN_DTYPE,
                generator=generator,
            )
            noise = torch.randn(
                clean.shape,
                device=DEVICE,
                dtype=TRAIN_DTYPE,
                generator=generator,
            )
            sigmas = torch.full(
                (int(batch_size),), float(sigma), device=DEVICE, dtype=torch.float32
            )
            timesteps = sigmas * float(WAN_NUM_TRAIN_TIMESTEPS)
            view_shape = (int(batch_size),) + (1,) * (clean.ndim - 1)
            x_t = (1.0 - sigmas.view(view_shape)) * clean + sigmas.view(
                view_shape
            ) * noise
            x_t = apply_first_frame_condition(x_t, clean)
            if requires_grad:
                x_t.requires_grad_(True)
            seq_len = latent_seq_len(clean, base.patch_size)
            timestep_tokens = first_frame_condition_timesteps(
                clean, timesteps, base.patch_size
            )
            if physics is None:
                physics_batch = torch.zeros(
                    int(batch_size), 2, device=DEVICE, dtype=torch.float32
                )
            else:
                physics_batch = physics.to(device=DEVICE, dtype=torch.float32)
                if physics_batch.dim() == 1:
                    physics_batch = physics_batch.unsqueeze(0)
                if physics_batch.shape[0] == 1 and int(batch_size) > 1:
                    physics_batch = physics_batch.expand(int(batch_size), -1)
                if tuple(physics_batch.shape) != (int(batch_size), 2):
                    raise ValueError(
                        f"physics must broadcast to ({int(batch_size)}, 2), "
                        f"got {tuple(physics_batch.shape)}"
                    )

            def run_forward():
                with grad_context():
                    pred = model_velocity_with_attention(
                        model,
                        x_t,
                        timestep_tokens,
                        contexts,
                        seq_len,
                        physics_batch,
                        window_left_frames=left_frames,
                        window_right_frames=right_frames,
                        first_frame=clean[:, :, 0:1],
                        delta_enabled=use_delta,
                        ssm_enabled=use_ssm,
                    )
                    if requires_grad:
                        pred.float().mean()
                    return pred

            warmup_times = []
            for _ in range(int(warmup)):
                _synchronize_if_cuda(DEVICE)
                start = time.perf_counter()
                output = run_forward()
                _synchronize_if_cuda(DEVICE)
                warmup_times.append((time.perf_counter() - start) * 1000.0)
                del output

            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats(DEVICE)
            gpu_times = []
            wall_times = []
            for _ in range(int(repeats)):
                if DEVICE.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    _synchronize_if_cuda(DEVICE)
                    wall_start = time.perf_counter()
                    start_event.record()
                    output = run_forward()
                    end_event.record()
                    _synchronize_if_cuda(DEVICE)
                    wall_times.append((time.perf_counter() - wall_start) * 1000.0)
                    gpu_times.append(float(start_event.elapsed_time(end_event)))
                else:
                    wall_start = time.perf_counter()
                    output = run_forward()
                    wall_times.append((time.perf_counter() - wall_start) * 1000.0)
                del output

            peak_allocated_mib = None
            peak_reserved_mib = None
            if DEVICE.type == "cuda":
                peak_allocated_mib = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
                peak_reserved_mib = torch.cuda.max_memory_reserved(DEVICE) / (1024**2)

            timing_source = "cuda_event" if gpu_times else "wall"
            times = gpu_times if gpu_times else wall_times
            results.append(
                {
                    "frames": frame_count,
                    "seq_len": int(seq_len),
                    "batch_size": int(batch_size),
                    "delta_enabled": bool(use_delta),
                    "ssm_enabled": bool(use_ssm),
                    "train_mode": bool(train_mode),
                    "requires_grad": bool(requires_grad),
                    "warmup_ms_total": sum(warmup_times),
                    "warmup_ms_last": warmup_times[-1] if warmup_times else None,
                    "forward_ms_mean": sum(times) / len(times),
                    "forward_ms_min": min(times),
                    "forward_ms_max": max(times),
                    "wall_ms_mean": sum(wall_times) / len(wall_times),
                    "timing_source": timing_source,
                    "peak_allocated_mib": peak_allocated_mib,
                    "peak_reserved_mib": peak_reserved_mib,
                }
            )

            del clean, noise, x_t, timestep_tokens, physics_batch
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        model.train(old_training)

    return results


def print_forward_benchmark_results(results) -> None:
    if not results:
        print("No benchmark results.")
        return
    columns = [
        "frames",
        "seq_len",
        "forward_ms_mean",
        "forward_ms_min",
        "forward_ms_max",
        "warmup_ms_total",
        "peak_allocated_mib",
    ]
    header = " | ".join(columns)
    print(header)
    print(" | ".join("---" for _ in columns))
    for row in results:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            elif value is None:
                values.append("-")
            else:
                values.append(str(value))
        print(" | ".join(values))


def trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def resolve_max_steps(max_steps: Optional[int] = None) -> int:
    max_steps = MAX_STEPS if max_steps is None else max_steps
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    return max_steps


def learning_rate_for_step(step: int) -> float:
    if LR_WARMUP_STEPS > 0 and step <= LR_WARMUP_STEPS:
        return float(LEARNING_RATE) * max(0.0, float(step) / float(LR_WARMUP_STEPS))
    if MAX_STEPS <= LR_WARMUP_STEPS:
        return float(LEARNING_RATE)
    decay_progress = (float(step) - float(LR_WARMUP_STEPS)) / float(
        MAX_STEPS - LR_WARMUP_STEPS
    )
    decay_progress = min(1.0, max(0.0, decay_progress))
    return float(LEARNING_RATE) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def load_wan_vae(dtype=torch.float32, device=DEVICE):
    Wan2_2_VAE = load_wan_vae_class()
    return Wan2_2_VAE(
        vae_pth=str(WAN_CHECKPOINT_DIR / "Wan2.2_VAE.pth"),
        dtype=dtype,
        device=device,
    )


@torch.no_grad()
def infer_latents_from_first_frame(
    model,
    clean_latents: torch.Tensor,
    *,
    text_contexts,
    physics: torch.Tensor,
    window_left_frames: Optional[int],
    window_right_frames: Optional[int],
    initial_noise: Optional[torch.Tensor] = None,
    steps: int = EVAL_INFERENCE_STEPS,
    guide_scale: float = WAN_SAMPLE_GUIDE_SCALE,
):
    was_training = model.training
    model.eval()

    clean = clean_latents.to(device=DEVICE, dtype=TRAIN_DTYPE)
    if initial_noise is None:
        z = torch.randn(clean.shape, device=DEVICE, dtype=TRAIN_DTYPE)
    else:
        z = initial_noise.to(device=DEVICE, dtype=TRAIN_DTYPE).clone()
    z[:, 0:1] = clean[:, 0:1]

    base = base_model(model)
    clean_batch = clean.unsqueeze(0)
    assert_frame_window_alignment(clean_batch, model)
    seq_len = latent_seq_len(clean_batch, base.patch_size)
    positive_context = prompt_context(1, text_contexts, key="positive")
    negative_context = prompt_context(1, text_contexts, key="negative")
    physics = physics.to(device=DEVICE, dtype=torch.float32).view(1, 2)
    solver = str(WAN_SAMPLE_SOLVER).lower()
    scheduler = make_wan_scheduler(steps=steps, solver=solver, device=DEVICE)

    with attention_window(model, window_left_frames, window_right_frames):
        for step_index, t in enumerate(scheduler.timesteps):
            timestep = torch.stack([t]).to(device=DEVICE)
            timestep_tokens = first_frame_condition_timesteps(
                clean_batch, timestep, base.patch_size
            )
            with torch.amp.autocast(
                "cuda", dtype=TRAIN_DTYPE, enabled=DEVICE.type == "cuda"
            ):
                pred_positive = model(
                    [z],
                    timestep_tokens,
                    context=positive_context,
                    seq_len=seq_len,
                    physics=physics,
                    first_frame=[clean[:, 0:1]],
                )[0]
                if float(guide_scale) == 1.0:
                    pred = pred_positive
                else:
                    pred_negative = model(
                        [z],
                        timestep_tokens,
                        context=negative_context,
                        seq_len=seq_len,
                        physics=physics,
                        first_frame=[clean[:, 0:1]],
                    )[0]
                    pred = pred_negative + float(guide_scale) * (
                        pred_positive - pred_negative
                    )
            if not torch.isfinite(pred).all():
                bad_count = int((~torch.isfinite(pred)).sum().detach().cpu())
                raise FloatingPointError(
                    f"non-finite model prediction at inference step {step_index}, "
                    f"timestep={float(t.detach().float().cpu())}, bad_values={bad_count}, "
                    f"solver={WAN_SAMPLE_SOLVER}, guide={float(guide_scale):g}"
                )
            z = (
                scheduler.step(
                    pred.unsqueeze(0),
                    t,
                    z.unsqueeze(0),
                    return_dict=False,
                )[0]
                .squeeze(0)
                .to(dtype=TRAIN_DTYPE)
            )
            if not torch.isfinite(z).all():
                bad_count = int((~torch.isfinite(z)).sum().detach().cpu())
                raise FloatingPointError(
                    f"non-finite latent sample after scheduler step {step_index}, "
                    f"timestep={float(t.detach().float().cpu())}, bad_values={bad_count}, "
                    f"solver={WAN_SAMPLE_SOLVER}, guide={float(guide_scale):g}"
                )
            z[:, 0:1] = clean[:, 0:1]

    if was_training:
        model.train()
    return z.float().cpu()


@torch.no_grad()
def diagnose_inference_nans(
    model,
    state: dict,
    *,
    sample: Optional[LatentSample] = None,
    seed: int = 0,
    steps: Optional[int] = None,
    use_forward_hooks: bool = True,
    print_report: bool = True,
) -> dict:
    """Trace latent inference and report the first non-finite source.

    This mirrors ``infer_latents_from_first_frame`` but records tensor ranges at
    every sampling step and, when hooks are enabled, identifies the first module
    whose forward output becomes non-finite.
    """
    if state is None:
        raise ValueError("Pass the training state returned by awl.train()")
    text_contexts = state["text_contexts"]
    physics_mean = state["physics_mean"]
    physics_std = state["physics_std"]
    holdout_samples = state["holdout_samples"]
    if sample is None:
        sample = holdout_samples[0]

    steps = EVAL_INFERENCE_STEPS if steps is None else int(steps)
    was_training = model.training
    model.eval()

    full_clean_latents = load_sample_latents(sample)
    clean_latents = truncate_latent_frames(full_clean_latents)
    clean = clean_latents.to(device=DEVICE, dtype=TRAIN_DTYPE)
    physics = sample_physics(sample, physics_mean, physics_std).to(
        device=DEVICE, dtype=torch.float32
    )

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(int(seed))
    z = torch.randn(clean.shape, device=DEVICE, dtype=TRAIN_DTYPE, generator=generator)
    z[:, 0:1] = clean[:, 0:1]

    base = base_model(model)
    clean_batch = clean.unsqueeze(0)
    assert_frame_window_alignment(clean_batch, model)
    seq_len = latent_seq_len(clean_batch, base.patch_size)
    positive_context = prompt_context(1, text_contexts, key="positive")
    negative_context = prompt_context(1, text_contexts, key="negative")
    physics = physics.view(1, 2)
    guide_scale = float(WAN_SAMPLE_GUIDE_SCALE)
    solver = str(WAN_SAMPLE_SOLVER).lower()
    scheduler = make_wan_scheduler(steps=steps, solver=solver, device=DEVICE)
    left_frames, right_frames = causal_window()
    recurrent_kind = recurrent_attention_kind(model)
    previous_recurrent = None
    if recurrent_kind is not None:
        previous_recurrent = set_recurrent_attention_enabled(
            model, recurrent_kind, True
        )

    records = []
    result = {
        "sample": sample.name,
        "seed": int(seed),
        "solver": WAN_SAMPLE_SOLVER,
        "guide_scale": guide_scale,
        "steps": steps,
        "recurrent_kind": recurrent_kind,
        "parameter_diagnostics": recurrent_parameter_diagnostics(model),
        "records": records,
        "first_failure": None,
    }

    trace_context = (
        first_nonfinite_forward_trace(model)
        if use_forward_hooks
        else nullcontext({"first": None})
    )
    try:
        with trace_context as module_trace:
            with attention_window(model, left_frames, right_frames):
                for step_index, t in enumerate(scheduler.timesteps):
                    timestep = torch.stack([t]).to(device=DEVICE)
                    timestep_tokens = first_frame_condition_timesteps(
                        clean_batch, timestep, base.patch_size
                    )
                    record = {
                        "step": step_index,
                        "timestep": float(t.detach().float().cpu()),
                        "z_before": tensor_finite_stats("z_before", z),
                    }
                    records.append(record)
                    with torch.amp.autocast(
                        "cuda", dtype=TRAIN_DTYPE, enabled=DEVICE.type == "cuda"
                    ):
                        pred_positive = model(
                            [z],
                            timestep_tokens,
                            context=positive_context,
                            seq_len=seq_len,
                            physics=physics,
                            first_frame=[clean[:, 0:1]],
                        )[0]
                        if guide_scale == 1.0:
                            pred = pred_positive
                        else:
                            pred_negative = model(
                                [z],
                                timestep_tokens,
                                context=negative_context,
                                seq_len=seq_len,
                                physics=physics,
                                first_frame=[clean[:, 0:1]],
                            )[0]
                            pred = pred_negative + guide_scale * (
                                pred_positive - pred_negative
                            )
                    record["pred_positive"] = tensor_finite_stats(
                        "pred_positive", pred_positive
                    )
                    record["pred"] = tensor_finite_stats("pred", pred)
                    if not record["pred"]["finite"]:
                        result["first_failure"] = {
                            "kind": "model_prediction",
                            "step": step_index,
                            "timestep": record["timestep"],
                            "module": module_trace["first"],
                            "record": record,
                        }
                        break

                    z = (
                        scheduler.step(
                            pred.unsqueeze(0),
                            t,
                            z.unsqueeze(0),
                            return_dict=False,
                        )[0]
                        .squeeze(0)
                        .to(dtype=TRAIN_DTYPE)
                    )
                    record["z_after"] = tensor_finite_stats("z_after", z)
                    if not record["z_after"]["finite"]:
                        result["first_failure"] = {
                            "kind": "scheduler_step",
                            "step": step_index,
                            "timestep": record["timestep"],
                            "module": module_trace["first"],
                            "record": record,
                        }
                        break
                    z[:, 0:1] = clean[:, 0:1]
                    record["z_after_condition"] = tensor_finite_stats(
                        "z_after_condition", z
                    )
    finally:
        if previous_recurrent is not None:
            for attn, old_value in previous_recurrent:
                if recurrent_kind == "ssm":
                    attn.ssm_attention_enabled = old_value
                elif recurrent_kind == "delta":
                    attn.delta_attention_enabled = old_value
        if was_training:
            model.train()

    if result["first_failure"] is None:
        result["final_latents"] = tensor_finite_stats("final_latents", z)

    if print_report:
        print(
            f"inference NaN diagnostic: sample={result['sample']} seed={result['seed']} "
            f"solver={result['solver']} guide={result['guide_scale']:g} "
            f"recurrent={result['recurrent_kind']}"
        )
        for diag in result["parameter_diagnostics"][:3]:
            print(f"block {diag['block']} {diag['kind']} parameters:")
            for key, value in diag.items():
                if isinstance(value, dict):
                    print("  " + format_finite_stats(value))
        if len(result["parameter_diagnostics"]) > 3:
            print(
                f"  ... {len(result['parameter_diagnostics']) - 3} more recurrent blocks"
            )
        for record in records[-5:]:
            print(
                f"step {record['step']} t={record['timestep']:.4g} | "
                + format_finite_stats(record["z_before"])
            )
            if "pred" in record:
                print("  " + format_finite_stats(record["pred"]))
            if "z_after" in record:
                print("  " + format_finite_stats(record["z_after"]))
        if result["first_failure"] is None:
            print("no non-finite values found")
            print(format_finite_stats(result["final_latents"]))
        else:
            failure = result["first_failure"]
            print(
                f"first failure: {failure['kind']} at step={failure['step']} "
                f"t={failure['timestep']:.4g}"
            )
            if failure["module"] is not None:
                print("first non-finite module:", failure["module"])
    return result


@torch.no_grad()
def decode_latents_to_video(vae, latents: torch.Tensor):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*"
        )
        video = vae.decode([latents.to(device=DEVICE, dtype=torch.float32)])[0]
    return video.detach().float().cpu().clamp(-1, 1)


def tensor_range_summary(name: str, tensor: torch.Tensor) -> str:
    values = tensor.detach().float()
    return (
        f"{name}: min={values.amin().item():.3f}, "
        f"max={values.amax().item():.3f}, "
        f"mean={values.mean().item():.3f}, "
        f"std={values.std(unbiased=False).item():.3f}"
    )


def tensor_finite_stats(name: str, tensor: torch.Tensor) -> dict:
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    stats = {
        "name": name,
        "shape": tuple(values.shape),
        "finite": bool(finite.all().detach().cpu()),
        "bad_values": int((~finite).sum().detach().cpu()),
    }
    if bool(finite.any().detach().cpu()):
        finite_values = values[finite]
        stats.update(
            {
                "min": float(finite_values.amin().detach().cpu()),
                "max": float(finite_values.amax().detach().cpu()),
                "absmax": float(finite_values.abs().amax().detach().cpu()),
                "mean": float(finite_values.mean().detach().cpu()),
                "std": float(finite_values.std(unbiased=False).detach().cpu()),
            }
        )
    return stats


def format_finite_stats(stats: dict) -> str:
    if "min" not in stats:
        return (
            f"{stats['name']}: shape={stats['shape']} finite={stats['finite']} "
            f"bad={stats['bad_values']}"
        )
    return (
        f"{stats['name']}: shape={stats['shape']} finite={stats['finite']} "
        f"bad={stats['bad_values']} min={stats['min']:.4g} max={stats['max']:.4g} "
        f"absmax={stats['absmax']:.4g} mean={stats['mean']:.4g} std={stats['std']:.4g}"
    )


def iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)


@contextmanager
def first_nonfinite_forward_trace(model):
    trace = {"first": None}
    handles = []

    def make_hook(module_name: str, module: nn.Module):
        def hook(_module, _inputs, output):
            if trace["first"] is not None:
                return
            for tensor_index, tensor in enumerate(iter_tensors(output)):
                if torch.isfinite(tensor).all():
                    continue
                trace["first"] = {
                    "module": module_name,
                    "module_type": type(module).__name__,
                    "tensor_index": tensor_index,
                    "output": tensor_finite_stats(
                        f"{module_name}.output[{tensor_index}]", tensor
                    ),
                }
                return

        return hook

    try:
        for module_name, module in model.named_modules():
            if not module_name:
                continue
            handles.append(module.register_forward_hook(make_hook(module_name, module)))
        yield trace
    finally:
        for handle in handles:
            handle.remove()


def recurrent_parameter_diagnostics(model) -> list[dict]:
    diagnostics = []
    base = base_model(model)
    for block_index, block in enumerate(base.blocks):
        block = unwrap_compiled_module(block)
        attn = block.self_attn
        ssm_memory = getattr(attn, "ssm_memory", None)
        if ssm_memory is not None:
            diagnostics.append(
                {
                    "block": block_index,
                    "kind": "ssm",
                    "rho": tensor_finite_stats(
                        f"blocks.{block_index}.self_attn.ssm_memory.rho",
                        ssm_memory.rho,
                    ),
                    "decay": tensor_finite_stats(
                        f"blocks.{block_index}.self_attn.ssm_memory.decay",
                        torch.sigmoid(ssm_memory.decay_logit),
                    ),
                    "input_gain": tensor_finite_stats(
                        f"blocks.{block_index}.self_attn.ssm_memory.input_gain",
                        torch.sigmoid(ssm_memory.input_logit),
                    ),
                    "output_gain": tensor_finite_stats(
                        f"blocks.{block_index}.self_attn.ssm_memory.output_gain",
                        ssm_memory.output_gain,
                    ),
                    "skip_gain": tensor_finite_stats(
                        f"blocks.{block_index}.self_attn.ssm_memory.skip_gain",
                        ssm_memory.skip_gain,
                    ),
                }
            )
        delta_memory = getattr(attn, "delta_memory", None)
        if delta_memory is not None:
            diagnostics.append(
                {
                    "block": block_index,
                    "kind": "delta",
                    "rho": tensor_finite_stats(
                        f"blocks.{block_index}.self_attn.delta_memory.rho",
                        delta_memory.rho,
                    ),
                    "beta": tensor_finite_stats(
                        f"blocks.{block_index}.self_attn.delta_memory.beta",
                        torch.sigmoid(delta_memory.beta_logit),
                    ),
                    "lambda": tensor_finite_stats(
                        f"blocks.{block_index}.self_attn.delta_memory.lambda",
                        torch.sigmoid(delta_memory.lambda_logit),
                    ),
                }
            )
    return diagnostics


def video_cthw_to_uint8_frames(
    video: torch.Tensor, output_size: int = VIDEO_OUTPUT_SIZE
) -> torch.Tensor:
    frames = ((video.float().clamp(-1, 1) + 1.0) * 0.5).permute(1, 0, 2, 3).contiguous()
    if frames.shape[-2:] != (int(output_size), int(output_size)):
        frames = F.interpolate(
            frames,
            size=(int(output_size), int(output_size)),
            mode="bilinear",
            align_corners=False,
        )
    return (
        (255.0 * frames.permute(0, 2, 3, 1)).round().clamp(0, 255).to(torch.uint8).cpu()
    )


def encode_rgb_frames_mp4(frames: torch.Tensor, fps: int = EVAL_FPS) -> bytes:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (time, height, width, 3)")
    frames = frames.detach().to(torch.uint8).cpu()
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        for frame in frames.numpy():
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _pad_video_frames(frames: torch.Tensor, frame_count: int) -> torch.Tensor:
    frame_count = int(frame_count)
    if frames.shape[0] >= frame_count:
        return frames[:frame_count]
    if frames.shape[0] < 1:
        raise ValueError("Cannot pad a video with zero frames")
    padding = frames[-1:].expand(frame_count - frames.shape[0], -1, -1, -1)
    return torch.cat([frames, padding], dim=0)


def error_to_uint8_frames(
    error: torch.Tensor, vmax: float = EVAL_ERROR_VMAX
) -> torch.Tensor:
    x = (error.float() / max(float(vmax), 1e-12)).clamp(0.0, 1.0)
    red = x
    green = (1.35 * x - 0.35).clamp(0.0, 1.0)
    blue = (2.0 * x - 1.5).clamp(0.0, 1.0)
    return (
        (255.0 * torch.stack((red, green, blue), dim=-1))
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
    )


def prediction_triplet_frames(
    pred_video,
    gt_video,
    output_size: int = VIDEO_OUTPUT_SIZE,
    frame_alignment: str = "shortest",
) -> torch.Tensor:
    pred = video_cthw_to_uint8_frames(pred_video, output_size=output_size)
    gt = video_cthw_to_uint8_frames(gt_video, output_size=output_size)
    error_frame_count = min(gt.shape[0], pred.shape[0])
    if error_frame_count < 1:
        raise ValueError("prediction and ground-truth videos need at least one frame")
    error = (
        (
            gt[:error_frame_count].float() / 255.0
            - pred[:error_frame_count].float() / 255.0
        )
        .abs()
        .mean(dim=-1)
    )
    err = error_to_uint8_frames(error)

    if frame_alignment == "shortest":
        frame_count = error_frame_count
        pred = pred[:frame_count]
        gt = gt[:frame_count]
        err = err[:frame_count]
    elif frame_alignment == "longest":
        frame_count = max(gt.shape[0], pred.shape[0])
        pred = _pad_video_frames(pred, frame_count)
        gt = _pad_video_frames(gt, frame_count)
        err = _pad_video_frames(err, frame_count)
    else:
        raise ValueError(
            "frame_alignment must be either 'shortest' or 'longest', "
            f"got {frame_alignment!r}"
        )
    return torch.cat((pred, gt, err), dim=2).contiguous()


def display_rgb_frames(frames: torch.Tensor, fps: int = EVAL_FPS):
    video_b64 = base64.b64encode(encode_rgb_frames_mp4(frames, fps=fps)).decode("ascii")
    height = int(frames.shape[1])
    width = int(frames.shape[2])
    html = HTML(
        f"<video autoplay loop muted playsinline controls "
        f'width="{width}" height="{height}" '
        f'style="display:block;width:{width}px;height:{height}px;margin:0;padding:0;border:0;line-height:0">'
        f'<source src="data:video/mp4;base64,{video_b64}" type="video/mp4">'
        f"</video>"
    )
    display(html)
    return html


def display_side_by_side_videos(
    *videos,
    output_size: int = VIDEO_OUTPUT_SIZE,
    fps: int = EVAL_FPS,
    frame_alignment: str = "shortest",
):
    if len(videos) < 1:
        raise ValueError("display_side_by_side_videos expects at least one video")
    video_frames = [
        video_cthw_to_uint8_frames(video, output_size=output_size) for video in videos
    ]
    if frame_alignment == "shortest":
        frame_count = min(frames.shape[0] for frames in video_frames)
        video_frames = [frames[:frame_count] for frames in video_frames]
    elif frame_alignment == "longest":
        frame_count = max(frames.shape[0] for frames in video_frames)
        video_frames = [
            _pad_video_frames(frames, frame_count) for frames in video_frames
        ]
    else:
        raise ValueError(
            "frame_alignment must be either 'shortest' or 'longest', "
            f"got {frame_alignment!r}"
        )
    frames = torch.cat(video_frames, dim=2).contiguous()
    return display_rgb_frames(frames, fps=fps)


def display_prediction_triplet(
    pred_video,
    gt_video,
    output_size: int = VIDEO_OUTPUT_SIZE,
    fps: int = EVAL_FPS,
    frame_alignment: str = "shortest",
):
    frames = prediction_triplet_frames(
        pred_video,
        gt_video,
        output_size=output_size,
        frame_alignment=frame_alignment,
    )
    return display_rgb_frames(frames, fps=fps)


def display_frame_loss_histogram(
    frame_loss_ema,
    frame_loss_indices=None,
    *,
    step: Optional[int] = None,
):
    if frame_loss_ema is None:
        return None
    values = torch.as_tensor(frame_loss_ema, dtype=torch.float32).detach().cpu()
    if values.numel() < 1:
        return None
    if frame_loss_indices is None:
        indices = list(range(1, int(values.numel()) + 1))
    else:
        indices = [int(index) for index in frame_loss_indices]
    if len(indices) != int(values.numel()):
        raise ValueError(
            f"frame_loss_indices length {len(indices)} does not match "
            f"frame_loss_ema length {int(values.numel())}"
        )
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn(f"Could not plot frame-loss histogram: {exc}")
        return None

    width = max(6.0, min(14.0, 0.45 * len(indices) + 3.0))
    fig, ax = plt.subplots(figsize=(width, 3.5))
    ax.bar(indices, values.numpy(), width=0.8)
    title = "Training loss EMA by latent frame"
    if step is not None:
        title = f"{title} at step {int(step)}"
    ax.set_title(title)
    ax.set_xlabel("latent frame index")
    ax.set_ylabel("velocity MSE EMA")
    ax.set_xticks(indices)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    display(fig)
    plt.close(fig)
    return fig


@torch.no_grad()
def evaluate_random_holdout(
    model,
    vae,
    holdout_samples,
    step: int,
    text_contexts,
    physics_mean,
    physics_std,
    *,
    frame_loss_ema=None,
    frame_loss_indices=None,
):
    sample = random.choice(holdout_samples)
    full_clean_latents = load_sample_latents(sample)
    clean_latents = truncate_latent_frames(full_clean_latents)
    physics = sample_physics(sample, physics_mean, physics_std)

    seed = random.randint(0, 2**31 - 1)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    initial_noise = torch.randn(
        clean_latents.shape, device=DEVICE, dtype=TRAIN_DTYPE, generator=generator
    )
    initial_noise[:, 0:1] = clean_latents.to(device=DEVICE, dtype=TRAIN_DTYPE)[:, 0:1]

    left_frames, right_frames = causal_window()
    recurrent_kind = recurrent_attention_kind(model)
    if recurrent_kind is not None:
        if lora_adapter_parameters(model):
            require_lora_adapter_state(
                model, enabled=True, label=f"{recurrent_kind} LoRA windowed inference"
            )
        set_recurrent_attention_enabled(model, recurrent_kind, True)
        eval_label = f"{recurrent_kind} causal window L={left_frames}"
    else:
        require_lora_adapter_state(model, enabled=True, label="LoRA windowed inference")
        eval_label = f"lora causal window L={left_frames}"
    lora_latents = infer_latents_from_first_frame(
        model,
        clean_latents,
        text_contexts=text_contexts,
        physics=physics,
        window_left_frames=left_frames,
        window_right_frames=right_frames,
        initial_noise=initial_noise,
        steps=EVAL_INFERENCE_STEPS,
    )

    lora_video = decode_latents_to_video(vae, lora_latents)
    ground_truth_video = decode_latents_to_video(vae, clean_latents)
    html = display_prediction_triplet(lora_video, ground_truth_video)
    print(
        f"step {step}: {sample.name} | "
        f"nu={sample.nu:.2e}, rho={sample.rho:.3g} | "
        f"left={eval_label}, solver={WAN_SAMPLE_SOLVER}, guide={WAN_SAMPLE_GUIDE_SCALE:g} | "
        "middle=ground truth | "
        "right=absolute error"
    )
    print(
        tensor_range_summary("pred_latents", lora_latents),
        "|",
        tensor_range_summary("gt_latents", clean_latents),
        "|",
        tensor_range_summary("pred_video", lora_video),
        "|",
        tensor_range_summary("gt_video", ground_truth_video),
    )
    frame_loss_histogram = display_frame_loss_histogram(
        frame_loss_ema,
        frame_loss_indices,
        step=step,
    )

    if LATENT_FRAME_COUNT is None:
        long_frame_count = int(full_clean_latents.shape[1])
    else:
        long_frame_count = int(LATENT_FRAME_COUNT) * 2
    long_condition_latents = clean_latents.new_zeros(
        clean_latents.shape[0],
        long_frame_count,
        clean_latents.shape[2],
        clean_latents.shape[3],
    )
    long_condition_latents[:, :1] = clean_latents[:, :1]
    long_seed = random.randint(0, 2**31 - 1)
    long_generator = torch.Generator(device=DEVICE)
    long_generator.manual_seed(long_seed)
    long_initial_noise = torch.randn(
        long_condition_latents.shape,
        device=DEVICE,
        dtype=TRAIN_DTYPE,
        generator=long_generator,
    )
    long_initial_noise[:, 0:1] = long_condition_latents.to(
        device=DEVICE, dtype=TRAIN_DTYPE
    )[:, 0:1]
    if recurrent_kind is not None:
        if lora_adapter_parameters(model):
            require_lora_adapter_state(
                model,
                enabled=True,
                label=f"{recurrent_kind} LoRA long windowed inference",
            )
        set_recurrent_attention_enabled(model, recurrent_kind, True)
    else:
        require_lora_adapter_state(
            model, enabled=True, label="LoRA long windowed inference"
        )
    long_windowed_latents = infer_latents_from_first_frame(
        model,
        long_condition_latents,
        text_contexts=text_contexts,
        physics=physics,
        window_left_frames=left_frames,
        window_right_frames=right_frames,
        initial_noise=long_initial_noise,
        steps=EVAL_INFERENCE_STEPS,
    )
    long_windowed_video = decode_latents_to_video(vae, long_windowed_latents)
    full_ground_truth_video = decode_latents_to_video(vae, full_clean_latents)
    long_video_frames = int(long_windowed_video.shape[1])
    long_ground_truth_video = full_ground_truth_video[
        :, :long_video_frames
    ].contiguous()
    html_long = display_prediction_triplet(
        long_windowed_video,
        long_ground_truth_video,
    )
    print(
        f"step {step}: {sample.name} | "
        f"left=long windowed rollout={long_frame_count} latent frames | "
        f"middle=ground truth truncated to {long_ground_truth_video.shape[1]} video frames "
        f"(from {full_ground_truth_video.shape[1]}) | "
        "right=absolute error | "
        f"solver={WAN_SAMPLE_SOLVER}, guide={WAN_SAMPLE_GUIDE_SCALE:g}, seed={long_seed}"
    )
    print(
        tensor_range_summary("long_pred_latents", long_windowed_latents),
        "|",
        tensor_range_summary("long_pred_video", long_windowed_video),
    )
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "sample": sample,
        "seed": seed,
        "long_seed": long_seed,
        "html": html,
        "html_long": html_long,
        "frame_loss_histogram": frame_loss_histogram,
    }


def build_training_objects():
    _validate_training_mode()
    _init_log(
        f"mode=flow_matching device={DEVICE} param_dtype={PARAM_DTYPE} train_dtype={TRAIN_DTYPE}"
    )
    with init_stage("discover latent samples"):
        samples = discover_latent_samples()
    _init_log(f"found {len(samples):,} samples in {LATENT_ROOT}")
    with init_stage("split dataset and compute physics stats"):
        train_samples, holdout_samples = make_splits(samples)
        physics_mean, physics_std = physics_stats(train_samples)
    _init_log(f"train={len(train_samples):,} holdout={len(holdout_samples):,}")
    with init_stage("load or compute prompt embeddings"):
        text_contexts = prepare_prompt_contexts_for_device(
            load_or_compute_prompt_contexts(
                cache_path=TEXT_EMBED_CACHE_PATH, device=DEVICE
            ),
            device=DEVICE,
            dtype=TRAIN_DTYPE,
        )
    with init_stage("create latent dataset and dataloader"):
        train_dataset = LatentDataset(train_samples, physics_mean, physics_std)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=DEVICE.type == "cuda",
            collate_fn=collate_batch,
            drop_last=True,
        )

    with init_stage(f"load Wan transformer from {WAN_CHECKPOINT_DIR}"):
        transformer = load_wan_transformer()
    with init_stage("attach rho/nu physics conditioning"):
        transformer = attach_physics_adaln_forward(transformer)
    with init_stage("configure gradient checkpointing"):
        configure_gradient_checkpointing(transformer)
    with init_stage("attach LoRA adapters"):
        model = add_lora_to_wan(transformer)
    with init_stage("set causal frame window"):
        set_frame_window(model, *causal_window())
        model.train()
    with init_stage("regionally compile repeated Wan blocks"):
        model = apply_regional_compile(model, label="flow model")

    with init_stage("create optimizer"):
        optimizer = torch.optim.AdamW(
            trainable_parameters(model), lr=LEARNING_RATE, weight_decay=1e-4
        )
    try:
        model.print_trainable_parameters()
    except Exception:
        total = sum(param.numel() for param in model.parameters())
        trainable = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )
        print(f"trainable params: {trainable:,} / {total:,}")
    _init_log("flow-matching training objects ready")

    return {
        "model": model,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "train_samples": train_samples,
        "holdout_samples": holdout_samples,
        "text_contexts": text_contexts,
        "physics_mean": physics_mean,
        "physics_std": physics_std,
    }


def build_delta_distillation_objects():
    _init_log(
        f"mode=delta_distill device={DEVICE} param_dtype={PARAM_DTYPE} train_dtype={TRAIN_DTYPE}"
    )
    with init_stage("discover latent samples"):
        samples = discover_latent_samples()
    _init_log(f"found {len(samples):,} samples in {LATENT_ROOT}")
    with init_stage("split dataset and compute physics stats"):
        train_samples, holdout_samples = make_splits(samples)
        physics_mean, physics_std = physics_stats(train_samples)
    _init_log(f"train={len(train_samples):,} holdout={len(holdout_samples):,}")
    with init_stage("load or compute prompt embeddings"):
        text_contexts = prepare_prompt_contexts_for_device(
            load_or_compute_prompt_contexts(
                cache_path=TEXT_EMBED_CACHE_PATH, device=DEVICE
            ),
            device=DEVICE,
            dtype=TRAIN_DTYPE,
        )
    with init_stage("create latent dataset and dataloader"):
        train_dataset = LatentDataset(train_samples, physics_mean, physics_std)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=DEVICE.type == "cuda",
            collate_fn=collate_batch,
            drop_last=True,
        )

    with init_stage(f"load student Wan transformer from {WAN_CHECKPOINT_DIR}"):
        student = load_wan_transformer()
    with init_stage("attach student physics conditioning and delta memory"):
        student = attach_physics_adaln_forward(student)
        student = attach_delta_attention(student)
    with init_stage("attach student LoRA adapters"):
        student = add_lora_to_wan(student)
    with init_stage("configure student gradient checkpointing"):
        configure_gradient_checkpointing(student)
    with init_stage("enable delta, LoRA, and causal window training"):
        enable_delta_attention_training(student, train_lora=True)
        set_frame_window(student, *causal_window())
        set_delta_attention_enabled(student, True)
        student.train()
    with init_stage("regionally compile student repeated Wan blocks"):
        student = apply_regional_compile(student, label="delta student")

    with init_stage("create optimizer"):
        optimizer = torch.optim.AdamW(
            trainable_parameters(student), lr=LEARNING_RATE, weight_decay=1e-4
        )
    total = sum(param.numel() for param in student.parameters())
    trainable = sum(
        param.numel() for param in student.parameters() if param.requires_grad
    )
    delta_trainable = sum(
        param.numel()
        for param in delta_attention_parameters(student)
        if param.requires_grad
    )
    lora_trainable = sum(
        param.numel()
        for param in lora_adapter_parameters(student)
        if param.requires_grad
    )
    physics_trainable = sum(
        param.numel()
        for param in physics_conditioning_parameters(student)
        if param.requires_grad
    )
    first_frame_trainable = sum(
        param.numel()
        for param in first_frame_conditioning_parameters(student)
        if param.requires_grad
    )
    print(
        f"delta window trainable params: {trainable:,} / {total:,} "
        f"(delta={delta_trainable:,}, lora={lora_trainable:,}, "
        f"physics={physics_trainable:,}, first_frame={first_frame_trainable:,})"
    )
    _init_log("delta window training objects ready")

    return {
        "model": student,
        "student": student,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "train_samples": train_samples,
        "holdout_samples": holdout_samples,
        "text_contexts": text_contexts,
        "physics_mean": physics_mean,
        "physics_std": physics_std,
    }


def build_ssm_training_objects():
    _init_log(
        f"mode={TRAINING_MODE} device={DEVICE} param_dtype={PARAM_DTYPE} train_dtype={TRAIN_DTYPE}"
    )
    with init_stage("discover latent samples"):
        samples = discover_latent_samples()
    _init_log(f"found {len(samples):,} samples in {LATENT_ROOT}")
    with init_stage("split dataset and compute physics stats"):
        train_samples, holdout_samples = make_splits(samples)
        physics_mean, physics_std = physics_stats(train_samples)
    _init_log(f"train={len(train_samples):,} holdout={len(holdout_samples):,}")
    with init_stage("load or compute prompt embeddings"):
        text_contexts = prepare_prompt_contexts_for_device(
            load_or_compute_prompt_contexts(
                cache_path=TEXT_EMBED_CACHE_PATH, device=DEVICE
            ),
            device=DEVICE,
            dtype=TRAIN_DTYPE,
        )
    with init_stage("create latent dataset and dataloader"):
        train_dataset = LatentDataset(train_samples, physics_mean, physics_std)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=DEVICE.type == "cuda",
            collate_fn=collate_batch,
            drop_last=True,
        )

    with init_stage(f"load student Wan transformer from {WAN_CHECKPOINT_DIR}"):
        student = load_wan_transformer()
    with init_stage("attach student physics conditioning and SSM memory"):
        student = attach_physics_adaln_forward(student)
        student = attach_ssm_attention(student)
    with init_stage("attach student LoRA adapters"):
        student = add_lora_to_wan(student)
    with init_stage("configure student gradient checkpointing"):
        configure_gradient_checkpointing(student)
    with init_stage("enable SSM, LoRA, and causal window training"):
        enable_ssm_attention_training(student, train_lora=True)
        set_frame_window(student, *causal_window())
        set_ssm_attention_enabled(student, True)
        student.train()
    with init_stage("regionally compile student repeated Wan blocks"):
        student = apply_regional_compile(student, label="SSM student")

    with init_stage("create optimizer"):
        optimizer = torch.optim.AdamW(
            trainable_parameters(student), lr=LEARNING_RATE, weight_decay=1e-4
        )
    total = sum(param.numel() for param in student.parameters())
    trainable = sum(
        param.numel() for param in student.parameters() if param.requires_grad
    )
    ssm_trainable = sum(
        param.numel()
        for param in ssm_attention_parameters(student)
        if param.requires_grad
    )
    lora_trainable = sum(
        param.numel()
        for param in lora_adapter_parameters(student)
        if param.requires_grad
    )
    physics_trainable = sum(
        param.numel()
        for param in physics_conditioning_parameters(student)
        if param.requires_grad
    )
    first_frame_trainable = sum(
        param.numel()
        for param in first_frame_conditioning_parameters(student)
        if param.requires_grad
    )
    print(
        f"SSM window trainable params: {trainable:,} / {total:,} "
        f"(ssm={ssm_trainable:,}, lora={lora_trainable:,}, "
        f"physics={physics_trainable:,}, first_frame={first_frame_trainable:,})"
    )
    _init_log("SSM window training objects ready")

    return {
        "model": student,
        "student": student,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "train_samples": train_samples,
        "holdout_samples": holdout_samples,
        "text_contexts": text_contexts,
        "physics_mean": physics_mean,
        "physics_std": physics_std,
    }


def save_lora_checkpoint(model, step: int):
    output_dir = OUTPUT_DIR / f"step_{step:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with temporarily_uncompiled_repeated_blocks(model):
        model.save_pretrained(str(output_dir))
    save_conditioning_checkpoint(model, output_dir)
    latest_dir = OUTPUT_DIR / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    with temporarily_uncompiled_repeated_blocks(model):
        model.save_pretrained(str(latest_dir))
    save_conditioning_checkpoint(model, latest_dir)
    return output_dir


def conditioning_state_dict(model):
    with temporarily_uncompiled_repeated_blocks(model):
        base = base_model(model)
        return {
            key: value.detach().cpu()
            for key, value in base.state_dict().items()
            if key.startswith("physics_adaln.")
            or key.startswith("first_frame_conditioner.")
        }


def save_conditioning_checkpoint(model, output_dir: Path):
    state = conditioning_state_dict(model)
    if not state:
        return None
    save_file(
        state,
        str(Path(output_dir) / "conditioning.safetensors"),
        metadata={
            "first_frame_conditioning": "patchified_zero_linear",
            "attention_frame0_anchor": "false",
            "window_left_frames": str(WINDOW_LEFT_FRAMES),
            "window_right_frames": "0",
        },
    )
    return Path(output_dir) / "conditioning.safetensors"


def delta_attention_state_dict(model):
    with temporarily_uncompiled_repeated_blocks(model):
        base = base_model(model)
        return {
            key: value.detach().cpu()
            for key, value in base.state_dict().items()
            if ".delta_memory." in key
        }


def save_delta_attention_checkpoint(model, step: int):
    output_dir = OUTPUT_DIR / f"delta_step_{step:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        delta_attention_state_dict(model),
        str(output_dir / "delta_attention.safetensors"),
        metadata={
            "training_mode": "delta_distill",
            "window_left_frames": str(WINDOW_LEFT_FRAMES),
            "window_right_frames": "0",
            "beta_init": str(DELTA_ATTENTION_BETA_INIT),
            "lambda_init": str(DELTA_ATTENTION_LAMBDA_INIT),
            "rho_init": str(DELTA_ATTENTION_RHO_INIT),
            "rho_frozen_zero": str(bool(DELTA_ATTENTION_FREEZE_RHO_ZERO)),
            "lora_r": str(LORA_R),
            "lora_alpha": str(LORA_ALPHA),
            "lora_dropout": str(LORA_DROPOUT),
            "lora_target_modules": json.dumps(LORA_TARGET_MODULES),
        },
    )
    if lora_adapter_parameters(model) and hasattr(model, "save_pretrained"):
        with temporarily_uncompiled_repeated_blocks(model):
            model.save_pretrained(str(output_dir / "lora"))
    save_conditioning_checkpoint(model, output_dir)

    latest_dir = OUTPUT_DIR / "delta_latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        delta_attention_state_dict(model),
        str(latest_dir / "delta_attention.safetensors"),
        metadata={
            "training_mode": "delta_distill",
            "window_left_frames": str(WINDOW_LEFT_FRAMES),
            "window_right_frames": "0",
            "rho_init": str(DELTA_ATTENTION_RHO_INIT),
            "rho_frozen_zero": str(bool(DELTA_ATTENTION_FREEZE_RHO_ZERO)),
            "lora_r": str(LORA_R),
            "lora_alpha": str(LORA_ALPHA),
            "lora_dropout": str(LORA_DROPOUT),
            "lora_target_modules": json.dumps(LORA_TARGET_MODULES),
        },
    )
    if lora_adapter_parameters(model) and hasattr(model, "save_pretrained"):
        with temporarily_uncompiled_repeated_blocks(model):
            model.save_pretrained(str(latest_dir / "lora"))
    save_conditioning_checkpoint(model, latest_dir)
    return output_dir


def ssm_attention_state_dict(model):
    with temporarily_uncompiled_repeated_blocks(model):
        base = base_model(model)
        return {
            key: value.detach().cpu()
            for key, value in base.state_dict().items()
            if ".ssm_memory." in key
        }


def save_ssm_attention_checkpoint(model, step: int):
    output_dir = OUTPUT_DIR / f"ssm_step_{step:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "training_mode": "ssm",
        "window_left_frames": str(WINDOW_LEFT_FRAMES),
        "window_right_frames": "0",
        "decay_init": str(SSM_DECAY_INIT),
        "input_init": str(SSM_INPUT_INIT),
        "output_init": str(SSM_OUTPUT_INIT),
        "skip_init": str(SSM_SKIP_INIT),
            "rho_init": str(SSM_RHO_INIT),
            "rho_frozen_zero": str(bool(SSM_FREEZE_RHO_ZERO)),
            "query_scale": "" if SSM_QUERY_SCALE is None else str(SSM_QUERY_SCALE),
            "recurrent_dim": str(SSM_RECURRENT_DIM),
            "lora_r": str(LORA_R),
        "lora_alpha": str(LORA_ALPHA),
        "lora_dropout": str(LORA_DROPOUT),
        "lora_target_modules": json.dumps(LORA_TARGET_MODULES),
    }
    save_file(
        ssm_attention_state_dict(model),
        str(output_dir / "ssm_attention.safetensors"),
        metadata=metadata,
    )
    if lora_adapter_parameters(model) and hasattr(model, "save_pretrained"):
        with temporarily_uncompiled_repeated_blocks(model):
            model.save_pretrained(str(output_dir / "lora"))
    save_conditioning_checkpoint(model, output_dir)

    latest_dir = OUTPUT_DIR / "ssm_latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        ssm_attention_state_dict(model),
        str(latest_dir / "ssm_attention.safetensors"),
        metadata=metadata,
    )
    if lora_adapter_parameters(model) and hasattr(model, "save_pretrained"):
        with temporarily_uncompiled_repeated_blocks(model):
            model.save_pretrained(str(latest_dir / "lora"))
    save_conditioning_checkpoint(model, latest_dir)
    return output_dir


def checkpoint_saving_enabled() -> bool:
    return SAVE_EVERY is not None and int(SAVE_EVERY) > 0


REPLAY_LOSS_KEYS = (
    "window",
    "latent_err",
    "noise_err",
    "replay_used",
    "bank_fill",
    "total",
)


def set_replay_progress_postfix(progress, ema_losses) -> None:
    progress.set_postfix(
        window_ema=(
            "-" if ema_losses["window"] is None else f"{ema_losses['window']:.6f}"
        ),
        latent_err=(
            "-"
            if ema_losses["latent_err"] is None
            else f"{ema_losses['latent_err']:.6f}"
        ),
        noise_err=(
            "-"
            if ema_losses["noise_err"] is None
            else f"{ema_losses['noise_err']:.6f}"
        ),
        replay=(
            "-"
            if ema_losses["replay_used"] is None
            else f"{ema_losses['replay_used']:.2f}"
        ),
        bank=(
            "-"
            if ema_losses["bank_fill"] is None
            else f"{ema_losses['bank_fill']:.0f}"
        ),
    )


def initialize_frame_loss_tracking(state) -> None:
    state["frame_loss_history"] = []
    state["frame_loss_ema"] = None
    state["frame_loss_indices"] = None


def collect_training_components(components, step_losses, step_frame_losses) -> None:
    for name, value in components.items():
        if name == "frame_loss":
            step_frame_losses.append(
                value.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            )
            continue
        if name in step_losses:
            step_losses[name].append(float(value.detach().cpu()))


def update_frame_loss_tracking(state, step: int, step_frame_losses) -> None:
    if not step_frame_losses:
        return
    lengths = {int(frame_loss.numel()) for frame_loss in step_frame_losses}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent per-frame loss lengths in step: {sorted(lengths)}")
    frame_loss = torch.stack(step_frame_losses, dim=0).mean(dim=0)
    if frame_loss.numel() < 1:
        return
    previous = state.get("frame_loss_ema")
    if previous is None:
        frame_loss_ema = frame_loss
    else:
        frame_loss_ema = (
            LOSS_EMA_BETA * previous + (1.0 - LOSS_EMA_BETA) * frame_loss
        )
    frame_indices = list(range(1, int(frame_loss.numel()) + 1))
    state["frame_loss_ema"] = frame_loss_ema
    state["frame_loss_indices"] = frame_indices
    state.setdefault("frame_loss_history", []).append(
        {
            "step": int(step),
            "frame_indices": frame_indices,
            "loss": [float(value) for value in frame_loss.tolist()],
            "ema": [float(value) for value in frame_loss_ema.tolist()],
        }
    )


def train_flow_matching(max_steps: Optional[int] = None):
    max_steps = resolve_max_steps(max_steps)
    total_start = time.perf_counter()
    _init_log(f"train_flow_matching(max_steps={int(max_steps)}) starting")
    with init_stage("build flow-matching objects"):
        state = build_training_objects()
    model = state["model"]
    optimizer = state["optimizer"]
    train_loader = state["train_loader"]
    holdout_samples = state["holdout_samples"]
    text_contexts = state["text_contexts"]
    physics_mean = state["physics_mean"]
    physics_std = state["physics_std"]
    with init_stage("load Wan VAE"):
        vae = load_wan_vae(dtype=torch.float32, device=DEVICE)

    with init_stage("create training iterator and bookkeeping"):
        loader_iter = iter(train_loader)
        optimizer.zero_grad(set_to_none=True)
        error_bank = ErrorReplayBank()
    state["error_replay_bank"] = error_bank
    tracked_losses = REPLAY_LOSS_KEYS
    loss_history = {name: [] for name in tracked_losses}
    ema_losses = {name: None for name in tracked_losses}
    state["loss_history"] = loss_history
    state["loss_ema"] = ema_losses
    initialize_frame_loss_tracking(state)
    _init_log(
        f"initialization complete in {time.perf_counter() - total_start:.1f}s; entering training loop"
    )

    progress = trange(1, int(max_steps) + 1, desc="fine-tuning", dynamic_ncols=True)
    for step in progress:
        lr = learning_rate_for_step(step)
        set_optimizer_lr(optimizer, lr)
        step_losses = {name: [] for name in tracked_losses}
        step_frame_losses = []

        for _ in range(GRAD_ACCUM_STEPS):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            compiler_mark_step_begin()
            loss, components = flow_matching_batch(
                batch, model, text_contexts, error_bank=error_bank, step=step
            )
            collect_training_components(components, step_losses, step_frame_losses)
            (loss / GRAD_ACCUM_STEPS).backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters(model),
            CLIP_GRAD_NORM,
            error_if_nonfinite=True,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        for mode, losses in step_losses.items():
            if not losses:
                continue
            mode_loss = sum(losses) / len(losses)
            previous = ema_losses[mode]
            ema_losses[mode] = (
                mode_loss
                if previous is None
                else LOSS_EMA_BETA * previous + (1.0 - LOSS_EMA_BETA) * mode_loss
            )
            loss_history[mode].append(
                {"step": step, "loss": mode_loss, "ema": ema_losses[mode]}
            )
        update_frame_loss_tracking(state, step, step_frame_losses)

        set_replay_progress_postfix(progress, ema_losses)

        if step % EVAL_EVERY == 0:
            evaluate_random_holdout(
                model,
                vae,
                holdout_samples,
                step,
                text_contexts,
                physics_mean,
                physics_std,
                frame_loss_ema=state.get("frame_loss_ema"),
                frame_loss_indices=state.get("frame_loss_indices"),
            )
            model.train()
            set_frame_window(model, *causal_window())

        if checkpoint_saving_enabled() and step % int(SAVE_EVERY) == 0:
            save_lora_checkpoint(model, step)

    if checkpoint_saving_enabled():
        save_lora_checkpoint(model, int(max_steps))
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return model, state


def train_delta_distillation(max_steps: Optional[int] = None):
    max_steps = resolve_max_steps(max_steps)
    total_start = time.perf_counter()
    _init_log(f"train_delta_window(max_steps={int(max_steps)}) starting")
    with init_stage("build delta window objects"):
        state = build_delta_distillation_objects()
    student = state["student"]
    optimizer = state["optimizer"]
    train_loader = state["train_loader"]
    holdout_samples = state["holdout_samples"]
    text_contexts = state["text_contexts"]
    physics_mean = state["physics_mean"]
    physics_std = state["physics_std"]
    with init_stage("load Wan VAE"):
        vae = load_wan_vae(dtype=torch.float32, device=DEVICE)

    with init_stage("create training iterator and bookkeeping"):
        loader_iter = iter(train_loader)
        optimizer.zero_grad(set_to_none=True)
        error_bank = ErrorReplayBank()
    state["error_replay_bank"] = error_bank
    tracked_losses = REPLAY_LOSS_KEYS
    loss_history = {name: [] for name in tracked_losses}
    ema_losses = {name: None for name in tracked_losses}
    state["loss_history"] = loss_history
    state["loss_ema"] = ema_losses
    initialize_frame_loss_tracking(state)
    _init_log(
        f"initialization complete in {time.perf_counter() - total_start:.1f}s; entering training loop"
    )

    progress = trange(1, int(max_steps) + 1, desc="delta-window", dynamic_ncols=True)
    for step in progress:
        lr = learning_rate_for_step(step)
        set_optimizer_lr(optimizer, lr)
        step_losses = {name: [] for name in tracked_losses}
        step_frame_losses = []

        for _ in range(GRAD_ACCUM_STEPS):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            compiler_mark_step_begin()
            loss, components = delta_flow_matching_batch(
                batch, student, text_contexts, error_bank=error_bank, step=step
            )
            collect_training_components(components, step_losses, step_frame_losses)
            (loss / GRAD_ACCUM_STEPS).backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters(student),
            CLIP_GRAD_NORM,
            error_if_nonfinite=True,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        for mode, losses in step_losses.items():
            if not losses:
                continue
            mode_loss = sum(losses) / len(losses)
            previous = ema_losses[mode]
            ema_losses[mode] = (
                mode_loss
                if previous is None
                else LOSS_EMA_BETA * previous + (1.0 - LOSS_EMA_BETA) * mode_loss
            )
            loss_history[mode].append(
                {"step": step, "loss": mode_loss, "ema": ema_losses[mode]}
            )
        update_frame_loss_tracking(state, step, step_frame_losses)

        set_replay_progress_postfix(progress, ema_losses)

        if step % EVAL_EVERY == 0:
            evaluate_random_holdout(
                student,
                vae,
                holdout_samples,
                step,
                text_contexts,
                physics_mean,
                physics_std,
                frame_loss_ema=state.get("frame_loss_ema"),
                frame_loss_indices=state.get("frame_loss_indices"),
            )
            student.train()
            set_frame_window(student, *causal_window())
            set_delta_attention_enabled(student, True)

        if checkpoint_saving_enabled() and step % int(SAVE_EVERY) == 0:
            save_delta_attention_checkpoint(student, step)

    if checkpoint_saving_enabled():
        save_delta_attention_checkpoint(student, int(max_steps))
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return student, state


def train_ssm_window(max_steps: Optional[int] = None):
    max_steps = resolve_max_steps(max_steps)
    total_start = time.perf_counter()
    _init_log(f"train_ssm_window(max_steps={int(max_steps)}) starting")
    with init_stage("build SSM window objects"):
        state = build_ssm_training_objects()
    student = state["student"]
    optimizer = state["optimizer"]
    train_loader = state["train_loader"]
    holdout_samples = state["holdout_samples"]
    text_contexts = state["text_contexts"]
    physics_mean = state["physics_mean"]
    physics_std = state["physics_std"]
    with init_stage("load Wan VAE"):
        vae = load_wan_vae(dtype=torch.float32, device=DEVICE)

    with init_stage("create training iterator and bookkeeping"):
        loader_iter = iter(train_loader)
        optimizer.zero_grad(set_to_none=True)
        error_bank = ErrorReplayBank()
    state["error_replay_bank"] = error_bank
    tracked_losses = REPLAY_LOSS_KEYS
    loss_history = {name: [] for name in tracked_losses}
    ema_losses = {name: None for name in tracked_losses}
    state["loss_history"] = loss_history
    state["loss_ema"] = ema_losses
    initialize_frame_loss_tracking(state)
    _init_log(
        f"initialization complete in {time.perf_counter() - total_start:.1f}s; entering training loop"
    )

    progress = trange(1, int(max_steps) + 1, desc="ssm-window", dynamic_ncols=True)
    for step in progress:
        lr = learning_rate_for_step(step)
        set_optimizer_lr(optimizer, lr)
        step_losses = {name: [] for name in tracked_losses}
        step_frame_losses = []

        for _ in range(GRAD_ACCUM_STEPS):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            compiler_mark_step_begin()
            loss, components = ssm_flow_matching_batch(
                batch, student, text_contexts, error_bank=error_bank, step=step
            )
            collect_training_components(components, step_losses, step_frame_losses)
            (loss / GRAD_ACCUM_STEPS).backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters(student),
            CLIP_GRAD_NORM,
            error_if_nonfinite=True,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        for mode, losses in step_losses.items():
            if not losses:
                continue
            mode_loss = sum(losses) / len(losses)
            previous = ema_losses[mode]
            ema_losses[mode] = (
                mode_loss
                if previous is None
                else LOSS_EMA_BETA * previous + (1.0 - LOSS_EMA_BETA) * mode_loss
            )
            loss_history[mode].append(
                {"step": step, "loss": mode_loss, "ema": ema_losses[mode]}
            )
        update_frame_loss_tracking(state, step, step_frame_losses)

        set_replay_progress_postfix(progress, ema_losses)

        if step % EVAL_EVERY == 0:
            evaluate_random_holdout(
                student,
                vae,
                holdout_samples,
                step,
                text_contexts,
                physics_mean,
                physics_std,
                frame_loss_ema=state.get("frame_loss_ema"),
                frame_loss_indices=state.get("frame_loss_indices"),
            )
            student.train()
            set_frame_window(student, *causal_window())
            set_ssm_attention_enabled(student, True)

        if checkpoint_saving_enabled() and step % int(SAVE_EVERY) == 0:
            save_ssm_attention_checkpoint(student, step)

    if checkpoint_saving_enabled():
        save_ssm_attention_checkpoint(student, int(max_steps))
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return student, state


def train(max_steps: Optional[int] = None):
    max_steps = resolve_max_steps(max_steps)
    _validate_training_mode()
    if TRAINING_MODE == "delta_distill":
        return train_delta_distillation(max_steps=max_steps)
    if TRAINING_MODE in {"ssm", "ssm_distill"}:
        return train_ssm_window(max_steps=max_steps)
    return train_flow_matching(max_steps=max_steps)
