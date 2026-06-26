# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn.functional as F

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask as create_flex_block_mask,
    )
    from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention
except Exception:
    create_flex_block_mask = None
    torch_flex_attention = None

__all__ = [
    'flash_attention',
    'attention',
    'frame_window_attention',
]

USE_FLEX_ATTENTION = True
FLEX_ATTENTION_BLOCK_SIZE = 128
FLEX_ATTENTION_DYNAMIC = True
FLEX_ATTENTION_RECOMPILE_LIMIT = 64


def _padding_mask(batch_size, q_len, k_len, q_lens, k_lens, device):
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
    return torch.where(q_valid.unsqueeze(2), mask, k_valid.unsqueeze(1))


def sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
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


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    if q.device.type != 'cuda' or not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return sdpa_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=version,
        )

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )


def frame_window_mask(
    grid_sizes,
    seq_len,
    left_frames,
    right_frames,
    *,
    q_lens=None,
    k_lens=None,
):
    left_frames = int(left_frames)
    right_frames = int(right_frames)
    if left_frames < 0 or right_frames < 0:
        raise ValueError('left_frames and right_frames must be >= 0')
    if right_frames != 0:
        raise ValueError('right_frames must be 0 for causal windowed attention')

    device = grid_sizes.device
    q_idx = torch.arange(seq_len, device=device)
    k_idx = torch.arange(seq_len, device=device)
    masks = []

    for batch_index, (frame_count, grid_h, grid_w) in enumerate(grid_sizes.tolist()):
        spatial_tokens = int(grid_h) * int(grid_w)
        valid_tokens = int(frame_count) * spatial_tokens
        if spatial_tokens <= 0:
            raise ValueError(f'Invalid spatial token count from grid {grid_h}x{grid_w}')
        if valid_tokens > seq_len:
            raise ValueError(
                f'Grid has {valid_tokens} tokens, but seq_len is {seq_len}')

        q_valid_len = int(q_lens[batch_index].item()) if q_lens is not None else valid_tokens
        k_valid_len = int(k_lens[batch_index].item()) if k_lens is not None else valid_tokens
        q_valid = q_idx < q_valid_len
        k_valid = k_idx < k_valid_len

        q_frame = torch.div(q_idx, spatial_tokens, rounding_mode='floor')
        k_frame = torch.div(k_idx, spatial_tokens, rounding_mode='floor')
        frame_mask = (k_frame.unsqueeze(0) >= q_frame.unsqueeze(1) - left_frames) & (
            k_frame.unsqueeze(0) <= q_frame.unsqueeze(1) + right_frames)
        mask = q_valid.unsqueeze(1) & k_valid.unsqueeze(0) & frame_mask
        mask = torch.where(q_valid.unsqueeze(1), mask, k_valid.unsqueeze(0))
        masks.append(mask)

    return torch.stack(masks, dim=0)


_FLEX_BLOCK_MASK_CACHE = {}
_COMPILED_FLEX_ATTENTION = None
_COMPILED_FLEX_ATTENTION_KEY = None


def _set_torch_dynamo_limit(name, value):
    dynamo = getattr(torch, '_dynamo', None)
    config = getattr(dynamo, 'config', None)
    if config is None or not hasattr(config, name):
        return
    try:
        current = int(getattr(config, name))
        setattr(config, name, max(current, int(value)))
    except Exception:
        return


def _compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION, _COMPILED_FLEX_ATTENTION_KEY
    if torch_flex_attention is None or not hasattr(torch, 'compile'):
        return None

    dynamic = bool(FLEX_ATTENTION_DYNAMIC)
    limit = FLEX_ATTENTION_RECOMPILE_LIMIT
    key = (dynamic, None if limit is None else int(limit))
    if _COMPILED_FLEX_ATTENTION is not None and _COMPILED_FLEX_ATTENTION_KEY == key:
        return _COMPILED_FLEX_ATTENTION

    if limit is not None:
        _set_torch_dynamo_limit('recompile_limit', int(limit))
        _set_torch_dynamo_limit('cache_size_limit', int(limit))
    _COMPILED_FLEX_ATTENTION = torch.compile(torch_flex_attention, dynamic=dynamic)
    _COMPILED_FLEX_ATTENTION_KEY = key
    return _COMPILED_FLEX_ATTENTION


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def flex_attention_available():
    return (
        USE_FLEX_ATTENTION
        and create_flex_block_mask is not None
        and _compiled_flex_attention() is not None
    )


def _flex_block_size(spatial_tokens):
    spatial_tokens = int(spatial_tokens)
    if spatial_tokens <= 0:
        raise ValueError(f'Invalid spatial token count {spatial_tokens}')
    min_block_size = 64
    if FLEX_ATTENTION_BLOCK_SIZE is not None:
        block_size = int(FLEX_ATTENTION_BLOCK_SIZE)
    elif spatial_tokens >= 64:
        block_size = 64
    else:
        block_size = min_block_size
    if not _is_power_of_two(block_size):
        raise ValueError(
            f'FLEX_ATTENTION_BLOCK_SIZE must be a power of two, got {block_size}')
    if block_size < min_block_size:
        raise ValueError(
            f'flex block size must be >= {min_block_size}, got {block_size}')
    return block_size


def _padded_flex_spatial_tokens(spatial_tokens, block_size):
    spatial_tokens = int(spatial_tokens)
    block_size = int(block_size)
    return ((spatial_tokens + block_size - 1) // block_size) * block_size


def _pad_frame_tokens_for_flex(x, frame_count, spatial_tokens, padded_spatial_tokens):
    if spatial_tokens == padded_spatial_tokens:
        return x
    batch_size, seq_len, num_heads, head_dim = x.shape
    expected_len = int(frame_count) * int(spatial_tokens)
    if seq_len != expected_len:
        raise ValueError(
            f'Expected {expected_len} tokens from frame/grid layout, got {seq_len}')
    x = x.reshape(batch_size, int(frame_count), int(spatial_tokens), num_heads, head_dim)
    pad_tokens = int(padded_spatial_tokens) - int(spatial_tokens)
    pad = x.new_zeros(batch_size, int(frame_count), pad_tokens, num_heads, head_dim)
    return torch.cat((x, pad), dim=2).reshape(
        batch_size, int(frame_count) * int(padded_spatial_tokens), num_heads, head_dim)


def _unpad_frame_tokens_from_flex(x, frame_count, spatial_tokens, padded_spatial_tokens):
    if spatial_tokens == padded_spatial_tokens:
        return x
    batch_size, seq_len, num_heads, head_dim = x.shape
    expected_len = int(frame_count) * int(padded_spatial_tokens)
    if seq_len != expected_len:
        raise ValueError(
            f'Expected {expected_len} padded tokens from flex attention, got {seq_len}')
    x = x.reshape(
        batch_size, int(frame_count), int(padded_spatial_tokens), num_heads, head_dim)
    return x[:, :, :int(spatial_tokens)].reshape(
        batch_size, int(frame_count) * int(spatial_tokens), num_heads, head_dim)


def frame_window_flex_block_mask(
    *,
    seq_len,
    spatial_tokens,
    padded_spatial_tokens=None,
    left_frames,
    right_frames,
    device,
):
    block_size = _flex_block_size(spatial_tokens)
    if padded_spatial_tokens is None:
        padded_spatial_tokens = _padded_flex_spatial_tokens(spatial_tokens, block_size)
    padded_spatial_tokens = int(padded_spatial_tokens)
    key = (
        str(device),
        int(seq_len),
        int(spatial_tokens),
        padded_spatial_tokens,
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
        raise ValueError('right_frames must be 0 for causal windowed attention')

    def mask_mod(batch, head, q_idx, kv_idx):
        del batch, head
        q_frame = q_idx // padded_spatial_tokens
        kv_frame = kv_idx // padded_spatial_tokens
        q_pos = q_idx - q_frame * padded_spatial_tokens
        kv_pos = kv_idx - kv_frame * padded_spatial_tokens
        q_valid = q_pos < spatial_tokens
        kv_valid = kv_pos < spatial_tokens
        frame_ok = (kv_frame >= q_frame - left_frames) & (
            kv_frame <= q_frame + right_frames)
        # Dummy padded query rows still need at least one legal key row so the
        # fused kernel can keep ROWS_GUARANTEED_SAFE enabled. Their outputs are
        # discarded before returning to the model.
        dummy_q_safe = (~q_valid) & kv_valid & frame_ok
        return (q_valid & kv_valid & frame_ok) | dummy_q_safe

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


def can_use_frame_window_flex_attention(q, grid_sizes, q_lens, k_lens):
    if not flex_attention_available() or q.device.type != 'cuda':
        return False, None
    if q_lens is not None and not torch.all(q_lens.to(device=q.device) == q.size(1)):
        return False, None
    if k_lens is not None and not torch.all(k_lens.to(device=q.device) == q.size(1)):
        return False, None

    spatial_tokens = []
    frame_counts = []
    for frame_count, grid_h, grid_w in grid_sizes.tolist():
        tokens = int(grid_h) * int(grid_w)
        if int(frame_count) * tokens != q.size(1):
            return False, None
        spatial_tokens.append(tokens)
        frame_counts.append(int(frame_count))
    if not spatial_tokens or any(
        tokens != spatial_tokens[0] for tokens in spatial_tokens):
        return False, None
    if any(count != frame_counts[0] for count in frame_counts):
        return False, None
    try:
        block_size = _flex_block_size(spatial_tokens[0])
    except ValueError:
        return False, None
    padded_spatial_tokens = _padded_flex_spatial_tokens(spatial_tokens[0], block_size)
    return True, {
        'frame_count': frame_counts[0],
        'spatial_tokens': spatial_tokens[0],
        'padded_spatial_tokens': padded_spatial_tokens,
        'block_size': block_size,
    }


def frame_window_flex_attention(
    q,
    k,
    v,
    *,
    grid_sizes,
    left_frames,
    right_frames,
    softmax_scale=None,
    q_scale=None,
    dtype=torch.bfloat16,
    layout,
):
    del grid_sizes
    out_dtype = q.dtype
    if q_scale is not None:
        q = q * q_scale
    frame_count = int(layout['frame_count'])
    spatial_tokens = int(layout['spatial_tokens'])
    padded_spatial_tokens = int(layout['padded_spatial_tokens'])
    block_size = int(layout['block_size'])
    q = _pad_frame_tokens_for_flex(
        q, frame_count, spatial_tokens, padded_spatial_tokens)
    k = _pad_frame_tokens_for_flex(
        k, frame_count, spatial_tokens, padded_spatial_tokens)
    v = _pad_frame_tokens_for_flex(
        v, frame_count, spatial_tokens, padded_spatial_tokens)
    block_mask = frame_window_flex_block_mask(
        seq_len=q.size(1),
        spatial_tokens=spatial_tokens,
        padded_spatial_tokens=padded_spatial_tokens,
        left_frames=int(left_frames),
        right_frames=int(right_frames),
        device=q.device,
    )
    rows_guaranteed_safe = True
    q = q.transpose(1, 2).to(dtype).contiguous()
    k = k.transpose(1, 2).to(dtype).contiguous()
    v = v.transpose(1, 2).to(dtype).contiguous()
    compiled_flex_attention = _compiled_flex_attention()
    if compiled_flex_attention is None:
        raise RuntimeError('compiled flex attention is not available')
    out = compiled_flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=softmax_scale,
        kernel_options={
            'BLOCK_M': block_size,
            'BLOCK_N': block_size,
            'ROWS_GUARANTEED_SAFE': rows_guaranteed_safe,
            'BLOCKS_ARE_CONTIGUOUS': True,
        },
    )
    out = out.transpose(1, 2).contiguous().to(out_dtype)
    return _unpad_frame_tokens_from_flex(
        out, frame_count, spatial_tokens, padded_spatial_tokens)


def frame_window_attention(
    q,
    k,
    v,
    *,
    grid_sizes,
    left_frames,
    right_frames,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    dtype=torch.bfloat16,
):
    out_dtype = q.dtype
    if int(right_frames) != 0:
        raise ValueError('right_frames must be 0 for causal windowed attention')
    if q.size(1) != k.size(1):
        raise ValueError('Frame-window attention expects equal query/key sequence lengths')

    use_flex, layout = can_use_frame_window_flex_attention(
        q, grid_sizes, q_lens, k_lens)
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
            layout=layout,
        )

    if q_scale is not None:
        q = q * q_scale
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
