# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention, frame_window_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        frame_window = getattr(self, 'frame_window', None)
        ssm_enabled = bool(getattr(self, 'ssm_attention_enabled', False))

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        if frame_window is None and not ssm_enabled:
            x = flash_attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size)
        else:
            if frame_window is None:
                frame_window = getattr(self, 'default_frame_window', (4, 0))
            left_frames, right_frames = frame_window
            attention_dtype = getattr(self, 'attention_dtype', torch.bfloat16)
            x = frame_window_attention(
                q=q,
                k=k,
                v=v,
                grid_sizes=grid_sizes,
                left_frames=int(left_frames),
                right_frames=int(right_frames),
                q_lens=seq_lens,
                k_lens=seq_lens,
                dtype=attention_dtype,
            )
            if ssm_enabled:
                ssm_memory = getattr(self, 'ssm_memory', None)
                if ssm_memory is None:
                    raise RuntimeError(
                        'ssm_attention_enabled=True but this attention layer has no ssm_memory module')
                x = x + ssm_memory(
                    q=q,
                    k=k,
                    v=v,
                    grid_sizes=grid_sizes,
                    seq_lens=seq_lens,
                )

        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def _forward_impl(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens,
            grid_sizes,
            freqs)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # FFN path. Text cross-attention is intentionally disabled for the
        # Navier setup.
        def ffn_fn(x, e):
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = ffn_fn(x, e)
        return x

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
    ):
        if not (self.training and getattr(self, 'gradient_checkpointing', False)):
            return self._forward_impl(
                x,
                e=e,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
            )

        def run(activation):
            return self._forward_impl(
                activation,
                e=e,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
            )

        return torch_checkpoint.checkpoint(run, x, use_reentrant=False)


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

    def _ensure_rope_capacity(self, grid_sizes, device):
        max_position = int(grid_sizes.max().item())
        if max_position <= int(self.freqs.size(0)):
            return
        head_dim = int(self.dim) // int(self.num_heads)
        self.freqs = torch.cat([
            rope_params(max_position, head_dim - 4 * (head_dim // 6)),
            rope_params(max_position, 2 * (head_dim // 6)),
            rope_params(max_position, 2 * (head_dim // 6))
        ],
                               dim=1).to(device)

    @staticmethod
    def _latent_list(value, name):
        if isinstance(value, torch.Tensor):
            if value.dim() == 5:
                return [u for u in value]
            if value.dim() == 4:
                return [value]
            raise ValueError(
                f'{name} tensor must have shape [B,C,F,H,W] or [C,F,H,W], got {tuple(value.shape)}')
        return list(value)

    def _patch_embedding_forward(self, x):
        tokens = self.patch_embedding(x)
        adapter = getattr(self, 'patch_embedding_lora', None)
        if adapter is not None:
            tokens = tokens + adapter(x).to(dtype=tokens.dtype)
        return tokens

    def _time_projection_forward(self, e):
        tokens = self.time_projection(e)
        adapter = getattr(self, 'timestep_adaln_lora', None)
        if adapter is not None:
            time_hidden = self.time_projection[0](e)
            tokens = tokens + adapter(time_hidden).to(dtype=tokens.dtype)
        return tokens

    def _first_frame_adaln_tokens(self, first_frame, token_batches, grid_sizes):
        if not hasattr(self, 'first_frame_adaln'):
            raise RuntimeError(
                'first_frame was provided, but this WanModel has no first_frame_adaln modules')

        device = self.patch_embedding.weight.device
        tokens_out = []
        for item_index, (tokens, condition_latent) in enumerate(
                zip(token_batches, first_frame)):
            condition_tokens = self._patch_embedding_forward(
                condition_latent.to(
                    device=device,
                    dtype=self.patch_embedding.weight.dtype).unsqueeze(0))
            if int(condition_tokens.shape[2]) != 1:
                raise ValueError(
                    'first_frame must contain exactly one latent frame after patchification')
            condition_tokens = condition_tokens.flatten(2).transpose(1, 2)
            frame_count, grid_h, grid_w = grid_sizes[item_index].tolist()
            spatial_tokens = int(grid_h) * int(grid_w)
            if condition_tokens.size(1) != spatial_tokens:
                raise ValueError(
                    f'first_frame produced {condition_tokens.size(1)} spatial tokens, expected {spatial_tokens}')
            condition_tokens = (
                condition_tokens.to(dtype=torch.float32)
                .reshape(1, 1, spatial_tokens, self.dim)
                .expand(1, int(frame_count), spatial_tokens, self.dim)
                .reshape(1, int(frame_count) * spatial_tokens, self.dim))
            tokens_out.append(condition_tokens)
        return tokens_out

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        physics=None,
        first_frame=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = self._latent_list(x, 'x')
        if first_frame is not None:
            first_frame = self._latent_list(first_frame, 'first_frame')
            if len(first_frame) != len(x):
                raise ValueError(
                    f'first_frame batch has {len(first_frame)} items but x has {len(x)}')

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self._patch_embedding_forward(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x])
        self._ensure_rope_capacity(grid_sizes, device)
        x = [u.flatten(2).transpose(1, 2) for u in x]
        first_frame_adaln_tokens = (
            self._first_frame_adaln_tokens(first_frame, x, grid_sizes)
            if first_frame is not None
            else None)
        seq_lens = torch.tensor([u.size(1) for u in x],
                                dtype=torch.long,
                                device=device)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        if first_frame_adaln_tokens is not None:
            first_frame_adaln_tokens = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                          dim=1) for u in first_frame_adaln_tokens
            ])

        # Text conditioning is disabled for the Navier setup.
        # Timestep AdaLN can be toggled from the training config while keeping
        # the same timestep tensor shape at call sites.
        del context
        timestep_conditioning_enabled = bool(
            getattr(self, 'timestep_conditioning_enabled', True))
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        if t.dim() != 2:
            raise ValueError(
                f't must have shape [B] or [B, seq_len], got {tuple(t.shape)}')
        if int(t.size(1)) != int(seq_len):
            raise ValueError(
                f't timestep token count {int(t.size(1))} does not match seq_len {int(seq_len)}')
        if int(t.size(0)) != int(x.size(0)):
            raise ValueError(
                f't batch size {int(t.size(0))} does not match x batch size {int(x.size(0))}')
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            if timestep_conditioning_enabled:
                t = t.to(device=device, dtype=torch.float32).flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).unflatten(
                        0, (bt, seq_len)).float().to(device))
                e0 = self._time_projection_forward(e).unflatten(2, (6, self.dim))
            else:
                e = torch.zeros(
                    bt, seq_len, self.dim, device=device, dtype=torch.float32)
                e0 = torch.zeros(
                    bt, seq_len, 6, self.dim, device=device, dtype=torch.float32)
            physics_condition = None
            if physics is not None:
                if not hasattr(self, 'physics_adaln'):
                    raise RuntimeError(
                        'physics was provided, but this WanModel has no physics_adaln modules')
                physics = physics.to(device=device, dtype=torch.float32)
                if physics.shape != (bt, 2):
                    raise ValueError(
                        f'physics must have shape ({bt}, 2), got {tuple(physics.shape)}')
                physics_condition = physics
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs)

        dropped_layers = set(int(index) for index in getattr(
            self, 'dropped_transformer_layers', ()))
        training_dropped_layer = getattr(
            self, '_dropped_transformer_layer_index', None)
        for block_index, block in enumerate(self.blocks):
            if block_index in dropped_layers:
                continue
            if training_dropped_layer == block_index:
                continue
            block_e0 = e0
            if physics_condition is not None:
                block_physics_e0 = self.physics_adaln[block_index](
                    physics_condition).view(bt, 1, 6, self.dim)
                block_e0 = block_e0 + block_physics_e0
            if first_frame_adaln_tokens is not None:
                first_frame_key = str(block_index)
                first_frame_adapter = (
                    self.first_frame_adaln[first_frame_key]
                    if first_frame_key in self.first_frame_adaln
                    else None)
                if first_frame_adapter is not None:
                    first_frame_e0 = first_frame_adapter(
                        first_frame_adaln_tokens).reshape(
                            bt, seq_len, 6, self.dim)
                    block_e0 = block_e0 + first_frame_e0
            kwargs['e'] = block_e0
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
