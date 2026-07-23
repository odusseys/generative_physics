from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import FluxPosEmbed
from diffusers.models.transformers.transformer_flux2 import apply_rotary_emb


class ThermalCoefficientModulator(nn.Module):
    def __init__(
        self,
        transformer_dim,
        num_double_blocks,
        num_single_blocks,
        bottleneck_dim=64,
        normalization_mean=0.0,
        normalization_std=1.0,
        parameter_names=("thermal_diffusivity",),
        parameter_transforms=("log",),
    ):
        super().__init__()
        self.transformer_dim = int(transformer_dim)
        self.parameter_names = tuple(str(name) for name in parameter_names)
        self.parameter_transforms = tuple(str(kind) for kind in parameter_transforms)
        if len(self.parameter_names) != len(self.parameter_transforms):
            raise ValueError("parameter_names and parameter_transforms must have the same length.")
        if not self.parameter_names:
            raise ValueError("at least one scalar conditioning parameter is required.")
        if any(kind not in {"linear", "log"} for kind in self.parameter_transforms):
            raise ValueError("parameter_transforms entries must be 'linear' or 'log'.")

        mean = torch.as_tensor(normalization_mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(normalization_std, dtype=torch.float32).flatten()
        if mean.numel() == 1 and len(self.parameter_names) != 1:
            mean = mean.repeat(len(self.parameter_names))
        if std.numel() == 1 and len(self.parameter_names) != 1:
            std = std.repeat(len(self.parameter_names))
        if mean.numel() != len(self.parameter_names) or std.numel() != len(self.parameter_names):
            raise ValueError("normalization statistics must be scalar or one value per conditioning parameter.")
        self.input_dim = len(self.parameter_names)
        self.register_buffer("normalization_mean", mean, persistent=True)
        self.register_buffer("normalization_std", std, persistent=True)
        self.double_blocks = nn.ModuleList(
            [self._make_mlp(bottleneck_dim, self.transformer_dim * 6) for _ in range(num_double_blocks)]
        )
        self.single_blocks = nn.ModuleList(
            [self._make_mlp(bottleneck_dim, self.transformer_dim * 6) for _ in range(num_single_blocks)]
        )

    def _make_mlp(self, bottleneck_dim, output_dim):
        mlp = nn.Sequential(
            nn.Linear(self.input_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, output_dim),
        )
        nn.init.zeros_(mlp[-1].weight)
        if mlp[-1].bias is not None:
            nn.init.zeros_(mlp[-1].bias)
        return mlp

    def _conditioning_matrix(self, conditioning_values, hidden_states, dtype):
        values = conditioning_values.to(device=hidden_states.device, dtype=dtype)
        values = values.reshape(hidden_states.shape[0], -1)
        if values.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} scalar conditioning values, got {values.shape[1]}.")

        columns = []
        tiny = torch.finfo(dtype).tiny
        for idx, transform in enumerate(self.parameter_transforms):
            column = values[:, idx]
            if transform == "log":
                column = torch.log(column.clamp_min(tiny))
            columns.append(column)
        transformed = torch.stack(columns, dim=1)
        mean = self.normalization_mean.to(device=hidden_states.device, dtype=dtype)
        std = self.normalization_std.to(device=hidden_states.device, dtype=dtype).clamp_min(torch.finfo(dtype).eps)
        return (transformed - mean) / std

    def double_delta(self, block_index, conditioning_values, hidden_states):
        block = self.double_blocks[block_index]
        values = self._conditioning_matrix(
            conditioning_values,
            hidden_states,
            next(block.parameters()).dtype,
        )
        return block(values).to(hidden_states.dtype).unsqueeze(1)

    def single_delta(self, block_index, conditioning_values, hidden_states):
        block = self.single_blocks[block_index]
        values = self._conditioning_matrix(
            conditioning_values,
            hidden_states,
            next(block.parameters()).dtype,
        )
        mod6 = block(values).to(hidden_states.dtype)
        first, second = mod6.chunk(2, dim=-1)
        return (first + second).unsqueeze(1)


class ConditionLatentAdaLNZero(nn.Module):
    """Spatial AdaLN-Zero deltas from unpooled condition latent tokens."""

    def __init__(self, condition_dim, transformer_dim, num_double_blocks, num_single_blocks):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.transformer_dim = int(transformer_dim)
        self.double_blocks = nn.ModuleList(
            [self._make_adapter(6 * self.transformer_dim) for _ in range(num_double_blocks)]
        )
        self.single_blocks = nn.ModuleList(
            [self._make_adapter(3 * self.transformer_dim) for _ in range(num_single_blocks)]
        )

    def _make_adapter(self, output_dim):
        adapter = nn.Linear(self.condition_dim, output_dim)
        nn.init.zeros_(adapter.weight)
        nn.init.zeros_(adapter.bias)
        return adapter

    def _delta(self, blocks, block_index, condition_latents, hidden_states):
        block = blocks[block_index]
        if condition_latents.ndim != 3 or condition_latents.shape[-1] != self.condition_dim:
            raise ValueError(
                "condition_latents must have shape "
                f"[batch, tokens, {self.condition_dim}]."
            )
        values = condition_latents.to(
            device=hidden_states.device,
            dtype=next(block.parameters()).dtype,
        )
        return block(values).to(hidden_states.dtype)

    @staticmethod
    def _validate_matching_tokens(delta, target_token_count):
        target_token_count = int(target_token_count)
        if delta.shape[1] != target_token_count:
            raise ValueError(
                "condition and target latent grids must match elementwise; "
                f"got {delta.shape[1]} and {target_token_count} tokens."
            )
        return target_token_count

    def double_delta(self, block_index, condition_latents, hidden_states, target_token_count):
        delta = self._delta(
            self.double_blocks,
            block_index,
            condition_latents,
            hidden_states,
        )
        target_token_count = self._validate_matching_tokens(delta, target_token_count)
        if target_token_count > hidden_states.shape[1]:
            raise ValueError("target_token_count exceeds the image token count.")

        output = hidden_states.new_zeros(
            hidden_states.shape[0],
            hidden_states.shape[1],
            delta.shape[-1],
        )
        output[:, :target_token_count] = delta
        return output

    def single_delta(
        self,
        block_index,
        condition_latents,
        hidden_states,
        num_text_tokens,
        target_token_count,
    ):
        delta = self._delta(
            self.single_blocks,
            block_index,
            condition_latents,
            hidden_states,
        )
        target_token_count = self._validate_matching_tokens(delta, target_token_count)
        num_text_tokens = int(num_text_tokens)
        if num_text_tokens + target_token_count > hidden_states.shape[1]:
            raise ValueError("text and target token counts exceed the joint token count.")

        output = hidden_states.new_zeros(
            hidden_states.shape[0],
            hidden_states.shape[1],
            delta.shape[-1],
        )
        target_slice = slice(num_text_tokens, num_text_tokens + target_token_count)
        output[:, target_slice] = delta
        return output


class ConditionLatentCrossAttentionBlock(nn.Module):
    def __init__(self, condition_dim, transformer_dim, attention_dim=512, num_heads=8):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.transformer_dim = int(transformer_dim)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        if self.attention_dim % self.num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads.")
        self.head_dim = self.attention_dim // self.num_heads
        self.to_q = nn.Linear(self.transformer_dim, self.attention_dim, bias=False)
        self.to_k = nn.Linear(self.condition_dim, self.attention_dim, bias=False)
        self.to_v = nn.Linear(self.condition_dim, self.attention_dim, bias=False)
        self.to_out = nn.Linear(self.attention_dim, self.transformer_dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(
        self,
        image_hidden_states,
        condition_latents,
        image_rotary_emb=None,
        condition_rotary_emb=None,
    ):
        if condition_latents.ndim != 3 or condition_latents.shape[-1] != self.condition_dim:
            raise ValueError(
                "condition_latents must have shape "
                f"[batch, tokens, {self.condition_dim}]."
            )
        if condition_latents.shape[0] != image_hidden_states.shape[0]:
            raise ValueError("condition and image token batches must match.")

        projection_dtype = self.to_q.weight.dtype
        queries = self.to_q(image_hidden_states.to(projection_dtype))
        condition_latents = condition_latents.to(
            device=image_hidden_states.device,
            dtype=projection_dtype,
        )
        keys = self.to_k(condition_latents)
        values = self.to_v(condition_latents)

        def split_heads(tensor):
            return tensor.reshape(
                tensor.shape[0],
                tensor.shape[1],
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        attention_dtype = image_hidden_states.dtype
        queries = split_heads(queries).to(attention_dtype)
        keys = split_heads(keys).to(attention_dtype)
        values = split_heads(values).to(attention_dtype)
        if image_rotary_emb is not None:
            queries = apply_rotary_emb(queries, image_rotary_emb, sequence_dim=2)
        if condition_rotary_emb is not None:
            keys = apply_rotary_emb(keys, condition_rotary_emb, sequence_dim=2)
        attended = F.scaled_dot_product_attention(queries, keys, values)
        attended = attended.transpose(1, 2).reshape(
            image_hidden_states.shape[0],
            image_hidden_states.shape[1],
            self.attention_dim,
        )
        return self.to_out(attended.to(self.to_out.weight.dtype)).to(image_hidden_states.dtype)


class ConditionLatentCrossAttention(nn.Module):
    """Block-end cross-attention residuals from an unpooled condition grid."""

    def __init__(
        self,
        condition_dim,
        transformer_dim,
        num_double_blocks,
        num_single_blocks,
        attention_dim=512,
        num_heads=8,
        rope_theta=2000,
        num_position_axes=4,
    ):
        super().__init__()
        block_kwargs = {
            "condition_dim": condition_dim,
            "transformer_dim": transformer_dim,
            "attention_dim": attention_dim,
            "num_heads": num_heads,
        }
        self.double_blocks = nn.ModuleList(
            [ConditionLatentCrossAttentionBlock(**block_kwargs) for _ in range(num_double_blocks)]
        )
        self.single_blocks = nn.ModuleList(
            [ConditionLatentCrossAttentionBlock(**block_kwargs) for _ in range(num_single_blocks)]
        )
        head_dim = int(attention_dim) // int(num_heads)
        if head_dim % int(num_position_axes) != 0:
            raise ValueError("cross-attention head dimension must divide evenly across position axes.")
        axis_dim = head_dim // int(num_position_axes)
        if axis_dim % 2 != 0:
            raise ValueError("each rotary position axis dimension must be even.")
        self.axes_dims_rope = [axis_dim] * int(num_position_axes)
        self.pos_embed = FluxPosEmbed(theta=int(rope_theta), axes_dim=self.axes_dims_rope)

    def rotary_embeddings(self, image_ids, condition_token_count):
        condition_token_count = int(condition_token_count)
        if image_ids.shape[0] < condition_token_count:
            raise ValueError("condition token count exceeds the image ID sequence length.")
        condition_ids = image_ids[-condition_token_count:]
        return self.pos_embed(image_ids), self.pos_embed(condition_ids)

    def double_delta(
        self,
        block_index,
        image_hidden_states,
        condition_latents,
        image_rotary_emb=None,
        condition_rotary_emb=None,
    ):
        return self.double_blocks[block_index](
            image_hidden_states,
            condition_latents,
            image_rotary_emb,
            condition_rotary_emb,
        )

    def single_delta(
        self,
        block_index,
        image_hidden_states,
        condition_latents,
        image_rotary_emb=None,
        condition_rotary_emb=None,
    ):
        return self.single_blocks[block_index](
            image_hidden_states,
            condition_latents,
            image_rotary_emb,
            condition_rotary_emb,
        )


def attach_thermal_coefficient_modulation(transformer, bottleneck_dim=64, log_alpha_mean=0.0, log_alpha_std=1.0):
    return attach_scalar_parameter_modulation(
        transformer,
        bottleneck_dim=bottleneck_dim,
        parameter_names=("thermal_diffusivity",),
        parameter_transforms=("log",),
        normalization_mean=(log_alpha_mean,),
        normalization_std=(log_alpha_std,),
    )


def attach_scalar_parameter_modulation(
    transformer,
    bottleneck_dim=64,
    parameter_names=("thermal_diffusivity",),
    parameter_transforms=("log",),
    normalization_mean=0.0,
    normalization_std=1.0,
):
    if hasattr(transformer, "thermal_modulation"):
        if getattr(transformer.thermal_modulation, "input_dim", None) != len(tuple(parameter_names)):
            raise ValueError("transformer already has scalar modulation with a different input dimension.")
        return transformer.thermal_modulation

    transformer.thermal_modulation = ThermalCoefficientModulator(
        transformer_dim=transformer.inner_dim,
        num_double_blocks=len(transformer.transformer_blocks),
        num_single_blocks=len(transformer.single_transformer_blocks),
        bottleneck_dim=bottleneck_dim,
        parameter_names=parameter_names,
        parameter_transforms=parameter_transforms,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )
    transformer.forward = MethodType(_thermal_modulated_flux2_forward, transformer)
    return transformer.thermal_modulation


def attach_condition_latent_adaln_zero(transformer):
    if hasattr(transformer, "condition_latent_adaln_zero"):
        return transformer.condition_latent_adaln_zero

    transformer.condition_latent_adaln_zero = ConditionLatentAdaLNZero(
        condition_dim=transformer.config.in_channels,
        transformer_dim=transformer.inner_dim,
        num_double_blocks=len(transformer.transformer_blocks),
        num_single_blocks=len(transformer.single_transformer_blocks),
    )
    transformer.forward = MethodType(_thermal_modulated_flux2_forward, transformer)
    return transformer.condition_latent_adaln_zero


def attach_condition_latent_cross_attention(transformer):
    if hasattr(transformer, "condition_latent_cross_attention"):
        return transformer.condition_latent_cross_attention

    transformer.condition_latent_cross_attention = ConditionLatentCrossAttention(
        condition_dim=transformer.config.in_channels,
        transformer_dim=transformer.inner_dim,
        attention_dim=512,
        num_heads=8,
        rope_theta=transformer.config.rope_theta,
        num_position_axes=len(transformer.config.axes_dims_rope),
        num_double_blocks=len(transformer.transformer_blocks),
        num_single_blocks=len(transformer.single_transformer_blocks),
    )
    transformer.forward = MethodType(_thermal_modulated_flux2_forward, transformer)
    return transformer.condition_latent_cross_attention


def _thermal_modulated_flux2_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
    thermal_diffusivity: torch.Tensor | None = None,
    conditioning_values: torch.Tensor | None = None,
    condition_latents: torch.Tensor | None = None,
    target_token_count: int | None = None,
) -> torch.Tensor | Transformer2DModelOutput:
    num_txt_tokens = encoder_hidden_states.shape[1]

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = self.time_guidance_embed(timestep, guidance)
    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    image_rotary_emb = self.pos_embed(img_ids)
    text_rotary_emb = self.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    if conditioning_values is None and thermal_diffusivity is not None:
        conditioning_values = thermal_diffusivity.reshape(-1, 1)
    has_thermal_modulation = conditioning_values is not None and hasattr(self, "thermal_modulation")
    has_condition_latent_modulation = condition_latents is not None and hasattr(
        self,
        "condition_latent_adaln_zero",
    )
    has_condition_cross_attention = condition_latents is not None and hasattr(
        self,
        "condition_latent_cross_attention",
    )
    if has_condition_latent_modulation and has_condition_cross_attention:
        raise RuntimeError("condition AdaLN-Zero and cross-attention modes are mutually exclusive.")
    if has_condition_latent_modulation and target_token_count is None:
        raise ValueError("target_token_count is required with condition_latents.")
    cross_image_rotary_emb = None
    cross_condition_rotary_emb = None
    if has_condition_cross_attention:
        cross_image_rotary_emb, cross_condition_rotary_emb = (
            self.condition_latent_cross_attention.rotary_embeddings(
                img_ids,
                condition_latents.shape[1],
            )
        )

    for index_block, block in enumerate(self.transformer_blocks):
        block_mod_img = double_stream_mod_img
        if has_thermal_modulation or has_condition_latent_modulation:
            block_mod_img = double_stream_mod_img.unsqueeze(1)
        if has_thermal_modulation:
            block_mod_img = block_mod_img + self.thermal_modulation.double_delta(
                index_block, conditioning_values, hidden_states
            )
        if has_condition_latent_modulation:
            block_mod_img = block_mod_img + self.condition_latent_adaln_zero.double_delta(
                index_block,
                condition_latents,
                hidden_states,
                target_token_count,
            )
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                block_mod_img,
                double_stream_mod_txt,
                concat_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=block_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        if has_condition_cross_attention:
            hidden_states = hidden_states + self.condition_latent_cross_attention.double_delta(
                index_block,
                hidden_states,
                condition_latents,
                cross_image_rotary_emb,
                cross_condition_rotary_emb,
            )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        block_mod = single_stream_mod
        if has_thermal_modulation or has_condition_latent_modulation:
            block_mod = single_stream_mod.unsqueeze(1)
        if has_thermal_modulation:
            block_mod = block_mod + self.thermal_modulation.single_delta(
                index_block, conditioning_values, hidden_states
            )
        if has_condition_latent_modulation:
            block_mod = block_mod + self.condition_latent_adaln_zero.single_delta(
                index_block,
                condition_latents,
                hidden_states,
                num_txt_tokens,
                target_token_count,
            )
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                None,
                block_mod,
                concat_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=block_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        if has_condition_cross_attention:
            text_hidden_states = hidden_states[:, :num_txt_tokens]
            image_hidden_states = hidden_states[:, num_txt_tokens:]
            image_hidden_states = image_hidden_states + self.condition_latent_cross_attention.single_delta(
                index_block,
                image_hidden_states,
                condition_latents,
                cross_image_rotary_emb,
                cross_condition_rotary_emb,
            )
            hidden_states = torch.cat([text_hidden_states, image_hidden_states], dim=1)

    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
