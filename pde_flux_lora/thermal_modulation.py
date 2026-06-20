from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput


class ThermalCoefficientModulator(nn.Module):
    def __init__(
        self,
        transformer_dim,
        num_double_blocks,
        num_single_blocks,
        bottleneck_dim=64,
        log_alpha_mean=0.0,
        log_alpha_std=1.0,
    ):
        super().__init__()
        self.transformer_dim = int(transformer_dim)
        self.register_buffer("log_alpha_mean", torch.tensor(float(log_alpha_mean)), persistent=True)
        self.register_buffer("log_alpha_std", torch.tensor(float(log_alpha_std)), persistent=True)
        self.double_blocks = nn.ModuleList(
            [self._make_mlp(bottleneck_dim, self.transformer_dim * 6) for _ in range(num_double_blocks)]
        )
        self.single_blocks = nn.ModuleList(
            [self._make_mlp(bottleneck_dim, self.transformer_dim * 6) for _ in range(num_single_blocks)]
        )

    @staticmethod
    def _make_mlp(bottleneck_dim, output_dim):
        mlp = nn.Sequential(
            nn.Linear(1, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, output_dim),
        )
        nn.init.zeros_(mlp[-1].weight)
        if mlp[-1].bias is not None:
            nn.init.zeros_(mlp[-1].bias)
        return mlp

    def _coefficient_column(self, thermal_diffusivity, hidden_states, dtype):
        coeff = thermal_diffusivity.to(device=hidden_states.device, dtype=dtype)
        log_coeff = torch.log(coeff.clamp_min(torch.finfo(dtype).tiny))
        mean = self.log_alpha_mean.to(device=hidden_states.device, dtype=dtype)
        std = self.log_alpha_std.to(device=hidden_states.device, dtype=dtype).clamp_min(torch.finfo(dtype).eps)
        normalized = (log_coeff - mean) / std
        return normalized.reshape(hidden_states.shape[0], 1)

    def double_delta(self, block_index, thermal_diffusivity, hidden_states):
        block = self.double_blocks[block_index]
        coeff = self._coefficient_column(thermal_diffusivity, hidden_states, block[0].weight.dtype)
        return block(coeff).to(hidden_states.dtype).unsqueeze(1)

    def single_delta(self, block_index, thermal_diffusivity, hidden_states):
        block = self.single_blocks[block_index]
        coeff = self._coefficient_column(thermal_diffusivity, hidden_states, block[0].weight.dtype)
        mod6 = block(coeff).to(hidden_states.dtype)
        first, second = mod6.chunk(2, dim=-1)
        return (first + second).unsqueeze(1)


def attach_thermal_coefficient_modulation(transformer, bottleneck_dim=64, log_alpha_mean=0.0, log_alpha_std=1.0):
    if hasattr(transformer, "thermal_modulation"):
        return transformer.thermal_modulation

    transformer.thermal_modulation = ThermalCoefficientModulator(
        transformer_dim=transformer.inner_dim,
        num_double_blocks=len(transformer.transformer_blocks),
        num_single_blocks=len(transformer.single_transformer_blocks),
        bottleneck_dim=bottleneck_dim,
        log_alpha_mean=log_alpha_mean,
        log_alpha_std=log_alpha_std,
    )
    transformer.forward = MethodType(_thermal_modulated_flux2_forward, transformer)
    return transformer.thermal_modulation


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

    has_thermal_modulation = thermal_diffusivity is not None and hasattr(self, "thermal_modulation")

    for index_block, block in enumerate(self.transformer_blocks):
        block_mod_img = double_stream_mod_img
        if has_thermal_modulation:
            block_mod_img = double_stream_mod_img.unsqueeze(1) + self.thermal_modulation.double_delta(
                index_block, thermal_diffusivity, hidden_states
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

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        block_mod = single_stream_mod
        if has_thermal_modulation:
            block_mod = single_stream_mod.unsqueeze(1) + self.thermal_modulation.single_delta(
                index_block, thermal_diffusivity, hidden_states
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

    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
