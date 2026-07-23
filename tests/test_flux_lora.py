import unittest
from copy import deepcopy

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from generative_physics.flux_lora import (
    compile_flux2_regions,
    distilled_timesteps_and_sigmas,
    inference_aligned_training_timesteps_and_sigmas,
    unwrap_flux2_regions,
)
from generative_physics.thermal_modulation import (
    attach_condition_latent_adaln_zero,
    attach_condition_latent_cross_attention,
    attach_scalar_parameter_modulation,
)


def make_tiny_flux2_transformer():
    return Flux2Transformer2DModel(
        in_channels=8,
        out_channels=8,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    )


class FluxRegionalCompilationTests(unittest.TestCase):
    def test_compiles_dit_and_adaln_regions_and_restores_lora_keys(self):
        transformer = make_tiny_flux2_transformer()
        transformer.add_adapter(
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=["to_q", "to_k", "to_v"],
            )
        )
        attach_scalar_parameter_modulation(
            transformer,
            parameter_names=("coefficient",),
            parameter_transforms=("linear",),
        )
        attach_condition_latent_adaln_zero(transformer)
        original_lora_keys = set(get_peft_model_state_dict(transformer))

        stats = compile_flux2_regions(transformer, backend="eager")

        self.assertEqual(stats["dit_blocks"], 4)
        self.assertEqual(stats["adaln_zero"], 3)
        self.assertEqual(stats["scalar_adaln_zero"], 4)
        self.assertEqual(stats["condition_adaln_zero"], 4)
        self.assertEqual(stats["condition_cross_attention"], 0)
        self.assertIn("_orig_mod", transformer.transformer_blocks.__dict__)
        self.assertIn("_orig_mod", transformer.double_stream_modulation_img.__dict__)

        output = transformer(
            hidden_states=torch.randn(1, 3, 8),
            encoder_hidden_states=torch.randn(1, 2, 16),
            timestep=torch.tensor([0.5]),
            img_ids=torch.zeros(3, 4),
            txt_ids=torch.zeros(2, 4),
            conditioning_values=torch.tensor([[0.25]]),
            condition_latents=torch.randn(1, 1, 8),
            target_token_count=1,
            return_dict=False,
        )[0]
        output.square().mean().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in transformer.parameters()
                if parameter.requires_grad
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in transformer.condition_latent_adaln_zero.parameters()
            )
        )

        unwrap_flux2_regions(transformer)

        self.assertEqual(set(get_peft_model_state_dict(transformer)), original_lora_keys)
        self.assertNotIn("_orig_mod", transformer.transformer_blocks.__dict__)
        self.assertNotIn("_orig_mod", transformer.double_stream_modulation_img.__dict__)


class ConditionLatentAdaLNZeroTests(unittest.TestCase):
    def test_keeps_latent_grid_and_only_modulates_matching_target_tokens(self):
        transformer = make_tiny_flux2_transformer()
        modulation = attach_condition_latent_adaln_zero(transformer)

        self.assertIsInstance(modulation.double_blocks[0], torch.nn.Linear)
        self.assertIsInstance(modulation.single_blocks[0], torch.nn.Linear)

        condition_latents = torch.randn(2, 3, 8)
        image_hidden_states = torch.randn(2, 5, transformer.inner_dim)
        joint_hidden_states = torch.randn(2, 7, transformer.inner_dim)
        double_zero = modulation.double_delta(
            0,
            condition_latents,
            image_hidden_states,
            target_token_count=3,
        )
        single_zero = modulation.single_delta(
            0,
            condition_latents,
            joint_hidden_states,
            num_text_tokens=2,
            target_token_count=3,
        )
        torch.testing.assert_close(double_zero, torch.zeros_like(double_zero))
        torch.testing.assert_close(single_zero, torch.zeros_like(single_zero))

        torch.nn.init.zeros_(modulation.double_blocks[0].weight)
        torch.nn.init.ones_(modulation.double_blocks[0].bias)
        torch.nn.init.zeros_(modulation.single_blocks[0].weight)
        torch.nn.init.ones_(modulation.single_blocks[0].bias)
        double_delta = modulation.double_delta(
            0,
            condition_latents,
            image_hidden_states,
            target_token_count=3,
        )
        single_delta = modulation.single_delta(
            0,
            condition_latents,
            joint_hidden_states,
            num_text_tokens=2,
            target_token_count=3,
        )

        torch.testing.assert_close(double_delta[:, :3], torch.ones_like(double_delta[:, :3]))
        torch.testing.assert_close(double_delta[:, 3:], torch.zeros_like(double_delta[:, 3:]))
        torch.testing.assert_close(single_delta[:, :2], torch.zeros_like(single_delta[:, :2]))
        torch.testing.assert_close(single_delta[:, 2:5], torch.ones_like(single_delta[:, 2:5]))
        torch.testing.assert_close(single_delta[:, 5:], torch.zeros_like(single_delta[:, 5:]))

    def test_rejects_nonmatching_condition_and_target_latent_grids(self):
        transformer = make_tiny_flux2_transformer()
        modulation = attach_condition_latent_adaln_zero(transformer)
        with self.assertRaisesRegex(ValueError, "must match elementwise"):
            modulation.double_delta(
                0,
                torch.randn(1, 2, 8),
                torch.randn(1, 5, transformer.inner_dim),
                target_token_count=3,
            )


class ConditionLatentCrossAttentionTests(unittest.TestCase):
    def test_zero_output_preserves_transformer_and_receives_gradients(self):
        transformer = make_tiny_flux2_transformer()
        baseline = deepcopy(transformer)
        cross_attention = attach_condition_latent_cross_attention(transformer)
        self.assertEqual(cross_attention.double_blocks[0].attention_dim, 512)
        self.assertEqual(cross_attention.double_blocks[0].num_heads, 8)
        self.assertEqual(cross_attention.axes_dims_rope, [16, 16, 16, 16])
        inputs = {
            "hidden_states": torch.randn(1, 5, 8),
            "encoder_hidden_states": torch.randn(1, 2, 16),
            "timestep": torch.tensor([0.5]),
            "img_ids": torch.zeros(5, 4),
            "txt_ids": torch.zeros(2, 4),
            "return_dict": False,
        }

        expected = baseline(**inputs)[0]
        actual = transformer(
            **inputs,
            condition_latents=torch.randn(1, 3, 8),
            target_token_count=2,
        )[0]
        torch.testing.assert_close(actual, expected)

        actual.square().mean().backward()
        self.assertTrue(
            any(
                block.to_out.weight.grad is not None
                for block in [*cross_attention.double_blocks, *cross_attention.single_blocks]
            )
        )

    def test_cross_attention_returns_one_residual_per_image_token(self):
        transformer = make_tiny_flux2_transformer()
        cross_attention = attach_condition_latent_cross_attention(transformer)
        image_hidden_states = torch.randn(2, 7, transformer.inner_dim)
        condition_latents = torch.randn(2, 3, 8)

        delta = cross_attention.double_delta(
            0,
            image_hidden_states,
            condition_latents,
        )

        self.assertEqual(delta.shape, image_hidden_states.shape)
        torch.testing.assert_close(delta, torch.zeros_like(delta))

    def test_cross_attention_uses_flux_image_ids_for_rotary_positions(self):
        transformer = make_tiny_flux2_transformer()
        cross_attention = attach_condition_latent_cross_attention(transformer)
        block = cross_attention.double_blocks[0]
        torch.nn.init.normal_(block.to_out.weight)
        image_hidden_states = torch.randn(1, 5, transformer.inner_dim)
        condition_latents = torch.randn(1, 3, 8)
        zero_ids = torch.zeros(5, 4)
        spatial_ids = zero_ids.clone()
        spatial_ids[:, 2] = torch.arange(5)
        zero_rotary = cross_attention.rotary_embeddings(zero_ids, 3)
        spatial_rotary = cross_attention.rotary_embeddings(spatial_ids, 3)

        zero_position_output = cross_attention.double_delta(
            0,
            image_hidden_states,
            condition_latents,
            *zero_rotary,
        )
        spatial_position_output = cross_attention.double_delta(
            0,
            image_hidden_states,
            condition_latents,
            *spatial_rotary,
        )

        self.assertFalse(torch.allclose(zero_position_output, spatial_position_output))

    def test_cross_attention_regions_compile_and_unwrap(self):
        transformer = make_tiny_flux2_transformer()
        attach_condition_latent_cross_attention(transformer)

        stats = compile_flux2_regions(transformer, backend="eager")

        self.assertEqual(stats["condition_cross_attention"], 4)
        unwrap_flux2_regions(transformer)
        self.assertNotIn(
            "_orig_mod",
            transformer.condition_latent_cross_attention.double_blocks.__dict__,
        )


class FluxScheduleTests(unittest.TestCase):
    def test_training_schedule_densifies_the_inference_schedule(self):
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=4096,
            time_shift_type="exponential",
        )

        _, training_sigmas, _ = inference_aligned_training_timesteps_and_sigmas(
            scheduler,
            image_seq_len=1024,
            device="cpu",
            dtype=torch.float32,
            inference_num_steps=20,
        )
        _, inference_sigmas = distilled_timesteps_and_sigmas(
            scheduler,
            image_seq_len=1024,
            device="cpu",
            dtype=torch.float32,
            num_inference_steps=20,
        )

        self.assertEqual(len(training_sigmas), 1000)
        torch.testing.assert_close(
            training_sigmas[::50],
            inference_sigmas,
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
