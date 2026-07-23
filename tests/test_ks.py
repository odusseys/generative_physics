import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from generative_physics.config import (
    FLUX2_KLEIN_BASE_MODEL_ID,
    FLUX2_KLEIN_DISTILLED_MODEL_ID,
    TrainingConfig,
)
from generative_physics.ks import (
    KS_GRID_SIZE,
    KS_HORIZON_T,
    KS_RANDOM_SAMPLING_MAE,
    KS_STEPS_PER_FRAME,
    hilbert_decode,
    hilbert_encode,
    initial_condition_to_y_constant_rgb,
    ks_effective_discretization,
    ks_integrate_cnab2_batch,
    ks_integrate_cnab2_torch,
    make_hilbert_lut,
)
from generative_physics.run import (
    resolve_pretrained_model_source,
    resolve_vae_lora_checkpoint,
)


class KsFluxConfigTests(unittest.TestCase):
    def test_cached_model_snapshot_is_used_without_hub_metadata_request(self):
        cached_index = "/cache/snapshots/commit/model_index.json"
        with patch(
            "generative_physics.run.try_to_load_from_cache",
            return_value=cached_index,
        ):
            source = resolve_pretrained_model_source("organization/model")

        self.assertEqual(source, Path(cached_index).parent)

    def test_uncached_model_keeps_remote_model_id(self):
        with patch(
            "generative_physics.run.try_to_load_from_cache",
            return_value=None,
        ):
            source = resolve_pretrained_model_source("organization/model")

        self.assertEqual(source, "organization/model")

    def test_ks_uses_base_model_with_inference_aligned_training_sigmas(self):
        config = TrainingConfig(pde_kind="ks")

        self.assertEqual(config.model_id, FLUX2_KLEIN_BASE_MODEL_ID)
        self.assertEqual(config.inference_num_steps, 20)
        self.assertEqual(config.inference_guidance_scale, 4.0)
        self.assertEqual(config.training_sigma_mode, "inference_aligned")
        self.assertFalse(config.transformer_gradient_checkpointing)
        self.assertTrue(config.transformer_compile_regions)
        self.assertEqual(config.transformer_compile_mode, "default")
        self.assertEqual(config.ks_condition_adapter_mode, "adaln_zero")

    def test_ks_condition_adapter_mode_is_validated(self):
        with self.assertRaisesRegex(ValueError, "ks_condition_adapter_mode"):
            TrainingConfig(pde_kind="ks", ks_condition_adapter_mode="invalid")

    def test_other_modes_keep_distilled_schedule(self):
        config = TrainingConfig(pde_kind="heat")

        self.assertEqual(config.model_id, FLUX2_KLEIN_DISTILLED_MODEL_ID)
        self.assertEqual(config.inference_num_steps, 4)
        self.assertEqual(config.inference_guidance_scale, 1.0)
        self.assertEqual(config.training_sigma_mode, "distilled")

    def test_latest_ks_vae_lora_checkpoint_is_selected(self):
        with TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            older = checkpoint_dir / "step-001000.safetensors"
            latest = checkpoint_dir / "step-002000.safetensors"
            older.touch()
            latest.touch()
            os.utime(older, ns=(1_000_000_000, 1_000_000_000))
            os.utime(latest, ns=(2_000_000_000, 2_000_000_000))
            config = TrainingConfig(
                pde_kind="ks",
                ks_vae_lora_dir=str(checkpoint_dir),
            )

            selected = resolve_vae_lora_checkpoint(config)

        self.assertEqual(selected, latest.resolve())

    def test_explicit_vae_lora_checkpoint_takes_precedence(self):
        with TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            discovered = checkpoint_dir / "step-002000.safetensors"
            explicit = checkpoint_dir / "chosen.safetensors"
            discovered.touch()
            explicit.touch()
            config = TrainingConfig(
                pde_kind="ks",
                ks_vae_lora_dir=str(checkpoint_dir),
            )

            selected = resolve_vae_lora_checkpoint(config, explicit)

        self.assertEqual(selected, explicit.resolve())


class KsEncodingTests(unittest.TestCase):
    def test_y_constant_encoding_repeats_grayscale_profile(self):
        encoded = initial_condition_to_y_constant_rgb(
            torch.tensor([-4.0, 0.0, 4.0]),
            image_size=5,
            value_bounds=(-4.0, 4.0),
        )

        self.assertEqual(encoded.shape, (5, 5, 3))
        np.testing.assert_allclose(encoded, np.repeat(encoded[:1], 5, axis=0), atol=0.0)
        np.testing.assert_allclose(encoded[..., 0], encoded[..., 1], atol=0.0)
        np.testing.assert_allclose(encoded[..., 1], encoded[..., 2], atol=0.0)
        np.testing.assert_allclose(encoded[0, :, 0], np.linspace(0.0, 1.0, 5), atol=1e-6)

    def test_hilbert_encoding_round_trips_curve_values(self):
        values = torch.linspace(0.0, 1.0, 65)
        lut = make_hilbert_lut(order=3)

        decoded = hilbert_decode(hilbert_encode(values, lut), lut)

        torch.testing.assert_close(decoded, values, atol=1e-6, rtol=0.0)


class KsDiscretizationTests(unittest.TestCase):
    def test_random_sampling_error_baseline(self):
        self.assertEqual(KS_RANDOM_SAMPLING_MAE, 0.20495)

    def test_batched_integrator_matches_single_state_integrator(self):
        x = torch.linspace(0.0, 2.0 * torch.pi, 32, dtype=torch.float64)
        initial_states = torch.stack([torch.sin(x), 0.5 * torch.cos(2.0 * x)])

        batched = ks_integrate_cnab2_batch(
            initial_states,
            Lx=32.0,
            dt=0.001,
            Nt=4,
            nsave=2,
        )
        individual = torch.stack(
            [
                ks_integrate_cnab2_torch(
                    state,
                    Lx=32.0,
                    dt=0.001,
                    Nt=4,
                    nsave=2,
                )
                for state in initial_states
            ]
        )

        torch.testing.assert_close(batched, individual)

    def test_high_resolution_render_uses_matching_simulation_grid_and_fixed_horizon(self):
        image_size = 1024
        nx, time_frames, dt = ks_effective_discretization(
            Nx=KS_GRID_SIZE,
            image_size=image_size,
            steps_per_frame=KS_STEPS_PER_FRAME,
        )

        self.assertEqual(nx, image_size)
        self.assertEqual(time_frames, image_size)
        self.assertAlmostEqual(dt * (time_frames - 1) * KS_STEPS_PER_FRAME, KS_HORIZON_T)


if __name__ == "__main__":
    unittest.main()
