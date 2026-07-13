import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from generative_physics.config import TrainingConfig
from generative_physics.data import make_pde_records, record_to_item
from generative_physics.flux_lora import (
    _prepare_joint_target_ids,
    infer_joint_solutions,
    pde_lora_loss,
)
from generative_physics.navier_stokes import (
    NAVIER_STOKES_DENSITY_MAX,
    NAVIER_STOKES_DENSITY_MIN,
    NAVIER_STOKES_MULTIPLE_TIMES,
    NAVIER_STOKES_VISCOSITY_MAX,
    NAVIER_STOKES_VISCOSITY_MIN,
    estimate_navier_stokes_initial_condition_coverage,
    sample_divergence_free_velocity_batch,
    solve_navier_stokes_2d,
    velocity_to_rgb,
)


class NavierStokesSamplingTests(unittest.TestCase):
    def test_sampled_velocity_is_divergence_free_and_speed_bounded(self):
        velocity, params = sample_divergence_free_velocity_batch(
            [3, 4],
            grid_size=32,
            device="cpu",
        )

        speed = torch.linalg.vector_norm(velocity, dim=1)
        self.assertLessEqual(float(speed.max()), 1.0)
        for sample_index, sample_params in enumerate(params):
            self.assertAlmostEqual(
                float(speed[sample_index].max()),
                sample_params["initial_max_speed"],
                places=5,
            )
            self.assertTrue(NAVIER_STOKES_DENSITY_MIN <= sample_params["density"] <= NAVIER_STOKES_DENSITY_MAX)
            self.assertTrue(
                NAVIER_STOKES_VISCOSITY_MIN
                <= sample_params["kinematic_viscosity"]
                <= NAVIER_STOKES_VISCOSITY_MAX
            )

        grid_size = velocity.shape[-1]
        frequency = 2.0 * torch.pi * torch.fft.fftfreq(grid_size, d=1.0 / grid_size)
        velocity_x_hat = torch.fft.fft2(velocity[:, 0])
        velocity_y_hat = torch.fft.fft2(velocity[:, 1])
        divergence = torch.fft.ifft2(
            1j * frequency[None, :] * velocity_x_hat
            + 1j * frequency[:, None] * velocity_y_hat
        ).real
        self.assertLess(float(divergence.abs().max()), 1e-4)

    def test_unforced_viscous_flow_loses_kinetic_energy(self):
        initial_velocity, params = sample_divergence_free_velocity_batch(
            [5],
            grid_size=32,
            device="cpu",
        )
        trajectory = solve_navier_stokes_2d(
            initial_velocity,
            [params[0]["kinematic_viscosity"]],
            save_times=[0.0, 0.02, 0.04],
        )

        energy = 0.5 * trajectory.square().sum(dim=2).mean(dim=(-2, -1))[0]
        self.assertTrue(torch.isfinite(trajectory).all())
        self.assertTrue(torch.all(energy[1:] <= energy[:-1] + 1e-6))

    def test_velocity_rgb_channels_encode_direction_and_speed(self):
        velocity = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, -0.5]],
            ]
        )

        encoded = velocity_to_rgb(velocity)

        torch.testing.assert_close(encoded[0, 0], torch.tensor([1.0, 0.5, 1.0]))
        torch.testing.assert_close(encoded[0, 1], torch.tensor([0.5, 0.0, 0.5]))

    def test_initial_condition_coverage_improves_with_sample_count(self):
        coverage = estimate_navier_stokes_initial_condition_coverage(
            range(100, 164),
            range(200, 216),
            simulation_grid_size=16,
            feature_grid_size=8,
            sample_sizes=(8, 16, 32, 64),
            distance_tolerances=None,
            mse_tolerances=(0.1,),
            coverage_quantiles=(0.5, 0.9),
            generation_batch_size=64,
            distance_batch_size=8,
        )

        empirical = coverage["empirical_coverage"][:, 0]
        self.assertTrue(np.all(empirical[1:] >= empirical[:-1]))
        self.assertTrue(np.all(coverage["distance_quantiles"][:, 1:] <= coverage["distance_quantiles"][:, :-1]))
        self.assertTrue(np.isfinite(coverage["estimated_sample_counts"]).all())
        np.testing.assert_allclose(coverage["mse_tolerances"], [0.1])


class NavierStokesModeTests(unittest.TestCase):
    def test_mode_generates_condition_and_final_solution_records(self):
        config = TrainingConfig(
            pde_kind="navier_stokes",
            train_image_size=32,
            output_image_size=32,
            navier_stokes_grid_size=32,
            navier_stokes_final_time=0.02,
        )

        record = make_pde_records(
            1,
            seed_offset=17,
            sim_device="cpu",
            config=config,
            progress=False,
        )[0]

        self.assertEqual(record["initial"].shape, (32, 32, 3))
        self.assertEqual(record["solution"].shape, (32, 32, 3))
        self.assertEqual(record["params"]["pde"], "navier_stokes")
        self.assertEqual(
            record["params"]["conditioning_names"],
            ("density", "kinematic_viscosity"),
        )
        self.assertTrue(np.isfinite(record["solution"]).all())
        self.assertGreater(float(np.abs(record["solution"] - record["initial"]).mean()), 0.0)

    def test_multiple_mode_generates_four_joint_targets(self):
        config = TrainingConfig(
            pde_kind="navier_stokes_multiple",
            train_image_size=16,
            output_image_size=16,
            navier_stokes_grid_size=16,
        )

        record = make_pde_records(
            1,
            seed_offset=18,
            sim_device="cpu",
            config=config,
            progress=False,
        )[0]
        item = record_to_item(record, image_size=16)

        self.assertEqual(record["params"]["pde"], "navier_stokes_multiple")
        self.assertEqual(record["params"]["solution_times"], NAVIER_STOKES_MULTIPLE_TIMES)
        self.assertEqual(len(record["solutions"]), 4)
        self.assertEqual(item["target_pixels"].shape, (4, 3, 16, 16))
        self.assertEqual(item["condition_pixels"][0].shape, (3, 16, 16))


class FluxJointTargetTests(unittest.TestCase):
    def test_joint_loss_concatenates_four_targets_before_one_condition(self):
        class RecordingTransformer:
            dtype = torch.float32

            def __init__(self):
                self.hidden_shape = None
                self.image_ids = None

            def __call__(self, hidden_states, img_ids, **kwargs):
                self.hidden_shape = hidden_states.shape
                self.image_ids = img_ids
                return (torch.zeros_like(hidden_states),)

        transformer = RecordingTransformer()
        pipe = SimpleNamespace(
            vae=SimpleNamespace(dtype=torch.float32),
            scheduler=object(),
            transformer=transformer,
        )
        batch = {
            "target_pixels": torch.randn(2, 4, 3, 8, 8),
            "condition_pixels": [torch.randn(2, 3, 8, 8)],
            "conditioning_values": torch.tensor([[1.0, 1e-3], [1.1, 2e-3]]),
        }

        def fake_schedule(scheduler, image_seq_len, batch_size, latent_dtype, device, num_inference_steps):
            timesteps = torch.full((batch_size,), 500.0, device=device)
            sigmas = torch.full((batch_size,), 0.5, device=device, dtype=latent_dtype)
            return timesteps, sigmas, timesteps[:1], sigmas[:1]

        with (
            patch("generative_physics.flux_lora.encode_flux2_latents", side_effect=lambda pipe, pixels, device: pixels),
            patch("generative_physics.flux_lora.sample_exact_distilled_training_step", side_effect=fake_schedule),
        ):
            loss = pde_lora_loss(
                pipe,
                batch,
                prompt_embeds=torch.zeros(1, 2, 3),
                text_ids=torch.zeros(1, 2, 4),
                device="cpu",
            )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(transformer.hidden_shape, (2, 5 * 8 * 8, 3))
        target_id_blocks = transformer.image_ids[:, : 4 * 8 * 8].split(8 * 8, dim=1)
        self.assertEqual(
            [torch.unique(block[..., 0]).item() for block in target_id_blocks],
            [1, 2, 3, 4],
        )
        self.assertEqual(torch.unique(transformer.image_ids[:, 4 * 8 * 8 :, 0]).item(), 10)

    def test_single_target_keeps_original_zero_time_id(self):
        ids = _prepare_joint_target_ids([torch.empty(2, 8, 3, 4)])
        self.assertEqual(torch.unique(ids[..., 0]).item(), 0)

    def test_joint_inference_splits_and_decodes_four_target_blocks(self):
        class RecordingTransformer:
            dtype = torch.float32
            config = SimpleNamespace(in_channels=16)
            training = True

            def __init__(self):
                self.image_ids = None

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

            def cache_context(self, name):
                return nullcontext()

            def __call__(self, hidden_states, img_ids, **kwargs):
                self.image_ids = img_ids
                return (torch.zeros_like(hidden_states),)

        class FakeScheduler:
            config = SimpleNamespace(use_flow_sigmas=True)

            def set_begin_index(self, index):
                self.begin_index = index

            def step(self, noise_pred, timestep, latents, return_dict=False):
                return (latents,)

        class FakePipe:
            vae_scale_factor = 1

            def __init__(self):
                self.transformer = RecordingTransformer()
                self.scheduler = FakeScheduler()
                self.vae = SimpleNamespace(
                    dtype=torch.float32,
                    bn=SimpleNamespace(
                        running_mean=torch.zeros(4),
                        running_var=torch.ones(4),
                    ),
                    config=SimpleNamespace(batch_norm_eps=1e-5),
                    decode=lambda latents, return_dict=False: (latents,),
                )
                self.image_processor = SimpleNamespace(
                    postprocess=lambda images, output_type: images.repeat(1, 3, 1, 1)
                    .permute(0, 2, 3, 1)
                    .numpy()
                )

            @staticmethod
            def _ids(time_id):
                ids = torch.cartesian_prod(
                    torch.tensor([time_id]),
                    torch.arange(2),
                    torch.arange(2),
                    torch.arange(1),
                )
                return ids.unsqueeze(0)

            def prepare_latents(self, **kwargs):
                latents = torch.randn(1, 4, 2, 2, generator=kwargs["generator"])
                return latents.flatten(2).permute(0, 2, 1), self._ids(0)

            def prepare_image_latents(self, **kwargs):
                return torch.zeros(1, 4, 4), self._ids(10)

        pipe = FakePipe()
        with patch(
            "generative_physics.flux_lora.retrieve_timesteps",
            return_value=(torch.tensor([1000.0, 500.0]), 2),
        ):
            images = infer_joint_solutions(
                pipe,
                np.zeros((8, 8, 3), dtype=np.float32),
                prompt_embeds=torch.zeros(1, 2, 3),
                text_ids=torch.zeros(1, 2, 4),
                device="cpu",
                num_outputs=4,
                train_image_size=8,
                conditioning_values=(1.0, 1e-3),
            )

        self.assertEqual(images.shape, (4, 4, 4, 3))
        target_blocks = pipe.transformer.image_ids[:, :16].split(4, dim=1)
        self.assertEqual(
            [torch.unique(block[..., 0]).item() for block in target_blocks],
            [1, 2, 3, 4],
        )
        self.assertEqual(torch.unique(pipe.transformer.image_ids[:, 16:, 0]).item(), 10)
        self.assertTrue(pipe.transformer.training)


if __name__ == "__main__":
    unittest.main()
