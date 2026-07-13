import unittest

import numpy as np
import torch

from generative_physics.ks import (
    KS_GRID_SIZE,
    KS_HORIZON_T,
    KS_STEPS_PER_FRAME,
    hilbert_decode,
    hilbert_encode,
    initial_condition_to_y_constant_rgb,
    ks_effective_discretization,
    make_hilbert_lut,
)


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
