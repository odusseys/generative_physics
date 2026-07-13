import unittest

import numpy as np

from generative_physics.fracture import random_initial_damage_with_defects


class FractureInitialDamageTests(unittest.TestCase):
    def test_initial_damage_has_no_random_defects(self):
        rng = np.random.default_rng(0)
        _, num_defects = random_initial_damage_with_defects(rng, 16)
        self.assertEqual(num_defects, 0)


if __name__ == "__main__":
    unittest.main()
