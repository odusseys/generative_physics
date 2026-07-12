import numpy as np

from generative_physics.fracture import random_initial_damage_with_defects


def test_fracture_initial_damage_has_no_random_defects():
    rng = np.random.default_rng(0)
    _, num_defects = random_initial_damage_with_defects(rng, 16)
    assert num_defects == 0
