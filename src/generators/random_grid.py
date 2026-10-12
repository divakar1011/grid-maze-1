"""
Deterministic seeded random grid obstacle generator.
"""

import random

import numpy as np


def generate_random_grid(
    rows: int,
    cols: int,
    obstacle_density: float,
    seed: int,
    start: tuple[int, int] = (0, 0),
    goal: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Generates a 2D binary obstacle grid deterministically from student seed.
    0 = Open cell, 1 = Wall / Obstacle.
    """
    if goal is None:
        goal = (rows - 1, cols - 1)

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    # Generate random binary grid
    grid = (np.random.rand(rows, cols) < obstacle_density).astype(int)

    # Enforce ground rules: Start and Goal are always open
    grid[start[0], start[1]] = 0
    grid[goal[0], goal[1]] = 0

    return grid
