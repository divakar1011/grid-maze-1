"""
Unit tests for maze generators, presets, solvability, and submission verification.
"""

import unittest

import numpy as np

from src.benchmark.integrity import (
    generate_submission_signature,
    verify_submission_signature,
)
from src.generators.presets import get_available_presets, get_preset_maze
from src.generators.prims import generate_prims_maze
from src.generators.random_grid import generate_random_grid
from src.generators.recursive_backtracker import generate_recursive_backtracker_maze
from src.generators.solvability import carve_guaranteed_solution, is_solvable


class TestGeneratorsAndIntegrity(unittest.TestCase):
    def test_random_grid_deterministic_reproducibility(self):
        """Identical student register numbers must generate 100% identical grid arrays."""
        seed = 113025148022
        grid1 = generate_random_grid(20, 20, 0.25, seed)
        grid2 = generate_random_grid(20, 20, 0.25, seed)

        np.testing.assert_array_equal(
            grid1, grid2, "Grids with identical seed must match bit-for-bit"
        )

    def test_different_seeds_produce_distinct_grids(self):
        """Different student seeds must produce distinct grid layouts."""
        grid1 = generate_random_grid(20, 20, 0.25, seed=11111)
        grid2 = generate_random_grid(20, 20, 0.25, seed=22222)

        self.assertFalse(np.array_equal(grid1, grid2))

    def test_start_and_goal_always_open(self):
        """Start (0,0) and Goal (rows-1, cols-1) must never be blocked."""
        for seed in [1, 999, 123456789]:
            grid = generate_random_grid(15, 15, 0.40, seed)
            self.assertEqual(grid[0, 0], 0)
            self.assertEqual(grid[14, 14], 0)

    def test_recursive_backtracker_generates_solvable_maze(self):
        """Recursive backtracker must produce a connected solvable maze."""
        grid = generate_recursive_backtracker_maze(15, 15, seed=42)
        self.assertEqual(grid[0, 0], 0)
        self.assertEqual(grid[14, 14], 0)
        self.assertTrue(is_solvable(grid, (0, 0), (14, 14)))

    def test_prims_generates_solvable_maze(self):
        """Randomized Prim's algorithm must produce a connected solvable maze."""
        grid = generate_prims_maze(15, 15, seed=99)
        self.assertEqual(grid[0, 0], 0)
        self.assertEqual(grid[14, 14], 0)
        self.assertTrue(is_solvable(grid, (0, 0), (14, 14)))

    def test_all_presets_valid_and_solvable(self):
        """All curated benchmark presets must generate valid binary grids."""
        presets = get_available_presets()
        for name in presets:
            grid = get_preset_maze(name, 21, 21)
            self.assertEqual(grid.shape, (21, 21))
            self.assertEqual(grid[0, 0], 0)
            self.assertEqual(grid[20, 20], 0)
            self.assertTrue(
                is_solvable(grid, (0, 0), (20, 20)), f"Preset '{name}' must be solvable"
            )

    def test_carve_guaranteed_solution(self):
        """Blocked grid is successfully made solvable by carving."""
        blocked_grid = np.ones((10, 10), dtype=int)
        blocked_grid[0, 0] = 0
        blocked_grid[9, 9] = 0

        self.assertFalse(is_solvable(blocked_grid, (0, 0), (9, 9)))
        carved = carve_guaranteed_solution(blocked_grid, (0, 0), (9, 9))
        self.assertTrue(is_solvable(carved, (0, 0), (9, 9)))

    def test_cryptographic_verification_signature(self):
        """SHA-256 signature accurately authenticates run report and detects tampering."""
        grid = np.zeros((10, 10), dtype=int)
        seed = 113025148022
        algo = "Breadth-First Search (BFS)"
        path = [(0, 0), (0, 1), (1, 1)]

        sig = generate_submission_signature(
            student_seed=seed,
            grid=grid,
            algorithm_name=algo,
            path_length=2,
            total_explored=5,
            path_coordinates=path,
        )

        report = {
            "student_register_number_seed": seed,
            "algorithm_name": algo,
            "execution_metrics": {"shortest_path_steps": 2, "total_nodes_explored": 5},
            "reconstructed_path_coordinates": path,
            "verification_signature": sig,
        }

        self.assertTrue(verify_submission_signature(report, grid))

        # Tampered report (falsifying path steps) must fail verification
        report["execution_metrics"]["shortest_path_steps"] = 1
        self.assertFalse(verify_submission_signature(report, grid))


if __name__ == "__main__":
    unittest.main()
