"""
Comprehensive Unit Tests for Breadth-First Search (BFS) Engine.
"""

import unittest

import numpy as np

from src.algorithms.bfs import BreadthFirstSearch


class TestBreadthFirstSearch(unittest.TestCase):
    def setUp(self):
        self.bfs = BreadthFirstSearch()

    def test_open_grid_shortest_path_length(self):
        """On a 5x5 open grid, 4-way Manhattan distance from (0,0) to (4,4) is exactly 8 steps."""
        grid = np.zeros((5, 5), dtype=int)
        res = self.bfs.search(grid, start=(0, 0), goal=(4, 4), allow_diagonal=False)

        self.assertTrue(res.metrics.found)
        self.assertEqual(res.metrics.path_length, 8)
        self.assertEqual(len(res.path), 9)  # 9 nodes = 8 edges
        self.assertEqual(res.path[0], (0, 0))
        self.assertEqual(res.path[-1], (4, 4))

    def test_diagonal_movement_reduces_path_length(self):
        """On a 5x5 open grid with 8-way movement, path length from (0,0) to (4,4) is 4 steps."""
        grid = np.zeros((5, 5), dtype=int)
        res = self.bfs.search(grid, start=(0, 0), goal=(4, 4), allow_diagonal=True)

        self.assertTrue(res.metrics.found)
        self.assertEqual(res.metrics.path_length, 4)
        self.assertEqual(len(res.path), 5)

    def test_completely_blocked_grid(self):
        """When a solid wall blocks all traversal, BFS correctly reports no path."""
        grid = np.zeros((5, 5), dtype=int)
        grid[2, :] = 1  # Solid wall across row 2

        res = self.bfs.search(grid, start=(0, 0), goal=(4, 4), allow_diagonal=False)
        self.assertFalse(res.metrics.found)
        self.assertEqual(len(res.path), 0)
        self.assertEqual(res.metrics.path_length, 0)

    def test_start_equals_goal(self):
        """When start == goal, path length is 0 and node is immediately found."""
        grid = np.zeros((4, 4), dtype=int)
        res = self.bfs.search(grid, start=(2, 2), goal=(2, 2))

        self.assertTrue(res.metrics.found)
        self.assertEqual(res.metrics.path_length, 0)
        self.assertEqual(res.path, [(2, 2)])

    def test_start_or_goal_on_obstacle(self):
        """If start or goal is blocked, return no path immediately."""
        grid = np.zeros((4, 4), dtype=int)
        grid[0, 0] = 1

        res = self.bfs.search(grid, start=(0, 0), goal=(3, 3))
        self.assertFalse(res.metrics.found)
        self.assertEqual(len(res.path), 0)

    def test_detour_around_wall(self):
        """BFS successfully navigates around an obstacle wall to find optimal route."""
        grid = np.zeros((5, 5), dtype=int)
        grid[0:4, 2] = 1

        res = self.bfs.search(grid, start=(0, 0), goal=(0, 4), allow_diagonal=False)
        self.assertTrue(res.metrics.found)
        self.assertEqual(res.metrics.path_length, 12)
        self.assertIn((4, 2), res.path)

    def test_fifo_monotonicity_invariant(self):
        """FIFO invariant verification: nodes are expanded in monotonic non-decreasing order of distance."""
        np.random.seed(42)
        grid = (np.random.rand(15, 15) < 0.25).astype(int)
        grid[0, 0] = 0
        grid[14, 14] = 0

        res = self.bfs.search(grid, start=(0, 0), goal=(14, 14))
        self.assertTrue(res.metrics.fifo_invariant_satisfied)

    def test_level_distribution_integrity(self):
        """Total discovered nodes across all BFS levels equals total nodes explored."""
        grid = np.zeros((10, 10), dtype=int)
        res = self.bfs.search(grid, start=(0, 0), goal=(9, 9))

        self.assertTrue(res.metrics.found)
        total_in_levels = sum(res.level_distribution.values())
        self.assertEqual(total_in_levels, res.metrics.total_explored)

    def test_queue_history_peak_matches_metric(self):
        """Max queue size in history matches recorded max_frontier_size metric."""
        grid = np.zeros((12, 12), dtype=int)
        res = self.bfs.search(grid, start=(0, 0), goal=(11, 11))

        self.assertTrue(res.metrics.found)
        self.assertEqual(max(res.queue_size_history), res.metrics.max_frontier_size)


if __name__ == "__main__":
    unittest.main()
