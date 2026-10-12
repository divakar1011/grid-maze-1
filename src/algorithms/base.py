"""
Base abstractions, data structures, and metrics for Breadth-First Search (BFS) graph engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Coordinate = tuple[int, int]


@dataclass
class AlgorithmMetrics:
    """Telemetry and performance metrics recorded during BFS graph search."""

    algorithm_name: str
    found: bool
    path_length: int
    path_cost: float
    total_explored: int
    max_frontier_size: int
    execution_time_ms: float
    search_efficiency: float  # (path_length / total_explored) * 100
    is_optimal: bool = True
    max_depth_reached: int = 0
    fifo_invariant_satisfied: bool = True


@dataclass
class SearchResult:
    """Comprehensive result object returned by BFS search."""

    algorithm_name: str
    path: list[Coordinate]
    visited_order: list[Coordinate]
    steps_history: list[dict[str, Any]]
    metrics: AlgorithmMetrics
    parent_map: dict[Coordinate, Coordinate | None] = field(default_factory=dict)
    distance_matrix: np.ndarray | None = None
    level_distribution: dict[int, int] = field(default_factory=dict)
    queue_size_history: list[int] = field(default_factory=list)


class SearchAlgorithm(ABC):
    """Abstract base class for grid-based search algorithms."""

    def __init__(self, name: str, is_optimal: bool = True):
        self.name = name
        self.is_optimal = is_optimal

    @staticmethod
    def get_neighbors(
        curr: Coordinate, grid: np.ndarray, allow_diagonal: bool = False
    ) -> list[tuple[Coordinate, float]]:
        """
        Returns valid unblocked neighbors and transition step costs.

        Orthogonal moves cost 1.0.
        Diagonal moves (if enabled) cost sqrt(2) ~ 1.4142.
        """
        r, c = curr
        rows, cols = grid.shape
        neighbors: list[tuple[Coordinate, float]] = []

        # 4-way orthogonal directions
        orthogonal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in orthogonal_dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                neighbors.append(((nr, nc), 1.0))

        # 8-way diagonal directions
        if allow_diagonal:
            diag_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dr, dc in diag_dirs:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and grid[nr, nc] == 0
                    and (grid[r, nc] == 0 or grid[nr, c] == 0)
                ):
                    neighbors.append(((nr, nc), 1.4142))

        return neighbors

    @staticmethod
    def reconstruct_path(
        parent: dict[Coordinate, Coordinate | None], goal: Coordinate
    ) -> list[Coordinate]:
        """Reconstructs the path from start to goal via backwards parent traversal."""
        path: list[Coordinate] = []
        curr: Coordinate | None = goal
        while curr is not None:
            path.append(curr)
            curr = parent.get(curr)
        path.reverse()
        return path

    @abstractmethod
    def search(
        self,
        grid: np.ndarray,
        start: Coordinate = (0, 0),
        goal: Coordinate | None = None,
        allow_diagonal: bool = False,
        record_history: bool = True,
    ) -> SearchResult:
        """Executes the search algorithm and returns the SearchResult."""
