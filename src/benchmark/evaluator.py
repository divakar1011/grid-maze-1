"""
BFS Scalability and Empirical Complexity Evaluator.
"""

import pandas as pd

from src.algorithms.bfs import BreadthFirstSearch
from src.generators.random_grid import generate_random_grid
from src.generators.solvability import carve_guaranteed_solution, is_solvable


def run_bfs_scaling_benchmark(
    seed: int,
    dimensions: list[int] | None = None,
    obstacle_density: float = 0.25,
    allow_diagonal: bool = False,
) -> pd.DataFrame:
    """
    Evaluates BFS across increasing grid dimensions to empirically verify
    O(V + E) time and O(V) space complexity.
    """
    if dimensions is None:
        dimensions = [10, 15, 20, 25, 30]

    rows = []
    bfs = BreadthFirstSearch()

    for dim in dimensions:
        grid = generate_random_grid(dim, dim, obstacle_density, seed)
        start = (0, 0)
        goal = (dim - 1, dim - 1)

        if not is_solvable(grid, start, goal, allow_diagonal):
            grid = carve_guaranteed_solution(grid, start, goal, seed)

        res = bfs.search(
            grid,
            start=start,
            goal=goal,
            allow_diagonal=allow_diagonal,
            record_history=False,
        )
        m = res.metrics
        v = dim * dim

        rows.append(
            {
                "Dimensions (N x N)": f"{dim} x {dim}",
                "Vertices (V)": v,
                "Path Length": m.path_length,
                "Nodes Explored": m.total_explored,
                "Exploration % of V": f"{(m.total_explored / v) * 100:.1f}%",
                "Max Queue Peak": m.max_frontier_size,
                "Runtime (ms)": f"{m.execution_time_ms:.2f}",
                "FIFO Invariant": "Verified (dist_u <= dist_v)"
                if m.fifo_invariant_satisfied
                else "Violated",
                "Efficiency": f"{m.search_efficiency:.1f}%",
            }
        )

    return pd.DataFrame(rows)


def run_bfs_density_benchmark(
    seed: int,
    dim: int = 20,
    densities: list[float] | None = None,
    allow_diagonal: bool = False,
) -> pd.DataFrame:
    """
    Evaluates BFS across varying obstacle densities to observe how maze constraints
    impact shortest path lengths, queue memory, and node expansion.
    """
    if densities is None:
        densities = [0.10, 0.20, 0.30, 0.40]

    rows = []
    bfs = BreadthFirstSearch()

    for density in densities:
        grid = generate_random_grid(dim, dim, density, seed)
        start = (0, 0)
        goal = (dim - 1, dim - 1)

        solvable_initially = is_solvable(grid, start, goal, allow_diagonal)
        if not solvable_initially:
            grid = carve_guaranteed_solution(grid, start, goal, seed)

        res = bfs.search(
            grid,
            start=start,
            goal=goal,
            allow_diagonal=allow_diagonal,
            record_history=False,
        )
        m = res.metrics

        rows.append(
            {
                "Obstacle Density": f"{int(density * 100)}%",
                "Solvable Naturally": "Yes" if solvable_initially else "Carved Path",
                "Path Length": m.path_length,
                "Nodes Explored": m.total_explored,
                "Max Queue Peak": m.max_frontier_size,
                "Runtime (ms)": f"{m.execution_time_ms:.2f}",
                "Efficiency": f"{m.search_efficiency:.1f}%",
            }
        )

    return pd.DataFrame(rows)
