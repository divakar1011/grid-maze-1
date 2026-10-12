"""
Randomized Prim's Algorithm Maze Generator.

Theoretical Properties:
Generates mazes using a randomized minimum spanning tree algorithm.
Characteristics:
  - High branching factor with numerous short dead ends.
  - Differs from recursive backtracker by avoiding long straight corridors.
  - Produces more central, radiating search patterns for BFS and A*.
"""

import random

import numpy as np


def generate_prims_maze(
    rows: int,
    cols: int,
    seed: int,
    start: tuple[int, int] = (0, 0),
    goal: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Generates a maze using Randomized Prim's Algorithm.
    0 = Open path, 1 = Wall.
    """
    if goal is None:
        goal = (rows - 1, cols - 1)

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    # Grid full of walls
    grid = np.ones((rows, cols), dtype=int)

    # Start cell
    sr, sc = start
    grid[sr, sc] = 0

    # Wall frontier: list of (wall_r, wall_c, from_r, from_c, to_r, to_c)
    frontier: list[tuple[int, int, int, int]] = []

    def add_frontier(r: int, c: int):
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
                frontier.append((r + dr // 2, c + dc // 2, nr, nc))

    add_frontier(sr, sc)

    while frontier:
        idx = random.randrange(len(frontier))
        wr, wc, nr, nc = frontier.pop(idx)

        if grid[nr, nc] == 1:
            grid[wr, wc] = 0
            grid[nr, nc] = 0
            add_frontier(nr, nc)

    # Ensure Goal is open and connected
    grid[goal[0], goal[1]] = 0
    gr, gc = goal
    adjacent = [(gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)]
    for ar, ac in adjacent:
        if 0 <= ar < rows and 0 <= ac < cols:
            grid[ar, ac] = 0
            break

    # Add extra shortcuts for loops (reducing strict dead-ends)
    num_shortcuts = max(1, int(rows * cols * 0.05))
    for _ in range(num_shortcuts):
        rr = random.randint(1, rows - 2)
        cc = random.randint(1, cols - 2)
        grid[rr, cc] = 0

    grid[start[0], start[1]] = 0
    grid[goal[0], goal[1]] = 0

    return grid
