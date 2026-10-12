"""
Solvability verification and guaranteed path carving utilities.
"""

import random
from collections import deque

import numpy as np


def is_solvable(
    grid: np.ndarray,
    start: tuple[int, int] = (0, 0),
    goal: tuple[int, int] | None = None,
    allow_diagonal: bool = False,
) -> bool:
    """Fast connectivity test returning True if a path exists between start and goal."""
    rows, cols = grid.shape
    if goal is None:
        goal = (rows - 1, cols - 1)

    if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
        return False

    queue = deque([start])
    visited = {start}

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        curr = queue.popleft()
        if curr == goal:
            return True

        r, c = curr
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and grid[nr, nc] == 0
                and (nr, nc) not in visited
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return False


def carve_guaranteed_solution(
    grid: np.ndarray,
    start: tuple[int, int] = (0, 0),
    goal: tuple[int, int] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    If the grid is blocked, carves a natural winding unblocked corridor from
    start to goal while preserving the rest of the obstacle layout.
    """
    rows, cols = grid.shape
    if goal is None:
        goal = (rows - 1, cols - 1)

    if is_solvable(grid, start, goal):
        return grid.copy()

    carved_grid = grid.copy()
    random.seed(seed)

    curr_r, curr_c = start
    carved_grid[curr_r, curr_c] = 0

    target_r, target_c = goal

    while (curr_r, curr_c) != (target_r, target_c):
        candidates = []
        # Move closer to target horizontally or vertically with some random wander
        if curr_r < target_r:
            candidates.append((curr_r + 1, curr_c))
        elif curr_r > target_r:
            candidates.append((curr_r - 1, curr_c))

        if curr_c < target_c:
            candidates.append((curr_r, curr_c + 1))
        elif curr_c > target_c:
            candidates.append((curr_r, curr_c - 1))

        # Pick one candidate
        if candidates:
            next_r, next_c = random.choice(candidates)
            curr_r, curr_c = next_r, next_c
            carved_grid[curr_r, curr_c] = 0
        else:
            break

    carved_grid[start[0], start[1]] = 0
    carved_grid[goal[0], goal[1]] = 0
    return carved_grid
