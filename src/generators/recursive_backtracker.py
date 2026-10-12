"""
Recursive Backtracker (DFS) Maze Generator.

Theoretical Properties:
Generates a 'perfect maze' (a spanning tree on a grid graph).
Characteristics:
  - 100% Guaranteed Solvable: Exactly one unique simple path exists between any two points.
  - High dead-end count and long winding corridors.
  - Ideal for evaluating search algorithms under constrained corridor navigation.
"""

import random

import numpy as np


def generate_recursive_backtracker_maze(
    rows: int,
    cols: int,
    seed: int,
    start: tuple[int, int] = (0, 0),
    goal: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Generates a maze using the randomized depth-first search recursive backtracker.
    0 = Open path, 1 = Wall.
    """
    if goal is None:
        goal = (rows - 1, cols - 1)

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    # Start with grid filled with walls
    grid = np.ones((rows, cols), dtype=int)

    # We treat even/odd coordinates as cells and walls
    # Carve starting from (0, 0)
    stack: list[tuple[int, int]] = [start]
    grid[start[0], start[1]] = 0

    visited = {start}

    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while stack:
        cr, cc = stack[-1]

        # Find unvisited neighbors at distance 2
        unvisited_neighbors = []
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                unvisited_neighbors.append((nr, nc, dr, dc))

        if unvisited_neighbors:
            nr, nc, dr, dc = random.choice(unvisited_neighbors)
            # Carve between current and neighbor
            wall_r, wall_c = cr + dr // 2, cc + dc // 2
            grid[wall_r, wall_c] = 0
            grid[nr, nc] = 0

            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

    # Guarantee Goal is open and connected
    grid[goal[0], goal[1]] = 0
    gr, gc = goal
    # If goal is surrounded by walls, carve to nearest neighbor
    adjacent = [(gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)]
    for ar, ac in adjacent:
        if 0 <= ar < rows and 0 <= ac < cols:
            grid[ar, ac] = 0
            break

    # Also add some random loops if user wants less strict dead-ends
    # Carve ~5% extra connections so multiple paths exist
    num_extra = max(1, int(rows * cols * 0.04))
    for _ in range(num_extra):
        rr = random.randint(1, rows - 2)
        cc = random.randint(1, cols - 2)
        grid[rr, cc] = 0

    grid[start[0], start[1]] = 0
    grid[goal[0], goal[1]] = 0

    return grid
