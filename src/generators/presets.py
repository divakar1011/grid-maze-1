"""
Curated benchmark topology presets for pathfinding algorithm evaluation.
"""

import numpy as np


def generate_bottleneck_preset(rows: int, cols: int) -> np.ndarray:
    """Two large chambers separated by a dividing wall with a single 1-cell chokepoint."""
    grid = np.zeros((rows, cols), dtype=int)
    mid_c = cols // 2
    # Build wall along mid_c
    grid[:, mid_c] = 1
    # Single opening in the middle
    opening_r = rows // 2
    grid[opening_r, mid_c] = 0
    grid[0, 0] = 0
    grid[rows - 1, cols - 1] = 0
    return grid


def generate_spiral_preset(rows: int, cols: int) -> np.ndarray:
    """Concentric winding spiral maze."""
    grid = np.zeros((rows, cols), dtype=int)
    top, bottom, left, right = 1, rows - 2, 1, cols - 2

    while top < bottom - 1 and left < right - 1:
        # Top wall
        grid[top, left:right] = 1
        # Right wall
        grid[top:bottom, right] = 1
        # Bottom wall
        grid[bottom, left + 1 : right + 1] = 1
        # Left wall
        grid[top + 2 : bottom + 1, left] = 1

        top += 2
        bottom -= 2
        left += 2
        right -= 2

    grid[0, 0] = 0
    grid[rows - 1, cols - 1] = 0
    return grid


def generate_slits_preset(rows: int, cols: int) -> np.ndarray:
    """Alternating horizontal walls with openings at opposite ends (Zig-zag snake)."""
    grid = np.zeros((rows, cols), dtype=int)
    row_step = max(3, rows // 6)

    for idx, r in enumerate(range(row_step, rows - 1, row_step)):
        grid[r, :] = 1
        if idx % 2 == 0:
            grid[r, cols - 1] = 0
            grid[r, cols - 2] = 0
        else:
            grid[r, 0] = 0
            grid[r, 1] = 0

    grid[0, 0] = 0
    grid[rows - 1, cols - 1] = 0
    return grid


def generate_trap_preset(rows: int, cols: int) -> np.ndarray:
    """Deceptive U-shaped obstacle barrier trapping greedy heuristics."""
    grid = np.zeros((rows, cols), dtype=int)
    mid_r = rows // 2
    mid_c = cols // 2
    w_h = max(2, rows // 4)
    w_w = max(2, cols // 4)

    # U-shape facing top-left
    # Top edge of cup
    grid[max(0, mid_r - w_h) : min(rows, mid_r + w_h), max(0, mid_c + w_w)] = 1
    grid[
        min(rows - 1, mid_r + w_h), max(0, mid_c - w_w) : min(cols, mid_c + w_w + 1)
    ] = 1
    grid[max(0, mid_r - w_h), max(0, mid_c - w_w) : min(cols, mid_c + w_w + 1)] = 1

    grid[0, 0] = 0
    grid[rows - 1, cols - 1] = 0
    return grid


def generate_pillars_preset(rows: int, cols: int) -> np.ndarray:
    """Regular matrix of pillar obstacles."""
    grid = np.zeros((rows, cols), dtype=int)
    for r in range(2, rows - 1, 3):
        for c in range(2, cols - 1, 3):
            grid[r, c] = 1

    grid[0, 0] = 0
    grid[rows - 1, cols - 1] = 0
    return grid


PRESET_GENERATORS = {
    "Chokepoint / Bottleneck": generate_bottleneck_preset,
    "Spiral Labyrinth": generate_spiral_preset,
    "Zig-Zag Slits": generate_slits_preset,
    "Deceptive U-Trap": generate_trap_preset,
    "Pillar Arena": generate_pillars_preset,
}


def get_preset_maze(name: str, rows: int, cols: int) -> np.ndarray:
    """Returns grid for named benchmark preset."""
    gen = PRESET_GENERATORS.get(name)
    if gen is None:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list(PRESET_GENERATORS.keys())}"
        )
    return gen(rows, cols)


def get_available_presets() -> list[str]:
    """Returns list of preset names."""
    return list(PRESET_GENERATORS.keys())
