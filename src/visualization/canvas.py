"""
Matplotlib Canvas Rendering Engine with Heatmap and Vector Field Overlays.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, ListedColormap

from src.algorithms.base import Coordinate


def render_maze_canvas(
    grid: np.ndarray,
    start: Coordinate = (0, 0),
    goal: Coordinate | None = None,
    visited_cells: set[Coordinate] | None = None,
    frontier_cells: set[Coordinate] | None = None,
    current_cell: Coordinate | None = None,
    path_cells: list[Coordinate] | None = None,
    distance_matrix: np.ndarray | None = None,
    parent_map: dict[Coordinate, Coordinate | None] | None = None,
    show_heatmap: bool = False,
    show_flow_field: bool = False,
    cell_size: float = 0.42,
    algorithm_title: str = "BFS Maze Visualizer",
) -> plt.Figure:
    """
    Renders the complete maze state on a high-DPI dark-themed Matplotlib canvas.
    """
    rows, cols = grid.shape
    if goal is None:
        goal = (rows - 1, cols - 1)

    visited_cells = visited_cells or set()
    frontier_cells = frontier_cells or set()
    path_cells = path_cells or []

    fig_w = max(7.0, cols * cell_size)
    fig_h = max(7.0, rows * cell_size)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)

    # Dark background theme
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")

    if show_heatmap and distance_matrix is not None:
        # HEATMAP OVERLAY MODE:
        # Base background: walls = dark slate, unvisited = near black
        base_rgb = np.full((rows, cols, 3), 0.08)  # near black
        # Walls
        base_rgb[grid == 1] = [0.15, 0.18, 0.24]

        # Normalize visited distance values
        valid_dists = distance_matrix[distance_matrix >= 0]
        max_d = float(np.max(valid_dists)) if len(valid_dists) > 0 else 1.0
        max_d = max(1.0, max_d)

        # Colormap for distance
        plasma_cmap = cm.get_cmap("plasma")

        for r, c in visited_cells:
            if (r, c) != start and (r, c) != goal and grid[r, c] == 0:
                d_val = distance_matrix[r, c]
                if d_val >= 0:
                    norm_val = min(1.0, d_val / max_d)
                    base_rgb[r, c] = plasma_cmap(norm_val)[:3]

        # Frontier cells highlight
        for r, c in frontier_cells:
            if (r, c) != start and (r, c) != goal and grid[r, c] == 0:
                base_rgb[r, c] = [0.75, 0.52, 0.99]  # bright purple

        # Path cells highlight
        for r, c in path_cells:
            if (r, c) != start and (r, c) != goal:
                base_rgb[r, c] = [0.99, 0.85, 0.2]  # gold

        ax.imshow(
            base_rgb,
            origin="upper",
            extent=(-0.5, float(cols) - 0.5, float(rows) - 0.5, -0.5),
        )

    else:
        # STANDARD HIGH-CONTRAST PALETTE MODE:
        # 0: Open, 1: Obstacle, 2: Explored, 3: Frontier, 4: Current, 5: Path
        display_grid = np.zeros((rows, cols), dtype=int)
        display_grid[grid == 1] = 1

        for r, c in visited_cells:
            if (r, c) != start and (r, c) != goal and grid[r, c] == 0:
                display_grid[r, c] = 2

        for r, c in frontier_cells:
            if (r, c) != start and (r, c) != goal and grid[r, c] == 0:
                display_grid[r, c] = 3

        if current_cell and current_cell != start and current_cell != goal:
            cr, cc = current_cell
            if 0 <= cr < rows and 0 <= cc < cols:
                display_grid[cr, cc] = 4

        for r, c in path_cells:
            if (r, c) != start and (r, c) != goal:
                display_grid[r, c] = 5

        cmap_colors = [
            "#f1f5f9",  # 0: Open (Crisp Off-White)
            "#1e293b",  # 1: Wall (Dark Slate)
            "#7dd3fc",  # 2: Explored (Sky Blue)
            "#c084fc",  # 3: Frontier (Soft Violet)
            "#fb923c",  # 4: Current Expanding (Amber/Orange)
            "#fef08a",  # 5: Shortest Path Cell (Glowing Yellow)
        ]
        cmap = ListedColormap(cmap_colors)
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

        ax.imshow(
            display_grid,
            cmap=cmap,
            norm=norm,
            origin="upper",
            extent=(-0.5, float(cols) - 0.5, float(rows) - 0.5, -0.5),
        )

    # Gridlines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#334155", linestyle="-", linewidth=0.6, alpha=0.6)
    ax.tick_params(which="minor", size=0)

    # Major ticks
    if rows <= 25 and cols <= 25:
        ax.set_xticks(np.arange(0, cols, max(1, cols // 8)))
        ax.set_yticks(np.arange(0, rows, max(1, rows // 8)))
        ax.tick_params(colors="#94a3b8", labelsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor("#475569")
        spine.set_linewidth(1.2)

    # VECTOR FLOW FIELD OVERLAY (Parent pointers)
    if (
        show_flow_field
        and parent_map
        and len(visited_cells) > 0
        and rows <= 35
        and cols <= 35
    ):
        for node, par in parent_map.items():
            if par is not None and node in visited_cells and node != start:
                nr, nc = node
                pr, pc = par
                # Arrow from node towards parent
                dr, dc = pr - nr, pc - nc
                if dr != 0 or dc != 0:
                    ax.arrow(
                        nc,
                        nr,
                        dc * 0.35,
                        dr * 0.35,
                        head_width=0.18,
                        head_length=0.18,
                        fc="#38bdf8",
                        ec="#38bdf8",
                        alpha=0.5,
                        zorder=5,
                    )

    # Start Cell Marker (S)
    sr, sc = start
    ax.add_patch(
        mpatches.Rectangle(
            (sc - 0.5, sr - 0.5), 1, 1, fill=True, color="#10b981", zorder=7
        )
    )
    ax.text(
        sc,
        sr,
        "S",
        color="white",
        weight="bold",
        fontsize=12,
        ha="center",
        va="center",
        zorder=8,
    )

    # Goal Cell Marker (G)
    gr, gc = goal
    ax.add_patch(
        mpatches.Rectangle(
            (gc - 0.5, gr - 0.5), 1, 1, fill=True, color="#f43f5e", zorder=7
        )
    )
    ax.text(
        gc,
        gr,
        "G",
        color="white",
        weight="bold",
        fontsize=12,
        ha="center",
        va="center",
        zorder=8,
    )

    # Shortest Path Vector Line
    if path_cells and len(path_cells) > 1:
        path_r = [p[0] for p in path_cells]
        path_c = [p[1] for p in path_cells]
        # Glow halo
        ax.plot(path_c, path_r, color="#fbbf24", linewidth=4.5, alpha=0.55, zorder=9)
        # Sharp centerline
        ax.plot(
            path_c,
            path_r,
            color="#dc2626",
            linewidth=2.4,
            linestyle="-",
            marker="o",
            markersize=4.0,
            markerfacecolor="#fef08a",
            markeredgecolor="#b91c1c",
            zorder=10,
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color="#10b981", label=f"Start ({start[0]},{start[1]})"),
        mpatches.Patch(color="#f43f5e", label=f"Goal ({goal[0]},{goal[1]})"),
        mpatches.Patch(color="#1e293b", label="Wall / Obstacle"),
        mpatches.Patch(color="#7dd3fc", label="Explored Node"),
        mpatches.Patch(color="#c084fc", label="Frontier Queue"),
        mpatches.Patch(color="#dc2626", label="Shortest Path"),
    ]
    if show_heatmap:
        legend_patches.append(mpatches.Patch(color="#fb923c", label="Distance Heatmap"))

    ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=min(7, len(legend_patches)),
        frameon=True,
        facecolor="#111827",
        edgecolor="#374151",
        labelcolor="#e5e7eb",
        fontsize=8.5,
    )

    ax.set_title(
        algorithm_title, color="#f8fafc", fontsize=11, pad=10, weight="semibold"
    )
    plt.tight_layout()
    return fig
