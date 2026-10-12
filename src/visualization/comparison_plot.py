"""
BFS Queue Dynamics and Wavefront Analytics visualization charts.
"""

import matplotlib.pyplot as plt

from src.algorithms.base import SearchResult


def render_bfs_analytics_charts(result: SearchResult) -> plt.Figure:
    """
    Renders detailed 2x2 analytical plots describing BFS queue dynamics,
    wavefront level expansion, and search progression.
    """
    m = result.metrics
    queue_history = result.queue_size_history
    level_dist = result.level_distribution

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=120)
    fig.patch.set_facecolor("#0f172a")

    def style_axis(ax, title, xlabel, ylabel):
        ax.set_facecolor("#1e293b")
        ax.set_title(title, color="#f8fafc", fontsize=11, weight="bold", pad=8)
        ax.set_xlabel(xlabel, color="#94a3b8", fontsize=9)
        ax.set_ylabel(ylabel, color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.grid(color="#334155", linestyle="--", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#475569")

    # 1. FIFO Queue Size vs Expansion Step
    ax1 = axes[0, 0]
    if queue_history:
        steps = list(range(len(queue_history)))
        ax1.plot(steps, queue_history, color="#38bdf8", linewidth=2.0)
        ax1.fill_between(steps, queue_history, color="#38bdf8", alpha=0.2)
        ax1.axhline(
            m.max_frontier_size,
            color="#f43f5e",
            linestyle=":",
            alpha=0.8,
            label=f"Peak: {m.max_frontier_size}",
        )
        ax1.legend(
            facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8
        )
    style_axis(
        ax1,
        "FIFO Frontier Queue Size over Exploration Step",
        "Step Number",
        "Queue Size (Nodes)",
    )

    # 2. Nodes Explored per Distance Level (Level-Order Wavefront)
    ax2 = axes[0, 1]
    if level_dist:
        levels = sorted(level_dist.keys())
        counts = [level_dist[lvl] for lvl in levels]
        ax2.bar(levels, counts, color="#a855f7", edgecolor="#0f172a", width=0.75)
    style_axis(
        ax2,
        "BFS Wavefront: Nodes per Distance Level k",
        "Distance Level k (Hops from Start)",
        "Node Count",
    )

    # 3. Cumulative Nodes Explored vs Search Step
    ax3 = axes[1, 0]
    if result.visited_order:
        cum_explored = list(range(1, len(result.visited_order) + 1))
        ax3.plot(
            range(1, len(cum_explored) + 1),
            cum_explored,
            color="#34d399",
            linewidth=2.0,
        )
    style_axis(
        ax3,
        "Cumulative Node Exploration Linearity (O(V))",
        "Expansion Step",
        "Total Nodes Visited",
    )

    # 4. Search Efficiency & Branching Factor Progression
    ax4 = axes[1, 1]
    if level_dist and len(level_dist) > 1:
        sorted_levels = sorted(level_dist.keys())
        branching_factors = []
        bf_levels = []
        for i in range(len(sorted_levels) - 1):
            l1, l2 = sorted_levels[i], sorted_levels[i + 1]
            if level_dist[l1] > 0:
                bf = level_dist[l2] / level_dist[l1]
                branching_factors.append(bf)
                bf_levels.append(l1)

        ax4.plot(
            bf_levels,
            branching_factors,
            color="#fbbf24",
            marker="o",
            markersize=3,
            linewidth=1.5,
        )
        ax4.axhline(
            1.0, color="#94a3b8", linestyle="--", alpha=0.5, label="Equilibrium (b=1)"
        )
        ax4.legend(
            facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8
        )
    style_axis(
        ax4,
        "Effective Wavefront Branching Factor (N_{k+1} / N_k)",
        "Distance Level k",
        "Branching Ratio",
    )

    plt.tight_layout()
    return fig
