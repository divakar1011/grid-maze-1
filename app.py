"""
================================================================================
🧭 BREADTH-FIRST SEARCH (BFS) MAZE SOLVER & PATH VISUALIZER
================================================================================
Academic Demonstration Project for Graph Search Algorithms, FIFO Queue Dynamics,
Level-Order Wavefront Expansion, and Deterministic Seed Reproducibility.

Features:
  - 100% Deterministic Grid Generation via Student Register Number (Seed)
  - Pure Python collections.deque FIFO Queue BFS Engine
  - Level-Order Distance Invariant & Wavefront Analytics
  - Procedural Maze Generation: Deterministic Random, Recursive Backtracker,
    Prim's Algorithm, and Curated Presets
  - Dynamic Step-by-Step Frontier & Explored Node Animation
  - Exploration Depth Heatmaps & Parent Vector Flow Fields
  - Empirical Complexity & Grid Scaling Benchmark Harness
  - Downloadable Academic Run Report with SHA-256 Cryptographic Verification
================================================================================
"""

import io
import json
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.algorithms.base import AlgorithmMetrics, SearchResult
from src.algorithms.bfs import BreadthFirstSearch
from src.benchmark.evaluator import run_bfs_density_benchmark, run_bfs_scaling_benchmark
from src.benchmark.integrity import generate_submission_signature
from src.generators.presets import get_available_presets, get_preset_maze
from src.generators.prims import generate_prims_maze
from src.generators.random_grid import generate_random_grid
from src.generators.recursive_backtracker import generate_recursive_backtracker_maze
from src.generators.solvability import carve_guaranteed_solution, is_solvable
from src.visualization.canvas import render_maze_canvas
from src.visualization.comparison_plot import render_bfs_analytics_charts

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BFS Maze Solver & Path Visualizer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }
    
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border-radius: 12px;
        padding: 22px 28px;
        color: #f8fafc;
        margin-bottom: 22px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        font-size: 0.98rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
    }
    
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }
    
    .badge-success {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-failure {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .theory-box {
        background: #111827;
        border-left: 4px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 16px 0;
        color: #cbd5e1;
    }
    
    .signature-badge {
        font-family: monospace;
        background: #0f172a;
        border: 1px solid #334155;
        padding: 6px 12px;
        border-radius: 6px;
        color: #38bdf8;
        font-size: 0.85rem;
        word-break: break-all;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    # Header Banner
    st.markdown(
        """
    <div class="header-banner">
        <h1 class="header-title">🧭 Breadth-First Search (BFS) Maze Visualizer</h1>
        <p class="header-subtitle">
            Academic Demonstration of BFS Graph Traversal, FIFO Queue Dynamics, Level-Order Wavefront Expansion & Uniform-Cost Shortest Path Optimality
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # SIDEBAR CONFIGURATION
    # -------------------------------------------------------------------------
    st.sidebar.header("⚙️ Configuration & Environment")

    # Anti-Copy Seed
    student_seed = st.sidebar.number_input(
        "Student Register Number / Seed",
        min_value=1,
        max_value=999999999999,
        value=113025148022,
        step=1,
        help="Deterministic RNG seed ensuring 100% reproducible obstacle grids.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🗺️ Maze Generation Mode")

    maze_mode = st.sidebar.selectbox(
        "Generation Algorithm",
        [
            "Deterministic Seeded Random",
            "Recursive Backtracker (DFS Perfect Maze)",
            "Randomized Prim's Algorithm",
            "Academic Benchmark Presets",
        ],
        index=0,
    )

    preset_name = None
    if maze_mode == "Academic Benchmark Presets":
        preset_name = st.sidebar.selectbox(
            "Select Benchmark Preset", get_available_presets()
        )

    col_r, col_c = st.sidebar.columns(2)
    with col_r:
        grid_rows = st.slider("Rows", min_value=10, max_value=35, value=20, step=1)
    with col_c:
        grid_cols = st.slider("Columns", min_value=10, max_value=35, value=20, step=1)

    obstacle_density = 0.25
    if maze_mode == "Deterministic Seeded Random":
        obstacle_density = st.sidebar.slider(
            "Obstacle Density",
            min_value=0.10,
            max_value=0.45,
            value=0.25,
            step=0.01,
            format="%.2f",
            help="Proportion of blocked wall cells randomly placed across the grid.",
        )

    guarantee_path = st.sidebar.checkbox(
        "Guarantee Solvable Path",
        value=True,
        help="If random obstacles block start and goal, automatically carve a natural open corridor.",
    )

    allow_diagonal = st.sidebar.checkbox(
        "8-Way Movement (Diagonal)",
        value=False,
        help="Standard BFS on a grid uses 4-directional orthogonal movement (c=1 per step). Toggle for 8-way traversal.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Visual Overlays & Display")

    show_heatmap = st.sidebar.checkbox(
        "BFS Exploration Depth Heatmap",
        value=False,
        help="Overlay distance gradient map dist(s, v) showing the concentric shells of BFS level expansion.",
    )

    show_flow_field = st.sidebar.checkbox(
        "Parent Vector Flow Field",
        value=False,
        help="Render directional arrows showing parent pointers leading back to the source Start node.",
    )

    anim_speed = st.sidebar.slider(
        "Animation Delay (ms)",
        min_value=10,
        max_value=250,
        value=30,
        step=5,
    )

    # -------------------------------------------------------------------------
    # GRID GENERATION & BFS CACHING
    # -------------------------------------------------------------------------
    cache_key = (
        student_seed,
        maze_mode,
        preset_name,
        grid_rows,
        grid_cols,
        obstacle_density,
        guarantee_path,
        allow_diagonal,
    )

    if (
        "current_cache_key" not in st.session_state
        or st.session_state["current_cache_key"] != cache_key
    ):
        if maze_mode == "Academic Benchmark Presets" and preset_name:
            grid = get_preset_maze(preset_name, grid_rows, grid_cols)
        elif maze_mode == "Recursive Backtracker (DFS Perfect Maze)":
            grid = generate_recursive_backtracker_maze(
                grid_rows, grid_cols, student_seed
            )
        elif maze_mode == "Randomized Prim's Algorithm":
            grid = generate_prims_maze(grid_rows, grid_cols, student_seed)
        else:
            grid = generate_random_grid(
                grid_rows, grid_cols, obstacle_density, student_seed
            )

        start = (0, 0)
        goal = (grid_rows - 1, grid_cols - 1)

        if guarantee_path and not is_solvable(grid, start, goal, allow_diagonal):
            grid = carve_guaranteed_solution(grid, start, goal, seed=student_seed)

        bfs = BreadthFirstSearch()
        search_res = bfs.search(
            grid=grid,
            start=start,
            goal=goal,
            allow_diagonal=allow_diagonal,
            record_history=True,
        )

        st.session_state["grid"] = grid
        st.session_state["search_res"] = search_res
        st.session_state["current_cache_key"] = cache_key
        st.session_state["current_step_idx"] = (
            len(search_res.steps_history) - 1 if search_res.steps_history else 0
        )

    grid = st.session_state["grid"]
    search_res: SearchResult = st.session_state["search_res"]
    metrics: AlgorithmMetrics = search_res.metrics
    steps_history = search_res.steps_history
    total_steps = len(steps_history) - 1 if steps_history else 0

    # -------------------------------------------------------------------------
    # ACTION CONTROL BAR
    # -------------------------------------------------------------------------
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns([1, 1, 1, 1])
    with c_btn1:
        run_animation = st.button(
            "▶ Run Step Animation", use_container_width=True, type="primary"
        )
    with c_btn2:
        instant_solve = st.button("⚡ Show Full Solution", use_container_width=True)
    with c_btn3:
        step_back = st.button("◀ Step Back", use_container_width=True)
    with c_btn4:
        reset_maze = st.button("🔄 Reset / Step 0", use_container_width=True)

    if reset_maze:
        st.session_state["current_step_idx"] = 0
        st.rerun()

    if instant_solve:
        st.session_state["current_step_idx"] = total_steps
        st.rerun()

    if step_back:
        cur_idx = st.session_state.get("current_step_idx", total_steps)
        st.session_state["current_step_idx"] = max(0, cur_idx - 1)
        st.rerun()

    # -------------------------------------------------------------------------
    # HUD METRIC CARDS
    # -------------------------------------------------------------------------
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)

    with col_m1:
        status_html = (
            '<span class="badge-success">🎯 GOAL REACHED</span>'
            if metrics.found
            else '<span class="badge-failure">🚫 NO PATH</span>'
        )
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">BFS Status</div>
            <div style="margin-top: 6px;">{status_html}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Shortest Path Length</div>
            <div class="metric-val">{metrics.path_length} <span style="font-size: 0.9rem; color: #94a3b8;">steps</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_m3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Total Explored Nodes</div>
            <div class="metric-val">{metrics.total_explored} <span style="font-size: 0.9rem; color: #94a3b8;">cells</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_m4:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Search Efficiency</div>
            <div class="metric-val">{metrics.search_efficiency:.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_m5:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Peak Queue Size</div>
            <div class="metric-val">{metrics.max_frontier_size} <span style="font-size: 0.9rem; color: #94a3b8;">frontier</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_m6:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Max Depth Level</div>
            <div class="metric-val">{metrics.max_depth_reached} <span style="font-size: 0.9rem; color: #94a3b8;">hops</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # CANVAS & INTERACTIVE SCRUBBER
    # -------------------------------------------------------------------------
    plot_placeholder = st.empty()

    if not metrics.found:
        st.warning(
            f"⚠️ **Maze is Disconnected!** Obstacles completely isolate Start `(0, 0)` from Goal `({grid_rows - 1}, {grid_cols - 1})`. "
            f"Enable **Guarantee Solvable Path** in the sidebar or lower the Obstacle Density.",
            icon="⚠️",
        )

    # Animation Playback Mode
    if run_animation and steps_history:
        pbar = st.progress(0)
        ptext = st.empty()
        stride = 1 if total_steps <= 120 else max(1, total_steps // 100)

        for idx in range(0, total_steps + 1, stride):
            s_data = steps_history[idx]
            fig = render_maze_canvas(
                grid=grid,
                start=(0, 0),
                goal=(grid_rows - 1, grid_cols - 1),
                visited_cells=s_data["visited"],
                frontier_cells=set(s_data["queue"]),
                current_cell=s_data["current"],
                path_cells=s_data["path"] if idx == total_steps else [],
                distance_matrix=search_res.distance_matrix,
                parent_map=search_res.parent_map,
                show_heatmap=show_heatmap,
                show_flow_field=show_flow_field,
                algorithm_title=f"Breadth-First Search (BFS) — Step {idx}/{total_steps}",
            )
            plot_placeholder.pyplot(fig)
            plt.close(fig)

            pbar.progress(min(1.0, idx / max(1, total_steps)))
            ptext.caption(
                f"Expanding BFS Frontier... Step {idx}/{total_steps} | Queue Size: {len(s_data['queue'])}"
            )
            time.sleep(anim_speed / 1000.0)

        # Final render
        final_step = steps_history[-1]
        fig = render_maze_canvas(
            grid=grid,
            start=(0, 0),
            goal=(grid_rows - 1, grid_cols - 1),
            visited_cells=final_step["visited"],
            frontier_cells=set(final_step["queue"]),
            current_cell=final_step["current"],
            path_cells=final_step["path"],
            distance_matrix=search_res.distance_matrix,
            parent_map=search_res.parent_map,
            show_heatmap=show_heatmap,
            show_flow_field=show_flow_field,
            algorithm_title=f"BFS Optimal Solution Complete: {metrics.path_length} steps",
        )
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        pbar.empty()
        ptext.empty()
        st.session_state["current_step_idx"] = total_steps

    else:
        # Step Scrubber / Manual Slider Mode
        if total_steps > 0:
            sc_col1, sc_col2 = st.columns([5, 1])
            with sc_col1:
                selected_step = st.slider(
                    "🔍 Interactive BFS Step Scrubber (Inspect Queue & Visited States)",
                    min_value=0,
                    max_value=total_steps,
                    value=min(
                        total_steps,
                        st.session_state.get("current_step_idx", total_steps),
                    ),
                    help="Drag slider to inspect the exact FIFO queue frontier and visited nodes at any step.",
                )
                st.session_state["current_step_idx"] = selected_step
            with sc_col2:
                st.markdown(
                    "<div style='margin-top: 28px;'></div>", unsafe_allow_html=True
                )
                st.caption(f"Showing Step **{selected_step}** of **{total_steps}**")

            s_data = steps_history[selected_step]
            fig = render_maze_canvas(
                grid=grid,
                start=(0, 0),
                goal=(grid_rows - 1, grid_cols - 1),
                visited_cells=s_data["visited"],
                frontier_cells=set(s_data["queue"]),
                current_cell=s_data["current"],
                path_cells=s_data["path"] if selected_step == total_steps else [],
                distance_matrix=search_res.distance_matrix,
                parent_map=search_res.parent_map,
                show_heatmap=show_heatmap,
                show_flow_field=show_flow_field,
                algorithm_title=f"BFS Traversal — Step {selected_step}/{total_steps}",
            )
            plot_placeholder.pyplot(fig)
            plt.close(fig)
        else:
            fig = render_maze_canvas(
                grid=grid,
                start=(0, 0),
                goal=(grid_rows - 1, grid_cols - 1),
                visited_cells=set(),
                frontier_cells=set(),
                current_cell=None,
                path_cells=[],
                algorithm_title="Breadth-First Search (BFS)",
            )
            plot_placeholder.pyplot(fig)
            plt.close(fig)

    # -------------------------------------------------------------------------
    # ACADEMIC MODULES & TAB SECTION
    # -------------------------------------------------------------------------
    st.markdown("---")

    tab_analytics, tab_scale, tab_theory, tab_path, tab_export = st.tabs(
        [
            "📊 BFS Queue & Wavefront Dynamics",
            "📈 Empirical Complexity & Grid Scaling",
            "📚 Mathematical Theory & Optimality Proof",
            "📍 Path Coordinates & Hop Breakdown",
            "📥 Submission Export & SHA-256 Verification",
        ]
    )

    # TAB 1: BFS QUEUE DYNAMICS & WAVEFRONT
    with tab_analytics:
        st.subheader("📊 BFS Queue Dynamics & Level-Order Wavefront Analytics")
        st.markdown(
            "In Breadth-First Search, nodes expand in concentric wavefront shells determined by their hop distance $k$ from the Start node. "
            "These analytics capture the live growth of the FIFO queue and the layer-by-layer expansion of the search tree."
        )

        fig_analytics = render_bfs_analytics_charts(search_res)
        st.pyplot(fig_analytics)
        plt.close(fig_analytics)

        # Level breakdown table
        if search_res.level_distribution:
            st.markdown("#### Wavefront Level Breakdown Table")
            level_rows = [
                {
                    "Distance Level (k)": lvl,
                    "Discovered Nodes N(k)": count,
                    "% of Visited Nodes": f"{(count / max(1, metrics.total_explored)) * 100:.1f}%",
                }
                for lvl, count in sorted(search_res.level_distribution.items())
            ]
            st.dataframe(
                pd.DataFrame(level_rows), use_container_width=True, hide_index=True
            )

    # TAB 2: GRID SCALING & EMPIRICAL COMPLEXITY
    with tab_scale:
        st.subheader("📈 Empirical Complexity & Scalability Analysis")
        st.markdown(
            "Theoretical computer science establishes that BFS runs in $\\mathcal{O}(V + E) = \\mathcal{O}(R \\times C)$ time. "
            "Execute the benchmark below to empirically measure runtime, node expansions, and queue memory across growing grid dimensions."
        )

        col_run1, col_run2 = st.columns(2)
        with col_run1:
            if st.button(
                "🚀 Run Dimension Scaling Benchmark (10x10 to 30x30)", type="primary"
            ):
                with st.spinner(
                    "Benchmarking BFS across increasing grid dimensions..."
                ):
                    df_scaling = run_bfs_scaling_benchmark(
                        seed=student_seed,
                        dimensions=[10, 15, 20, 25, 30],
                        obstacle_density=obstacle_density,
                        allow_diagonal=allow_diagonal,
                    )
                    st.session_state["df_scaling"] = df_scaling

        with col_run2:
            if st.button("🔬 Run Obstacle Density Sweep (10% to 40%)"):
                with st.spinner(
                    "Benchmarking BFS across varying obstacle densities..."
                ):
                    df_density = run_bfs_density_benchmark(
                        seed=student_seed,
                        dim=grid_rows,
                        densities=[0.10, 0.20, 0.30, 0.40],
                        allow_diagonal=allow_diagonal,
                    )
                    st.session_state["df_density"] = df_density

        if "df_scaling" in st.session_state:
            st.markdown("#### Grid Dimension Scaling Results")
            st.dataframe(
                st.session_state["df_scaling"],
                use_container_width=True,
                hide_index=True,
            )

        if "df_density" in st.session_state:
            st.markdown("#### Obstacle Density Sweep Results")
            st.dataframe(
                st.session_state["df_density"],
                use_container_width=True,
                hide_index=True,
            )

    # TAB 3: THEORETICAL ANALYSIS & PROOFS
    with tab_theory:
        st.markdown(
            r"""
        ### Theoretical Foundations & Mathematical Proof of BFS Optimality

        <div class="theory-box">
            <h4>1. Mathematical Proof of BFS Optimality on Uniform-Cost Graphs</h4>
            <p>
            Let $G = (V, E)$ be an unweighted graph where every transition between adjacent unblocked cells has 
            an identical step cost $c(u, v) = 1$. Let $\delta(s, v)$ denote the true shortest-path distance from source $s$ to vertex $v$.
            </p>
            <ul>
                <li><strong>Lemma 1 (FIFO Queue Invariant):</strong> Let the vertices in queue $Q$ at any time be 
                $\langle v_1, v_2, \dots, v_r \rangle$. Then:
                $$\text{dist}(s, v_r) \le \text{dist}(s, v_1) + 1 \quad \text{and} \quad \text{dist}(s, v_i) \le \text{dist}(s, v_{i+1}) \quad \forall \; i$$
                Nodes are expanded in strictly non-decreasing order of their distance from the source $s$.</li>
                <li><strong>Theorem 1 (Shortest Path Correctness — CLRS Theorem 20.5):</strong> When BFS dequeues node $v$, 
                its discovered distance $\text{dist}(s, v)$ is identical to the true shortest-path distance $\delta(s, v)$. 
                Upon the first discovery of the Goal node $G = (R-1, C-1)$, the reconstructed path via back-pointers 
                $\text{parent}[u]$ is mathematically guaranteed to possess minimal edge count.</li>
            </ul>
        </div>

        <div class="theory-box">
            <h4>2. Asymptotic Complexity Breakdown</h4>
            <p>
            On a 2D grid with dimensions $R \times C$:
            </p>
            <ul>
                <li><strong>Total Vertices ($V$):</strong> $V = R \times C$</li>
                <li><strong>Total Edges ($E$):</strong> In 4-way orthogonal movement, each cell has at most 4 edges, so $E \le 4V$.</li>
                <li><strong>Time Complexity:</strong> $\mathcal{O}(V + E) = \mathcal{O}(R \times C + 4 R \times C) = \mathcal{O}(R \times C)$. Every cell is enqueued and dequeued at most once.</li>
                <li><strong>Space Complexity:</strong> $\mathcal{O}(V) = \mathcal{O}(R \times C)$ required by the FIFO queue, visited set, and parent mapping dictionary.</li>
            </ul>
        </div>

        #### Formal BFS Algorithmic Invariants
        | Property | Specification | Verification in Visualizer |
        | :--- | :--- | :--- |
        | **Queue Discipline** | FIFO (`collections.deque`) | $\mathcal{O}(1)$ amortized `popleft()` and `append()` |
        | **Level Traversal** | Monotonic level expansion | Verified runtime check: $\text{dist}(s, u) \le \text{dist}(s, v)$ |
        | **Optimality** | Guaranteed shortest path | Backtracking backwards along parent pointer mapping |
        | **Completeness** | Guaranteed termination | Explores all connected components in finite time |
        """,
            unsafe_allow_html=True,
        )

    # TAB 4: PATH COORDINATES
    with tab_path:
        if search_res.path and metrics.found:
            st.markdown(
                f"#### Shortest Path Trajectory ({len(search_res.path)} Waypoints)"
            )
            coords_data = []
            for idx, (r, c) in enumerate(search_res.path):
                node_type = (
                    "Start (S)"
                    if idx == 0
                    else (
                        "Goal (G)" if idx == len(search_res.path) - 1 else "Path Step"
                    )
                )
                coords_data.append(
                    {
                        "Step Index": idx,
                        "Node Type": node_type,
                        "Coordinate (r, c)": f"({r}, {c})",
                        "Row": r,
                        "Column": c,
                        "BFS Distance Level": idx,
                    }
                )
            df_path = pd.DataFrame(coords_data)
            st.dataframe(df_path, use_container_width=True, hide_index=True)

            csv_data = df_path.to_csv(index=False)
            st.download_button(
                label="📥 Download Path Coordinates (CSV)",
                data=csv_data,
                file_name=f"bfs_path_seed_{student_seed}.csv",
                mime="text/csv",
            )
        else:
            st.info("No connected path found for the current configuration.")

    # TAB 5: SUBMISSION EXPORT & VERIFICATION
    with tab_export:
        st.subheader("📥 Academic Submission & Cryptographic Verification")
        st.markdown(
            "Generate an officially verifiable JSON submission report. "
            "The report includes a **SHA-256 Cryptographic Signature** binding your student seed, "
            "grid topology, BFS metrics, and reconstructed path sequence. "
            "Instructors can verify submission authenticity using the CLI command below."
        )

        sig = generate_submission_signature(
            student_seed=student_seed,
            grid=grid,
            algorithm_name=metrics.algorithm_name,
            path_length=metrics.path_length,
            total_explored=metrics.total_explored,
            path_coordinates=[list(p) for p in search_res.path],
        )

        report_data = {
            "assignment_title": "BFS Maze Solver & Path Visualizer",
            "version": "2.0.0",
            "student_register_number_seed": student_seed,
            "grid_dimensions": {
                "rows": grid_rows,
                "columns": grid_cols,
                "total_cells": grid_rows * grid_cols,
            },
            "generation_mode": maze_mode,
            "obstacle_density_setting": obstacle_density,
            "movement_model": "8-Way (Diagonal)"
            if allow_diagonal
            else "4-Way (Orthogonal)",
            "algorithm_name": metrics.algorithm_name,
            "execution_metrics": {
                "goal_reached": metrics.found,
                "shortest_path_steps": metrics.path_length,
                "path_cost": metrics.path_cost,
                "total_nodes_explored": metrics.total_explored,
                "max_frontier_queue_size": metrics.max_frontier_size,
                "max_depth_level_reached": metrics.max_depth_reached,
                "fifo_monotonicity_satisfied": metrics.fifo_invariant_satisfied,
                "search_efficiency_percent": metrics.search_efficiency,
                "algorithm_runtime_ms": metrics.execution_time_ms,
                "optimality_guaranteed": True,
            },
            "reconstructed_path_coordinates": [list(p) for p in search_res.path],
            "verification_signature": sig,
        }

        json_bytes = json.dumps(report_data, indent=2).encode("utf-8")

        st.markdown(
            f"**Verification Signature:** <span class='signature-badge'>{sig}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            st.download_button(
                label="📥 Download Verified BFS Report (JSON)",
                data=json_bytes,
                file_name=f"bfs_maze_report_seed_{student_seed}.json",
                mime="application/json",
                type="primary",
                use_container_width=True,
            )
        with col_d2:
            buf = io.BytesIO()
            final_fig = render_maze_canvas(
                grid=grid,
                start=(0, 0),
                goal=(grid_rows - 1, grid_cols - 1),
                visited_cells=set(search_res.visited_order),
                frontier_cells=set(),
                current_cell=None,
                path_cells=search_res.path,
                distance_matrix=search_res.distance_matrix,
                parent_map=search_res.parent_map,
                show_heatmap=show_heatmap,
                show_flow_field=show_flow_field,
                algorithm_title=f"BFS Solution (Seed: {student_seed})",
            )
            final_fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
            plt.close(final_fig)
            buf.seek(0)

            st.download_button(
                label="🖼️ Download High-Res Solved Maze (PNG)",
                data=buf,
                file_name=f"bfs_maze_solution_seed_{student_seed}.png",
                mime="image/png",
                use_container_width=True,
            )

        st.markdown("#### Automated Verification Command for Instructors")
        st.code(
            f"python cli.py --verify bfs_maze_report_seed_{student_seed}.json",
            language="bash",
        )

        st.json(report_data, expanded=False)


if __name__ == "__main__":
    main()
