"""
================================================================================
BREADTH-FIRST SEARCH (BFS) MAZE SOLVER & PATH VISUALIZER
================================================================================
Academic Demonstration Project for Pathfinding & Graph Search Algorithms
Coursework & Submission Visualizer with Deterministic Seed Reproducibility

Author: Senior Full-Stack AI Engineer
Features:
  - 100% Deterministic Grid Generation via Student Register Number (Seed)
  - Pure Python collections.deque FIFO Queue BFS Engine
  - Parent-Pointer Backtracking for Optimal Shortest Path Reconstruction
  - Dynamic Step-by-Step Frontier & Explored Node Animation
  - High-Resolution Matplotlib Canvas Rendering
  - Interactive Step Scrubber & Real-time Execution Statistics
  - Downloadable Academic Run Report (JSON)
================================================================================
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import random
from collections import deque
import time
import json
import pandas as pd


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BFS Maze Solver Visualizer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphic aesthetics and polished academic layout
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    
    /* Header gradient banner */
    .header-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1e1b4b 100%);
        border-radius: 12px;
        padding: 24px 30px;
        color: #f8fafc;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-top: 8px;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Status Badges */
    .badge-success {
        background-color: #065f46;
        color: #34d399;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .badge-failure {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    /* Info Box */
    .theory-box {
        background: #0f172a;
        border-left: 4px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 16px;
        margin: 15px 0;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CORE ALGORITHM & HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def generate_reproducible_maze(rows: int, cols: int, obstacle_density: float, seed: int) -> np.ndarray:
    """
    Generates a 2D binary grid representing the maze with 100% reproducibility.
    
    Parameters:
      - rows (int): Number of grid rows
      - cols (int): Number of grid columns
      - obstacle_density (float): Probability [0.0, 1.0] of a cell being an obstacle
      - seed (int): Student register number / random seed
      
    Returns:
      - grid (np.ndarray): 2D array where 0 = Open Cell, 1 = Obstacle (Wall)
    """
    # Deterministic seeding across both standard Python and NumPy RNGs
    random.seed(seed)
    # Ensure seed is within NumPy uint32 bounds [0, 2**32 - 1] for long student IDs
    np.random.seed(int(seed) % (2**32 - 1))
    
    # Generate random binary grid based on obstacle density
    grid = (np.random.rand(rows, cols) < obstacle_density).astype(int)
    
    # Enforce Ground Rule: Start (0, 0) and Goal (rows-1, cols-1) must ALWAYS be unblocked
    grid[0, 0] = 0
    grid[rows - 1, cols - 1] = 0
    
    return grid


def run_bfs_solver(grid: np.ndarray, start=(0, 0), goal=None, allow_diagonal=False):
    """
    Executes Breadth-First Search (BFS) on a uniform-cost 2D grid.
    
    ===========================================================================
    THEORETICAL OPTIMALITY OF BREADTH-FIRST SEARCH (BFS):
    ===========================================================================
    1. Uniform Cost Property:
       Every transition between adjacent unblocked cells has an identical edge
       weight c(u, v) = 1.
    2. FIFO Queue Invariant:
       BFS utilizes a First-In-First-Out (FIFO) queue (collections.deque).
       Nodes are discovered and expanded strictly in non-decreasing order of
       their distance from the start node:
         dist(start, v) <= dist(start, w) for all nodes v expanded before w.
    3. Shortest Path Guarantee:
       When the Goal node is first discovered/enqueued, the path traced back
       via parent pointers is mathematically guaranteed to be the shortest path
       (minimal number of steps/edges).
    4. Time & Space Complexity:
       - Time Complexity:  O(V + E) = O(R * C + 4 * R * C) = O(R * C)
       - Space Complexity: O(V) = O(R * C) for visited set, queue, and parents
    ===========================================================================
    
    Returns:
      - path (list of tuples): Sequence of (r, c) coordinates from start to goal, or []
      - visited_order (list of tuples): All cells visited in chronological expansion order
      - steps_history (list of dicts): Step snapshots for frame-by-frame animation & scrubber
      - stats (dict): Summary performance metrics
    """
    rows, cols = grid.shape
    if goal is None:
        goal = (rows - 1, cols - 1)
        
    start_r, start_c = start
    goal_r, goal_c = goal
    
    # Quick sanity validation
    if grid[start_r, start_c] == 1 or grid[goal_r, goal_c] == 1:
        return [], [], [], {
            "found": False,
            "total_explored": 0,
            "path_length": 0,
            "max_queue_size": 0,
            "execution_time_ms": 0.0
        }
        
    # Directions: 4-way orthogonal (Up, Right, Down, Left) or 8-way if enabled
    if allow_diagonal:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        # Standard Orthogonal grid movements (Cost = 1)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
    # Data structures
    # 1. FIFO Queue: stores frontier cells to be explored
    queue = deque([start])
    
    # 2. Visited set: prevents cycles and redundant re-expansions
    visited = {start}
    
    # 3. Parent Map: parent[child] = curr; enables exact backwards path reconstruction
    parent = {start: None}
    
    # Step-by-step history tracker for animation and scrubbing
    visited_order = []
    steps_history = []
    
    max_queue_size = 1
    found_goal = False
    
    start_time = time.perf_counter()
    
    # Record initial state (Step 0)
    steps_history.append({
        "step": 0,
        "current": None,
        "queue": list(queue),
        "visited": set(visited),
        "path": []
    })
    
    step_count = 0
    
    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        
        # Dequeue the oldest node from the front of the FIFO queue (O(1) operation)
        curr = queue.popleft()
        visited_order.append(curr)
        step_count += 1
        
        # Check if we have reached the destination
        if curr == goal:
            found_goal = True
            # Snapshot final step
            steps_history.append({
                "step": step_count,
                "current": curr,
                "queue": list(queue),
                "visited": set(visited),
                "path": []
            })
            break
            
        r, c = curr
        
        # Explore all valid neighboring cells
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            
            # Boundary check
            if 0 <= nr < rows and 0 <= nc < cols:
                # Obstacle & visited check
                if grid[nr, nc] == 0 and neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)
                    
        # Log snapshot every expansion for smooth visualization playback
        steps_history.append({
            "step": step_count,
            "current": curr,
            "queue": list(queue),
            "visited": set(visited),
            "path": []
        })
        
    end_time = time.perf_counter()
    execution_time_ms = (end_time - start_time) * 1000.0
    
    # Reconstruct shortest path by following parent pointers backwards from goal to start
    final_path = []
    if found_goal:
        curr_node = goal
        while curr_node is not None:
            final_path.append(curr_node)
            curr_node = parent.get(curr_node)
        final_path.reverse()  # Reverse to obtain Start -> Goal order
        
        # Update the final step history with the complete reconstructed path
        if steps_history:
            steps_history[-1]["path"] = final_path
            
    stats = {
        "found": found_goal,
        "total_explored": len(visited),
        "path_length": len(final_path) - 1 if found_goal else 0,
        "max_queue_size": max_queue_size,
        "execution_time_ms": round(execution_time_ms, 3)
    }
    
    return final_path, visited_order, steps_history, stats


# -----------------------------------------------------------------------------
# MATPLOTLIB RENDERING ENGINE
# -----------------------------------------------------------------------------

def render_maze_matplotlib(
    grid: np.ndarray,
    start=(0, 0),
    goal=None,
    visited_cells=None,
    frontier_cells=None,
    current_cell=None,
    path_cells=None,
    cell_size=0.45
):
    """
    Renders the maze grid state with clear color contrast and sharp gridlines.
    
    Color Scheme:
      - Open Unvisited: Pure White / Light Slate (#f8fafc)
      - Obstacles / Walls: Deep Dark Slate (#1e293b)
      - Visited Nodes: Light Cyan / Sky Blue (#7dd3fc)
      - Frontier (In Queue): Soft Purple / Violet (#c084fc)
      - Current Node: Electric Orange (#fb923c)
      - Start Cell (0,0): Emerald Green (#10b981)
      - Goal Cell (Max, Max): Crimson / Amber Star (#f43f5e)
      - Shortest Path: Bold Crimson Line (#e11d48) with Gold Markers (#fbbf24)
    """
    rows, cols = grid.shape
    if goal is None:
        goal = (rows - 1, cols - 1)
        
    visited_cells = visited_cells or set()
    frontier_cells = frontier_cells or set()
    path_cells = path_cells or []
    
    # Create display matrix:
    # 0: Open, 1: Obstacle, 2: Visited, 3: Frontier, 4: Current, 5: Path
    display_grid = np.zeros((rows, cols), dtype=int)
    
    # 1. Obstacles
    display_grid[grid == 1] = 1
    
    # 2. Visited
    for r, c in visited_cells:
        if (r, c) != start and (r, c) != goal and grid[r, c] == 0:
            display_grid[r, c] = 2
            
    # 3. Frontier Queue
    for r, c in frontier_cells:
        if (r, c) != start and (r, c) != goal and grid[r, c] == 0:
            display_grid[r, c] = 3
            
    # 4. Current cell being expanded
    if current_cell and current_cell != start and current_cell != goal:
        cr, cc = current_cell
        if 0 <= cr < rows and 0 <= cc < cols:
            display_grid[cr, cc] = 4
            
    # 5. Final Path
    for r, c in path_cells:
        if (r, c) != start and (r, c) != goal:
            display_grid[r, c] = 5
            
    # Define Colormap
    # Indices: 0: Open, 1: Obstacle, 2: Visited, 3: Frontier, 4: Current, 5: Path
    cmap_colors = [
        '#f8fafc',  # 0: Open (Crisp Off-White)
        '#1e293b',  # 1: Wall (Dark Slate)
        '#bae6fd',  # 2: Explored / Visited (Soft Cyan)
        '#e9d5ff',  # 3: Frontier Queue (Soft Purple)
        '#fdba74',  # 4: Current Expanding Node (Peach/Orange)
        '#fef08a'   # 5: Shortest Path Cell (Glowing Yellow)
    ]
    cmap = ListedColormap(cmap_colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    
    fig, ax = plt.subplots(figsize=(max(7, cols * cell_size), max(7, rows * cell_size)), dpi=120)
    fig.patch.set_facecolor('#0f172a')  # Dark backdrop
    ax.set_facecolor('#0f172a')
    
    # Display the grid
    ax.imshow(display_grid, cmap=cmap, norm=norm, origin='upper', extent=[-0.5, cols - 0.5, rows - 0.5, -0.5])
    
    # Draw gridlines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='#334155', linestyle='-', linewidth=0.75, alpha=0.7)
    ax.tick_params(which='minor', size=0)
    
    # Major ticks with coordinates
    if rows <= 25 and cols <= 25:
        ax.set_xticks(np.arange(0, cols, max(1, cols // 10)))
        ax.set_yticks(np.arange(0, rows, max(1, rows // 10)))
        ax.tick_params(colors='#94a3b8', labelsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        
    for spine in ax.spines.values():
        spine.set_edgecolor('#475569')
        spine.set_linewidth(1.5)
        
    # Mark Start (0, 0)
    start_r, start_c = start
    ax.add_patch(plt.Rectangle((start_c - 0.5, start_r - 0.5), 1, 1, fill=True, color='#10b981', zorder=4))
    ax.text(start_c, start_r, "S", color='white', weight='bold', fontsize=12, ha='center', va='center', zorder=5)
    
    # Mark Goal (R-1, C-1)
    goal_r, goal_c = goal
    ax.add_patch(plt.Rectangle((goal_c - 0.5, goal_r - 0.5), 1, 1, fill=True, color='#f43f5e', zorder=4))
    ax.text(goal_c, goal_r, "G", color='white', weight='bold', fontsize=12, ha='center', va='center', zorder=5)
    
    # Trace Shortest Path Vector Line
    if path_cells and len(path_cells) > 1:
        path_r = [p[0] for p in path_cells]
        path_c = [p[1] for p in path_cells]
        # Glowing shadow line
        ax.plot(path_c, path_r, color='#fbbf24', linewidth=4.5, alpha=0.5, zorder=6)
        # Core sharp line
        ax.plot(path_c, path_r, color='#dc2626', linewidth=2.5, linestyle='-', marker='o', markersize=4.5,
                markerfacecolor='#fef08a', markeredgecolor='#b91c1c', zorder=7)
        
    # Legend
    legend_patches = [
        mpatches.Patch(color='#10b981', label='Start (0,0)'),
        mpatches.Patch(color='#f43f5e', label=f'Goal ({goal[0]},{goal[1]})'),
        mpatches.Patch(color='#1e293b', label='Obstacle / Wall'),
        mpatches.Patch(color='#bae6fd', label='Explored / Visited'),
        mpatches.Patch(color='#e9d5ff', label='Frontier Queue'),
        mpatches.Patch(color='#dc2626', label='Shortest Path')
    ]
    ax.legend(
        handles=legend_patches,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(6, len(legend_patches)),
        frameon=True,
        facecolor='#1e293b',
        edgecolor='#334155',
        labelcolor='#e2e8f0',
        fontsize=8.5
    )
    
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# STREAMLIT UI & INTERACTION LOGIC
# -----------------------------------------------------------------------------

def main():
    # Header Banner
    st.markdown("""
    <div class="header-banner">
        <h1 class="header-title">🧭 Breadth-First Search (BFS) Maze Visualizer</h1>
        <p class="header-subtitle">
            Academic Demonstration of BFS Graph Traversal, FIFO Queue Dynamics & Uniform-Cost Shortest Path Optimality
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # SIDEBAR: PARAMETERS & SEED CONFIGURATION
    # -------------------------------------------------------------------------
    st.sidebar.header("⚙️ Configuration & Anti-Copy Seed")
    
    # Ground Rule 1: Student Register Number / Seed
    student_seed = st.sidebar.number_input(
        "Student Register Number / Seed",
        min_value=1,
        max_value=999999999999,
        value=113025148022,
        step=1,
        help="Use your official student registration ID to ensure 100% reproducible obstacle generation."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Grid Dimensions & Density")
    
    col_r, col_c = st.sidebar.columns(2)
    with col_r:
        grid_rows = st.slider("Rows", min_value=10, max_value=35, value=20, step=1)
    with col_c:
        grid_cols = st.slider("Columns", min_value=10, max_value=35, value=20, step=1)
        
    obstacle_density = st.sidebar.slider(
        "Obstacle Density",
        min_value=0.10,
        max_value=0.45,
        value=0.25,
        step=0.01,
        format="%.2f",
        help="Proportion of randomly placed wall obstacles in the grid (e.g. 0.25 = 25%)."
    )
    
    allow_diagonal = st.sidebar.checkbox(
        "Allow 8-Way Movement (Diagonal)",
        value=False,
        help="Standard BFS on a grid uses 4-directional orthogonal movement (Cost = 1 per step)."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎬 Playback Controls")
    
    anim_speed = st.sidebar.slider(
        "Animation Speed (Delay in ms)",
        min_value=10,
        max_value=300,
        value=35,
        step=5,
        help="Delay between step rendering during automatic playback."
    )
    
    # Initialize Session State
    if "grid" not in st.session_state or st.session_state.get("current_seed") != student_seed or \
       st.session_state.get("current_dims") != (grid_rows, grid_cols) or \
       st.session_state.get("current_density") != obstacle_density or \
       st.session_state.get("current_diag") != allow_diagonal:
        
        # Generate new maze
        st.session_state["grid"] = generate_reproducible_maze(
            grid_rows, grid_cols, obstacle_density, student_seed
        )
        st.session_state["current_seed"] = student_seed
        st.session_state["current_dims"] = (grid_rows, grid_cols)
        st.session_state["current_density"] = obstacle_density
        st.session_state["current_diag"] = allow_diagonal
        
        # Run BFS solver once to cache steps and solution
        path, visited_order, steps_history, stats = run_bfs_solver(
            st.session_state["grid"],
            start=(0, 0),
            goal=(grid_rows - 1, grid_cols - 1),
            allow_diagonal=allow_diagonal
        )
        st.session_state["path"] = path
        st.session_state["visited_order"] = visited_order
        st.session_state["steps_history"] = steps_history
        st.session_state["stats"] = stats
        st.session_state["current_step_idx"] = len(steps_history) - 1 if steps_history else 0

    grid = st.session_state["grid"]
    path = st.session_state["path"]
    steps_history = st.session_state["steps_history"]
    stats = st.session_state["stats"]
    total_steps = len(steps_history) - 1 if steps_history else 0

    # -------------------------------------------------------------------------
    # ACTION BUTTONS
    # -------------------------------------------------------------------------
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    
    with btn_col1:
        run_animation = st.button("▶ Run Step Animation", use_container_width=True, type="primary")
    with btn_col2:
        instant_solve = st.button("⚡ Show Full Solution", use_container_width=True)
    with btn_col3:
        reset_maze = st.button("🔄 Reset / Step 0", use_container_width=True)
        
    if reset_maze:
        st.session_state["current_step_idx"] = 0
        st.rerun()
        
    if instant_solve:
        st.session_state["current_step_idx"] = total_steps
        st.rerun()

    # -------------------------------------------------------------------------
    # STATS & METRICS BAR
    # -------------------------------------------------------------------------
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        status_html = '<span class="badge-success">🎯 GOAL REACHED</span>' if stats["found"] else '<span class="badge-failure">🚫 NO PATH (BLOCKED)</span>'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Status</div>
            <div style="margin-top: 6px;">{status_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Shortest Path Length</div>
            <div class="metric-val">{stats['path_length']} <span style="font-size: 1rem; color: #94a3b8;">steps</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Nodes Explored</div>
            <div class="metric-val">{stats['total_explored']} <span style="font-size: 1rem; color: #94a3b8;">cells</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with m4:
        efficiency = (stats['path_length'] / max(1, stats['total_explored'])) * 100 if stats['found'] else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Search Efficiency</div>
            <div class="metric-val">{efficiency:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max Queue Size</div>
            <div class="metric-val">{stats['max_queue_size']} <span style="font-size: 1rem; color: #94a3b8;">frontier</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # INTERACTIVE VISUALIZER CANVAS
    # -------------------------------------------------------------------------
    plot_placeholder = st.empty()
    
    # Safety Check: Display Warning if maze is blocked
    if not stats["found"]:
        st.warning(
            f"⚠️ **No Valid Path Exists!** For Register Number / Seed `{student_seed}`, grid obstacles completely disconnected "
            f"Start `(0, 0)` from Goal `({grid_rows - 1}, {grid_cols - 1})`. "
            f"Try lowering the Obstacle Density or changing the Register Number.",
            icon="⚠️"
        )
        
    # Animation Mode
    if run_animation and steps_history:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step increment for large grids to keep animation swift
        step_stride = 1 if total_steps <= 150 else max(1, total_steps // 120)
        
        for idx in range(0, total_steps + 1, step_stride):
            step_data = steps_history[idx]
            fig = render_maze_matplotlib(
                grid=grid,
                start=(0, 0),
                goal=(grid_rows - 1, grid_cols - 1),
                visited_cells=step_data["visited"],
                frontier_cells=set(step_data["queue"]),
                current_cell=step_data["current"],
                path_cells=step_data["path"] if idx == total_steps else []
            )
            plot_placeholder.pyplot(fig)
            plt.close(fig)
            
            progress = min(1.0, idx / max(1, total_steps))
            progress_bar.progress(progress)
            status_text.caption(f"Expanding BFS Frontier... Step {idx}/{total_steps} | Queue Size: {len(step_data['queue'])}")
            time.sleep(anim_speed / 1000.0)
            
        # Final render with full path
        final_step = steps_history[-1]
        fig = render_maze_matplotlib(
            grid=grid,
            start=(0, 0),
            goal=(grid_rows - 1, grid_cols - 1),
            visited_cells=final_step["visited"],
            frontier_cells=set(final_step["queue"]),
            current_cell=final_step["current"],
            path_cells=final_step["path"]
        )
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        progress_bar.empty()
        status_text.empty()
        st.session_state["current_step_idx"] = total_steps
        
    else:
        # Step Scrubber / Manual Slider Mode
        if total_steps > 0:
            step_col1, step_col2 = st.columns([4, 1])
            with step_col1:
                selected_step = st.slider(
                    "🔍 Interactive Step-by-Step Scrubber (Explore Search Frontier)",
                    min_value=0,
                    max_value=total_steps,
                    value=st.session_state.get("current_step_idx", total_steps),
                    help="Drag to inspect the exact state of the BFS queue and visited set at any step."
                )
                st.session_state["current_step_idx"] = selected_step
            with step_col2:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                st.caption(f"Showing Step **{selected_step}** of **{total_steps}**")
                
            step_data = steps_history[selected_step]
            fig = render_maze_matplotlib(
                grid=grid,
                start=(0, 0),
                goal=(grid_rows - 1, grid_cols - 1),
                visited_cells=step_data["visited"],
                frontier_cells=set(step_data["queue"]),
                current_cell=step_data["current"],
                path_cells=step_data["path"] if selected_step == total_steps else []
            )
            plot_placeholder.pyplot(fig)
            plt.close(fig)
        else:
            # Step 0 or empty grid
            fig = render_maze_matplotlib(
                grid=grid,
                start=(0, 0),
                goal=(grid_rows - 1, grid_cols - 1),
                visited_cells=set(),
                frontier_cells=set(),
                current_cell=None,
                path_cells=[]
            )
            plot_placeholder.pyplot(fig)
            plt.close(fig)

    # -------------------------------------------------------------------------
    # ACADEMIC INSIGHTS & TABS SECTION
    # -------------------------------------------------------------------------
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs([
        "📚 Theoretical Analysis & Optimality Proof",
        "📍 Path Coordinates & Step Table",
        "📥 Submission Export (JSON Report)"
    ])
    
    with tab1:
        st.markdown(r"""
        ### Breadth-First Search (BFS) Algorithmic Proof & Complexity
        
        <div class="theory-box">
            <h4>1. Mathematical Proof of Optimality on Uniform-Cost Grids</h4>
            <p>
            In a uniform grid where every step cost $c(u, v) = 1$, Breadth-First Search guarantees the 
            <strong>shortest path</strong>.
            </p>
            <ul>
                <li><strong>Monotonic Level Expansion:</strong> BFS expands all vertices at distance $k$ from the Start node before expanding any vertex at distance $k + 1$.</li>
                <li><strong>First-Discovery Property:</strong> If node $v$ is discovered by neighbor $u$, the length of the discovered path is $\text{dist}(s, u) + 1$. Because $u$ is processed in non-decreasing order of distance, this value equals the true shortest distance $\delta(s, v)$.</li>
                <li><strong>Optimality:</strong> Upon first reaching Goal $G = (R-1, C-1)$, the reconstructed path via back-pointers $\text{parent}[child]$ has minimal edge count.</li>
            </ul>
        </div>
        
        #### Complexity Analysis
        | Property | Value / Formula | Note |
        | :--- | :--- | :--- |
        | **Time Complexity** | $\mathcal{O}(V + E) = \mathcal{O}(R \times C)$ | Every cell and valid edge is enqueued and dequeued at most once. |
        | **Space Complexity** | $\mathcal{O}(V) = \mathcal{O}(R \times C)$ | Queue storage, visited hash set, and parent mapping table. |
        | **Queue Structure** | `collections.deque` (FIFO) | $\mathcal{O}(1)$ amortized popleft and append operations. |
        | **Completeness** | **Guaranteed** | Always finds a path if one exists on finite graphs. |
        """, unsafe_allow_html=True)
        
    with tab2:
        if path and stats["found"]:
            st.markdown(f"#### Reconstructed Shortest Path ({len(path)} Waypoints)")
            path_df = pd.DataFrame([
                {
                    "Step Index": idx,
                    "Node Type": "Start (S)" if idx == 0 else ("Goal (G)" if idx == len(path) - 1 else "Path Step"),
                    "Grid Coordinate (Row, Col)": f"({r}, {c})",
                    "Row (r)": r,
                    "Col (c)": c,
                    "Cost from Start": idx
                }
                for idx, (r, c) in enumerate(path)
            ])
            st.dataframe(path_df, use_container_width=True, hide_index=True)
        else:
            st.info("No reconstructed path available for the current configuration.")
            
    with tab3:
        # Create academic report dictionary for assignment submission
        report_data = {
            "assignment_title": "BFS Maze Solver & Path Visualizer",
            "student_register_number_seed": student_seed,
            "grid_dimensions": {
                "rows": grid_rows,
                "columns": grid_cols,
                "total_cells": grid_rows * grid_cols
            },
            "obstacle_density_setting": obstacle_density,
            "movement_model": "8-Way (Diagonal)" if allow_diagonal else "4-Way (Orthogonal)",
            "execution_metrics": {
                "goal_reached": stats["found"],
                "shortest_path_steps": stats["path_length"],
                "total_nodes_explored": stats["total_explored"],
                "max_frontier_queue_size": stats["max_queue_size"],
                "search_efficiency_percent": round(efficiency, 2),
                "algorithm_runtime_ms": stats["execution_time_ms"]
            },
            "reconstructed_path_coordinates": [list(p) for p in path]
        }
        
        json_report = json.dumps(report_data, indent=2)
        
        st.markdown("#### Academic Verification & Grading Export")
        st.caption("Download this deterministic JSON report containing your register seed, parameters, and shortest path proof for submission.")
        
        st.download_button(
            label="📥 Download Run Report (JSON)",
            data=json_report,
            file_name=f"bfs_maze_report_seed_{student_seed}.json",
            mime="application/json",
            type="primary"
        )
        
        st.json(report_data, expanded=False)


if __name__ == "__main__":
    main()
