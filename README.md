# 🧭 BFS Maze Solver & Path Visualizer

An interactive, self-contained **Breadth-First Search (BFS) Maze Solver Visualizer** built with Python and Streamlit for academic assignments, algorithm verification, and grading reproducibility.

---

## 🌟 Key Features

1. **Anti-Copy & Deterministic Reproducibility**:
   - Sidebar input for **Student Register Number / Seed** directly seeds `random` and `numpy.random`.
   - Any given student ID generates a unique, 100% reproducible obstacle grid.
2. **Ground Rules & Grid Setup**:
   - Customizable grid size (e.g., $15 \times 15$ to $35 \times 35$) and obstacle density ($10\%$ to $45\%$).
   - Guaranteed open Start at `(0, 0)` [Top-Left] and Goal at `(Max, Max)` [Bottom-Right].
3. **Pure BFS Algorithm Engine**:
   - Implemented with Python's `collections.deque` for true $\mathcal{O}(1)$ FIFO queue operations.
   - Parent pointer mapping (`parent[child] = curr`) for accurate backward path reconstruction.
   - Comprehensive documentation of BFS optimality on uniform-cost graphs ($c=1$).
4. **Interactive High-Resolution Visualization**:
   - Embedded Matplotlib canvas with custom color schemes:
     - **Start**: Emerald Green `(S)`
     - **Goal**: Crimson Star `(G)`
     - **Obstacles / Walls**: Dark Slate Blocks
     - **Explored Nodes**: Soft Cyan
     - **Frontier Queue**: Soft Purple
     - **Shortest Path**: Glowing Crimson/Gold vector line
   - **Step-by-Step Animation**: Real-time frontier expansion animation.
   - **Interactive Step Scrubber**: Drag slider to inspect queue and visited states at any arbitrary step.
5. **Academic Submission Tools**:
   - Real-time performance metrics (Path Length, Explored Count, Queue Peak, Search Efficiency %).
   - Step-by-step path coordinates breakdown table.
   - One-click downloadable JSON Academic Run Report.
   - Safety alerts when random obstacles completely disconnect Start and Goal.

---

## 📦 Installation & Setup

### 1. Prerequisites
- Python 3.9+ installed on your system.

### 2. Install Dependencies
Navigate to the project root directory and install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Application
Launch the Streamlit web application:

```bash
streamlit run app.py
```

The web visualizer will automatically open in your default browser at `http://localhost:8501`.

---

## ⚙️ Parameter Configuration Guide

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **Student Register Number / Seed** | Integer | `113025148022` | Official student ID used as the deterministic RNG seed for obstacle generation. |
| **Rows & Columns** | Sliders | `20 x 20` | Dimensions of the 2D grid ($N \times M$). |
| **Obstacle Density** | Slider | `0.25` (25%) | Percentage of blocked wall cells randomly placed across the grid. |
| **8-Way Movement (Diagonal)** | Checkbox | `False` | Toggle between standard 4-way orthogonal ($c=1$) or 8-way diagonal movement. |
| **Animation Speed** | Slider | `35 ms` | Playback delay per expansion step during live animation. |

---

## 📚 Theoretical Background & BFS Optimality

### Uniform-Cost Grid Property
On a grid where moving from any cell to an adjacent unblocked neighbor incurs an identical cost of $c(u, v) = 1$, the graph is unweighted / uniform-cost.

### FIFO Queue Invariant
Breadth-First Search maintains a FIFO queue ($Q$). Nodes are dequeued and processed in non-decreasing order of their distance from the source $s$:
$$\text{dist}(s, u) \le \text{dist}(s, v) \quad \forall \; u \text{ dequeued before } v$$

### Proof of Optimality
1. **Level-Order Traversal**: BFS visits all nodes at distance $k$ before any node at distance $k + 1$.
2. **First Discovery Guarantee**: When the Goal node $g = (R-1, C-1)$ is first discovered and placed into the queue, the length of the discovered path is $\text{dist}(s, u) + 1 = \delta(s, g)$.
3. **Reconstruction**: By backtracking along parent pointers from $g$ back to $s$, the reconstructed path is guaranteed to have minimal edge count (shortest path).

### Complexity
- **Time Complexity**: $\mathcal{O}(V + E) = \mathcal{O}(R \times C)$ where $V = R \times C$ and $E \le 4V$.
- **Space Complexity**: $\mathcal{O}(V) = \mathcal{O}(R \times C)$ for the FIFO queue, visited set, and parent dictionary.

---

## 📁 Project Structure

```
grid-maze/
├── app.py              # Complete Streamlit UI, BFS Solver & Matplotlib Visualizer
├── requirements.txt    # Project dependencies (streamlit, matplotlib, numpy, pandas)
└── README.md           # Academic documentation, setup guide, and theory
```

---

## 📝 Academic Submission & Citations

### Run Report Export
Students can download a verified JSON run report directly from the application (under the **📥 Submission Export** tab). The report records the student's register number seed, grid dimensions, obstacle density, nodes explored, and exact coordinate sequence of the reconstructed shortest path.

### Academic Integrity & References
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. (Chapter 20: Elementary Graph Algorithms - Breadth-First Search).
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. (Chapter 3: Solving Problems by Searching).
