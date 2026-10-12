# 🧭 Breadth-First Search (BFS) Maze Solver & Path Visualizer

An academic-grade, production-quality **Breadth-First Search (BFS) Maze Solver & Graph Traversal Visualizer** built in Python with **Streamlit** and **Matplotlib**. Features interactive step-by-step queue scrubbing, FIFO queue dynamics, level-order wavefront telemetry, procedural maze generators, exploration depth heatmaps, parent flow fields, empirical $\mathcal{O}(V+E)$ complexity benchmarks, and tamper-proof **SHA-256 cryptographic submission verification**.

---

## 🌟 Key Technical Features

### 1. Pure Breadth-First Search (BFS) Engine
- **FIFO Queue Invariant**: Implemented using Python's `collections.deque` for true $\mathcal{O}(1)$ amortized `popleft()` and `append()` operations.
- **Level-Order Monotonicity**: Verifies at runtime that nodes are expanded in strictly non-decreasing order of their distance from the source $s$:
  $$\text{dist}(s, u) \le \text{dist}(s, v) \quad \forall \; u \text{ dequeued before } v$$
- **Shortest Path Guarantee**: Proved via CLRS Theorem 20.5 — on unweighted/uniform-cost graphs ($c(u, v) = 1$), the first time the Goal node is discovered, the path reconstructed via parent back-pointers $\text{parent}[u]$ is mathematically guaranteed to be the shortest path.
- **Wavefront Distribution Tracking**: Collects telemetry on the number of nodes discovered per distance level $k$ ($N(k)$).

### 2. Procedural Maze Generators & Benchmark Topologies
- **100% Deterministic Seeded Random Grid**: Sidebar input for **Student Register Number / Seed** directly seeds `random` and `numpy.random`, ensuring exact assignment reproducibility across runs.
- **Recursive Backtracker (DFS Spanning Tree Maze)**: Generates 'perfect mazes' with guaranteed single unique solutions and long winding corridors for BFS to traverse.
- **Randomized Prim's Algorithm (Minimum Spanning Tree)**: Generates branching organic mazes featuring numerous short dead ends.
- **Curated Benchmark Presets**:
  - *Chokepoint / Bottleneck*: Two large open chambers separated by a single 1-cell slit.
  - *Spiral Labyrinth*: Concentric square walls testing frontier discipline.
  - *Zig-Zag Slits*: Alternating horizontal slit walls.
  - *Deceptive U-Trap*: Concave obstacle barrier testing BFS backtracking.
  - *Pillar Arena*: Regular lattice of obstacle columns.
- **Guaranteed Solvability Engine**: Auto-carver punches natural corridors if random density isolates the goal.

### 3. Deep Analytical Visualizations & Overlays
- **BFS Exploration Depth Heatmap**: Color-gradients ($\text{dist}(s, v)$ layer by layer) visualizing the concentric shells of BFS level expansion.
- **Parent Vector Flow Field**: Directional arrows on explored nodes showing back-pointers directing all traversals to the source Start node.
- **FIFO Queue Dynamics Plots**:
  1. *Queue Size over Exploration Step* with peak frontier marker.
  2. *BFS Wavefront: Nodes per Distance Level $k$*.
  3. *Cumulative Exploration Linearity* confirming linear $\mathcal{O}(V+E)$ progress.
  4. *Effective Wavefront Branching Factor* ($N_{k+1} / N_k$).
- **Interactive Step Scrubber**: Drag slider or step buttons to inspect frontier queue contents and visited set at any arbitrary expansion tick.
- **Live Playback Animation**: Real-time rendering with user-controlled animation delays.

### 4. Empirical Complexity & Scalability Analysis
- **Grid Scaling Benchmark**: Measures runtime, explored node count, and peak queue memory as grid size scales from $10\times 10$ to $30\times 30$, empirically validating theoretical $\mathcal{O}(V+E)$ bounds.
- **Obstacle Density Sweep**: Evaluates BFS efficiency under varying obstacle densities ($10\%$ to $40\%$).

### 5. Academic Verification & CLI Tooling
- **SHA-256 Cryptographic Signature**: Automatically generated hash binding student ID, grid binary digest, BFS telemetry, and coordinate sequence to prevent assignment report tampering.
- **Headless CLI (`cli.py`)**: Run automated batch benchmarks, generate reports, or authenticate submitted student JSON reports directly from the terminal.
- **Automated Unit Test Suite**: 17 unit tests verifying BFS FIFO invariants, level distribution, obstacle detours, and reproducibility (`python -m unittest discover tests -v`).

---

## 📦 Installation & Setup

### 1. Prerequisites
- Python 3.9+ installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 💻 Command-Line Interface (CLI)

The repository includes a headless CLI tool (`cli.py`) for automated grading and benchmarking:

### Run BFS on a Grid & Display Telemetry
```bash
python cli.py --seed 113025148022 --rows 20 --cols 20 --density 0.25
```

### Run Complexity Scaling Benchmark
```bash
python cli.py --seed 113025148022 --scaling
```

### Export a Verifiable Academic JSON Report
```bash
python cli.py --seed 113025148022 --rows 20 --cols 20 --report submission.json
```

### Authenticate a Submitted Report (Grading Verification)
```bash
python cli.py --verify submission.json
```

---

## 🧪 Running Automated Unit Tests

Run the full automated test suite using Python's standard `unittest` framework:

```bash
python -m unittest discover tests -v
```

All 17 tests verify:
- BFS correctness, boundary checks, and 4-way/8-way movement.
- FIFO monotonicity invariant satisfaction.
- BFS level distribution integrity ($N(k)$ sum equals total explored).
- Queue peak bounds matching recorded maximums.
- Procedural maze solvability (Recursive Backtracker and Prim's).
- Deterministic seed reproducibility.
- Cryptographic SHA-256 verification and tamper detection.

---

## 📚 Theoretical Complexity & BFS Invariants

| Property | Value / Formula | Note |
| :--- | :--- | :--- |
| **Time Complexity** | $\mathcal{O}(V + E) = \mathcal{O}(R \times C)$ | Every cell and edge is enqueued and dequeued at most once. |
| **Space Complexity** | $\mathcal{O}(V) = \mathcal{O}(R \times C)$ | Queue storage, visited hash set, and parent mapping table. |
| **Queue Structure** | `collections.deque` (FIFO) | $\mathcal{O}(1)$ amortized `popleft()` and `append()` operations. |
| **Optimality** | **Guaranteed** | Proven for uniform-cost graphs ($c=1$). |
| **Completeness** | **Guaranteed** | Always finds a path if one exists on finite graphs. |

---

## 📁 Modular Project Architecture

```
grid-maze/
├── app.py                      # Main interactive Streamlit visualizer (Pure BFS)
├── cli.py                      # Headless CLI for BFS benchmarking and grading verification
├── requirements.txt            # Project dependencies (streamlit, matplotlib, numpy, pandas)
├── README.md                   # Complete academic documentation and instructions
│
├── src/                        # Modular source package
│   ├── algorithms/             # BFS algorithm implementation
│   │   ├── base.py             # SearchAlgorithm ABC, SearchResult, and AlgorithmMetrics
│   │   ├── bfs.py              # Pure Breadth-First Search engine with level tracking
│   │   └── registry.py         # Factory and registry for BFS
│   │
│   ├── generators/             # Procedural grid & maze generation for BFS
│   │   ├── random_grid.py      # Deterministic seeded random grid
│   │   ├── recursive_backtracker.py # Randomized DFS perfect maze generator
│   │   ├── prims.py            # Randomized Prim's minimum spanning tree maze
│   │   ├── presets.py          # Benchmark topologies (Chokepoint, Spiral, Trap, etc.)
│   │   └── solvability.py      # Fast BFS connectivity validator & path carver
│   │
│   ├── visualization/          # High-contrast visual rendering engine
│   │   ├── canvas.py           # Matplotlib dark canvas with heatmaps & flow fields
│   │   └── comparison_plot.py  # BFS queue dynamics and wavefront analytical plots
│   │
│   └── benchmark/              # Empirical complexity & cryptographic integrity
│       ├── evaluator.py        # BFS dimension scaling and density sweep benchmarks
│       └── integrity.py        # SHA-256 assignment signature generation & verification
│
└── tests/                      # Comprehensive automated test suite
    ├── test_bfs.py             # BFS unit tests (optimality, bounds, FIFO, queue peak, levels)
    └── test_generators.py      # Reproducibility, solvability, and integrity verification tests
```

---

## 📝 Academic References & Citations

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. (Chapter 20: Elementary Graph Algorithms - Breadth-First Search).
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. (Chapter 3: Solving Problems by Searching - Uninformed Search).
