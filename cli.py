"""
CLI Tool for Breadth-First Search (BFS) Benchmarking and Academic Submission Verification.
"""

import argparse
import json
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError, OSError):
        pass

from src.algorithms.bfs import BreadthFirstSearch
from src.benchmark.evaluator import run_bfs_scaling_benchmark
from src.benchmark.integrity import (
    generate_submission_signature,
    verify_submission_signature,
)
from src.generators.presets import get_available_presets, get_preset_maze
from src.generators.random_grid import generate_random_grid
from src.generators.solvability import carve_guaranteed_solution, is_solvable


def main():
    parser = argparse.ArgumentParser(
        description="🧭 Breadth-First Search (BFS) Maze Solver - Benchmarking & Academic Verification"
    )

    parser.add_argument(
        "--seed", type=int, default=113025148022, help="Student register number seed"
    )
    parser.add_argument("--rows", type=int, default=20, help="Grid rows (default: 20)")
    parser.add_argument(
        "--cols", type=int, default=20, help="Grid columns (default: 20)"
    )
    parser.add_argument(
        "--density", type=float, default=0.25, help="Obstacle density (default: 0.25)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=get_available_presets(),
        help="Use a benchmark preset topology",
    )
    parser.add_argument("--diagonal", action="store_true", help="Allow 8-way movement")
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run BFS empirical complexity scaling benchmark across increasing grid dimensions",
    )
    parser.add_argument(
        "--report", type=str, default=None, help="Save verified JSON report to file"
    )
    parser.add_argument(
        "--verify",
        type=str,
        default=None,
        help="Verify cryptographic SHA-256 signature of an existing JSON report file",
    )

    args = parser.parse_args()

    # 1. Verification Mode
    if args.verify:
        print(f"[*] Verifying BFS report integrity: {args.verify}")
        try:
            with open(args.verify, encoding="utf-8") as f:
                report = json.load(f)

            seed = report["student_register_number_seed"]
            rows = report["grid_dimensions"]["rows"]
            cols = report["grid_dimensions"]["columns"]
            density = report["obstacle_density_setting"]

            grid = generate_random_grid(rows, cols, density, seed)
            is_valid = verify_submission_signature(report, grid)

            if is_valid:
                print(
                    "[OK] Signature VALID: Authentic submission report verified against student seed."
                )
                sys.exit(0)
            else:
                print(
                    "[FAIL] Signature INVALID: Report has been tampered with or parameters do not match!"
                )
                sys.exit(1)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"Error during verification: {e}")
            sys.exit(1)

    # 2. Scaling Benchmark Mode
    if args.scaling:
        print(
            f"\n[*] Executing BFS Complexity Scaling Benchmark (Seed: {args.seed})...\n"
        )
        df_scaling = run_bfs_scaling_benchmark(
            seed=args.seed, allow_diagonal=args.diagonal
        )
        print(df_scaling.to_string(index=False))
        return

    # 3. Standard BFS Execution
    if args.preset:
        print(f"Loading benchmark preset '{args.preset}' ({args.rows}x{args.cols})...")
        grid = get_preset_maze(args.preset, args.rows, args.cols)
    else:
        print(
            f"Generating grid from seed {args.seed} ({args.rows}x{args.cols}, density={args.density})..."
        )
        grid = generate_random_grid(args.rows, args.cols, args.density, args.seed)

    start = (0, 0)
    goal = (args.rows - 1, args.cols - 1)

    if not is_solvable(grid, start, goal, args.diagonal):
        print(
            "[!] Grid was naturally disconnected. Carving guaranteed solution path..."
        )
        grid = carve_guaranteed_solution(grid, start, goal, seed=args.seed)

    print(f"\nRunning Breadth-First Search (BFS) on {args.rows}x{args.cols} grid...")
    bfs = BreadthFirstSearch()
    res = bfs.search(
        grid, start=start, goal=goal, allow_diagonal=args.diagonal, record_history=False
    )
    m = res.metrics

    print("\n" + "=" * 60)
    print("🧭 BREADTH-FIRST SEARCH (BFS) EXECUTION TELEMETRY")
    print("=" * 60)
    print(f"Algorithm:               {m.algorithm_name}")
    print(f"Goal Reached:            {'Yes' if m.found else 'No'}")
    print(f"Shortest Path Length:    {m.path_length} steps")
    print(f"Path Cost:               {m.path_cost}")
    print(f"Total Nodes Explored:    {m.total_explored} cells")
    print(f"Max Frontier Queue Size: {m.max_frontier_size} nodes")
    print(f"Max Depth Level Reached: {m.max_depth_reached} hops")
    print(
        f"FIFO Invariant Check:    {'Verified (Monotonic)' if m.fifo_invariant_satisfied else 'Violated'}"
    )
    print(f"Execution Runtime:       {m.execution_time_ms:.3f} ms")
    print(f"Search Efficiency:       {m.search_efficiency:.1f}%")
    print("=" * 60)

    # 4. Report Export
    if args.report:
        sig = generate_submission_signature(
            student_seed=args.seed,
            grid=grid,
            algorithm_name=m.algorithm_name,
            path_length=m.path_length,
            total_explored=m.total_explored,
            path_coordinates=[list(p) for p in res.path],
        )
        report_data = {
            "assignment_title": "BFS Maze Solver & Path Visualizer",
            "student_register_number_seed": args.seed,
            "grid_dimensions": {"rows": args.rows, "columns": args.cols},
            "obstacle_density_setting": args.density,
            "movement_model": "8-Way" if args.diagonal else "4-Way",
            "algorithm_name": m.algorithm_name,
            "execution_metrics": {
                "goal_reached": m.found,
                "shortest_path_steps": m.path_length,
                "path_cost": m.path_cost,
                "total_nodes_explored": m.total_explored,
                "max_frontier_queue_size": m.max_frontier_size,
                "max_depth_level": m.max_depth_reached,
                "fifo_monotonicity_satisfied": m.fifo_invariant_satisfied,
                "search_efficiency_percent": m.search_efficiency,
                "algorithm_runtime_ms": m.execution_time_ms,
            },
            "reconstructed_path_coordinates": [list(p) for p in res.path],
            "verification_signature": sig,
        }

        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n[+] Verifiable BFS run report saved to: {args.report}")


if __name__ == "__main__":
    main()
