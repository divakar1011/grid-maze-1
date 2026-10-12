"""
Breadth-First Search (BFS) Algorithm Engine.

Optimality Guarantee & Mathematical Invariants:
On uniform-cost graphs (where all edge transition costs c(u, v) = 1), BFS guarantees
the shortest path (minimal edge count) because it explores nodes in strictly non-decreasing
order of their distance from the start node:
    dist(s, u) <= dist(s, v) for all nodes u dequeued before v.

Time Complexity: O(V + E) = O(R * C)
Space Complexity: O(V) = O(R * C)
Queue Implementation: collections.deque (O(1) popleft & append operations)
"""

import time
from collections import deque
from typing import Any

import numpy as np

from src.algorithms.base import (
    AlgorithmMetrics,
    Coordinate,
    SearchAlgorithm,
    SearchResult,
)


class BreadthFirstSearch(SearchAlgorithm):
    """Pure Breadth-First Search implementation with queue dynamics and level tracking."""

    def __init__(self):
        super().__init__(name="Breadth-First Search (BFS)", is_optimal=True)

    def search(
        self,
        grid: np.ndarray,
        start: Coordinate = (0, 0),
        goal: Coordinate | None = None,
        allow_diagonal: bool = False,
        record_history: bool = True,
    ) -> SearchResult:
        rows, cols = grid.shape
        if goal is None:
            goal = (rows - 1, cols - 1)

        start_time = time.perf_counter()

        # Distance matrix for level tracking and heatmap (-1 = unvisited)
        dist_matrix = np.full((rows, cols), -1.0, dtype=np.float32)

        # Sanity validation: start or goal blocked
        if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
            exec_time = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(
                algorithm_name=self.name,
                path=[],
                visited_order=[],
                steps_history=[],
                metrics=AlgorithmMetrics(
                    algorithm_name=self.name,
                    found=False,
                    path_length=0,
                    path_cost=0.0,
                    total_explored=0,
                    max_frontier_size=0,
                    execution_time_ms=round(exec_time, 3),
                    search_efficiency=0.0,
                    is_optimal=True,
                    max_depth_reached=0,
                    fifo_invariant_satisfied=True,
                ),
                parent_map={},
                distance_matrix=dist_matrix,
                level_distribution={},
                queue_size_history=[],
            )

        # Data structures
        # 1. FIFO Queue: O(1) popleft & append
        queue: deque[Coordinate] = deque([start])

        # 2. Visited set: prevents cycles and redundant re-expansions
        visited: set[Coordinate] = {start}

        # 3. Parent Map: parent[child] = curr; enables exact backwards path reconstruction
        parent: dict[Coordinate, Coordinate | None] = {start: None}
        dist_matrix[start[0], start[1]] = 0.0

        visited_order: list[Coordinate] = []
        steps_history: list[dict[str, Any]] = []
        queue_size_history: list[int] = [1]
        level_distribution: dict[int, int] = {0: 1}

        max_queue_size = 1
        max_depth_reached = 0
        fifo_invariant_satisfied = True
        last_dequeued_dist = 0.0
        found_goal = False
        step_count = 0

        if record_history:
            steps_history.append(
                {
                    "step": 0,
                    "current": None,
                    "queue": list(queue),
                    "visited": set(visited),
                    "path": [],
                }
            )

        while queue:
            max_queue_size = max(max_queue_size, len(queue))
            queue_size_history.append(len(queue))

            # Dequeue from front of FIFO queue
            curr = queue.popleft()
            visited_order.append(curr)
            step_count += 1

            curr_dist = float(dist_matrix[curr[0], curr[1]])
            curr_depth_int = round(curr_dist)
            max_depth_reached = max(max_depth_reached, curr_depth_int)

            # Mathematical Invariant check: dist must be monotonically non-decreasing
            if curr_dist < last_dequeued_dist - 1e-5:
                fifo_invariant_satisfied = False
            last_dequeued_dist = curr_dist

            if curr == goal:
                found_goal = True
                if record_history:
                    steps_history.append(
                        {
                            "step": step_count,
                            "current": curr,
                            "queue": list(queue),
                            "visited": set(visited),
                            "path": [],
                        }
                    )
                break

            # Explore valid unblocked neighbors
            for neighbor, cost in self.get_neighbors(curr, grid, allow_diagonal):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    n_dist = curr_dist + cost
                    dist_matrix[neighbor[0], neighbor[1]] = n_dist
                    queue.append(neighbor)

                    n_depth = round(n_dist)
                    level_distribution[n_depth] = level_distribution.get(n_depth, 0) + 1

            if record_history:
                steps_history.append(
                    {
                        "step": step_count,
                        "current": curr,
                        "queue": list(queue),
                        "visited": set(visited),
                        "path": [],
                    }
                )

        end_time = time.perf_counter()
        exec_time_ms = (end_time - start_time) * 1000.0

        path: list[Coordinate] = []
        path_cost = 0.0
        if found_goal:
            path = self.reconstruct_path(parent, goal)
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i + 1]
                dr, dc = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
                path_cost += 1.4142 if (dr == 1 and dc == 1) else 1.0

            if record_history and steps_history:
                steps_history[-1]["path"] = path

        path_len = len(path) - 1 if found_goal else 0
        total_explored = len(visited_order)
        efficiency = (path_len / max(1, total_explored)) * 100.0 if found_goal else 0.0

        metrics = AlgorithmMetrics(
            algorithm_name=self.name,
            found=found_goal,
            path_length=path_len,
            path_cost=round(path_cost, 2),
            total_explored=total_explored,
            max_frontier_size=max_queue_size,
            execution_time_ms=round(exec_time_ms, 3),
            search_efficiency=round(efficiency, 2),
            is_optimal=True,
            max_depth_reached=max_depth_reached,
            fifo_invariant_satisfied=fifo_invariant_satisfied,
        )

        return SearchResult(
            algorithm_name=self.name,
            path=path,
            visited_order=visited_order,
            steps_history=steps_history,
            metrics=metrics,
            parent_map=parent,
            distance_matrix=dist_matrix,
            level_distribution=level_distribution,
            queue_size_history=queue_size_history,
        )
