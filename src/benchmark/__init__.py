"""
Benchmark package for BFS analysis and cryptographic verification.
"""

from src.benchmark.evaluator import (
    run_bfs_density_benchmark,
    run_bfs_scaling_benchmark,
)
from src.benchmark.integrity import (
    compute_grid_hash,
    generate_submission_signature,
    verify_submission_signature,
)

__all__ = [
    "compute_grid_hash",
    "generate_submission_signature",
    "run_bfs_density_benchmark",
    "run_bfs_scaling_benchmark",
    "verify_submission_signature",
]
