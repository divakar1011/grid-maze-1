"""
Academic submission integrity and SHA-256 cryptographic verification.
"""

import hashlib
import json
from typing import Any

import numpy as np


def compute_grid_hash(grid: np.ndarray) -> str:
    """Computes SHA-256 digest of the raw binary grid representation."""
    return hashlib.sha256(grid.tobytes()).hexdigest()[:16]


def generate_submission_signature(
    student_seed: int,
    grid: np.ndarray,
    algorithm_name: str,
    path_length: int,
    total_explored: int,
    path_coordinates: list,
) -> str:
    """
    Generates a cryptographically secure SHA-256 verification signature.
    Prevents tampering with assignment submission reports.
    """
    payload = {
        "student_seed": student_seed,
        "grid_hash": compute_grid_hash(grid),
        "grid_shape": list(grid.shape),
        "algorithm": algorithm_name,
        "path_length": path_length,
        "total_explored": total_explored,
        "path": path_coordinates,
    }
    payload_str = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def verify_submission_signature(report_data: dict[str, Any], grid: np.ndarray) -> bool:
    """Verifies that an exported JSON report matches the computed signature."""
    try:
        expected_sig = generate_submission_signature(
            student_seed=report_data["student_register_number_seed"],
            grid=grid,
            algorithm_name=report_data["algorithm_name"],
            path_length=report_data["execution_metrics"]["shortest_path_steps"],
            total_explored=report_data["execution_metrics"]["total_nodes_explored"],
            path_coordinates=report_data["reconstructed_path_coordinates"],
        )
        return expected_sig == report_data.get("verification_signature")
    except (KeyError, TypeError, ValueError):
        return False
