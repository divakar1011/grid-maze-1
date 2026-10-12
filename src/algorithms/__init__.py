"""
Algorithms package: pure Breadth-First Search (BFS) engine.
"""

from src.algorithms.base import (
    AlgorithmMetrics,
    Coordinate,
    SearchAlgorithm,
    SearchResult,
)
from src.algorithms.bfs import BreadthFirstSearch
from src.algorithms.registry import get_algorithm, get_available_algorithms

__all__ = [
    "AlgorithmMetrics",
    "BreadthFirstSearch",
    "Coordinate",
    "SearchAlgorithm",
    "SearchResult",
    "get_algorithm",
    "get_available_algorithms",
]
