"""
Registry and factory for Breadth-First Search (BFS) engine.
"""

from src.algorithms.base import SearchAlgorithm
from src.algorithms.bfs import BreadthFirstSearch

ALGORITHM_FACTORY = {
    "BFS": lambda: BreadthFirstSearch(),
}


def get_algorithm(name: str = "BFS") -> SearchAlgorithm:
    """Returns an instance of the BFS algorithm."""
    return BreadthFirstSearch()


def get_available_algorithms() -> list[str]:
    """Returns list of algorithm display keys."""
    return ["Breadth-First Search (BFS)"]
