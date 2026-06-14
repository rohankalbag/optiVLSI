"""
Lee Maze Routing algorithm implementations.

Provides three variants:
- lee(): Pure Python implementation
- lee_networkx(): NetworkX-based BFS implementation
- lee_numba(): Numba-accelerated implementation
"""

from .algorithms import (
    lee_algorithm,
    lee_algorithm_networkx,
    accelerated_lee_algorithm,
)


def lee(maze, sx, sy, ex, ey):
    """Find path in maze using pure Python Lee algorithm (BFS).

    Args:
        maze: 2D numpy array where 0 = open, 1 = blocked.
        sx, sy: Start coordinates.
        ex, ey: End coordinates.

    Returns:
        Distance (number of steps) if path found, -1 otherwise.
    """
    return lee_algorithm(maze, sx, sy, ex, ey)


def lee_networkx(maze, sx, sy, ex, ey):
    """Find path in maze using NetworkX BFS.

    Args:
        maze: 2D numpy array where 0 = open, 1 = blocked.
        sx, sy: Start coordinates.
        ex, ey: End coordinates.

    Returns:
        Distance (number of steps) if path found, -1 otherwise.
    """
    return lee_algorithm_networkx(maze, sx, sy, ex, ey)


def lee_numba(maze, sx, sy, ex, ey):
    """Find path in maze using Numba-accelerated Lee algorithm.

    Args:
        maze: 2D numpy array where 0 = open, 1 = blocked.
        sx, sy: Start coordinates.
        ex, ey: End coordinates.

    Returns:
        Tuple of (distance, visited_matrix) if path found, (-1, visited) otherwise.
    """
    from numba.typed import List

    queue_x = List()
    queue_x.append(1)
    queue_x.pop()

    queue_y = List()
    queue_y.append(1)
    queue_y.pop()

    return accelerated_lee_algorithm(maze, sx, sy, ex, ey, queue_x, queue_y)


__all__ = [
    "lee",
    "lee_networkx",
    "lee_numba",
]