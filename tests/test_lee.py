"""Tests for Lee Maze Routing algorithm."""

import numpy as np
import pytest
from optivlsi.lee import (
    lee,
    lee_networkx,
    lee_numba,
)


def test_path_found_on_empty_grid(simple_maze):
    """Path found on open grid."""
    maze, sx, sy, ex, ey = simple_maze
    distance = lee(maze, sx, sy, ex, ey)
    assert distance > 0, "Should find a path"


def test_no_path_when_blocked():
    """No path when destination is completely blocked."""
    maze = np.zeros((3, 3), dtype=np.int64)
    maze[1, :] = 1  # Block entire middle row
    distance = lee(maze, 0, 0, 2, 2)
    assert distance == -1, "Should return -1 when no path exists"


def test_path_length_correctness(simple_maze):
    """Path length should be correct on a simple grid."""
    maze, sx, sy, ex, ey = simple_maze
    distance = lee(maze, sx, sy, ex, ey)
    # At minimum, path length should be >= Manhattan distance
    manhattan = abs(ex - sx) + abs(ey - sy)
    assert distance >= manhattan, "Path length must be at least Manhattan distance"


def test_implementations_match(simple_maze):
    """Regular vs NetworkX vs Numba produce same result."""
    maze, sx, sy, ex, ey = simple_maze
    py_result = lee(maze, sx, sy, ex, ey)
    nx_result = lee_networkx(maze, sx, sy, ex, ey)
    nb_result = lee_numba(maze, sx, sy, ex, ey)

    assert py_result == nx_result, f"Python {py_result} != NetworkX {nx_result}"
    assert py_result == nb_result[0], f"Python {py_result} != Numba {nb_result[0]}"


def test_one_by_one_grid():
    """1x1 grid with start = end should return distance 1."""
    maze = np.zeros((1, 1), dtype=np.int64)
    distance = lee(maze, 0, 0, 0, 0)
    assert distance == 1


def test_all_blocked_except_start():
    """Grid with all cells blocked except start should return -1."""
    maze = np.ones((3, 3), dtype=np.int64)
    maze[0, 0] = 0  # Only start is open
    distance = lee(maze, 0, 0, 2, 2)
    assert distance == -1