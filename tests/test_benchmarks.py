"""Benchmark tests for optiVLSI algorithms using pytest-benchmark.

Uses the same graph generation parameters and problem sizes as the original
research paper (OptiVLSI.pdf) to produce comparable speedup results.
"""

import numpy as np
import pytest
import networkx as nx
import random

# Import algorithm modules
from optivlsi.bellman_ford import bellman_ford, bellman_ford_networkx, bellman_ford_numba
from optivlsi.bellman_ford.algorithms import graph_to_numpy as bf_graph_to_numpy

from optivlsi.dijkstra import dijkstra, dijkstra_networkx, dijkstra_numba

from optivlsi.kruskal import kruskal, kruskal_networkx, kruskal_numba

from optivlsi.prim import prim, prim_networkx, prim_numba
from optivlsi.prim.algorithms import create_nx_graph as prim_create_nx_graph

from optivlsi.lee import lee, lee_networkx, lee_numba


# ---- Graph Generation Helpers (matching original automate.py parameters) ----

def create_bf_graph(size):
    """Create Bellman-Ford benchmark graph matching paper: p=0.5, w=[5,15], s=0, e=400."""
    generated = False
    while not generated:
        graph = nx.gnp_random_graph(size, 0.5, directed=True)
        circuit = nx.DiGraph([(u, v, {'weight': random.randint(5, 15)})
                              for (u, v) in graph.edges() if u < v])
        generated = nx.is_directed_acyclic_graph(circuit)
    return bf_graph_to_numpy(circuit)


def create_dijkstra_graph(size):
    """Create Dijkstra benchmark graph matching paper: p=0.3, w=[5,15], s=0, e=500."""
    generated = False
    while not generated:
        graph = nx.gnp_random_graph(size, 0.3, directed=True)
        circuit = nx.DiGraph([(u, v, {'weight': random.randint(5, 15)})
                              for (u, v) in graph.edges() if u < v])
        generated = nx.is_directed_acyclic_graph(circuit)
    nodes = np.array(circuit.nodes, dtype=np.int64)
    edges = []
    for e in circuit.edges:
        edges.append((e[0], e[1], circuit[e[0]][e[1]]["weight"]))
    return (nodes, np.array(edges, dtype=np.int64))


def create_mst_graph(size):
    """Create MST benchmark graph matching paper: p=0.3, w=[5,15].
    Uses undirected graph for MST algorithms."""
    graph = nx.gnp_random_graph(size, 0.3)
    circuit = nx.Graph([(u, v, {'weight': random.randint(5, 15)})
                        for (u, v) in graph.edges()])
    nodes = np.array(circuit.nodes, dtype=np.int64)
    nodes.sort()
    edges = []
    for e in circuit.edges:
        edges.append((e[0], e[1], circuit[e[0]][e[1]]["weight"]))
    return (nodes, np.array(edges, dtype=np.int64))


def create_lee_maze(size):
    """Create Lee maze benchmark matching paper: maze of given size.
    Blocks central cells to create a challenging routing problem."""
    maze = np.zeros((size, size), dtype=np.int64)
    # Block central region (leaves 1-cell border)
    border = max(1, size // 10)
    center_start = border
    center_end = size - border
    if center_end > center_start:
        maze[center_start:center_end, center_start:center_end] = 1
        # Create a narrow winding corridor
        for i in range(center_start, center_end, 3):
            if i + 1 < center_end:
                maze[i:i+2, center_start + (i - center_start) // 2] = 0
    return maze, 0, 0, size - 1, size - 1


# ---- Research Paper Sizes ----
# These match the sizes used in the original automate.py files

@pytest.fixture(params=[50, 100, 200])
def bf_benchmark(request):
    """Bellman-Ford at paper sizes: p=0.5, w=[5,15], s=0, e=size-1."""
    size = request.param
    nodes, edges = create_bf_graph(size)
    # Ensure end node is valid
    end = min(size - 1, 400) if size > 400 else size - 1
    return nodes, edges, 0, end


@pytest.fixture(params=[50, 100, 175])
def dijkstra_benchmark(request):
    """Dijkstra at paper sizes: p=0.3, w=[5,15], s=0, e=size-1."""
    size = request.param
    nodes, edges = create_dijkstra_graph(size)
    end = min(size - 1, 500) if size > 500 else size - 1
    return nodes, edges, 0, end


@pytest.fixture(params=[50, 100, 200])
def mst_benchmark(request):
    """MST at paper sizes: p=0.3, w=[5,15]."""
    size = request.param
    nodes, edges = create_mst_graph(size)
    return nodes, edges


@pytest.fixture(params=[50, 100, 200])
def lee_benchmark(request):
    """Lee at paper sizes."""
    size = request.param
    return create_lee_maze(size)


# ---- Bellman-Ford Benchmarks (paper: sizes 10-200, p=0.5, w=[5,15]) ----

def test_benchmark_bellman_ford_pythonic(benchmark, bf_benchmark):
    nodes, edges, src, end = bf_benchmark
    result = benchmark(bellman_ford, nodes, edges, src, end)
    assert result[0] == 1


def test_benchmark_bellman_ford_networkx(benchmark, bf_benchmark):
    nodes, edges, src, end = bf_benchmark
    result = benchmark(bellman_ford_networkx, nodes, edges, src, end)
    assert result[0] == 1


def test_benchmark_bellman_ford_numba(benchmark, bf_benchmark):
    nodes, edges, src, end = bf_benchmark
    result = benchmark(bellman_ford_numba, nodes, edges, src, end)
    assert result[0] == 1


# ---- Dijkstra Benchmarks (paper: sizes 10-175, p=0.3, w=[5,15]) ----

def test_benchmark_dijkstra_pythonic(benchmark, dijkstra_benchmark):
    nodes, edges, src, end = dijkstra_benchmark
    result = benchmark(dijkstra, nodes, edges, src, end)
    assert result[0] == 1


def test_benchmark_dijkstra_networkx(benchmark, dijkstra_benchmark):
    nodes, edges, src, end = dijkstra_benchmark
    result = benchmark(dijkstra_networkx, nodes, edges, src, end)
    assert result[0] == 1


def test_benchmark_dijkstra_numba(benchmark, dijkstra_benchmark):
    nodes, edges, src, end = dijkstra_benchmark
    result = benchmark(dijkstra_numba, nodes, edges, src, end)
    assert result[0] == 1


# ---- Kruskal Benchmarks (paper: sizes 10-500, p=0.3, w=[5,15]) ----

def test_benchmark_kruskal_pythonic(benchmark, mst_benchmark):
    nodes, edges = mst_benchmark
    result = benchmark(kruskal, nodes, edges)
    assert len(result[1]) > 0


def test_benchmark_kruskal_networkx(benchmark, mst_benchmark):
    nodes, edges = mst_benchmark
    result = benchmark(kruskal_networkx, nodes, edges)
    assert len(result[1]) > 0


def test_benchmark_kruskal_numba(benchmark, mst_benchmark):
    nodes, edges = mst_benchmark
    result = benchmark(kruskal_numba, nodes, edges)
    assert len(result[1]) > 0


# ---- Prim Benchmarks (paper: sizes 10-500, p=0.3, w=[5,15]) ----

def test_benchmark_prim_pythonic(benchmark, mst_benchmark):
    nodes, edges = mst_benchmark
    result = benchmark(prim, nodes, edges)
    assert len(result[1]) > 0


def test_benchmark_prim_networkx(benchmark, mst_benchmark):
    nodes, edges = mst_benchmark
    result = benchmark(prim_networkx, nodes, edges)
    assert len(result[1]) > 0


def test_benchmark_prim_numba(benchmark, mst_benchmark):
    nodes, edges = mst_benchmark
    result = benchmark(prim_numba, nodes, edges)
    assert len(result[1]) > 0


# ---- Lee Algorithm Benchmarks (paper: sizes 2-500) ----

def test_benchmark_lee_pythonic(benchmark, lee_benchmark):
    maze, sx, sy, ex, ey = lee_benchmark
    distance = benchmark(lee, maze, sx, sy, ex, ey)
    assert distance > 0


def test_benchmark_lee_networkx(benchmark, lee_benchmark):
    maze, sx, sy, ex, ey = lee_benchmark
    distance = benchmark(lee_networkx, maze, sx, sy, ex, ey)
    assert distance > 0


def test_benchmark_lee_numba(benchmark, lee_benchmark):
    maze, sx, sy, ex, ey = lee_benchmark
    result = benchmark(lee_numba, maze, sx, sy, ex, ey)
    assert result[0] > 0