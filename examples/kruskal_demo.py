#!/usr/bin/env python3
"""Example: Kruskal minimum spanning tree on an undirected graph."""

import numpy as np
from optivlsi.kruskal import kruskal, kruskal_networkx

# Small undirected graph: 4 nodes
nodes = np.array([0, 1, 2, 3], dtype=np.int64)
edges = np.array([
    [0, 1, 10],
    [0, 2, 6],
    [0, 3, 5],
    [1, 3, 15],
    [2, 3, 4],
], dtype=np.int64)

mst_edges, mst_weight = kruskal(nodes, edges)
print(f"Kruskal MST edges: {mst_edges}")
print(f"Kruskal MST weight: {mst_weight}")

mst_edges_nx, mst_weight_nx = kruskal_networkx(nodes, edges)
print(f"NetworkX MST weight: {mst_weight_nx}")
print(f"Weights match: {mst_weight == mst_weight_nx}")