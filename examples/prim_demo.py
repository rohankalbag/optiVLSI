#!/usr/bin/env python3
"""Example: Prim minimum spanning tree on an undirected graph."""

import numpy as np
from optivlsi.prim import prim, prim_networkx

# Small undirected graph: 4 nodes
nodes = np.array([0, 1, 2, 3], dtype=np.int64)
edges = np.array([
    [0, 1, 10],
    [0, 2, 6],
    [0, 3, 5],
    [1, 3, 15],
    [2, 3, 4],
], dtype=np.int64)

mst_edges, mst_weight = prim(nodes, edges)
print(f"Prim MST edges: {mst_edges}")
print(f"Prim MST weight: {mst_weight}")

mst_edges_nx, mst_weight_nx = prim_networkx(nodes, edges)
print(f"NetworkX Prim weight: {mst_weight_nx}")
print(f"Weights match: {mst_weight == mst_weight_nx}")