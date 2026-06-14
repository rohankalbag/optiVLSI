"""
Dijkstra shortest path algorithm implementations.

Provides three variants: pure Python, NetworkX-based, and Numba-accelerated.
"""

import networkx as nx
import numpy as np
import random
import heapq
import numba
from numba.typed import List


def create_nx_graph(size, prob, wt_min, wt_max):
    """Create a random directed acyclic graph.

    Args:
        size: Number of nodes.
        prob: Probability of edge existence.
        wt_min: Minimum edge weight.
        wt_max: Maximum edge weight.

    Returns:
        NetworkX DiGraph.
    """
    generated = False
    while not generated:
        graph = nx.gnp_random_graph(size, prob, directed=True)
        circuit = nx.DiGraph([(u, v, {'weight': random.randint(wt_min, wt_max)}) for
                              (u, v) in graph.edges() if u < v])
        generated = nx.is_directed_acyclic_graph(circuit)
    return circuit


def graph_to_numpy(graph):
    """Convert a NetworkX graph to numpy arrays.

    Args:
        graph: NetworkX graph.

    Returns:
        Tuple of (nodes, edges) as numpy arrays.
    """
    nodes = np.array(graph.nodes, dtype=np.int64)
    edges = graph.edges
    edge_list = []
    for e in edges:
        edge_list.append((e[0], e[1], graph[e[0]][e[1]]["weight"]))
    return (nodes, np.array(edge_list, dtype=np.int64))


def numpy_to_graph(nodes, edges):
    """Convert numpy arrays to a NetworkX graph.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.

    Returns:
        NetworkX DiGraph.
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for e in edges:
        G.add_edge(e[0], e[1], weight=e[2])
    return G


def dijkstra_nx(nodes, edges, src, end):
    """Find shortest path using NetworkX Dijkstra.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.
        src: Source node index.
        end: Destination node index.

    Returns:
        Tuple of (status, path) where status is 1 if path found, -1 otherwise.
    """
    graph = numpy_to_graph(nodes, edges)
    try:
        path = nx.dijkstra_path(graph, src, end)
        return (1, path)
    except nx.NetworkXNoPath:
        return (-1, None)


def dijkstra_pythonic(nodes, edges, src, end):
    """Find shortest path using pure Python Dijkstra.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.
        src: Source node index.
        end: Destination node index.

    Returns:
        Tuple of (status, path) where status is 1 if path found, -1 otherwise.
    """
    if src == end:
        return (1, [src])

    visited = {}
    distance = {}
    next_hop = {}
    priority_queue = []

    for i in nodes:
        visited[i] = False
        distance[i] = np.iinfo(np.int64).max
        next_hop[i] = None

    visited[src] = True
    distance[src] = 0

    adj_list = {}

    for e in edges:
        if e[0] not in adj_list.keys():
            adj_list[e[0]] = [(e[1], e[2])]
        else:
            adj_list[e[0]].append((e[1], e[2]))

    if src in adj_list:
        for x in adj_list[src]:
            heapq.heappush(priority_queue, (x[1], src, x[0]))

    while (len(priority_queue) > 0):
        wt, start, n_end = heapq.heappop(priority_queue)
        if (not visited[n_end]):
            visited[n_end] = True
            distance[n_end] = wt
            next_hop[n_end] = start
            if n_end in adj_list.keys():
                for x in adj_list[n_end]:
                    if not visited[x[0]]:
                        heapq.heappush(
                            priority_queue, (wt + x[1], n_end, x[0]))

    if (not visited[end]):
        return (-1, None)
    else:
        x = end
        path = []
        while next_hop[x] is not None:
            path.append(x)
            x = next_hop[x]
        path.append(x)
        path.reverse()
        return (1, path)


@numba.njit
def dijkstra_numba_accelerated(nodes, edges, src, end, priority_queue, path):
    """Find shortest path using Numba-accelerated Dijkstra.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.
        src: Source node index.
        end: Destination node index.
        priority_queue: Numba typed list for heap operations.
        path: Numba typed list for output path.

    Returns:
        Tuple of (status, path) where status is 1 if path found, -1 otherwise.
    """
    n = len(nodes)
    visited = np.zeros(n, dtype=np.bool_)
    distance = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    next_hop = np.full(n, -1, dtype=np.int64)

    visited[src] = True
    distance[src] = 0

    for i in range(edges.shape[0]):
        if edges[i, 0] == src:
            heapq.heappush(priority_queue, (edges[i, 2], src, edges[i, 1]))

    while len(priority_queue) > 0:
        wt, start, n_end = heapq.heappop(priority_queue)
        if not visited[n_end]:
            visited[n_end] = True
            distance[n_end] = wt
            next_hop[n_end] = start

            for i in range(edges.shape[0]):
                if edges[i, 0] == n_end and not visited[edges[i, 1]]:
                    heapq.heappush(priority_queue, (wt + edges[i, 2], n_end, edges[i, 1]))

    if not visited[end]:
        return (-1, None)

    x = end
    while next_hop[x] != -1:
        path.append(x)
        x = next_hop[x]
    path.append(x)
    return (1, path)
