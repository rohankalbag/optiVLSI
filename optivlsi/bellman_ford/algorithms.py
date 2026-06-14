"""
Bellman-Ford shortest path algorithm implementations.

Provides three variants: pure Python, NetworkX-based, and Numba-accelerated.
"""

import networkx as nx
import numpy as np
import numba


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
    import random
    generated = False
    while not generated:
        graph = nx.gnp_random_graph(size, prob, directed=True)
        circuit = nx.DiGraph([(u, v,
                                {'weight': random.randint(wt_min, wt_max)})
                              for (u, v) in graph.edges() if u < v])
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


def bellman_ford_nx(nodes, edges, src, end):
    """Find shortest path using NetworkX Bellman-Ford.

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
        path = nx.bellman_ford_path(graph, src, end)
        return (1, path)
    except nx.NetworkXNoPath:
        return (-1, None)


def bellman_ford_pythonic(nodes, edges, src, end):
    """Find shortest path using pure Python Bellman-Ford.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.
        src: Source node index.
        end: Destination node index.

    Returns:
        Tuple of (status, path) where status is 1 if path found, -1 otherwise.
    """
    dist = [float("Inf")] * len(nodes)
    dist[src] = 0
    prev_node = np.zeros_like(nodes)
    prev_node.fill(-1)
    prev_node[src] = 0
    path = [end]

    for i in range(len(nodes)-1):
        for e in edges:
            if dist[e[0]] != float("Inf") and dist[e[0]] + e[2] < dist[e[1]]:
                dist[e[1]] = dist[e[0]] + e[2]
                prev_node[e[1]] = e[0]

    if (prev_node[end] == -1):
        return (-1, None)
    else:
        curr_node = end
        while (curr_node != src):
            curr_node = prev_node[curr_node]
            path.append(curr_node)
        return (1, path[::-1])


@numba.njit
def bellman_ford_numba_accelerated(nodes, edges, src, end, path):
    """Find shortest path using Numba-accelerated Bellman-Ford.

    Args:
        nodes: Array of node indices.
        edges: Array of edges as (src, dst, weight) tuples.
        src: Source node index.
        end: Destination node index.
        path: Numba typed list for output path.

    Returns:
        Tuple of (status, path) where status is 1 if path found, -1 otherwise.
    """
    dist = np.ones_like(nodes)
    dist.fill(np.iinfo(np.int64).max)
    dist[src] = 0
    prev_node = np.zeros_like(nodes)
    prev_node.fill(-1)
    prev_node[src] = 0
    path = [end]

    for i in range(len(nodes)-1):
        for e in edges:
            if dist[e[0]] != np.iinfo(np.int64).max and dist[e[0]] + e[2] < dist[e[1]]:
                dist[e[1]] = dist[e[0]] + e[2]
                prev_node[e[1]] = e[0]

    if (prev_node[end] == -1):
        return (-1, None)
    else:
        curr_node = end
        while (curr_node != src):
            curr_node = prev_node[curr_node]
            path.append(curr_node)
        return (1, path[::-1])