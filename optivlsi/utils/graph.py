"""
Utility functions for graph creation and conversion used across algorithms.
"""

import random
import networkx as nx
import numpy as np

def create_nx_graph(size: int, prob: float, wt_min: int, wt_max: int) -> nx.DiGraph:
    """
    Create a random directed acyclic graph.

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
        circuit = nx.DiGraph(
            [
                (u, v, {"weight": random.randint(wt_min, wt_max)})
                for (u, v) in graph.edges()
                if u < v
            ]
        )
        generated = nx.is_directed_acyclic_graph(circuit)
    return circuit


def graph_to_numpy(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a NetworkX graph to numpy arrays.

    Args:
        graph: NetworkX graph.

    Returns:
        Tuple of (nodes, edges) as numpy arrays.
    """
    nodes = np.array(graph.nodes, dtype=np.int64)
    edges = graph.edges
    edge_list = [(e[0], e[1], graph[e[0]][e[1]]["weight"]) for e in edges]
    return (nodes, np.array(edge_list, dtype=np.int64))


def numpy_to_graph(nodes: np.ndarray, edges: np.ndarray) -> nx.DiGraph:
    """
    Convert numpy arrays to a NetworkX graph.

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