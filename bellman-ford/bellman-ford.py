import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import time
import numba
from numba.typed import List


def command_line_fetcher():
    # function to fetch command line arguments
    parser = ArgumentParser(description="bellman-ford")
    parser.add_argument(
        '-m', '--file', help="choose the filename for saving graph as npz/inputting file as npz")
    parser.add_argument("-n", '--size', type=int,
                        help="create a directed acyclic graph of n nodes")
    parser.add_argument("-p", '--prob', type=float,
                        help="probability of edge")
    parser.add_argument("-w1", '--wmin', type=int, help="min edge weight")
    parser.add_argument("-w2", '--wmax', type=int, help="max edge weight")
    parser.add_argument('--c', action='store_true', help="create graph")
    parser.add_argument('-s', "--source", type=int,
                        help="source node", required=True)
    parser.add_argument('-e', "--end", type=int,
                        help="end node", required=True)
    parser.add_argument('--f', action='store_true',
                        help='use npz input benchmark')
    parser.add_argument('--b', action='store_true',
                        help='print benchmark results')
    return parser.parse_args()


def create_nx_graph(size, prob, wt_min, wt_max):
    # reference: https://stackoverflow.com/a/13546785
    # erdos renyi random graph generates a directed acyclic graph
    # every vlsi circuit can be represented as a directed acyclic graph
    generated = False
    while not generated:
        graph = nx.gnp_random_graph(size, prob, directed=True)
        circuit = nx.DiGraph([(u, v,
                               {'weight': random.randint(wt_min, wt_max)})
                              for (u, v) in graph.edges() if u < v])
        generated = nx.is_directed_acyclic_graph(circuit)
    return circuit


def graph_to_numpy(graph):
    nodes = np.array(graph.nodes, dtype=np.int64)
    edges = graph.edges
    edge_list = []
    for e in edges:
        edge_list.append((e[0], e[1], graph[e[0]][e[1]]["weight"]))
    return (nodes, np.array(edge_list, dtype=np.int64))


def numpy_to_graph(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for e in edges:
        G.add_edge(e[0], e[1], weight=e[2])
    return G


def bellman_ford_nx(nodes, edges, src, end):
    # including time of creation of graph in networkx from numpy arrays for fairness
    graph = numpy_to_graph(nodes, edges)

    try:
        path = nx.bellman_ford_path(graph, src, end)
        return (1, path)
    except nx.NetworkXNoPath:
        return (-1, None)


def bellman_ford_pythonic(nodes, edges, src, end):
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


if __name__ == "__main__":
    args = command_line_fetcher()
    n = args.size
    f = args.file
    p = args.prob
    w1 = args.wmin
    w2 = args.wmax
    src = args.source
    end = args.end
    create = args.c
    use_file = args.f
    bench = args.b


    G = None
    nodes = None
    edgelist = None

    if create:
        G = create_nx_graph(n, p, w1, w2)
        nodes, edgelist = graph_to_numpy(G)
        G = numpy_to_graph(nodes, edgelist)
        pos = nx.shell_layout(G)
        nx.draw_networkx(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        # nodes, edgelist = graph_to_numpy(G)
        np.savez(f'{f}.npz', nodes=nodes, edgelist=edgelist)
        plt.savefig(f"{f}.pdf")

    elif use_file:
        graph_data = np.load(f'{f}.npz')
        nodes = graph_data['nodes']
        edgelist = graph_data['edgelist']
        G = numpy_to_graph(nodes, edgelist)
        pos = nx.shell_layout(G)
        nx.draw_networkx(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        nodes, edgelist = graph_to_numpy(G)
        plt.savefig(f"{f}.pdf")
        n = len(nodes)
    
    if end >= n: end = n-1

    # networkx bellman_ford
    t1 = time.perf_counter()
    nx_path = bellman_ford_nx(nodes, edgelist, src, end)
    t1 = time.perf_counter() - t1

    # pythonic bellman_ford
    t2 = time.perf_counter()
    py_path = bellman_ford_pythonic(nodes, edgelist, src, end)
    t2 = time.perf_counter() - t2

    path = List()
    path.append(1)
    path.pop()

    nb_path = bellman_ford_numba_accelerated(
        nodes, edgelist, src, end, path)

    # accelerated bellman_ford

    path = List()
    path.append(1)
    path.pop()

    t3 = time.perf_counter()
    nb_path = bellman_ford_numba_accelerated(
        nodes, edgelist, src, end, path)
    t3 = time.perf_counter() - t3

    if (bench):
        print(t1)
        print(t2)
        print(t3)
    else:
        if nx_path[0] > 0:
            print(nx_path[1])
        if py_path[0] > 0:
            print(py_path[1])
        if nb_path[0] > 0:
            print(nb_path[1])

    # visualize the numba accelerated path on graph
    G = numpy_to_graph(nodes, edgelist)
    color_map = []
    for node in nodes:
        if node in py_path[1]:
            color_map.append('red')
        else:
            color_map.append('#1f78b4')
    pos = nx.shell_layout(G)
    nx.draw_networkx(G, pos, node_color=color_map)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nodes, edgelist = graph_to_numpy(G)
    path = "--:>".join([str(k) for k in py_path[1]])
    plt.suptitle("Bellman Ford Algorithm")
    plt.title(f"Shortest path: {path}")
    plt.savefig(f"{f}_shortest_path.pdf")
