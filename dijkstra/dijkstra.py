import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def command_line_fetcher():
    # function to fetch command line arguments
    parser = ArgumentParser(description="dijkstra")
    parser.add_argument(
        '-f', '--file', help="choose the filename for saving graph as npz")
    parser.add_argument("-n", '--size', type=int,
                        help="create a directed acyclic graph of n nodes")
    parser.add_argument("-p", '--prob', type=float,
                        help="probability of edge")
    parser.add_argument("-w1", '--wmin', type=int, help="min edge weight")
    parser.add_argument("-w2", '--wmax', type=int, help="max edge weight")
    parser.add_argument('--c', action='store_true', help="create graph")
    return parser.parse_args()


def create_nx_graph(size, prob, wt_min, wt_max):
    # reference: https://stackoverflow.com/a/13546785
    # erdos renyi random graph generates a directed acyclic graph
    # every vlsi circuit can be represented as a directed acyclic graph
    generated = False
    while not generated:
        graph = nx.gnp_random_graph(size, prob, directed=True)
        circuit = nx.DiGraph([(u, v, {'weight': random.randint(wt_min, wt_max)}) for (
            u, v) in graph.edges() if u < v])
        generated = nx.is_directed_acyclic_graph(circuit)
    return circuit


def graph_to_numpy(graph):
    nodes = np.array(graph.nodes)
    edge_list = np.array(graph.edges)
    return (nodes, edge_list)


if __name__ == "__main__":
    args = command_line_fetcher()
    n = args.size
    f = args.file
    p = args.prob
    w1 = args.wmin
    w2 = args.wmax
    create = args.c
    if create:
        G = create_nx_graph(n, p, w1, w2)
        pos = nx.shell_layout(G)
        nx.draw_networkx(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        n, m = graph_to_numpy(G)
        np.savez(f'{f}.npz', nodes=n, edgelist=m)
        plt.savefig(f"{f}.pdf")
