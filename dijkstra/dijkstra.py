import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import time
import heapq # numba has support for heapq

def command_line_fetcher():
    # function to fetch command line arguments
    parser = ArgumentParser(description="dijkstra")
    parser.add_argument(
        '-m', '--file', help="choose the filename for saving graph as npz/inputting file as npz")
    parser.add_argument("-n", '--size', type=int,
                        help="create a directed acyclic graph of n nodes")
    parser.add_argument("-p", '--prob', type=float,
                        help="probability of edge")
    parser.add_argument("-w1", '--wmin', type=int, help="min edge weight")
    parser.add_argument("-w2", '--wmax', type=int, help="max edge weight")
    parser.add_argument('--c', action='store_true', help="create graph")
    parser.add_argument('-s', "--source", type=int, help="source node", required=True)
    parser.add_argument('-e', "--end", type=int, help="end node", required=True)
    parser.add_argument('--f', action='store_true', help='use npz input benchmark')
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
    edges = graph.edges
    edge_list = []
    for e in edges:
        edge_list.append((e[0], e[1], graph[e[0]][e[1]]["weight"]))
    return (nodes, np.array(edge_list))

def numpy_to_graph(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for e in edges:
        G.add_edge(e[0], e[1], weight=e[2])
    return G

def dijkstra_nx(graph, src, end):
    try:
        path = nx.dijkstra_path(graph, src, end)
        return (1, path)
    except nx.NetworkXNoPath:
        return (-1, [])

def dijkstra_pythonic(nodes, edges, src, end):
    visited = {}
    distance = {}
    next_hop = {}
    priority_queue = []

    for i in nodes:
        visited[i] = False
        distance[i] = np.inf
        next_hop[i] = None

    visited[src] = True
    distance[src] = 0

    adj_list = {}

    for e in edges:
        if e[0] not in adj_list.keys():
            adj_list[e[0]] = [(e[1], e[2])]
        else:
            adj_list[e[0]].append((e[1], e[2]))


    for x in adj_list[src]:
        heapq.heappush(priority_queue, (x[1], src, x[0]))

    while(len(priority_queue) > 0):
        wt, start, n_end = heapq.heappop(priority_queue)
        if(not visited[n_end]):
            visited[n_end] = True
            distance[n_end] = wt
            next_hop[n_end] = start
            if n_end in adj_list.keys():
                for x in adj_list[n_end]:
                    if not visited[x[0]]:
                        heapq.heappush(priority_queue, (wt + x[1], n_end, x[0]))

    if(not visited[end]):
        return (-1, [])
    else:
        x = end
        path = []
        while next_hop[x] != None:
            path.append(x)
            x = next_hop[x]
        path.append(x)
        path.reverse()
        return (1, path)

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
    G = None
    nodes = None
    edgelist = None
    
    if create:
        G = create_nx_graph(n, p, w1, w2)
        pos = nx.shell_layout(G)
        nx.draw_networkx(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        nodes, edgelist = graph_to_numpy(G)
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

    # networkx dijkstra

    nx_path = dijkstra_nx(G, src, end)
    if nx_path[0] > 0: print(nx_path[1])

    # pythonic dijkstra

    py_path = dijkstra_pythonic(nodes, edgelist, src, end)
    if py_path[0] > 0: print(py_path[1])