import time
import numba
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from networkx.drawing.nx_agraph import graphviz_layout
import pygraphviz
from numba.typed import List

# if facing issues for pygraphvis 
# https://stackoverflow.com/questions/40266604/pip-install-pygraphviz-fails-failed-building-wheel-for-pygraphviz

def command_line_fetcher():
    # function to fetch command line arguments
    parser = ArgumentParser(description="dijkstra")
    parser.add_argument(
        '-m', '--file', help="choose the filename for saving graph as npz/inputting file as npz")
    parser.add_argument(
        '-t', '--mst', help="store the mst finally here")
    parser.add_argument("-n", '--size', type=int,
                        help="create a directed acyclic graph of n nodes")
    parser.add_argument("-p", '--prob', type=float,
                        help="probability of edge")
    parser.add_argument("-w1", '--wmin', type=int, help="min edge weight")
    parser.add_argument("-w2", '--wmax', type=int, help="max edge weight")
    parser.add_argument('--c', action='store_true', help="create graph")
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
        circuit = nx.DiGraph([(u, v, {'weight': random.randint(wt_min, wt_max)}) for
                              (u, v) in graph.edges() if u < v])
        generated = nx.is_directed_acyclic_graph(circuit)
    return circuit


def graph_to_numpy(graph):
    nodes = np.array(graph.nodes, dtype=np.int64)
    nodes.sort()
    edges = graph.edges
    edge_list = []
    for e in edges:
        edge_list.append((e[0], e[1], graph[e[0]][e[1]]["weight"]))
    return (nodes, np.array(edge_list, dtype=np.int64))


def numpy_to_graph(nodes, edges):
    G = nx.Graph()  # modified to non directed from DiGraph() as Kruskal works only for Non Directed Graphs
    G.add_nodes_from(nodes)
    for e in edges:
        G.add_edge(e[0], e[1], weight=e[2])
    return G


def kruskal_pythonic(nodes, edges):
    # create a dsu (disjoint set union)
    # it is capable of union and find operation

    parent = np.array(range(len(nodes)))
    rank = [0]*len(nodes)

    def find(i):
        if not (parent[i] == i):
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        x, y = find(i), find(j)
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[x] = y
            rank[y] += 1

    mst = []
    wts = edges[:, 2]
    sorted_indices = np.argsort(wts)
    sorted_edges = edges[sorted_indices]

    for e in sorted_edges:
        src, end, wt = e
        if find(src) != find(end):
            union(src, end)
            mst.append(e)

    return (nodes, np.array(mst))


def kruskal_networkx(nodes, edges):
    graph = numpy_to_graph(nodes, edges)
    mst = nx.minimum_spanning_tree(graph)
    n, e = graph_to_numpy(mst)
    return (n, e)

@numba.njit
def find(i, parent):
    if not (parent[i] == i):
        parent[i] = find(parent[i], parent)
    return parent[i]

@numba.njit
def union(i, j, rank, parent):
    x, y = find(i, parent), find(j, parent)
    if rank[x] < rank[y]:
        parent[x] = y
    elif rank[x] > rank[y]:
        parent[y] = x
    else:
        parent[x] = y
        rank[y] += 1
    
@numba.njit
def kruskal_numba_accelerated(nodes, edges, mst):
    # create a dsu (disjoint set union)
    # it is capable of union and find operation
    
    parent = np.copy(nodes)
    rank = np.zeros_like(nodes)

    wts = edges[:, 2]
    sorted_indices = np.argsort(wts)
    sorted_edges = edges[sorted_indices]

    for e in sorted_edges:
        src, end, wt = e
        if find(src, parent) != find(end, parent):
            union(src, end, rank, parent)
            mst.append((e[0], e[1], e[2]))

    return (nodes, mst)


def draw_mst(mst_n, mst_edges, t):
    plt.figure()
    G = numpy_to_graph(mst_n, mst_edges)
    pos = graphviz_layout(G, prog="dot")
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig(f"{t}.pdf")
    np.savez(f'{t}.npz', nodes=mst_n, edgelist=mst_edges)



if __name__ == "__main__":
    args = command_line_fetcher()
    n = args.size
    f = args.file
    t = args.mst
    p = args.prob
    w1 = args.wmin
    w2 = args.wmax
    create = args.c
    use_file = args.f
    bench = args.b

    G = None
    nodes = None
    edgelist = None

    if create:
        
        plt.figure()
        G = create_nx_graph(n, p, w1, w2)
        nodes, edgelist = graph_to_numpy(G)
        G = numpy_to_graph(nodes, edgelist)
        pos = nx.shell_layout(G)
        nx.draw_networkx(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        np.savez(f'{f}.npz', nodes=nodes, edgelist=edgelist)
        plt.savefig(f"{f}.pdf")
    
    elif use_file:
        
        plt.figure()
        graph_data = np.load(f'{f}.npz')
        nodes = graph_data['nodes']
        edgelist = graph_data['edgelist']
        G = numpy_to_graph(nodes, edgelist)
        pos = nx.shell_layout(G)
        nx.draw_networkx(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.savefig(f"{f}.pdf")
        n = len(nodes)
    
    t1 = time.perf_counter()
    mst_n, mst_edges = kruskal_pythonic(nodes, edgelist)
    t1 = time.perf_counter() - t1 

    t2 = time.perf_counter()
    mst_n, mst_edges = kruskal_networkx(nodes, edgelist)
    t2 = time.perf_counter() - t2 

    
    # dummy calls for numba
    parent = np.array(range(len(nodes)))
    rank = np.zeros_like(nodes)

    find(0, parent)
    union(0, 1, rank, parent)

    mst = List()
    mst.append((1, 1, 1))
    mst.pop()

    kruskal_numba_accelerated(nodes, edgelist, mst)

    # actual numba call
    
    mst = List()
    mst.append((1, 1, 1))
    mst.pop()

    t3 = time.perf_counter()
    mst_n, mst_e = kruskal_numba_accelerated(nodes, edgelist, mst)
    t3 = time.perf_counter() - t3
    
    mst_edges = np.array(mst_e)

    draw_mst(mst_n, mst_edges, t)
    
    if(bench):
        print(t1)
        print(t2)
        print(t3)