from argparse import ArgumentParser
import numpy as np
from numba import njit
from numba.typed import List
import time
import networkx as nx

def command_line_fetcher():
    # function to fetch command line arguments
    parser = ArgumentParser(description="lee algorithm")
    
    parser.add_argument('-m', '--maze', required=True,
                        help="choose the maze filename for benchmarking")
    
    parser.add_argument('-sx', "--startx", type=int, default=0, required=True, help="start x index for algorithm")
    parser.add_argument('-sy', "--starty", type=int, default=0, required=True, help="start y index for algorithm")

    parser.add_argument('-ex', "--endx", type=int, default=0, required=True, help="destination x index for algorithm")
    parser.add_argument('-ey', "--endy", type=int, default=0, required=True, help="destination y index for algorithm")
    
    parser.add_argument('--b', action='store_true', help="benchmark and compare it with networkx BFS")

    return parser.parse_args()



def lee_algorithm_networkx(maze, sx, sy, ex, ey):

    graph = nx.Graph()

    n = maze.shape[0]
    d = {}

    for i in range(n):
        for j in range(n):
            node = i * n + j
            graph.add_node(node)
            d[node] = (i, j)

    for i in range(n):
        for j in range(n):
            if maze[i][j] == 0:
                node1 = i * n + j
                if i > 0 and maze[i-1][j] == 0:
                    node2 = (i - 1) * n + j
                    graph.add_edge(node1, node2)
                if i < n - 1 and maze[i+1][j] == 0:
                    node2 = (i + 1) * n + j
                    graph.add_edge(node1, node2)
                if j > 0 and maze[i][j-1] == 0:
                    node2 = i * n + (j - 1)
                    graph.add_edge(node1, node2)
                if j < n - 1 and maze[i][j+1] == 0:
                    node2 = i * n + (j + 1)
                    graph.add_edge(node1, node2)


    queue = [(sx, sy)]

    visited = np.zeros_like(maze)
    visited[sx, sy] = 1

    while len(queue) > 0:

        current_node = queue.pop(0)
        node_index = current_node[0]*n + current_node[1]


        if current_node[0] == ex and current_node[1] == ey:
            return visited[ex, ey]

        for neighbor in graph.neighbors(node_index):
            if visited[d[neighbor][0], d[neighbor][1]] == 0:
                queue.append((d[neighbor][0], d[neighbor][1]))
                visited[d[neighbor][0], d[neighbor][1]] = visited[d[node_index][0], d[node_index][1]] + 1

    return -1


def lee_algorithm(maze, sx, sy, ex, ey):
    n = maze.shape[0]
    visited = np.zeros_like(maze)
    visited[sx, sy] = 1
    queue = [(sx, sy)]

    while(len(queue) > 0):
        curr_x, curr_y = queue.pop(0)
        
        if(curr_x == ex and curr_y == ey):
            return visited[curr_x, curr_y]
        
         # move up
        next_x = curr_x
        next_y = curr_y + 1
        if((next_x < n and next_x >= 0) and (next_y < n and next_y >= 0)):
            if(visited[next_x, next_y] == 0 and maze[next_x, next_y] == 0):
                visited[next_x, next_y] = visited[curr_x, curr_y] + 1
                queue.append((next_x, next_y))

        # move down
        next_x = curr_x
        next_y = curr_y + 1
        if((next_x < n and next_x >= 0) and (next_y < n and next_y >= 0)):
            if(visited[next_x, next_y] == 0 and maze[next_x, next_y] == 0):
                visited[next_x, next_y] = visited[curr_x, curr_y] + 1
                queue.append((next_x, next_y))
        
         # move right
        next_x = curr_x + 1
        next_y = curr_y
        if((next_x < n and next_x >= 0) and (next_y < n and next_y >= 0)):
            if(visited[next_x, next_y] == 0 and maze[next_x, next_y] == 0):
                visited[next_x, next_y] = visited[curr_x, curr_y] + 1
                queue.append((next_x, next_y))
        
        # move left
        next_x = curr_x - 1
        next_y = curr_y
        if((next_x < n and next_x >= 0) and (next_y < n and next_y >= 0)):
            if(visited[next_x, next_y] == 0 and maze[next_x, next_y] == 0):
                visited[next_x, next_y] = visited[curr_x, curr_y] + 1
                queue.append((next_x, next_y))

    return -1

@njit
def accelerated_lee_algorithm(maze, sx, sy, ex, ey, queue_x, queue_y):
    n = maze.shape[0]
    visited = np.zeros_like(maze)
    visited[sx, sy] = 1
    queue_x.append(sx)
    queue_y.append(sy)

    while(len(queue_x) > 0):
        curr_x = queue_x.pop(0)
        curr_y = queue_y.pop(0)
        
        if(curr_x == ex and curr_y == ey):
            return visited[curr_x, curr_y]
        
         # move up
        next_x = curr_x
        next_y = curr_y + 1
        if((next_x < n and next_x >= 0) and (next_y < n and next_y >= 0)):
            if(visited[next_x, next_y] == 0 and maze[next_x, next_y] == 0):
                visited[next_x, next_y] = visited[curr_x, curr_y] + 1
                queue_x.append(next_x)
                queue_y.append(next_y)

        # move down
        next_x = curr_x
        next_y = curr_y + 1
        if((next_x < n and next_x >= 0) and (next_y < n and next_y >= 0)):
            if(visited[next_x, next_y] == 0 and maze[next_x, next_y] == 0):
                visited[next_x, next_y] = visited[curr_x, curr_y] + 1
                queue_x.append(next_x)
                queue_y.append(next_y)
        
         # move right
        next_x = curr_x + 1
        next_y = curr_y
        if((next_x < n and next_x >= 0) and (next_y < n and next_y >= 0)):
            if(visited[next_x, next_y] == 0 and maze[next_x, next_y] == 0):
                visited[next_x, next_y] = visited[curr_x, curr_y] + 1
                queue_x.append(next_x)
                queue_y.append(next_y)
        
        # move left
        next_x = curr_x - 1
        next_y = curr_y
        if((next_x < n and next_x >= 0) and (next_y < n and next_y >= 0)):
            if(visited[next_x, next_y] == 0 and maze[next_x, next_y] == 0):
                visited[next_x, next_y] = visited[curr_x, curr_y] + 1
                queue_x.append(next_x)
                queue_y.append(next_y)

    return -1


if __name__ == "__main__":
    args = command_line_fetcher()
    
    file = args.maze
    bench = args.b
    
    sx = args.startx
    sy = args.starty
    ex = args.endx
    ey = args.endy

    maze = np.load(file + '.npz')['arr_0']
    n = maze.shape[0]

    ex = n-1 if ex >= n else ex
    ey = n-1 if ey >= n else ey
    
    # dummy call if benchmarking
    queue_x = List()
    queue_x.append(1)
    queue_x.pop()

    queue_y = List()
    queue_y.append(1)
    queue_y.pop()

    distance = accelerated_lee_algorithm(maze, sx, sy, ex, ey, queue_x, queue_y)

    if(bench):
        # benchmark for accelerated code
        queue_x = List()
        queue_x.append(1)
        queue_x.pop()

        queue_y = List()
        queue_y.append(1)
        queue_y.pop()
        
        x = time.perf_counter()
        distance = accelerated_lee_algorithm(maze, sx, sx, ex, ex, queue_x, queue_y)
        x = time.perf_counter() - x
        
        # benchmark for non accelerated code

        y = time.perf_counter()
        distance = lee_algorithm(maze, sx, sx, ex, ex)
        y = time.perf_counter() - y

        # benchmark for networkx code

        z = time.perf_counter()
        distance = lee_algorithm_networkx(maze, sx, sx, ex, ex)
        z = time.perf_counter() - z
        
        print(x)
        print(y)
        print(z)
    else:
        print(distance)
