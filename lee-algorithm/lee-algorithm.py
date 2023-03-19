from argparse import ArgumentParser
import numpy as np
from numba import njit
from numba.typed import List
import time

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

@njit
def lee_algorithm(maze, sx, sy, ex, ey, queue_x, queue_y):
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
    
    # dummy call if benchmarking
    queue_x = List()
    queue_x.append(1)
    queue_x.pop()

    queue_y = List()
    queue_y.append(1)
    queue_y.pop()

    distance = lee_algorithm(maze, 0, 0, 9, 9, queue_x, queue_y)

    if(bench):
        queue_x = List()
        queue_x.append(1)
        queue_x.pop()

        queue_y = List()
        queue_y.append(1)
        queue_y.pop()
        
        x = time.perf_counter()
        distance =lee_algorithm(maze, 0, 0, 9, 9, queue_x, queue_y)
        x = time.perf_counter() - x
        print(distance)
        print(x)
    else:
        print(distance)
