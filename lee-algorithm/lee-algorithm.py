from argparse import ArgumentParser
import numpy as np
import numba

def command_line_fetcher():
    # function to fetch command line arguments
    parser = ArgumentParser(description="lee algorithm")
    
    parser.add_argument('-m', '--maze', required=True,
                        help="choose the maze filename for benchmarking")
    
    parser.add_argument('-sx', "--startx", type=int, default=0, required=True, help="start x index for algorithm")
    parser.add_argument('-sy', "--starty", type=int, default=0, required=True, help="start y index for algorithm")
    
    parser.add_argument('--benchmark', action='store_true', help="benchmark and compare it with networkx BFS")

    return parser.parse_args()

if __name__ == "__main__":
    args = command_line_fetcher()
    file = args.maze
    bench = args.benchmark
    sx = args.startx
    sy = args.starty

    maze = np.load(file + '.npz')['arr_0']
    n = maze.shape[0]
    
    print(maze)

