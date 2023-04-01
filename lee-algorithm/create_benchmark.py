from argparse import ArgumentParser
import numpy as np
import random


def command_line_fetcher():
    # function to fetch command line arguments
    parser = ArgumentParser(description="lee algorithm benchmark")

    parser.add_argument('-n', '--size',
                        type=int, default=100, required=True,
                        help="maze size")

    parser.add_argument('-p', '--prob', type=float, default=0.1,
                        required=True, help="prob of obstacle")

    parser.add_argument('-f', '--file', required=True, help="save maze")

    return parser.parse_args()


def create_maze(n, p):
    maze = np.zeros(shape=(n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if random.random() < p:
                maze[i][j] = 1
    return maze


if __name__ == "__main__":
    args = command_line_fetcher()
    n = args.size
    p = args.prob
    file = args.file
    maze = create_maze(n, p)
    np.savez(file + '.npz', maze)
