"""Example: Lee maze routing algorithm."""

import numpy as np
from optivlsi.lee import lee

# Create a 5x5 maze with some obstacles
maze = np.zeros((5, 5), dtype=np.int64)
# Add some obstacles
maze[1, 1] = 1
maze[1, 2] = 1
maze[2, 2] = 1
maze[3, 2] = 1

distance = lee(maze, 0, 0, 4, 4)
print(f"Shortest path distance: {distance}")