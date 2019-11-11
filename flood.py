import numpy as np
from collections import deque
import numpy.random


def neighbors(node, m, n):
    r, c = node
    node_east = r, c+1
    node_west = r, c-1
    node_south = r+1, c
    node_north = r-1, c
    neighbors = []
    for n_ in [node_east, node_west, node_south, node_north]:
        if 0 <= n_[0] < m and 0 <= n_[1] < n:
            neighbors.append(n_)
    return neighbors

def flood_fill(grid: np.array, replacement: int, start=(0, 0)):
    """
    Flood fills the grid starting top left
    """
    grid = grid.copy()
    target = grid[start]
    if target == replacement:
        return

    grid[start]=replacement
    #initialize a queue
    queue = deque()
    queue.append(start)
    while queue:
        n = queue.popleft()
        for neighbor in neighbors(n, grid.shape[0], grid.shape[1]):
            if grid[neighbor] == target:
                grid[neighbor] = replacement
                queue.append(neighbor)
    return grid


if __name__ == '__main__':
    m, n = 12, 12
    grid = numpy.random.randint(0, 6, size=(m, n))
    flood_fill(grid, 1)