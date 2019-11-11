import numpy as np
from collections import deque
import numpy.random
import networkx

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

def solve(grid, start=(0, 0)):
    queue= deque()
    if np.all(grid==grid[0,0]):
        return []
    queue.append((grid, []))
    while queue:
        grid, moves = queue.popleft()
        for replacement in range(6):
            if replacement == grid[start]:
                continue
            new_grid = flood_fill(grid, replacement, start)
            new_moves = moves + [ replacement ]
            if np.all(new_grid==new_grid[0, 0]):
                return new_moves
            queue.append((new_grid, new_moves))





if __name__ == '__main__':
    m, n = 5, 5
    grid = numpy.random.randint(0, 6, size=(m, n), dtype='int8')
    # flood_fill(grid, 1)
    print(grid)
    moves = solve(grid)
    for move in moves:
        grid = flood_fill(grid)
        print(grid)