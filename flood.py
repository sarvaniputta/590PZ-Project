import numpy as np
from collections import deque, defaultdict
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from functools import lru_cache
import time

# matplotlib colors we are going to use. Typically we use six.
colormap = ListedColormap(
    [
        "red",
        "green",
        "blue",
        "yellow",
        "darkorange",
        "purple",
        "deeppink",
        "cyan",
        "saddlebrown",
    ]
)


class Board:
    """Class that contains the state and helper functions to simulate a flood it board.
    Actual drawing does not happen here.
    """

    def __init__(
        self, nrows: int = 12, ncols: int = 12, ncolors: int = 6, start=(0, 0)
    ):
        self.nrows = nrows
        self.ncols = ncols
        self.ncolors = min(
            ncolors, colormap.N
        )  # limiting colors so that the colormap is distinct
        self.start = start[0] * nrows + start[1]

        # since grid is fixed through out the game, precompute neighbors for each cell as an optimization
        self._neighbors = {
            i: self.grid_neighbor(i, nrows, ncols) for i in range(nrows * ncols)
        }

        self.total_moves = 0
        self.soln = []

        # trial and error says 3 is good enough
        self.lookahead_depth = 3
        self.moves_till_now = 0
        self.grid_history = []

        # init a new (random game)
        self.new_game()

    def new_game(self):
        """
        Creates a random grid and solves it to obtain a threshold number of moves to prompt the user with.
        """
        self.grid = bytes(
            np.random.randint(
                0, self.ncolors, size=self.nrows * self.ncols, dtype="int8"
            )
        )
        # After a lot of experimenting, I settled on using bytes to store the grid as a 1d bytestring.
        # This was mainly because I wanted a hashable grid so that I can use it as a key or in sets
        # Using bytes was an memory optimization as I initially did a brute force search which resulted in a lot of
        # child grids I needed to store.
        # To further optimize for this representation, I computed the neighbors of each cell in linear index and store
        # it in a dict. This almost entirely eliminates converting linear index to 2d index

        # store the generated grid so that we can undo.
        self.grid_history.append(self.grid)
        print("Solving")
        self.soln = self.autosolve()
        self.total_moves = len(self.soln)
        print("Initialized game")

    @staticmethod
    def grid_neighbor(linear_index, m, n):
        """Static method that returns the linear indices of neighbors of a cell at linear_index
        in a mxn 2d array. Used to precompute the neighbor list as an optimization.
         """
        two_d_index = np.unravel_index(linear_index, dims=(m, n), order="C")
        neighbor_list = []
        for direction in np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]):
            nn = two_d_index + direction
            if 0 <= nn[0] < m and 0 <= nn[1] < n:
                neighbor_list.append(np.ravel_multi_index(nn, dims=(m, n), order="C"))
        return neighbor_list

    def flood_fill_delta(self, grid):
        """
        A flood fill algorithm that returns the cells that will be flood filled in the next move and
        also all the colors that border the flood fill cluster. Flood fill is started from the coordinate
        specified when initing the board.

        Rather than returning the flood filled grid (original design), I found that returning a
        list of all the cells that are contiguous to the start cell and have same color means that I need
        to do the BFS only once. Also by returning the colors on the frontier of the flood fill, I can restrict
        the moves (and the branching). Any color not on the frontier of the flood fill should not be investigated
        as a possible move as it does not increase the territory.

        Implementation wise, the code is a breadth first search (adapted from the flood fill pseudocode in wikipedia)
        with changes to return the cells rather than the changed grid.
        """
        start = self.start  # cell to start flood fill from.
        target = grid[start]  # color that will be filled
        changed = {start}
        border = set()
        queue = deque([start])
        while queue:
            n = queue.popleft()
            for nn in self._neighbors[n]:
                if nn in changed:
                    continue
                if grid[nn] == target:
                    # this cell is to be floodfilled
                    changed.add(nn)
                    queue.append(nn)
                else:
                    # this cell is a frontier cell
                    border.add(grid[nn])
        return changed, border

    @staticmethod
    def fill_grid(grid, changed, replacement):
        """Companion of the flood_fill_delta method that returns a new grid after flood filling with replacement."""
        new_grid = bytearray(grid)  # Convert to a mutable array
        for index in changed:
            new_grid[index] = replacement
        return bytes(new_grid)  # Back to immutable

    def flood_fill(self, move):
        """
        Execute move in current game. Results in new board state with board flood filled with color specified by move.
        """
        changed, _ = self.flood_fill_delta(self.grid)
        self.grid = self.fill_grid(self.grid, changed, move)
        self.grid_history.append(self.grid)

    @lru_cache(maxsize=200)
    def djikstras(self, grid):
        """
        Compute the shortest distance from the flood fill start cell for all cells in the grid.

        Distance between nodes is measured as zero if they share the same color or 1 if they are different.
        This ensures that the number for any given cell is the minimum number of moves required to bring it into the flood cluster.
        This means this is also a lower bound for the number of moves left in the game and can be used with A*

        The implementation is a straight up port from the pseudocode for djikstras algorithm in wikipedia. I tried using a
        lru_cache to get some of the benefits of a transposition table without storing one. Initial and final moves tend to transpose a lot,
        so hoping that we can utilize caching rather than computing this everytime.
        Experience says the code is faster using the lru_cache.
        """
        start = self.start
        vertex_set = list(i for i in range(self.nrows * self.ncols))
        removed_vertices = []
        dist = 1000 * np.ones(self.nrows * self.ncols, dtype="int")
        dist[start] = 0
        while vertex_set:
            dist_ = dist.copy()
            dist_[removed_vertices] = 1000
            u = np.argmin(dist_)
            vertex_set.remove(u)
            removed_vertices.append(u)
            for v in self._neighbors[u]:
                if v in vertex_set:
                    alt = dist[u] + (1 if grid[v] != grid[u] else 0)
                    if alt < dist[v]:
                        dist[v] = alt
        return dist

    def evaluation(self, grid):
        """Static evaluation of the grid used for searching best moves.

        This was by far the hardest part of the exercise. While distinct color distance,
        was easy to intuit, trying out various scalar evaluations I found that the solvers were extremely inefficient
        in the number of steps taken. Finally, I used a tuple of scores to prioritize board features.

        The features in decreasing order of importance are:
        The max number of moves required to bring any cell on the board into the flood.
        Number of cells at the max distance.
        If both the above being equal, I prefer the move that has the smallest cluster area or trying to maximize the
        perimeter to area of the flood fill cluster. Overall, this leads to a reasonable search time and close to optimal solutions.

        These features were tried based on discussion here.https://stackoverflow.com/questions/1430962/how-to-optimally-solve-the-flood-fill-puzzle/1431035
        """
        dist = self.djikstras(grid)
        max_dist = np.max(dist)
        number_at_max_dist = (dist == max_dist).sum()
        number_at_zero_dist = (dist == 0).sum()
        return (max_dist, number_at_max_dist, number_at_zero_dist)

    def best_heuristic_move(self, grid, depth=0):
        """Recursive depth first search to find move that minimizes the evaluation.
        Depth of search limited to lookahead_depth (3)
        """
        changed, border = self.flood_fill_delta(grid)
        best_val = (len(grid) + 1, 0, 0, -1)
        for move in border:
            new_grid = self.fill_grid(grid, changed, move)
            if self.is_filled(new_grid):
                return (-1, depth, len(new_grid)), move

            if depth >= self.lookahead_depth:
                return self.evaluation(new_grid), None
            else:
                val, _ = self.best_heuristic_move(new_grid, depth + 1)

            if val < best_val:
                best_val = val
                best_move = move
        return best_val, best_move

    @staticmethod
    def is_filled(grid):
        """Return True if game is complete (all grid cells are the same color)"""
        return all(val == grid[0] for val in grid)

    def autosolve(self):
        """
        Find solution of the game.

        Uses A* and returns the optimal solution for small board sizes. Else uses a lookahead move optimizer
        to compute a solution (not guaranteed to be optimal)
        """
        soln = []
        grid = self.grid
        if self.nrows * self.ncols <= 36:
            return self.a_star(grid)
        while not self.is_filled(self.grid):
            _, move = self.best_heuristic_move(self.grid)
            self.flood_fill(move)
            soln.append(move)
        self.grid = grid
        return soln

    def a_star_heuristic(self, grid):
        """Return a lower bound on the number of moves to finsh the game."""
        dist = self.djikstras(grid)
        return dist.max()

    def a_star(self, grid):
        """A* search with heuristic to search for an optimal solution.
        Slow with large grids. Implemented as a straight up port of the A* pseudo code
        from wikipedia.
        """
        open_set = {grid}
        came_from = {}
        gscore = dict()
        gscore[grid] = 0
        fscore = defaultdict(lambda: 1000)
        fscore[grid] = self.a_star_heuristic(grid)

        while open_set:
            current = min(fscore, key=lambda x: fscore[x] if x in open_set else 1000)
            if self.is_filled(current):
                return self.reconstruct_path(came_from, current)
            open_set.remove(current)
            changed, border = self.flood_fill_delta(current)
            for color in border:
                new_grid = self.fill_grid(current, changed, color)
                tentative_gscore = gscore[current] + 1
                if tentative_gscore < gscore.get(new_grid, 1000):
                    came_from[new_grid] = current
                    gscore[new_grid] = tentative_gscore
                    fscore[new_grid] = gscore[new_grid] + self.a_star_heuristic(
                        new_grid
                    )
                    if new_grid not in open_set:
                        open_set.add(new_grid)
        return []

    def reconstruct_path(self, came_from, current):
        """Compute the solution from the A* output."""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        soln = []
        for grid in total_path[1:]:
            soln.append(grid[0])
        return soln


class GameWindow:
    """Tkinter based GUI.
    Extended from the matplotlib embedding within tkinter code here.
    https://matplotlib.org/3.1.0/gallery/user_interfaces/embedding_in_tk_sgskip.html
    """

    def __init__(self, m, n, ncolors=6, start=(0, 0)):
        self.root = tkinter.Tk()
        self.root.wm_title("IS 590 Final-FloodIt")
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        self.board = Board(m, n, ncolors, start)
        self.cid = self.canvas.mpl_connect("button_press_event", self.on_click)
        self._display_grid = self.grid_to_array()
        self.label = tkinter.Label(
            master=self.root,
            text=f"{self.board.moves_till_now}/{self.board.total_moves} Moves",
            font="SegoeUI 22 bold",
        )
        self.label.pack()
        self.solution_btn = tkinter.Button(
            master=self.root,
            text="Show Solution",
            font="SegoeUI 22 bold",
            command=self.play_solution,
        )
        self.solution_btn.pack()
        self.refresh_display()
        tkinter.mainloop()

    def grid_to_array(self):
        """Convert the 1d bytes grid into 2d grid."""
        return (
            np.asarray(self.board.grid, dtype="|c")
            .astype("int8")
            .reshape(self.board.nrows, self.board.ncols)
        )

    def play_solution(self):
        """Visualize the solution for the current game."""
        self.board.grid = self.board.grid_history[0]
        self.board.moves_till_now = 0
        self.refresh_display()
        for move in self.board.soln:
            time.sleep(1)
            self.board.flood_fill(move)
            self.board.moves_till_now += 1
            self.refresh_display()

    def refresh_display(self):
        """Refresh display with updated board and moves done"""
        m, n = self.board.nrows, self.board.ncols
        self._display_grid = self.grid_to_array()
        self.ax.matshow(
            self._display_grid,
            cmap=colormap,
            vmin=0,
            vmax=colormap.N,
            aspect="equal",
            origin="upper",
            extent=(0, m, n, 0),
        )
        self.label.config(
            text=f"{self.board.moves_till_now}/{self.board.total_moves} Moves"
        )
        self.canvas.draw()

    def on_click(self, event):
        """Callback for making move."""
        row, col = int(event.ydata), int(event.xdata)
        if 0 <= row < self.board.nrows and 0 <= col < self.board.ncols:
            chosen_color = self._display_grid[(row, col)]
            self.board.flood_fill(chosen_color)
            self.board.moves_till_now += 1
            self.refresh_display()
            if self.board.moves_till_now >= self.board.total_moves:
                self.label.config(fg="red")


if __name__ == "__main__":
    GameWindow(12, 12)
