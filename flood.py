import numpy as np
import numpy.random
from collections import deque, Counter, defaultdict
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from functools import lru_cache
import time

colormap = ListedColormap(["red", "green", "blue", "yellow", "darkorange", "purple"])


class Board:
    def __init__(self, m, n, ncolors, start=(0, 0)):
        self.m = m
        self.n = n
        self.ncolors = ncolors
        self.start = start[0] * m + start[1]
        self._neighbors = {i: self.grid_neighbor(i, m, n) for i in range(m * n)}
        self.total_moves = 0
        self.soln = []
        self.lookahead_depth = 3
        self.moves_till_now = 0
        self.grid_history = []
        self.new_game()

    def new_game(self):
        self.grid = bytes(
            np.random.randint(0, self.ncolors, size=self.m * self.n, dtype="int8")
        )
        print("Solving")
        self.soln = self.autosolve()
        self.total_moves = len(self.soln)
        print("Initialized game")

    @staticmethod
    def grid_neighbor(linear_index, m, n):
        two_d_index = np.unravel_index(linear_index, dims=(m, n), order="C")
        neighbor_list = []
        for direction in np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]):
            nn = two_d_index + direction
            if 0 <= nn[0] < m and 0 <= nn[1] < n:
                neighbor_list.append(np.ravel_multi_index(nn, dims=(m, n), order="C"))
        return neighbor_list

    def flood_fill_delta(self, grid):
        start = self.start
        target = grid[start]
        changed = {start}
        border = set()
        queue = deque([start])
        while queue:
            n = queue.popleft()
            for nn in self._neighbors[n]:
                if nn in changed:
                    continue
                if grid[nn] == target:
                    changed.add(nn)
                    queue.append(nn)
                else:
                    border.add(grid[nn])
        return changed, border

    @staticmethod
    def fill_grid(grid, changed, replacement):
        new_grid = bytearray(grid)
        for index in changed:
            new_grid[index] = replacement
        return bytes(new_grid)

    def flood_fill(self, move):
        changed, _ = self.flood_fill_delta(self.grid)
        self.grid_history.append(self.grid)
        self.grid = self.fill_grid(self.grid, changed, move)

    @lru_cache(maxsize=200)
    def djikstras(self, grid):
        start = self.start
        vertex_set = list(i for i in range(self.m * self.n))
        removed_vertices = []
        dist = 1000 * np.ones(self.m * self.n, dtype="int")
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
        dist = self.djikstras(grid)
        max_dist = np.max(dist)
        number_at_max_dist = (dist == max_dist).sum()
        number_at_zero_dist = (dist == 0).sum()
        return (max_dist, number_at_max_dist, number_at_zero_dist)

    def best_heuristic_move(self, grid, depth=0):
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
        return all(val == grid[0] for val in grid)

    def autosolve(self):
        soln = []
        grid = self.grid
        while not self.is_filled(self.grid):
            _, move = self.best_heuristic_move(self.grid)
            self.flood_fill(move)
            soln.append(move)
        self.grid = grid
        return soln



class GameWindow:
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
        return (
            np.asarray(self.board.grid, dtype="|c")
            .astype("int8")
            .reshape(self.board.m, self.board.n)
        )

    def play_solution(self):
        self.board.grid = self.board.grid_history[0]
        self.board.moves_till_now = 0
        self.refresh_display()
        for move in self.board.soln:
            time.sleep(1)
            self.board.flood_fill(move)
            self.board.moves_till_now += 1
            self.refresh_display()

    def refresh_display(self):
        m, n = self.board.m, self.board.n
        self._display_grid = self.grid_to_array()
        self.ax.matshow(
            self._display_grid,
            cmap=colormap,
            aspect="equal",
            vmin=0,
            vmax=self.board.ncolors,
            origin="upper",
            extent=(0, m, n, 0),
        )
        self.label.config(
            text=f"{self.board.moves_till_now}/{self.board.total_moves} Moves"
        )
        self.canvas.draw()

    def on_click(self, event):
        row, col = int(event.ydata), int(event.xdata)
        if 0 <= row < self.board.m and 0 <= col < self.board.n:
            chosen_color = self._display_grid[(row, col)]
            self.board.flood_fill(chosen_color)
            self.board.moves_till_now += 1
            self.refresh_display()
            if self.board.moves_till_now >= self.board.total_moves:
                self.label.config(fg="red")


if __name__ == "__main__":
    GameWindow(12, 12)
