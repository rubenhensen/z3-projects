START_CELL_OBS = 0
TARGET_CELL_OBS = 1
LAVA_CELL_OBS = 2
ICY_CELL_OBS = 3
STICKY_CELL_OBS = 4
ACTIONS = ["N", "S", "W", "E"]
DIR_TO_CARET = { "W" : 4, "E" : 5, "N": 6, "S": 7 }

import csv
import math
from datetime import datetime as dt
import argparse

from z3 import sat, And, Implies, Or, Int, Bool, Solver, If, Sum, print_matrix, Not

COLOR_NAMES = ['green', 'black', 'crimson', 'white', 'white', 'violet', 'lightskyblue', 'orange', 'blue', 'yellow',  'deepskyblue', 'teal', 'navajowhite', 'darkgoldenrod', 'lightsalmon']


class Cell:
    """
    Individual cells in a grid, a basic container for (x,y) tuples with very limited operations.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"<{self.x}, {self.y}>"

    def __hash__(self):
        return hash(self.x) ^ hash(self.y)

    @classmethod
    def distance(cls, lhs, rhs):
        return abs(lhs.x - rhs.x) + abs(lhs.y - rhs.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Grid:
    def __init__(self, grid):
        """
        Internal initialization method for grid class. Requires a 2d matrix with the grid. Do not use.

        :param grid: A 2d matrix of Cells.
        """
        self._grid = grid
        self._colors = set()
        for row in grid:
            for entry in row:
                self._colors.add(entry)
        self._valid = True
        if 0 not in self._colors:
            self._valid = False
        if 1 not in self._colors:
            self._valid = False
        self._cells = []
        for x in range(0, self.xdim):
            for y in range(0, self.ydim):
                self._cells.append(Cell(x, y))
    
    def __iter__(self):
        for row in self._grid:
            for item in row:
                yield item


    @classmethod
    def from_csv(cls, path):
        """
        Creates a grid from a CSV file.

        :param path:
        :return:
        """
        grid = []
        with open(path) as csvfile:
            tablereader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in tablereader:
                grid.append([int(entry) for entry in row])  
        grid2 = [list(x) for x in zip(*grid)]
        return cls(grid2)

    @classmethod
    def random(cls, size_x : int, size_y : int, diff_terrains : int,
               prob_start : float, prob_target : float, prob_lava : float, prob_ice : float, prob_sticky : float):
        """
        Creates a random grid size. Requires the SciPy package to be installed.

        :param size_x: The size of the grid in x-dimension
        :param size_y: The size of the grid in y-dimension
        :param diff_terrains: The number of different terrains (including start, target, lava, and ice)
        :param prob_start: The probability that a given cell is a potential start field.
        :param prob_target: The probability that a given cell is a target field.
        :param prob_lava: The probability that a given cell is a lava field.
        :param prob_ice: The probability that a given cell is an ice field.
        :param prob_sticky: The probability that a given cell is a sticky field.
        :return: A grid.
        """
        try:
            from scipy.stats import rv_discrete
        except ImportError as e:
            raise RuntimeError("Cannot generate random grids, requires the SciPy package.")
        assert diff_terrains >= 5
        assert prob_start > 0
        assert prob_target > 0
        remaining_prob = 1.0 - (prob_start + prob_target + prob_lava + prob_ice + prob_sticky)
        assert remaining_prob >= 0
        def convert_1d_to_2d(l, cols):
            return [l[j:j + cols] for j in range(0, len(l), cols)]
        values = list(range(0, diff_terrains))
        probabilities = [prob_start, prob_target, prob_lava, prob_ice, prob_sticky] + ([remaining_prob/(diff_terrains-5)] * (diff_terrains - 5))
        distrib = rv_discrete(values=(values, probabilities))
        while True:
            """
            Rejection sampling.
            """
            result = distrib.rvs(size=size_x * size_y)
            grid = convert_1d_to_2d(result, size_y)
            candidate = cls(grid)
            if candidate.valid:
                return candidate

    def store_as_csv(self, path):
        """
        Store the grid into a CSV file.
        :param path: A path where to store the file.
        :return:
        """
        with open(path, "w") as csvfile:
            tablewriter = csv.writer(csvfile, delimiter=',', quotechar='|')
            tablewriter.writerows(self._grid)

    @property
    def valid(self):
        return self._valid

    @property
    def colors(self):
        return self._colors

    @property
    def xdim(self):
        return len(self._grid)

    @property
    def ydim(self):
        return len(self._grid[0])

    @property
    def maxx(self):
        return self.xdim - 1

    @property
    def maxy(self):
        return self.ydim - 1

    @property
    def cells(self):
        return self._cells

    def __str__(self):
        return "\n".join(["\t".join([f"{self._grid[x][y]}" for y in range(0, self.ydim)]) for x in range(0, self.ydim)])

    @property
    def lower_bound(self):
        """
        Computes the shortest possible solution length.
        :return:
        """
        dist = 0
        for c in self.cells:
            if not self.is_start(c):
                continue
            dist = max(dist, min([Cell.distance(c, d) for d in self.cells if self.is_target(d)]))
        return dist

    @property
    def upper_bound(self):
        """
        Computes the longest possible solution length.
        :return:
        """
        return self.maxx * self.maxy

    def is_target(self, cell):
        return self.get_color(cell) == TARGET_CELL_OBS

    def is_lava(self, cell):
        return self.get_color(cell) == LAVA_CELL_OBS

    def is_start(self, cell):
        return self.get_color(cell) == START_CELL_OBS

    def is_ice(self, cell):
        return self.get_color(cell) == ICY_CELL_OBS

    def is_sticky(self, cell):
        return self.get_color(cell) == STICKY_CELL_OBS

    def get_color(self, cell):
        assert self._cell_is_valid(cell)
        return self._grid[cell.x][cell.y]

    def _cell_is_valid(self, cell):
        return cell.x <= self.maxx and cell.y <= self.maxy

    def neighbours(self, cell, dir):
        assert dir in ["N", "S", "W", "E", 0, 1, 2, 3]
        if dir == "W" or dir == 3:
            result = [Cell(max(cell.x-1, 0), cell.y)]
            if self.is_ice(cell):
                result += [Cell(max(cell.x-2, 0), cell.y)]
        elif dir == "E" or dir == 1:
            result = [Cell(min(cell.x+1, self.maxx), cell.y)]
            if self.is_ice(cell):
                result += [Cell(min(cell.x+2, self.maxx), cell.y)]
        elif dir == "N" or dir == 0:
            result = [Cell(cell.x, max(cell.y-1, 0))]
            if self.is_ice(cell):
                result += [Cell(cell.x, max(cell.y-2, 0))]
        elif dir == "S" or dir == 2:
            result = [Cell(cell.x, min(cell.y+1, self.maxy))]
            if self.is_ice(cell):
                result += [Cell(cell.x, min(cell.y+2, self.maxy))]
        return result

    def inv_neighbours(self, cell, dir):
        result = []
        if dir == "E" and cell.y > 0:
            result.append(Cell(cell.x, cell.y - 1))
            if cell.y > 1:
                cand = Cell(cell.x, cell.y - 2)
                if self.is_ice(cand):
                    result.append(cand)
        if dir == "W" and cell.y < self.maxy:
            result.append(Cell(cell.x, cell.y + 1))
            if cell.y < self.maxy - 1:
                cand = Cell(cell.x, cell.y + 2)
                if self.is_ice(cand):
                    result.append(cand)
        if dir == "S" and cell.x > 0:
            result.append(Cell(cell.x - 1, cell.y))
            if cell.x > 1:
                cand = Cell(cell.x - 2, cell.y)
                if self.is_ice(cand):
                    result.append(cand)
        if dir == "N" and cell.x < self.maxx:
            result.append(Cell(cell.x + 1, cell.y))
            if cell.x < self.maxx - 1:
                cand = Cell(cell.x + 2, cell.y)
                if self.is_ice(cand):
                    result.append(cand)
        return [n for n in result if not self.is_target(n)]
    
    def pred_reach(self, cell, dir, reach):
        assert dir in [0, 1, 2, 3]
        result = []
        if dir == 2 and cell.y > 0:
            result.append(Cell(cell.x, cell.y - 1))
        if dir == 0 and cell.y < self.maxy:
            result.append(Cell(cell.x, cell.y + 1))
        if dir == 1 and cell.x > 0:
            result.append(Cell(cell.x - 1, cell.y))
        if dir == 3 and cell.x < self.maxx:
            result.append(Cell(cell.x + 1, cell.y))
        if result == []:
            return []
        else:
            return [reach[result[0].x][result[0].y]]
    
    def succ_reach(self, cell, dir, reach):
        assert dir in [0, 1, 2, 3]
        result = []
        if dir == 0 and cell.y > 0:
            result.append(Cell(cell.x, cell.y - 1))
        if dir == 2 and cell.y < self.maxy:
            result.append(Cell(cell.x, cell.y + 1))
        if dir == 3 and cell.x > 0:
            result.append(Cell(cell.x - 1, cell.y))
        if dir == 1 and cell.x < self.maxx:
            result.append(Cell(cell.x + 1, cell.y))
        if result == []:
            return False
        else:
            return reach[result[0].x][result[0].y]
        
    def succ_time(self, cell, dir, time):
        assert dir in [0, 1, 2, 3]
        result = []
        if dir == 0 and cell.y > 0:
            result.append(Cell(cell.x, cell.y - 1))
        if dir == 2 and cell.y < self.maxy:
            result.append(Cell(cell.x, cell.y + 1))
        if dir == 3 and cell.x > 0:
            result.append(Cell(cell.x - 1, cell.y))
        if dir == 1 and cell.x < self.maxx:
            result.append(Cell(cell.x + 1, cell.y))
        if result == []:
            return []
        else:
            return [time[result[0].x][result[0].y]]
        
    def pred_obs_act(self, cell, dir, obsact):
        assert dir in [0, 1, 2, 3]
        result = []
        if dir == 2 and cell.y > 0:
            result.append(Cell(cell.x, cell.y - 1))
        if dir == 0 and cell.y < self.maxy:
            result.append(Cell(cell.x, cell.y + 1))
        if dir == 1 and cell.x > 0:
            result.append(Cell(cell.x - 1, cell.y))
        if dir == 3 and cell.x < self.maxx:
            result.append(Cell(cell.x + 1, cell.y))
        if result == []:
            return []
        else:
            color = self.get_color(result[0])
            return [obsact[color][dir]]

    def plot(self, path, policy = None, count = None):
        """
        This method requires MatPlotLib and NumPy

        :param path:
        :param policy:
        :param count:
        :return:
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np

        def surface_to_text(surface):
            if surface == START_CELL_OBS:
                return "S"
            if surface == TARGET_CELL_OBS:
                return "T"
            if surface == LAVA_CELL_OBS:
                return "L"
            if surface == ICY_CELL_OBS:
                return "I"
            if surface == STICKY_CELL_OBS:
                return "D"
            return surface

        def dir_to_carret(dir):
            return DIR_TO_CARET[dir]

        def xdir_offset(dir):
            if dir == "W":
                return -0.6
            if dir == "E":
                return 0.6
            return 0.0

        def ydir_offset(dir):
            if dir == "N":
                return -0.6
            if dir == "S":
                return 0.6
            return 0.0

        fig = plt.figure()
        ax = fig.subplots()
        column_labels = list(range(0, self.ydim))
        ax.set_xlim([-0.4, self.maxx + 1.4])
        ax.set_ylim([-0.4, self.maxy + 1.4])
        row_labels = list(range(0, self.xdim))
        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(self.xdim) + 0.5, minor=False)
        ax.set_yticks(np.arange(self.ydim) + 0.5, minor=False)

        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(row_labels, minor=False)
        ax.set_yticklabels(column_labels, minor=False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect(1)
        if count is not None:
            ax.set_title(f"Solution for {count} steps")
        ax.pcolor([list(x) for x in zip(*self._grid)], cmap=mpl.colors.ListedColormap(COLOR_NAMES),
                  edgecolors='k', linestyle='dashed', linewidths=0.2, vmin=0, vmax=len(COLOR_NAMES))
        bid = dict(boxstyle='round', facecolor='white', alpha=0.7)

        for c in self.cells:
            if self.is_target(c):
                ax.scatter(c.x + 0.5, c.y + 0.5, s=320, marker='*', color='gold')
            elif self.is_lava(c):
                ax.scatter(c.x + 0.5, c.y + 0.5, s=320, marker='X', color='black')
            elif self.is_ice(c):
                ax.scatter(c.x + 0.5, c.y + 0.5, s=320, marker=(6, 2, 0), color="lightblue")
            elif self.is_sticky(c):
                ax.scatter(c.x + 0.5, c.y + 0.5, s=320, marker='D', color="lightblue")
            elif self.is_start(c):
                ax.scatter(c.x + 0.5, c.y + 0.5, s=320, marker="*", color="lightgreen")
            else:
                ax.text(c.x + 0.6, c.y + 0.6, surface_to_text(self._grid[c.x][c.y]), fontsize=10,
                    verticalalignment='top', bbox=bid)
            if policy is not None and c in policy and not self.is_target(c) and not self.is_lava(c):
                ax.scatter(c.x + 0.5 + xdir_offset(policy[c]), c.y + 0.5 + ydir_offset(policy[c]), s=220, marker=dir_to_carret(policy[c]), color="black")
                if self.is_ice(c):
                    ax.scatter(c.x + 0.5 + xdir_offset(policy[c])*0.6, c.y + 0.5 + ydir_offset(policy[c])*0.6, s=220,
                               marker=dir_to_carret(policy[c]), color="black")
        fig.savefig(path)
        plt.close(fig)


class GridEncoding:
    def __init__(self, grid):
        self._grid = grid
        self._s = Solver()

    def solve(self):
        """
        A solution is a tuple (nr_steps, policy) where the nr_steps is the number of steps actually necessary,
        and a policy, which is a dictionary from grid cells to directions. Returning a policy is optional, but helpful for debugging.

        :return:
        """
        # 0 = Start
        # 1 = Finish
        # 2 = Lava
        # 3 = Unassigned
        # 4 = Sticky
        # 5 - ? = Everything else

        # Grab max value from grid
        # Go through all values and create integers for all field types (skip three)
        # 0 = North
        # 1 = East
        # 2 = South
        # 3 = West

        # One boolean (Z3.Bool) for each observation o ∈ O and action a ∈ A - Po,a:
        self._observation_action = [[Bool(f"P_{o}_{a}") for a in range(4)] for o in range(max(self._grid)+1)]
        # print("self._observation_action", self._observation_action)

        # Each observation has 1 action
        each_observation_one_action_cs = [Sum([If(self._observation_action[o][a], 1, 0) 
                                               for a in range(4)]) == 1 
                                               for o in range(max(self._grid)+1) 
                                               if o != 2]
        # print("each_observation_one_action_cs", each_observation_one_action_cs)
        self._s.add(each_observation_one_action_cs)

        # Lava observation has 0 action
        lava_observation_zero_action_cs = [Not(self._observation_action[2][a])
                                               for a in range(4)]
        # print("lava_observation_zero_action_cs", lava_observation_zero_action_cs)
        self._s.add(lava_observation_zero_action_cs)

        # One boolean for each cell 〈x, y〉: Reachx,y
        self._reach = [[Bool(f"Reach_{x}_{y}") for y in range(self._grid.xdim)] for x in range(self._grid.ydim)]
        # print("self._reach", self._reach)

        # Initial states are reachable
        initial_states_reach_cs = [self._reach[x][y] for x in range(self._grid.xdim) for y in range(self._grid.ydim) if self._grid._grid[x][y] == 0]
        # print("initial_states_reach_cs", initial_states_reach_cs)
        self._s.add(initial_states_reach_cs)

        # Lava is never reachable
        lava_not_reach_cs = [Not(self._reach[x][y]) for x in range(self._grid.xdim) for y in range(self._grid.ydim) if self._grid._grid[x][y] == 2]
        # print("lava_not_reach_cs", lava_not_reach_cs)
        self._s.add(lava_not_reach_cs)

        # If this cell is reachable, then an adjacent cell should be reachable with a action to this cell
        reachable_cs = [Implies(
            self._reach[x][y], 
            Or(
                [And(self._grid.pred_reach(Cell(x,y), a, self._reach)[0], 
                     self._grid.pred_obs_act(Cell(x,y), a, self._observation_action)[0]
                     ) 
                 for a in range(4) 
                 if self._grid.pred_reach(Cell(x,y), a, self._reach) != [] 
                 and self._grid.pred_obs_act(Cell(x,y), a, self._observation_action) != []
                ]
            )) 
            for x in range(self._grid.xdim) 
            for y in range(self._grid.ydim) 
            if self._grid.get_color(Cell(x,y)) != 0]

        # print("reachable_cs")
        # for x in reachable_cs:
        #     print(x)
        self._s.add(reachable_cs)

        # If this cell is reachable, and it has a action to an adjacent cell. Then the adjacent cell is reachable.
        reachable_inv = [Implies(
            And(self._reach[x][y], self._observation_action[self._grid.get_color(Cell(x,y))][a]), 
            self._grid.succ_reach(Cell(x,y), a, self._reach)) 
            for x in range(self._grid.xdim) 
            for y in range(self._grid.ydim) 
            for a in range(4) 
            if self._grid._grid[x][y] != 1 ]
        # print("reachable_inv")
        # for x in reachable_inv:
        #     print(x)
        self._s.add(reachable_inv)

        # Time variable matrix
        self._time = [[Int(f"T_{x}_{y}") for y in range(self._grid.xdim)] for x in range(self._grid.ydim)]
        # print("self._time", self._time)

        # Initial states have t >= 1
        initial_states_time_cs = [self._time[x][y] >= 1 
             for x in range(self._grid.xdim) 
             for y in range(self._grid.ydim) 
             if self._grid._grid[x][y] == 0]
        # print("initial_states_time_cs")
        # for x in initial_states_time_cs:
        #     print(x)
        self._s.add(initial_states_time_cs)

        # All successive states have a t higher then there predecessor
        time_cs = [Implies(
            And(self._reach[x][y], self._observation_action[self._grid.get_color(Cell(x,y))][a]), 
            self._grid.succ_time(Cell(x,y), a, self._time)[0] > self._time[x][y]
            ) 
            for x in range(self._grid.xdim) 
            for y in range(self._grid.ydim) 
            for a in range(4) 
            if self._grid.get_color(Cell(x,y)) != 1 
            and self._grid.get_color(self._grid.neighbours(Cell(x,y), a)[0]) != 1 
            and self._grid.succ_time(Cell(x,y), a, self._time) != []]
        # print("time_cs")
        # for t in time_cs:
        #     print(t)
        self._s.add(time_cs)

        # All sticky successive states have a t + 7 higher then there predecessor
        time_sticky_cs = [Implies(
            And(self._reach[x][y], self._observation_action[self._grid.get_color(Cell(x,y))][a]), 
            self._grid.succ_time(Cell(x,y), a, self._time)[0] > self._time[x][y] + 6
            ) 
            for x in range(self._grid.xdim) 
            for y in range(self._grid.ydim) 
            for a in range(4) 
            if self._grid.get_color(Cell(x,y)) != 1 
            and self._grid.get_color(self._grid.neighbours(Cell(x,y), a)[0]) == 4
            and self._grid.succ_time(Cell(x,y), a, self._time) != []]
        # print("time_cs")
        # for t in time_cs:
        #     print(t)
        self._s.add(time_sticky_cs)

        # Max step restriction
        def max_step(n):
            return [And(self._time[x][y] >= 0, self._time[x][y] <= n) 
                    for x in range(self._grid.xdim) 
                    for y in range(self._grid.ydim) 
                    if self._grid._grid[x][y] != 1]

        policy = {}
        actions = {}
        for mx_stp in range(self._grid.lower_bound, 100):
            print("trying max step:", mx_stp)
            self._s.push()
            self._s.add(max_step(mx_stp))
            if self._s.check() == sat:
                print("sat")
                m = self._s.model()
                # Evaluate policy
                for c, colors in enumerate(self._observation_action):
                    for a, action in enumerate(colors):
                        if m.evaluate(action) == True:
                            cardinal = ""
                            if a == 0:
                                cardinal = "N"
                            if a == 1:
                                cardinal = "E"
                            if a == 2:
                                cardinal = "S"
                            if a == 3:
                                cardinal = "W"
                            actions[c] = cardinal

                # get a list of all initial states
                # for all init states
                    # follow with succ function untill final state
                initial_states = [Cell(x,y) for x in range(self._grid.xdim) for y in range(self._grid.ydim) if self._grid._grid[x][y] == 0]
                final_states = [Cell(x,y) for x in range(self._grid.xdim) for y in range(self._grid.ydim) if self._grid._grid[x][y] == 1]
                for init_state in initial_states:
                    current_state = init_state
                    while current_state not in final_states:
                        # print(current_state)
                        color = self._grid.get_color(current_state)
                        policy[current_state] = actions[color]
                        prev_state = current_state
                        current_state = self._grid.neighbours(current_state, actions[color])[0]
                        if prev_state == current_state:
                            print("Error same state!")
                            break
                print(actions)
                # print("Evaluate(self._observation_action)")
                # print_matrix([[(y,m.evaluate(y))for y in x] for x in self._observation_action])
                # print("Evaluate(self._reach)")
                # print_matrix([[(y,m.evaluate(y)) for y in x] for x in self._reach])
                # print("Evaluate(self._time)")
                # print_matrix([[(y,m.evaluate(y))for y in x] for x in self._time])
                return mx_stp, policy
            self._s.pop()
        print("unsat")
        return math.inf, []

def decide(grid, nr_steps = None):
    """
    The method takes a grid and should return a tuple (nr_steps, policy) where the nr_steps is the number of steps actually necessary,
    and a policy, which is a dictionary from grid cells to directions. Returning a policy is optional, but helpful for debugging.

    :param grid: The Grid
    :param nr_steps: The maximal number of steps that are ok to take.
    :return:
    """
    # TODO you may change this code.
    return solve(grid)[0] <= nr_steps

def solve(grid):
    """
    The method takes a grid and should return a tuple (nr_steps, policy) where the nr_steps is the number of steps actually necessary,
    and a policy, which is a dictionary from grid cells to directions. Returning a policy is optional, but helpful for debugging.

    :param grid: The Grid
    :return: The nr of steps and the policy that induces this solution.
    """
    encoding = GridEncoding(grid)
    # encoding.create_encoding()
    result, policy = encoding.solve()
    return result, policy

def main():
    """
    To ensure that this is remains a single-file project without advanced dependencies such as click,
     the main method is somewhat awkward.
    """
    parser = argparse.ArgumentParser(prog="ruSBP",
                                     description="Small stub for a program for step-bounded planning, used at the Radboud Automated Reasoning class.")
    parser.add_argument("--nr-steps", '-S', help="The maximal number of steps", type=int)
    parser.add_argument("--load-grid", help="A grid file to be loaded. If no grid is loaded, a random grid is generated.")
    parser.add_argument("--rg-xsize", help="For random grids, the x-size dimension.", type=int, default=6)
    parser.add_argument("--rg-ysize", help="For random grids, the y-size dimension.", type=int, default=6)
    parser.add_argument("--rg-terraintypes", help="For random grids, the number of terrain types", type=int, default=8)
    parser.add_argument("--batch-generation-mode", help="[ADVANCED] In batch-generation mode, we create and test multiple puzzles.", type=int, default=0)
    args = parser.parse_args()

    def generate_random_grid(random_grid_code_initialized = False):
        print("Generating a random grid...(NOTICE: Many grids will not allow reaching the finish in any number of steps).")
        if not random_grid_code_initialized:
            try:
                import numpy
                numpy.random.seed(48)
            except ImportError:
                raise RuntimeError("Generating random grids requires numpy.")
        grid = Grid.random(args.rg_xsize, args.rg_ysize, args.rg_terraintypes, 0.10, 0.08, 0.05, 0.0, 0.12)
        return grid

    def print_grid_info(grid):
        print(f"...The grid has dimensions {grid.xdim}x{grid.ydim}")
        print(f"...A trivial lower bound on the solution is {grid.lower_bound}")

    if args.batch_generation_mode == 0:
        if args.load_grid:
            print(f"Loading a grid from file {args.load_grid}...")
            grid = Grid.from_csv(args.load_grid)
        else:
            grid = generate_random_grid()
        print_grid_info(grid)
        if args.nr_steps:
            print("Decide: Is there a solution?")
            answer = decide(grid, args.nr_steps)
            res_as_string = "a" if answer else "no"
            print(f"*******************************************")
            print(f"**** Found {res_as_string} solution")
            print(f"******************************************")
        else:
            print("Computing number of steps instead")
            nr_steps, policy = solve(grid)
            print(f"*******************************************")
            print(f"**** Found a solution with {nr_steps} steps.")
            print(f"******************************************")
            dt_string = args.load_grid
            grid.plot(f"grid_{dt_string}.png")
            grid.plot(f"grid_{dt_string}_solution.png", policy=policy, count=nr_steps)
    else:
        print("Enter batch-generation mode...")
        i = 0
        random_grid_code_initialized = False

        while i < args.batch_generation_mode:

            print(f"* Batch generation mode, round {i+1} out of {args.batch_generation_mode}")
            grid = generate_random_grid(random_grid_code_initialized)
            random_grid_code_initialized = True
            print_grid_info(grid)
            nr_steps, policy = solve(grid)
            if nr_steps < math.inf and nr_steps >= grid.lower_bound + 4 and nr_steps >= 9 and len(policy) >= (grid.ydim * grid.xdim)*0.3:
                print(f"* Success! Solution requires {nr_steps} and reaches at least {len(policy)}/{grid.ydim * grid.xdim} cells.")
                #print(grid)
                #print(result)
                #dt_string = dt.now().strftime("%Y-%m-%d-%H:%M:%S")
                dt_string = "2023-A-" + str(i)
                grid.store_as_csv(f"grid_{dt_string}.csv")
                grid.plot(f"grid_{dt_string}.png")
                grid.plot(f"grid_{dt_string}_solution.png", policy=policy, count=nr_steps)
                i += 1
            else:
                print(f"* Randomly created grid does not admit a nice solution (nr_steps={nr_steps})...")


if __name__ == "__main__":
    main()
