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
