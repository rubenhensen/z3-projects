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


mydict = {Cell(1,2): "N", Cell(3,2): "S", Cell(1,9): "W"}

for cell in mydict:
    print(cell)