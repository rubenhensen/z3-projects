"""
This file contains a sudoku class and a parser/serializer.
"""
import csv
import sys
import typing
from itertools import product as cart_product

"""
Contains the SmtBasedSolver class.
"""
import typing

# TODO sadly, no typing support for z3 based types right now.
import z3 as smtapi  # type: ignore

# from radboudsudoku.sudoku import Sudoku, SudokuRectangularBlock



def _remove_none(list_with_none: list[int | None]) -> list[int]:
    """
    Auxiliary function to remove None from a list
    :param list_with_none: A list
    :return: The list with all occurences of None filtered out.
    """
    return [i for i in list_with_none if i is not None]


class SudokuCell:
    """
    Thin wrapper, basically a named tuple, that represents a cell in a Sudoku.
    """

    def __init__(self, row, column):
        self.row = row
        self.column = column

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column

    def __hash__(self):
        return hash(self.row) ^ hash(self.column)


class SudokuRectangularBlock:
    """
    This class represents a rectangular block inside a Sudoku, where every value may exists at most once.
    """

    def __init__(self, min_row: int, max_row: int, min_column: int, max_column: int) -> None:
        self._min_row = min_row
        self._max_row = max_row
        self._min_column = min_column
        self._max_column = max_column

    @property
    def nr_rows(self) -> int:
        """
        Number of rows in the puzzle.
        """
        return self._max_row - self._min_row + 1

    @property
    def nr_columns(self) -> int:
        """
        Number of columns in the block.
        """
        return self._max_column - self._min_column + 1

    @property
    def size(self) -> int:
        """
        The number of cells in a block.s
        """
        return self.nr_rows * self.nr_columns

    @property
    def _row_indices(self) -> list[int]:
        return list(range(self._min_row, self._max_row + 1))

    @property
    def _column_indices(self) -> list[int]:
        return list(range(self._min_column, self._max_column + 1))

    @property
    def cells(self) -> typing.Iterable[SudokuCell]:
        """
        An Iterable over all cells in the block.
        """
        return [SudokuCell(row, col) for row, col in cart_product(self._row_indices, self._column_indices)]

    def __repr__(self):
        return f"<RectangularBlock [{self._min_row},{self._max_row}]x[{self._min_column},{self._max_column}]>"


BLOCK_ROWS = [[0, 2], [3, 5], [6, 8]]
BLOCK_COLUMNS = [[0, 2], [3, 5], [6, 8]]


class Sudoku:
    """
    This class provides access to a sudoku.

    Rows and columns start with zero. The dimension is currently fixed to be 9.
    """

    def __init__(self) -> None:
        dimension = 9
        allowed_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self._nr_rows = dimension
        self._nr_columns = dimension
        self._allowed_values = allowed_values
        self._field: list[list[int | None]] = [[None for _ in range(self._nr_rows)] for _ in range(self._nr_columns)]
        self._blocks = [
            SudokuRectangularBlock(block_rows[0], block_rows[1], block_columns[0], block_columns[1])
            for block_rows, block_columns in cart_product(BLOCK_ROWS, BLOCK_COLUMNS)
        ]

    @property
    def allowed_values(self) -> list[int]:
        """

        :return:
        """
        return self._allowed_values

    @property
    def nr_rows(self) -> int:
        """
        Public read-only access to the number of rows.
        """
        return self._nr_rows

    @property
    def nr_columns(self) -> int:
        """
        Public read-only access to the number of columns.
        """
        return self._nr_columns

    @property
    def blocks(self) -> list[SudokuRectangularBlock]:
        """
        Returns the blocks, i.e., the partitions of the grid in which every number can occur at most once.
        """
        return self._blocks

    @property
    def row_indices(self) -> typing.Iterable[int]:
        """
        The row indices
        """
        return range(self._nr_rows)

    @property
    def column_indices(self) -> typing.Iterable[int]:
        """
        The column indices
        """
        return range(self._nr_columns)

    @property
    def cells(self) -> typing.Iterable[SudokuCell]:
        """
        Provides an iterable over all cells of the sudoku.
        """
        return [SudokuCell(row, col) for row, col in cart_product(self.row_indices, self.column_indices)]

    def cells_for_row(self, row_index: int) -> list[SudokuCell]:
        """
        Provides an iterable over all cells of the fixed row in the Sudoku.
        :param row_index: The index for the row
        """
        return [SudokuCell(row_index, c) for c in self.column_indices]

    def cells_for_column(self, column_index: int) -> list[SudokuCell]:
        """
        Provides an iterable over all cells of the fixed column in the Sudoku
        :param column_index: The index for the column
        """
        return [SudokuCell(r, column_index) for r in self.row_indices]

    def cell_exists(self, row: int, column: int) -> bool:
        """
        Does the cell with the given row and column index exist?
        """
        return self.row_exists(row) and self.column_exists(column)

    def row_exists(self, row: int) -> bool:
        """
        Check whether a given row index is valid.
        """
        return 0 <= row < self.nr_rows

    def column_exists(self, column: int) -> bool:
        """
        Check whether a given column index is valid.
        """
        return 0 <= column < self.nr_columns

    def is_filled(self) -> bool:
        """
        Checks whether all cells have been filled.
        :return:
        """
        for cell in self.cells:
            if self.get_value(cell.row, cell.column) is None:
                return False
        return True

    def set_value(self, row: int, column: int, value: int | None) -> None:
        """
        Write a value to the given field.
        :param row: 0-indexed row
        :param column: 0-indexed column
        :param value: the value to be written, must be a valid value or None
        """
        if value is not None and value not in self._allowed_values:
            msg = f"Value {value} is not allowed."
            raise ValueError(msg)
        if not self.cell_exists(row, column):
            msg = f"row={row},column={column} is not a valid cell."
            raise ValueError(msg)
        self._field[row][column] = value

    def get_value(self, row: int, column: int) -> int | None:
        """
        Obtain the value from the specified cell.
        Checks whether cell exists, otherwise raises an Exception
        :param row: The index of the row
        :param column: The index of the column
        :return: an allowed_value or None.
        """
        if not self.cell_exists(row, column):
            msg = f"row={row},column={column} is not a valid cell."
            raise ValueError(msg)
        return self._field[row][column]

    def check_is_valid(self) -> None:
        """
        Checks whether the (partially filled) puzzle is valid. Raises a RuntimeError with diagnostic info if not
        """
        for row in self.row_indices:
            self.check_row_is_valid(row)
        for col in self.column_indices:
            self.check_column_is_valid(col)
        for block in self._blocks:
            self.check_block_is_valid(block)

    def check_row_is_valid(self, row: int) -> None:
        """
        Checks whether a particular row is valid. Raises a RuntimeError with diagnostic info otherwise.
        :param row: The row index.
        """
        all_values = _remove_none([self.get_value(row, c) for c in self.column_indices])
        if len(all_values) != len(set(all_values)):
            msg = f"Row with index {row} and values {sorted(all_values)} contains duplicate."
            raise RuntimeError(msg)

    def check_column_is_valid(self, column: int) -> None:
        """
        Checks whether a particular column is valid. Raises a RuntimeError with diagnostic info otherwise.
        :param column: The row index.
        """
        all_values = _remove_none([self.get_value(r, column) for r in self.row_indices])
        if len(all_values) != len(set(all_values)):
            msg = f"Column with index {column} and values {sorted(all_values)} contains duplicate."
            raise RuntimeError(msg)

    def check_block_is_valid(self, block: SudokuRectangularBlock) -> None:
        all_values = _remove_none([self.get_value(cell.row, cell.column) for cell in block.cells])
        if len(all_values) != len(set(all_values)):
            msg = f"Block {block} and values {sorted(all_values)} contains duplicate."
            raise RuntimeError(msg)


def parse_from_csv(location) -> Sudoku:
    """
    Parse a CSV with a sudoku. Throws an error if the input is no longer valid.
    :param location: The location of the csv.
    :return: The puzzle in a Sudoku object.
    """

    # TODO This can be made more robust
    result = Sudoku()
    with open(location, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row_index, row in enumerate(reader):
            for column_index, value in enumerate(row):
                if value.strip() != "":
                    result.set_value(row_index, column_index, int(value))
    result.check_is_valid()
    return result


def write_to_csv(location, puzzle: Sudoku) -> None:
    """
    Exports a sudoku puzzle to a sudoku.
    :param location: The location of the csv. Can be None, then the csv is written to stdout
    :param puzzle: the puzzle that is to be written
    """
    if location is not None:
        with open(location, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for row in puzzle.row_indices:
                writer.writerow([puzzle.get_value(row, column) for column in puzzle.column_indices])
    else:
        writer = csv.writer(sys.stdout)
        for row in puzzle.row_indices:
            writer.writerow([puzzle.get_value(row, column) for column in puzzle.column_indices])




class SmtBasedSolver:
    """
    This SMT-based Sudoku solver uses the z3py API.
    """

    def __init__(self):
        self._smt_solver = smtapi.Solver()
        self._cell_variables = {}

    def encode(self, puzzle: Sudoku):
        """
        Takes a puzzle and encodes it into the SMT solver.
        :param puzzle:
        :return:
        """
        # We first create the variables and store them for future access in the object.
        self._create_variables(puzzle)
        # We then encode the puzzle
        self._add_constraints(puzzle)

    def _create_variables(self, puzzle: Sudoku) -> None:
        """
        Creates an integer variable for every cell
        :param puzzle: The sudoku.
        """
        for cell in puzzle.cells:
            self._cell_variables[cell] = smtapi.Int(f"v-{cell.row}-{cell.column}")

    def _add_constraints(self, puzzle: Sudoku) -> None:
        """

        :param puzzle:
        :return:
        """
        # 1. Each cell should be filled with an allowed value.
        for cell in puzzle.cells:
            current_value = puzzle.get_value(cell.row, cell.column)
            if current_value is None:
                self._add_either_value(self._cell_variables[cell], puzzle.allowed_values)
            else:
                self._smt_solver.add(self._cell_variables[cell] == current_value)
        # 2. The values for the cells in every row should be distinct.
        for row in puzzle.row_indices:
            self._add_all_different(self._row_variables(puzzle, row))
        # 3. The values for the cells in every column should be distinct.
        for column in puzzle.column_indices:
            self._add_all_different(self._column_variables(puzzle, column))
        # 4. The values for the cells in every block should be distinct.
        for block in puzzle.blocks:
            self._add_all_different(self._block_variables(block))
        # Note that in standard sudokus, 1+2 encodes that every value must be used in every row.
        # Likewise, 1+3 and 1+4 ensure this for columns and blocks, respectively.
        # Debugging tip: Here you can export all constraints via print(self._smt_solver)

    def _row_variables(self, puzzle: Sudoku, row: int) -> list[smtapi.Int]:
        return [self._cell_variables[cell] for cell in puzzle.cells_for_row(row)]

    def _column_variables(self, puzzle: Sudoku, column: int) -> list[smtapi.Int]:
        return [self._cell_variables[cell] for cell in puzzle.cells_for_column(column)]

    def _block_variables(self, block: SudokuRectangularBlock):
        return [self._cell_variables[cell] for cell in block.cells]

    def _add_all_different(self, variables) -> None:
        for index, lhs_var in enumerate(variables):
            for rhs_var in variables[index + 1 :]:
                self._smt_solver.add(lhs_var != rhs_var)

    def _add_either_value(self, variable: smtapi.Int, values: typing.Iterable[int]) -> None:
        self._smt_solver.add(smtapi.Or([variable == v for v in values]))

    def solve(self, puzzle) -> tuple[bool, typing.Any]:
        """
        Solves a puzzle after encoding
        :param puzzle:
        :return:
        """
        solved = self._smt_solver.check()
        if solved == smtapi.sat:
            model = self._smt_solver.model()
            result = {}
            for cell in puzzle.cells:
                result[cell] = model.evaluate(self._cell_variables[cell])
            return True, result
        return False, None