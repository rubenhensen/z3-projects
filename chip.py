from typing import List, Iterable
import typing
from z3 import *
from itertools import product as cart_product
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy import random
import numpy



REGULAR_CHIPS = [[4, 5], [4, 6], [5, 20], [6, 9], [6, 10], [6, 11],
[7, 8], [7, 12], [10, 10], [10, 20]]

# DISTANCE = 16
DISTANCE = 17
# DISTANCE = 18 # this breaks




class RegularComponent:
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2

class PowerComponent:
    def __init__(self, l1, l2):
        self.l1 = l1 # 4
        self.l2 = l2 # 3

class Chip:
    def __init__(self, h: int, w: int, nr_power_comps: int, rcomps: List[List[int]]):
        self.h = h
        self.w = w
        self._pcomps = [PowerComponent(4,3) for _ in range(nr_power_comps)]
        self._rcomps = [RegularComponent(rc_l1, rc_l2) for [rc_l1,rc_l2] in rcomps]

    @property
    def power_comps(self) -> List[PowerComponent]:
        """
        Provides an iterable over all power components of the chip.
        """
        return self._pcomps
    
    @property
    def regular_comps(self) -> List[RegularComponent]:
        """
        Provides an iterable over all regular components of the chip.
        """
        return self._rcomps

class SMT:
    """
    This SMT-based Sudoku solver uses the z3py API.
    """

    def __init__(self):
        self._smt_solver = Solver()
        self._comp_variables = {}
        self._chipheight = 0 
        self._chipwidth = 0

    def encode(self, chip: Chip):
        """
        Takes a chip and encodes it into the SMT solver.
        :param chip:
        :return:
        """
        # We first create the variables and store them for future access in the object.
        self._create_variables(chip)
        # We then encode the puzzle
        self._add_constraints(chip)

    def vars(self):
        print("Component vars:\n", self._comp_variables)

    def _create_variables(self, chip: Chip) -> None:
        """
        Creates four integer variable for every component.
        :param puzzle: The .
        """
        for rcomp in chip.regular_comps:
            self._comp_variables[rcomp] = Ints('rc-%sx%s-X rc-%sx%s-Y rc-%sx%s-h rc-%sx%s-w rc-%sx%s-r rc-%sx%s-u' % (rcomp.l1, rcomp.l2, rcomp.l1, rcomp.l2, rcomp.l1, rcomp.l2, rcomp.l1, rcomp.l2, rcomp.l1, rcomp.l2, rcomp.l1, rcomp.l2))
        for i, pcomp in enumerate(chip.power_comps):
            self._comp_variables[pcomp] = Ints('pc-%sx%s_%s-X pc-%sx%s_%s-Y pc-%sx%s_%s-h pc-%sx%s_%s-w pc-%sx%s_%s-r pc-%sx%s_%s-u' % (pcomp.l1, pcomp.l2, i, pcomp.l1, pcomp.l2, i, pcomp.l1, pcomp.l2, i, pcomp.l1, pcomp.l2, i, pcomp.l1, pcomp.l2, i, pcomp.l1, pcomp.l2, i))

        self._chipheight = chip.h
        self._chipwidth = chip.w


    def _add_constraints(self, chip: Chip) -> None:
        """

        :param chip:
        :return:
        """
        all_comps = chip.power_comps + chip.regular_comps

        for comp in all_comps:
            x = self._comp_variables[comp][0]
            y = self._comp_variables[comp][1]
            h = self._comp_variables[comp][2]
            w = self._comp_variables[comp][3]
            r = self._comp_variables[comp][4]
            u = self._comp_variables[comp][5]
            

            # Left Right Up Down
            self._smt_solver.add(r == x + w)
            self._smt_solver.add(u == y + h)

            # 1. Chips can be turned 90 degrees (Height and width are interchangeable. But heigth can only be l1 iff width is l2.)
            self._smt_solver.add(Or(h == comp.l1, h == comp.l2))
            self._smt_solver.add(Or(w == comp.l1, w == comp.l2))
            self._smt_solver.add(Implies(h == comp.l1, w == comp.l2))
            self._smt_solver.add(Implies(h == comp.l2, w == comp.l1))
            self._smt_solver.add(Implies(w == comp.l1, h == comp.l2))
            self._smt_solver.add(Implies(w == comp.l2, h == comp.l1))

            # 2. Components need to reside within the bounds of the chip
            self._smt_solver.add(u <= self._chipheight)
            self._smt_solver.add(y <= self._chipheight)
            self._smt_solver.add(u >= 0)
            self._smt_solver.add(y >= 0)
            self._smt_solver.add(r <= self._chipwidth)
            self._smt_solver.add(x <= self._chipwidth)
            self._smt_solver.add(r >= 0)
            self._smt_solver.add(x >= 0)
        
        # 3. Components may not overlap
        for c1, c2 in cart_product(all_comps, all_comps):
            self._no_overlap(c1, c2)

        # 4. Components should align to power components
        for rc in chip.regular_comps:
            self._with_power(chip.power_comps[0], chip.power_comps[1], rc)

        # 5. Power components should be x apart
        self._power_distance(chip.power_comps[0], chip.power_comps[1], DISTANCE)

    def _power_distance(self, pc1:PowerComponent, pc2:PowerComponent, dist: int ) -> None:
        pcL = self._comp_variables[pc1][0]
        pcR = self._comp_variables[pc1][4]
        pcU = self._comp_variables[pc1][5]
        pcD = self._comp_variables[pc1][1]
        pcH = self._comp_variables[pc1][2]
        pcW = self._comp_variables[pc1][3]

        _pcL = self._comp_variables[pc2][0]
        _pcR = self._comp_variables[pc2][4]
        _pcU = self._comp_variables[pc2][5]
        _pcD = self._comp_variables[pc2][1]
        _pcH = self._comp_variables[pc2][2]
        _pcW = self._comp_variables[pc2][3]
        self._smt_solver.add(
            Or(
                Abs((pcL+0.5*pcW) - (_pcL+0.5*_pcW)) >= dist,
                Abs((pcD+0.5*pcH) - (_pcD+0.5*_pcH)) >= dist
            )
        )

        # print(Or(
        #     Abs((pcL+0.5*pcW) - (_pcL+0.5*_pcW)) >= dist,
        #     Abs((pcD+0.5*pcH) - (_pcD+0.5*_pcH)) >= dist
        # ))
        return

    
    def _with_power(self, pc1:PowerComponent, pc2:PowerComponent, rc: RegularComponent) -> None:
        pcL = self._comp_variables[pc1][0]
        pcR = self._comp_variables[pc1][4]
        pcU = self._comp_variables[pc1][5]
        pcD = self._comp_variables[pc1][1]

        _pcL = self._comp_variables[pc2][0]
        _pcR = self._comp_variables[pc2][4]
        _pcU = self._comp_variables[pc2][5]
        _pcD = self._comp_variables[pc2][1]

        rcL = self._comp_variables[rc][0]
        rcR = self._comp_variables[rc][4]
        rcU = self._comp_variables[rc][5]
        rcD = self._comp_variables[rc][1]

        self._smt_solver.add(
            Or(
                # pc1
                Or(
                    And(pcL == rcR, Or(
                                    And(rcU >= pcU, rcD <= pcD),
                                    And(rcU > pcD, rcU < pcU),
                                    And(rcD > pcD, rcD < pcU))),
                    And(pcR == rcL, Or(
                                    And(rcU >= pcU, rcD <= pcD),
                                    And(rcU > pcD, rcU < pcU),
                                    And(rcD > pcD, rcD < pcU))),
                    And(pcD == rcU, Or(
                                    And(rcR >= pcR, rcL <= pcL),
                                    And(rcR > pcL, rcR < pcR),
                                    And(rcL > pcL, rcL < pcR))),
                    And(pcU == rcD, Or(
                                    And(rcR >= pcR, rcL <= pcL),
                                    And(rcR > pcL, rcR < pcR),
                                    And(rcL > pcL, rcL < pcR)))
                ),
                 # pc2
                Or(
                    And(_pcL == rcR, Or(
                                    And(rcU >= _pcU, rcD <= _pcD),
                                    And(rcU > _pcD, rcU < _pcU),
                                    And(rcD > _pcD, rcD < _pcU))),
                    And(_pcR == rcL, Or(
                                    And(rcU >= _pcU, rcD <= _pcD),
                                    And(rcU > _pcD, rcU < _pcU),
                                    And(rcD > _pcD, rcD < _pcU))),
                    And(_pcD == rcU, Or(
                                    And(rcR >= _pcR, rcL <= _pcL),
                                    And(rcR > _pcL, rcR < _pcR),
                                    And(rcL > _pcL, rcL < _pcR))),
                    And(_pcU == rcD, Or(
                                    And(rcR >= _pcR, rcL <= _pcL),
                                    And(rcR > _pcL, rcR < _pcR),
                                    And(rcL > _pcL, rcL < _pcR))),
                )
            )
        )
        return

    def _no_overlap(self, comp1: RegularComponent | PowerComponent, comp2: RegularComponent | PowerComponent) -> None:
        l1 = self._comp_variables[comp1][0]
        r1 = self._comp_variables[comp1][4]
        u1 = self._comp_variables[comp1][5]
        d1 = self._comp_variables[comp1][1]

        l2 = self._comp_variables[comp2][0]
        r2 = self._comp_variables[comp2][4]
        u2 = self._comp_variables[comp2][5]
        d2 = self._comp_variables[comp2][1]

        if (comp1 != comp2):
            self._smt_solver.add(
                Or(
                    # Of links, of recht van de ander
                    Or(
                        And(l1 >= r2, r1 >= r2),
                        And(l1 <= l2, r1 <= l2)
                    )
                    ,
                    # of boven, of onder van de ander
                    Or(
                        And(d1 >= u2, u1 >= u2),
                        And(d1 <= d2, u1 <= d2)
                    )
                )
            )

        return
    
    def solve(self, chip: Chip) -> tuple[bool, typing.Any]:
        """
        Solves a chip after encoding
        :param chip:
        :return:
        """
        solved = self._smt_solver.check()

        #define Matplotlib figure and axis
        fig, ax = plt.subplots()
        ax.set(ylim=(0, 30), xlim=(0, 30))

        if solved == sat:
            model = self._smt_solver.model()
            result = {}
            for comp in chip._rcomps:
                # result[comp] = model.evaluate(self._comp_variables[comp])
                #add rectangle to plot
                x = model.evaluate(self._comp_variables[comp][0]).as_long()
                y = model.evaluate(self._comp_variables[comp][1]).as_long()
                h = model.evaluate(self._comp_variables[comp][2]).as_long()
                w = model.evaluate(self._comp_variables[comp][3]).as_long()
                r = model.evaluate(self._comp_variables[comp][4]).as_long()
                u = model.evaluate(self._comp_variables[comp][5]).as_long()
                # print("(x:", x, " ,y:", y, ") - r:", r, " u:", u, " h:", h, " w:", w)
                ax.add_patch(Rectangle((x, y), w, h, color=random.rand(3,), alpha=0.5))

            for comp in chip._pcomps:
                # result[comp] = model.evaluate(self._comp_variables[comp])
                #add rectangle to plot
                x = model.evaluate(self._comp_variables[comp][0]).as_long()
                y = model.evaluate(self._comp_variables[comp][1]).as_long()
                h = model.evaluate(self._comp_variables[comp][2]).as_long()
                w = model.evaluate(self._comp_variables[comp][3]).as_long()
                r = model.evaluate(self._comp_variables[comp][4]).as_long()
                u = model.evaluate(self._comp_variables[comp][5]).as_long()
                # print("(x:", x, " ,y:", y, ") - r:", r, " u:", u, " h:", h, " w:", w)
                ax.add_patch(Rectangle((x, y), w, h, facecolor=numpy.array([1,0,0]), edgecolor=numpy.array([0,0,0]), linewidth=1, alpha=1))

            print(model)
            plt.show()
            return True, result
        return False, None
        
    #     # 1. Each cell should be filled with an allowed value.
    #     for cell in puzzle.cells:
    #         current_value = puzzle.get_value(cell.row, cell.column)
    #         if current_value is None:
    #             self._add_either_value(self._cell_variables[cell], puzzle.allowed_values)
    #         else:
    #             self._smt_solver.add(self._cell_variables[cell] == current_value)
    #     # 2. The values for the cells in every row should be distinct.
    #     for row in puzzle.row_indices:
    #         self._add_all_different(self._row_variables(puzzle, row))
    #     # 3. The values for the cells in every column should be distinct.
    #     for column in puzzle.column_indices:
    #         self._add_all_different(self._column_variables(puzzle, column))
    #     # 4. The values for the cells in every block should be distinct.
    #     for block in puzzle.blocks:
    #         self._add_all_different(self._block_variables(block))
    #     # Note that in standard sudokus, 1+2 encodes that every value must be used in every row.
    #     # Likewise, 1+3 and 1+4 ensure this for columns and blocks, respectively.
    #     # Debugging tip: Here you can export all constraints via print(self._smt_solver)


chip = Chip(30, 30, 2, REGULAR_CHIPS)
smt = SMT()
smt.encode(chip)
isSat, result = smt.solve(chip)
print(isSat)
# smt.show()
# smt.vars()