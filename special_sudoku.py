###############################################################################
# This is an example program handling the input and output for the Sudoku     #
# assignment.  It is up to you to fill in the middle part. :)                 #
# (You may also use your own reading/writing if you prefer.)                  #
###############################################################################

from z3 import *

import sys
import csv

########## read input ##########

inputfile = "sudoku.csv"
if len(sys.argv) == 2:
    inputfile = sys.argv[1]

lines = []
with open(inputfile, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lines.append(row)

# line 1: ships
ships = [int(w) for w in lines[0]]
ships_dict = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
for ship in ships:
    ships_dict[ship] = ships_dict[ship] + 1
    

# line 2-10: clues and colours
clues = [[int(lines[i+1][j][1]) for j in range(9)] for i in range(9)]
colours = [[lines[i+1][j][0] for j in range(9)] for i in range(9)]

########## determining the correct solution! ##########

# uncomment to see the shape of the ships list
print("ships:", ships)
print("ships_dict:",ships_dict)

# uncomment to see the shape of the clues matrix
print("clues")
print_matrix(clues)

# uncomment to see the shape of the colours matrix
print("colours")
print_matrix(colours)

# YOUR CODE GOES HERE
solver = Solver()

S = [[Int("s_%s_%s" % (i, j)) for j in range(9)]  # Sudoku
     for i in range(9)]
B = [[Bool("b_%s_%s" % (i, j)) for j in range(9)]  # Ships
     for i in range(9)]


def ship_cs_gen(x, y):
    constraints = []
    # above
    # print("x:",x)
    # print("y:",y)
    # print(S[x][y])
    if y != 0:
        # If the cel above is consecutive, then it is a boat, and next to the boat cannot be another boat.
        constraints.append(
            (Abs(S[x][y] - S[x][y-1]) == 1) == (And(B[x][y] == True, B[x][y-1] == True))
        )
        if x != 0:
          constraints.append(
              Implies(Abs(S[x][y] - S[x][y-1]) == 1, And(B[x-1][y] == False, B[x-1][y-1] == False))
          )
        if x != 8:
          constraints.append(
              Implies(Abs(S[x][y] - S[x][y-1]) == 1, And(B[x+1][y] == False, B[x+1][y-1] == False))
          )

    if y != 8 :
        # If the cel above is consecutive, then it is a boat, and next to the boat cannot be another boat.
        constraints.append(
            (Abs(S[x][y] - S[x][y+1]) == 1) == (And(B[x][y] == True, B[x][y+1] == True))
        )
        if x != 0:
          constraints.append(
              Implies(Abs(S[x][y] - S[x][y+1]) == 1, And(B[x-1][y] == False, B[x-1][y+1] == False))
          )
        if x != 8:
          constraints.append(
              Implies(Abs(S[x][y] - S[x][y+1]) == 1, And(B[x+1][y] == False, B[x+1][y+1] == False))
          )

    if x != 0 :
        # If the cel above is consecutive, then it is a boat, and next to the boat cannot be another boat.
        constraints.append(
            (Abs(S[x][y] - S[x-1][y]) == 1) == (And(B[x][y] == True, B[x-1][y] == True))
        )
        if y != 0:
          constraints.append(
              Implies(Abs(S[x][y] - S[x-1][y]) == 1, And(B[x][y-1] == False, B[x-1][y-1] == False))
          )
        if y != 8:
          constraints.append(
              Implies(Abs(S[x][y] - S[x-1][y]) == 1, And(B[x][y+1] == False, B[x-1][y+1] == False))
          )

    if x != 8 :
        # If the cel above is consecutive, then it is a boat, and next to the boat cannot be another boat.
        constraints.append(
            (Abs(S[x][y] - S[x+1][y]) == 1) == (And(B[x][y] == True, B[x+1][y] == True))
        )
        if y != 0:
          constraints.append(
              Implies(Abs(S[x][y] - S[x+1][y]) == 1, And(B[x][y-1] == False, B[x+1][y-1] == False))
          )
        if y != 8:
          constraints.append(
              Implies(Abs(S[x][y] - S[x+1][y]) == 1, And(B[x][y+1] == False, B[x+1][y+1] == False))
          )
          
    return (And(constraints))

# works for 2-7
def ship_nr_constraint_gen(ship_length):
    nr_of_ships = ships_dict[ship_length]
    constraints = []
    if ship_length != 9:
      # check left
      constraints.append(Sum([If(And(And([B[i][j] for i in range(ship_length)]), Not(B[ship_length][j])),1,0) for j in range(9)]))
      # check middle 
      if ship_length != 8:
        constraints.append(Sum([If(And(Not(B[0+k][j]), And([B[i+k+1][j] for i in range(ship_length)]), Not(B[ship_length+k+1][j])),1,0) for j in range(9) for k in range(8-ship_length)]))
      # check right
      constraints.append(Sum([If(And(And([B[i][j] for i in range(9-ship_length, 9)]), Not(B[8-ship_length][j])),1,0) for j in range(9)]))

      # check up
      constraints.append(Sum([If(And(And([B[j][i] for i in range(ship_length)]), Not(B[j][ship_length])),1,0) for j in range(9)]))
      # check middle 
      if ship_length != 8:
        constraints.append(Sum([If(And(Not(B[j][0+k]), And([B[j][i+k+1] for i in range(ship_length)]), Not(B[j][ship_length+k+1])),1,0) for j in range(9) for k in range(8-ship_length)]))
      # check right
      constraints.append(Sum([If(And(And([B[j][i] for i in range(9-ship_length, 9)]), Not(B[j][8-ship_length])),1,0) for j in range(9)]))
    else:
      constraints.append(Sum([If(And(B[i]),1,0) for i in range(9)]))
      constraints.append(Sum([If(And([B[i][j]for i in range(9)]) ,1,0) for j in range(9)]))
    return Sum(constraints) == nr_of_ships


def check_around(x, y):
    constraints = []
    if x != 0:
        constraints.append(B[x-1][y] == False)
    if x != 8:
        constraints.append(B[x+1][y] == False)
    if y != 0:
        constraints.append(B[x][y-1] == False)
    if y != 8:
        constraints.append(B[x][y+1] == False)

    return Implies(B[x][y] == True, Not(And(constraints)))

    

# Lock in clues to z3 sudoku variables
lock_in_clues = [ Implies(Not(clues[i][j] == 0), S[j][i] == clues[i][j]) for i in range(9) for j in range(9)]
print(lock_in_clues)
solver.add(lock_in_clues)

# Lock in colours to z3 ship variable
for (y, c1) in enumerate(colours):
    for (x, c2) in enumerate(c1):
        # print(c2)
        if c2 == 'b':
            cs = B[x][y] == False
            # print(cs)
            solver.add(cs)
        if c2 == 'y':
            cs = B[x][y] == True
            # print(cs)
            solver.add(cs)

# Every cell, only numbers 1 through 9
one_through_nine = [And(S[i][j] >= 1, S[i][j] <= 9)
                    for i in range(9)
                    for j in range(9)]
# print(one_through_nine)
solver.add(one_through_nine)

# Every number in a column is distinct
distinct_column = [S[i][j1] != S[i][j2]
                   for i in range(9)
                   for j1 in range(9)
                   for j2 in range(0, j1)]
# print(distinct_column)
solver.add(distinct_column)

# Every number in a row is distinct
distinct_row = [S[i1][j] != S[i2][j]
                for j in range(9)
                for i1 in range(9)
                for i2 in range(0, i1)]
# print(distinct_row)
solver.add(distinct_row)

# Every number at least once in a 3x3
three_by_three = [Or([S[i1+i2*3][j1+j2*3] == v
                      for i1 in range(3)
                      for j1 in range(3)])
                  for i2 in range(3)
                  for j2 in range(3)
                  for v in range(1, 10)]
# print(three_by_three)
solver.add(three_by_three)

# Numbers are consecutive iff there is a ship
consecutive_numbers = [ship_cs_gen(i, j)
                       for i in range(9)
                       for j in range(9)]
# print(consecutive_numbers)
solver.add(consecutive_numbers)


# Check the number of total ships.
count_ships = [ship_nr_constraint_gen(ship_length)
                       for ship_length in range(2,10)]
# print(count_ships)
solver.add(count_ships)

# No ships of length one.
no_ships_of_one = [check_around(i, j)
                       for i in range(9)
                       for j in range(9)]
# print(no_ships_of_one)
solver.add(no_ships_of_one)



# Remove the following -- this is just to set up variables for the output part
# solved = True

# answer = [ [3,6,1,2,7,5,8,4,9],
#            [7,2,5,9,4,8,3,6,1],
#            [9,4,8,3,6,1,5,2,7],
#            [2,9,4,7,3,6,1,8,5],
#            [8,5,6,1,9,2,4,7,3],
#            [1,7,3,8,5,4,2,9,6],
#            [5,3,7,4,2,9,6,1,8],
#            [4,1,9,6,8,3,7,5,2],
#            [6,8,2,5,1,7,9,3,4] ]

# ship = [ [False,False,True, True, False,False,False,False,False],
#          [False,False,False,False,False,False,False,False,False],
#          [False,False,False,False,False,False,False,False,False],
#          [False,False,False,False,False,False,False,True, False],
#          [False,True, True, False,False,False,False,True, False],
#          [False,False,False,False,True, True, False,False,False],
#          [True, False,False,False,False,False,True, False,False],
#          [True, False,False,True, False,False,True, False,False],
#          [False,False,False,True, False,False,False,True, True ] ]

########## output ##########
isSat = solver.check()
if isSat == sat:
    print("Solution found.")
    m = solver.model()
    for y in range(9):
        for x in range(9):
            print(m.evaluate(S[x][y]), end='')
            # print(m.evaluate(B[x][y]))
            if m.evaluate(B[x][y]) == True:
                print('*', end='')
            else:
                print(' ', end='')
            if (x == 2) | (x == 5):
                print('|', end='')
            else:
                print(' ', end='')
        print()
        if (y == 2) | (y == 5):
            print('--------+--------+--------')
else:
    print("No solution could be found.")
