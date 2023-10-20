from z3 import *

# Example 1
# x = Int('x')
# y = Int('y')
# solve(x > 2, y < 10, x + 2*y == 7)

# Example 2
# x = Int('x')
# y = Int('y')
# print (simplify(x + y + 2*x + 3))
# print (simplify(x < y + x + 2))
# print (simplify(And(x + 1 >= 3, x**2 + x**2 + y**2 + 2 >= 5)))

# HTML options (does not seem to work)
# x = Int('x')
# y = Int('y')
# print (x**2 + y**2 >= 1)
# set_option(html_mode=False)
# print (x**2 + y**2 >= 1)

# Formula travesal
# x = Int('x')
# y = Int('y')
# n = x + y >= 3
# print ("num args: ", n.num_args())
# print ("children: ", n.children())
# print ("1st child:", n.arg(0))
# print ("2nd child:", n.arg(1))
# print ("operator: ", n.decl())
# print ("op name:  ", n.decl().name())

# Example 3 - Non linear polynomials
# x = Real('x')
# y = Real('y')
# solve(x**2 + y**2 > 3, x**3 + y < 5)

# Example 4 - real precision
# x = Real('x')
# y = Real('y')
# solve(x**2 + y**2 == 3, x**3 == 2)

# set_option(precision=30)
# print("Solving, and displaying result with 30 decimal places")
# solve(x**2 + y**2 == 3, x**3 == 2)

# Error not z3 real
# print(1/3)
# print(RealVal(1)/3)
# print(Q(1,3))

# x = Real('x')
# print (x + 1/3)
# print (x + Q(1,3))
# print (x + "1/3")
# print (x + 0.25)

# Different notation
# x = Real('x')
# solve(3*x == 1)

# set_option(rational_to_decimal=True)
# solve(3*x == 1)

# set_option(precision=30)
# solve(3*x == 1)

# Unsat
# x = Real('x')
# solve(x > 4, x < 0)

# Some logic
# p = Bool('p')
# q = Bool('q')
# r = Bool('r')
# solve(Implies(p, q), r == Not(q), Or(Not(p), r))

# Combining polynomial and boolean
# p = Bool('p')
# x = Real('x')
# solve(Or(x < 5, x > 10), Or(p, x**2 == 2), Not(p))

# Z3 api
# x = Int('x')
# y = Int('y')

# s = Solver()
# print (s)

# s.add(x > 10, y == x + 2)
# print (s)
# print ("Solving constraints in the solver s ...")
# print (s.check())

# print ("Create a new scope...")
# s.push()
# s.add(y < 11)
# print (s)
# print ("Solving updated set of constraints...")
# print (s.check())

# print ("Restoring state...")
# s.pop()
# print (s)
# print ("Solving restored set of constraints...")
# print (s.check())

# Not a polynomial
# x = Real('x')
# s = Solver()
# s.add(2**x == 3)
# print (s.check())

# Statistics
# x = Real('x')
# y = Real('y')
# s = Solver()
# s.add(x > 1, y > 1, Or(x + y > 3, x - y < 2))
# print("asserted constraints...")
# for c in s.assertions():
#     print(c)

# print(s.check())
# print("statistics for the last check method...")
# print(s.statistics())
# # Traversing statistics
# for k, v in s.statistics():
#     print (k, " : ", v)

# model
# x, y, z = Reals('x y z')
# s = Solver()
# s.add(x > 1, y > 1, x + y > 3, z - x < 10)
# print (s.check())

# m = s.model()
# print ("x = %s" % m[x])

# print("traversing model...")
# for d in m.decls():
#     print ("%s = %s" % (d.name(), m[d]))


# Reals and ints
# x = Real('x')
# y = Int('y')
# a, b, c = Reals('a b c')
# s, r = Ints('s r')
# print (x + y + 1 + (a + s))
# print (ToReal(y) + c)

# a, b, c = Ints('a b c')
# d, e = Reals('d e')
# solve(a > b + 2,
#       a == 2*c + 10,
#       c + b <= 1000,
#       d >= e)

# x, y = Reals('x y')
# # Put expression in sum-of-monomials form
# t = simplify((x + y)**3, som=True)
# print(t)
# # Use power operator
# t = simplify(t, mul_to_power=True)
# print(t)

# x, y = Reals('x y')
# # Using Z3 native option names
# print (simplify(x == y + 2, ':arith-lhs', True))
# # Using Z3Py option names
# print (simplify(x == y + 2, arith_lhs=True))

# print ("\nAll available options:")
# help_simplify()

# lisp?
# x, y = Reals('x y')
# solve(x + 10000000000000000000000 == y, y > 20000000000000000)

# print (Sqrt(2) + Sqrt(3))
# print (simplify(Sqrt(2) + Sqrt(3)))
# print (simplify(Sqrt(2) + Sqrt(3)).sexpr())
# # The sexpr() method is available for any Z3 expression
# print ((x + Sqrt(y) * 2).sexpr())

# bitvec
# x = BitVec('x', 16)
# y = BitVec('y', 16)
# print (x + 2)
# # Internal representation
# print ((x + 2).sexpr())

# # -1 is equal to 65535 for 16-bit integers
# print (simplify(x + y - 1))

# # Creating bit-vector constants
# a = BitVecVal(-1, 16)
# b = BitVecVal(65535, 16)
# print (simplify(a == b))

# a = BitVecVal(-1, 32)
# b = BitVecVal(65535, 32)
# # -1 is not equal to 65535 for 32-bit integers
# print (simplify(a == b))

# signed and unsigned
# # Create to bit-vectors of size 32
# x, y = BitVecs('x y', 32)

# solve(x + y == 2, x > 0, y > 0)

# # Bit-wise operators
# # & bit-wise and
# # | bit-wise or
# # ~ bit-wise not
# solve(x & y == ~y)

# solve(x < 0)

# # using unsigned version of <
# solve(ULT(x, 0))

# # Create to bit-vectors of size 32
# x, y = BitVecs('x y', 32)

# solve(x >> 2 == 3)

# solve(x << 2 == 3)

# solve(x << 2 == 24)


# uninterpreted functions
# x = Int('x')
# y = Int('y')
# f = Function('f', IntSort(), IntSort())
# solve(f(f(x)) == x, f(x) == y, x != y)

# evaluate
# x = Int('x')
# y = Int('y')
# f = Function('f', IntSort(), IntSort())
# s = Solver()
# s.add(f(f(x)) == x, f(x) == y, x != y)
# print (s.check())
# m = s.model()
# print ("f(f(x)) =", m.evaluate(f(f(x))))
# print ("f(x)    =", m.evaluate(f(x)))

# demorgan
# p, q = Bools('p q')
# demorgan = And(p, q) == Not(Or(Not(p), Not(q)))
# print (demorgan)

# def prove(f):
#     s = Solver()
#     s.add(Not(f))
#     if s.check() == unsat:
#         print ("proved")
#     else:
#         print ("failed to prove")

# print ("Proving demorgan...")
# prove(demorgan)

# list comprehension
# Create list [1, ..., 5]
# print ([ x + 1 for x in range(5) ])

# # Create two lists containg 5 integer variables
# X = [ Int('x%s' % i) for i in range(5) ]
# Y = [ Int('y%s' % i) for i in range(5) ]
# print (X)

# # Create a list containing X[i]+Y[i]
# X_plus_Y = [ X[i] + Y[i] for i in range(5) ]
# print (X_plus_Y)

# # Create a list containing X[i] > Y[i]
# X_gt_Y = [ X[i] > Y[i] for i in range(5) ]
# print (X_gt_Y)

# print (And(X_gt_Y))

# # Create a 3x3 "matrix" (list of lists) of integer variables
# X = [ [ Int("x_%s_%s" % (i+1, j+1)) for j in range(3) ]
#       for i in range(3) ]
# pp(X)

# something lists?
# X = IntVector('x', 5)
# Y = RealVector('y', 5)
# P = BoolVector('p', 5)
# print (X)
# print (Y)
# print (P)
# print ([ y**2 for y in Y ])
# print (Sum([ y**2 for y in Y ]))

# kinematic problem 1
# d, v_i, t, a, v_f = Reals("d v_i t a v_f")
# s = Solver()
# print (s)
# s.add(a == -8)
# s.add(v_i == 30)
# s.add(d == v_i * t + (a * t**2) / 2)
# s.add(v_f == v_i + a*t)
# print (s.check())

# m = s.model()
# print(m)
# print ("d = %s" % m[d])


# d, a, t, v_i, v_f = Reals('d a t v__i v__f')

# equations = [
#    d == v_i * t + (a*t**2)/2,
#    v_f == v_i + a*t,
# ]
# print ("Kinematic equations:")
# print (equations)

# # Given v_i, v_f and a, find d
# problem = [
#     v_i == 30,
#     v_f == 0,
#     a   == -8
# ]
# print ("Problem:")
# print (problem)

# print ("Solution:")
# solve(equations + problem)


# # problem 2
# d, v_i, t, a, v_f = Reals("d v_i t a v_f")
# s = Solver()
# print (s)
# s.add(a == 6)
# s.add(v_i == 0)
# s.add(t == 4.1)
# s.add(d == v_i * t + (a * t**2) / 2)
# s.add(v_f == v_i + a*t)
# print (s.check())

# m = s.model()
# print(m)
# print ("d = %s" % m[d])

# d, a, t, v_i, v_f = Reals('d a t v__i v__f')

# equations = [
#    d == v_i * t + (a*t**2)/2,
#    v_f == v_i + a*t,
# ]

# # Given v_i, t and a, find d
# problem = [
#     v_i == 0,
#     t   == 4.10,
#     a   == 6
# ]
# # print(equations + problem) just adds the 2 list together
# solve(equations + problem)

# # Display rationals in decimal notation
# set_option(rational_to_decimal=True)

# solve(equations + problem)


# C hack
# x      = BitVec('x', 32)
# powers = [ 2**i for i in range(32) ]
# fast   = And(x != 0, x & (x - 1) == 0)
# slow   = Or([ x == p for p in powers ])
# print (fast)
# prove(fast == slow)

# print ("trying to prove buggy version...")
# fast   = x & (x - 1) == 0
# prove(fast == slow)

# Opposite signs
# x      = BitVec('x', 32)
# y      = BitVec('y', 32)

# # Claim: (x ^ y) < 0 iff x and y have opposite signs
# trick  = (x ^ y) < 0

# # Naive way to check if x and y have opposite signs
# opposite = Or(And(x < 0, y >= 0),
#               And(x >= 0, y < 0))

# prove(trick == opposite)

# cats, dogs, mice = Ints('cats dogs mice')
# s = Solver()
# s.add(cats * 1 + dogs * 15 + mice * 0.25 == 100)
# s.add(cats + dogs  + mice  == 100)
# s.add(cats >=  1)
# s.add(dogs >= 1)
# s.add(mice >= 1)

# s.check()
# m = s.model()
# print(m)

# # Create 3 integer variables
# dog, cat, mouse = Ints('dog cat mouse')
# solve(dog >= 1,   # at least one dog
#       cat >= 1,   # at least one cat
#       mouse >= 1, # at least one mouse
#       # we want to buy 100 animals
#       dog + cat + mouse == 100,
#       # We have 100 dollars (10000 cents):
#       #   dogs cost 15 dollars (1500 cents),
#       #   cats cost 1 dollar (100 cents), and
#       #   mice cost 25 cents
#       1500 * dog + 100 * cat + 25 * mouse == 10000)

# sudoku

# # Create a 9x9 "matrix" (list of lists) of integer variables
# s = Solver()
# X = [ [ Int("x_%s_%s" % (j+1, i+1)) for j in range(9) ]
#       for i in range(9) ]
# # pp(X)

# for idx, i in enumerate(X):
#     for jdx, j in enumerate(i):
#         # print(j)
#         s.add(j >= 1, j <= 9) # Add a [1,9] restriction to all vals
#         for xdx, x in enumerate(i):
#             if jdx < xdx:
#                 # print("J", j, " - ", "X", x)
#                 s.add(j != x) # Add a "not the same as row" restriction
#         for ydx, y in enumerate(X):
#             if idx < ydx:
#                 s.add(j != y[jdx])
#                 # print("J", j, " - ", "y[jdx]", y[jdx])

# for cy in range(3):
#     for cx in range(3):
#         for y1 in range(1,4):
#             for x1 in range(1,4):
#                 for y2 in range(1,4):
#                     for x2 in range(1,4):
#                         if x1 + y1*3 < x2 + y2*3:
#                             a = X[(x1+cx*3)-1][(y1+cy*3)-1]
#                             b = X[(x2+cx*3)-1][(y2+cy*3)-1]
#                             # print(a, " - ", b)
#                             s.add(a != b)

# # sudoku instance, we use '0' for empty cells
# instance = ((0,0,0,0,9,4,0,3,0),
#             (0,0,0,5,1,0,0,0,7),
#             (0,8,9,0,0,0,0,4,0),
#             (0,0,0,0,0,0,2,0,8),
#             (0,6,0,2,0,1,0,5,0),
#             (1,0,2,0,0,0,0,0,0),
#             (0,7,0,0,0,0,5,2,0),
#             (9,0,0,0,6,5,0,0,0),
#             (0,4,0,9,7,0,0,0,0))

# instance_c = [ If(instance[i][j] == 0,
#                   True,
#                   X[i][j] == instance[i][j])
#                for i in range(9) for j in range(9) ]

# s.add(instance_c)
# if s.check() == sat:
#     m = s.model()
#     r = [ [ m.evaluate(X[i][j]) for j in range(9) ]
#           for i in range(9) ]
#     print_matrix(r)
# else:
#     print ("failed to solve")


# # 9x9 matrix of integer variables
# X = [ [ Int("x_%s_%s" % (i+1, j+1)) for j in range(9) ]
#       for i in range(9) ]

# # each cell contains a value in {1, ..., 9}
# cells_c  = [ And(1 <= X[i][j], X[i][j] <= 9)
#              for i in range(9) for j in range(9) ]

# # each row contains a digit at most once
# rows_c   = [ Distinct(X[i]) for i in range(9) ]

# # each column contains a digit at most once
# cols_c   = [ Distinct([ X[i][j] for i in range(9) ])
#              for j in range(9) ]

# # each 3x3 square contains a digit at most once
# sq_c     = [ Distinct([ X[3*i0 + i][3*j0 + j]
#                         for i in range(3) for j in range(3) ])
#              for i0 in range(3) for j0 in range(3) ]

# sudoku_c = cells_c + rows_c + cols_c + sq_c

# # sudoku instance, we use '0' for empty cells
# instance = ((0,0,0,0,9,4,0,3,0),
#             (0,0,0,5,1,0,0,0,7),
#             (0,8,9,0,0,0,0,4,0),
#             (0,0,0,0,0,0,2,0,8),
#             (0,6,0,2,0,1,0,5,0),
#             (1,0,2,0,0,0,0,0,0),
#             (0,7,0,0,0,0,5,2,0),
#             (9,0,0,0,6,5,0,0,0),
#             (0,4,0,9,7,0,0,0,0))

# instance_c = [ If(instance[i][j] == 0,
#                   True,
#                   X[i][j] == instance[i][j])
#                for i in range(9) for j in range(9) ]

# print("class solution:")
# s = Solver()
# s.add(sudoku_c + instance_c)
# if s.check() == sat:
#     m = s.model()
#     r = [ [ m.evaluate(X[i][j]) for j in range(9) ]
#           for i in range(9) ]
#     print_matrix(r)
# else:
#     print ("failed to solve")


# 9x9 matrix of boolean variables
X = [[Bool("x_%s_%s" % (i+1, j+1)) for j in range(8)]
     for i in range(8)]

at_least_row = [Or(X[i]) for i in range(8)]
at_least_col = [Or([X[i][j] for i in range(8)])
                for j in range(8)]
at_most_diag = [And([Or(Not(X[i][j]), Not(X[k][l]))
                     for i in range(8) for j in range(8) for k in range(8) for l in range(8) if ((i+j == k+l) or (i-j == k-l)) and (i != k and j != l)
                     ])]

at_most_row = [And([Or(Not(X[i][j]), Not(X[i][k])) for j in range(8) for k in range(8) if (j != k)])
               for i in range(8)]
at_most_col = [And([Or(Not(X[j][i]), Not(X[k][i])) for j in range(8) for k in range(8) if (j != k)])
               for i in range(8)]

# at_most_row  = [ And([Or(Not(X[i][j]), Not(X[i][k])) for j in range(8) for k in range(8)])
#              for i in range(8)]
# at_most_col = [ And([Or(Not(X[j][i]), Not(X[k][i])) for j in range(8) for k in range(8)])
#              for i in range(8)]

eight_queens = at_least_col + at_least_row + \
    at_most_row + at_most_col + at_most_diag
s = Solver()
s.add(eight_queens)
if s.check() == sat:
    m = s.model()
    r = [[m.evaluate(X[i][j]) for j in range(8)] for i in range(8)]
#     print_matrix(r)
    for i in r:
        print("")
        for j in i:
            # print(i)
            if j == True:
                print("Q ", end="")
            else:
                print(". ", end="")
else:
    print("failed to solve")

# a = [And(Or(11, 12), Or(11,13), Or(11,14), Or(12, 13), Or(12,14)),
#      And(Or(21, 22), Or(21,23), Or(21,24), Or(22, 23), Or(22,24))]

# print("row:")
# pp(at_least_row)
# print("col:")
# pp(at_least_col)
# print("most_row:")
# pp(at_most_row)
# print("most_col:")
# pp(at_most_col)
# print("most_diag:")
# pp(at_most_diag)
