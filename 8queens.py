from z3 import *

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
