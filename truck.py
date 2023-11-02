from z3 import *
from tabulate import tabulate

# 8 trucks - max 8000kg - 8 pallets - 3 cooled

# To be delivered
# - 4x 700kg (nuzzles, distributed over trucks) 
# - ?x 400kg (prittles)
# - 8x 1000kg (skipples, cooled)
# - 10x 2500kg (crottles)
# - 20x 200kg (dupples)

#                T1  T2  T3  T4  T5  T6  T7  T8
# Nuzzles (1)    0   0   0   0   0   0   0   0
# Prittles (2)   0   0   0   0   0   0   0   0
# Skipples (3)   0   0   0   0   0   0   0   0
# Crottles (4)   0   0   0   0   0   0   0   0
# Dupples (5)    0   0   0   0   0   0   0   0


# 8x5 matrix of int variables
distribution = [[Int("T%s_%s" % (t+1, i+1)) for i in range(5)]
    for t in range(8)]
trucks = [Int("Truck_%s" % (i+1)) for i in range(8)]
def create_solver(e):
    s = Solver()

    truck_weight_limit_min = [trucks[i] >= 0 for i in range(8)] # truck min capacity of 0 kg
    truck_weight_limit_max = [trucks[i] <= 8000 for i in range(8)] # truck max capacity of 8000 kg


    distribution_min = [distribution[t][i] >= 0 for i in range(5)
        for t in range(8)]
    truck_weight = [(trucks[i] == 
                    distribution[i][0] * 700 + 
                    distribution[i][1] * 400 +  # type: ignore
                    distribution[i][2] * 1000 + 
                    distribution[i][3] * 2500 + 
                    distribution[i][4] * 200) for i in range(8) ] # truck capacity of 8000 kg

    sum_nuzzles = [Sum([distribution[i][0] for i in range(8)]) == 4]
    # sum_prittles = [Sum([distribution[i][1] for i in range(8)]) == 4]
    sum_skipples = [Sum([distribution[i][2] for i in range(8)]) == 8]
    sum_crottles = [Sum([distribution[i][3] for i in range(8)]) == 10]
    sum_dupples = [Sum([distribution[i][4] for i in range(8)]) == 20]
    sum_skipples_cooled = [Sum([distribution[i][2] for i in range(5)]) == 0]
    sum_nuzzles_max_one = [distribution[i][0] <= 1 for i in range(8)]
    at_most_eight_pallets = [Sum([distribution[t][p] for p in range(5)]) <= 8 for t in range(8)]

    s.add(distribution_min) 
    s.add(truck_weight_limit_min) 
    s.add(truck_weight_limit_max) 
    s.add(truck_weight) 
    s.add(sum_nuzzles) 
    s.add(sum_skipples) 
    s.add(sum_crottles) 
    s.add(sum_dupples) 
    s.add(sum_skipples_cooled) 
    s.add(sum_nuzzles_max_one)
    s.add(at_most_eight_pallets)

    if e == True:
        explosive = [Or(distribution[i][1] == distribution[i][1] + distribution[i][3], distribution[i][3] == distribution[i][1] + distribution[i][3])   for i in range(8)]
        s.add(explosive)
    return s


def check_nr_of_prittles(nr_prittles, m):
    s.push()
    s.add([Sum([distribution[i][1] for i in range(8)]) == nr_prittles])
    if s.check() == sat:
        m = s.model()
        nr_prittles += 1
        s.pop()
        check_nr_of_prittles(nr_prittles, m)
    else:
        s.pop()
        print("Succeeded with ", nr_prittles-1 ," prittles")
        # print(m)
    
        R = [[0 for t in range(8)] for p in range(6) ]

        for product in range(5):
            for truck in range(8):
                R[product][truck] = m.evaluate(distribution[truck][product]).as_long()
        for weight in range(8):
            R[5][weight] = m.evaluate(trucks[weight]).as_long()

        #define header names
        col_names = ["t1", "t2", "t3", "t4", "t5", "t6 (c)", "t7 (c)", "t8 (c)"]

        #display table
        print(tabulate(R, headers=col_names, showindex="always"))
        print()

s = create_solver(False)
check_nr_of_prittles(0, None)

# Investigate what is the maximum number of pallets of prittles that can be delivered,
# and show how for that number all pallets may be divided over the eight trucks

#                   T1     T2     T3    T4      T5      T6(c)   T7(c)   T8(c)   Total
# (1) Nuzzles       0      0      1     1       1       1       0       0       4
# (2) Prittles      7      1      18    12      3       12      0       7       60 
# (3) Skipples (c)  0      0      0     0       0       0       8       0       8
# (4) Crottles      2      3      0     1       1       1       0       2       10
# (5) Dupples       1      0      0     0       18      0       0       1       20
# Total weight      8000   7900   8000  8000    7900    8000    8000    8000    

# Do the same, with the extra information that prittles and crottles are an explosive
# combination: they are not allowed to be put in the same truck.

s = create_solver(True)
check_nr_of_prittles(0, None)

#                   T1     T2     T3    T4      T5      T6(c)   T7(c)   T8(c)   Total
# (1) Nuzzles       1      0      0     1       1       0       1       0       4
# (2) Prittles      18     0      0     18      0       5       18      0       59
# (3) Skipples (c)  0      0      0     0       0       5       0       3       8
# (4) Crottles      0      3      3     0       2       0       0       2       10
# (5) Dupples       0      2      2     0       11      5       0       0       20
# Total weight      7900   7900   7900  7900    7900    8000    7900    8000    
