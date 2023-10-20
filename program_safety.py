from typing import List, Iterable
import typing
from z3 import *
from itertools import product as cart_product
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy import random
import numpy

total_steps = 43
A = [ Int("a_%s" % (i+1)) for i in range(total_steps) ] 
B = [ Int("b_%s" % (i+1)) for i in range(total_steps) ] 
I = [ Int("i_%s" % (i+1)) for i in range(total_steps) ] 
T = [ Bool("t_%s" % (i+1)) for i in range(total_steps) ] 
N = Int('n')
C = [ Bool("c_%s" % (i+1)) for i in range(total_steps) ]

def assignment(var, val, s):
    res = []
    if type(val) == int:
        res.append(var[s] == val)
    else:
        res.append(var[s] == val[s-1])
    r, s =  step(s, var)
    return (res + r, s)

def plus(var, val1, val2, s):
    res = []
    # print("plus: ", s)
    if type(val1) == int and type(val2) == int:
        res.append(var[s] == val1 + val2)
    elif type(val2) == int:
        res.append(var[s] == val1[s-1] + val2) # type: ignore
    elif type(val1) == int and type(val2) == ArithRef:
        res.append(var[s] == val1 + val2)
    elif type(val1) == int: 
        res.append(var[s] == val1 + val2[s-1])
    elif type(val2) == ArithRef:
        res.append(var[s] == val1[s-1] + val2) # type: ignore
    else:
        res.append(var[s] == val1[s-1] + val2[s-1]) # type: ignore

    r, s = step(s, var)
    return (res + r, s)

def mul(var, val1, val2, s):
    # print("mul: ", s)
    res = []
    if type(val1) == int and type(val2) == int:
        res.append(var[s] == val1 * val2)
    elif type(val2) == int:
        res.append(var[s] == val1[s-1] * val2) # type: ignore
    elif type(val1) == int and type(var) == ArithRef:
        res.append(var == val1 * val2[s-1])
    elif type(val1) == int:
        res.append(var[s] == val1 * val2[s-1])
    else:
        res.append(var[s] == val1[s-1] * val2[s-1]) # type: ignore
    r, s = step(s, var)
    
    return (res + r, s)


def step(s, var=None):
    res = []
    if s != 0:
        if var is not A:
            res.append(A[s] == A[s -1])
        if var is not B:
            res.append(B[s] == B[s -1])
        if var is not I:
            res.append(I[s] == I[s -1])
        # if var is not T:
        #     res.append(T[s] == T[s -1])
        if var is not C:
            res.append(C[s] == C[s -1])
    else:
        if var is not A:
            res.append(A[s] == 0)
        if var is not B:
            res.append(B[s] == 0)
        if var is not I:
            res.append(I[s] == 0)
        if var is not C:
            res.append(C[s] == False)
        # if var != T:
        #     res.append(T[s] == False)
    s += 1
    return (res, s)

def ifstatement(s):
    # print("if")
    #     if ? then                 # non deterministic 
    #         a := a + 2b;          (3i + 1)
    #         b := b + i;           (3i + 2)
    temp = Int("temp_%s" % (s))
    m, s = mul(temp, 2, B, s)
    p, s = plus(A, A, temp, s)

    p2, s = plus(B, B, I, s)
    return (And(m + p + p2), s)

def elsestatement(s):
#     else 
#         b := a + b;           (3i)
#         a := a + i;           (3i + 1)
#         skip;                 (3i + 2)
    p, s = plus(B, A, B, s)
    p2, s = plus(A, A, I, s)
    res, s = step(s)
    return (And(p + p2 + res), s)

solver = Solver()
s = 0
r1, s = assignment(A, 1, s)
solver.add(r1)
print(r1)
r2, s = assignment(B, 1, s)
solver.add(r2)
print(r2)
for i in range(1,11):
    # print("inloop:", i)
    res, s = assignment(I, i, s)
    print(res)
    solver.add(res)
    ifres, s = ifstatement(s)
    s -= 3
    elseres, s = elsestatement(s)
    print(If(T[s], ifres, elseres))
    solver.add(If(T[s], ifres, elseres))

# r3, s = plus(B, 700, N, s)
# solver.add(r3)
# print(r3)
r4 = If(B[41] == 700 + N, C[42] == True, C[42] == False)
solver.add(r4)
print(r4)

r5 = Not(C[42] == False)
print(r5)
solver.add(r5) # postcondition



# S:
# a = 1;                        (1)
# b = 1;                        (2)
# for i := 1; i <= 10           (4i - 1)
#     if ? then                 # non deterministic 
#         temp := 2*b;          (4i)
#         a := a + temp;        (4i + 1)
#         b := b + i;           (4i + 2)
#     else 
#         b := a + b;           (4i)
#         a := a + i;           (4i + 1)
#         skip;                 (4i + 2)
# if b == 700 + n then crash    (4i + 3)
# 43 steps total

# Precond: True
# Postcond: Not(crash == False)

n = 1
for _ in range(10):
    solver.push()
    solver.add(N == n)
    print("Checking n = ", n)
    isSat = solver.check()
    if isSat == sat:
        print("n:", n, "is sat") # this means an unsafe sequence was found
        model = solver.model()
        bval = model.evaluate(B[41])
        print("bval:", bval)
        for t in T:
            print(model.evaluate(t))
        solver.pop()
        n += 1
    else:
        solver.pop()
        n += 1
