from typing import List, Iterable
import typing
from z3 import *
from itertools import product as cart_product
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy import random
import numpy
from tabulate import tabulate

# Z3py constraints that solve the following problem: Five couples each living in a separate house want to organize a dinner.
# Since all restaurants are closed due to some lock-down, they will do it in their own houses.
# The dinner will consist of 5 rounds. Due to the 1.5 meter restriction in every house presence
# of at most 5 people is allowed, by which every round has to be prepared and served in two
# houses simultaneously, each with the corresponding couple and three guests. Every couple
# will serve two rounds in their house, and in between the rounds participants may move from
# one house to another.
# Every two people among the 10 participants meet each other at most 4 times during these
# 5 rounds. Further there are four desired properties:
# (A) Every two people among the 10 participants meet each other at least once.
# (B) Every two people among the 10 participants meet each other at most 3 times.
# (C) Couples never meet outside their own houses.
# (D) For every house the six guests (three for each of the two rounds) are distinct; i.e. no
# person can be a guest in the same house twice.
# The problem is to show that:
# 1. Show that (A) is possible, both in combination with (C) and (D), but not with both(C) and (D).
# 2. Show that (B) is possible in combination with both (C) and (D).

# The following code is a solution to the problem above using the z3py API.
# The code is based on the following paper: https://www.cs.ru.nl/bachelors-theses/2019/Sjoerd_Bosch.pdf
# The code is written by Sjoerd Bosch, student number 10217584, for the course Heuristieken at the University of Amsterdam.

s = Solver()

# matrix of boolean variables of houses and people
x = [[[Int("h_%s_p%s_r%s" % (i+1, j+1, k+1)) for i in range(5)]
        for j in range(10)]
        for k in range(5)]

# Every house should house 5 people, every column is has 5 T or 0 T.

# each row contains a digit at most once
# rows_c   = [ X[i] for i in range(9) ]

# A person is in a house (1), or is not is a house (0)
# Or(h_1_p1_r1, h_2_p1_r1, h_3_p1_r1, h_4_p1_r1, h_5_p1_r1)
one_or_zero   = [Or(x[round][people][house] == 0, x[round][people][house] == 1)  for house in range(5) 
                for people in range(10) 
                for round in range(5) ]

# A person must be in at most 1 house a time
only_one_house_at_the_same_time = [Sum([(x[round][people][house]) for house in range(5) ]) == 1
                                        for people in range(10) 
                                        for round in range(5)]

# There are 5 people at one house, or there are none
five_in_a_house = [Or(Sum([(x[round][people][house]) for people in range(10) ]) == 5, Sum([(x[round][people][house])  for people in range(10) ]) == 0)
                                        for house in range(5) 
                                        for round in range(5)]

# 2 rounds at every house
two_rounds_per_house = [Sum([Sum([(x[round][people][house]) for people in range(10) ])
                                        for round in range(5)]) == 10
                                        for house in range(5)]

# A couple serves at their own house
couples_at_their_house_are_there_together = [And(Implies(x[round][people1][house] == 1, x[round][people2][house] == 1),Implies(x[round][people2][house] == 1, x[round][people1][house] == 1)) 
                          for people1 in range(10) 
                          for people2 in range(10)
                          for round in range(5)
                          for house in range(5) 
                          if people1 != people2 and people1 % 2 == 0 and people2 == people1+1 and people1 / 2  == house]

couple_must_be_in_their_house = [Implies(Sum([(x[round][people][house]) for people in range(10) ]) == 5, x[round][house*2][house] == 1)
                                        for house in range(5) 
                                        for round in range(5)]

# 2 people meet at most 4 times
# The rule says that for all house and round combinations. There must be at least one instance where they are not both 1.
# meet_at_most_four_times = [
#     Or([x[round][people1][house] - x[round][people2][house] != 1
#         for house in range(5) 
#         for round in range(5)])
     
#      for people1 in range(10) 
#      for people2 in range(10) 
#      if people1 != people2
#         ]
meet_at_most_four_times = [
        Sum([Abs(x[round][people1][house] - x[round][people2][house])
        for house in range(5) 
        for round in range(5)]) >= 2
     
     for people1 in range(10) 
     for people2 in range(10) 
     if people1 != people2
        ]


# Desirable properties
# (A) Every two people among the 10 participants meet each other at least once.
# meet_at_least_one_time = [
#     Or([x[round][people1][house] - x[round][people2][house] == 0
#         for house in range(5) 
#         for round in range(5)])
     
#      for people1 in range(10) 
#      for people2 in range(10) 
#      if people1 != people2
#         ]
meet_at_least_one_time =  [
        Sum([Abs(x[round][people1][house] - x[round][people2][house])
        for house in range(5) 
        for round in range(5)]) <= 8
     
     for people1 in range(10) 
     for people2 in range(10) 
     if people1 != people2
        ]

# (B) Every two people among the 10 participants meet each other at most 3 times.
meet_at_most_three_times = [
        Sum([Abs(x[round][people1][house] - x[round][people2][house])
        for house in range(5) 
        for round in range(5)]) >= 4
     
     for people1 in range(10) 
     for people2 in range(10) 
     if people1 != people2
        ]

# (C) Couples never meet outside their own houses.
couples_not_meet_outside_house = [And(Implies(x[round][people1][house] == 1, x[round][people2][house] == 0),Implies(x[round][people2][house] == 1, x[round][people1][house] == 0)) 
                          for people1 in range(10) 
                          for people2 in range(10)
                          for round in range(5)
                          for house in range(5) 
                          if people1 != people2 
                                and people1 % 2 == 0 
                                and people2 == people1+1 
                                and  people1 / 2  != house 
                                and (people2 - 1) / 2  != house]

# couple_must_be_in_their_house = [Implies(Sum([(x[round][people][house]) for people in range(10) ]) == 5, x[round][house*2][house] == 1)
#                                         for house in range(5) 
#                                         for round in range(5)]

# (D) For every house the six guests (three for each of the two rounds) are distinct; i.e. no
# person can be a guest in the same house twice.
distinct_guests = [Sum([x[round][people][house] for round in range(5) ]) <= 1
     for people in range(10) 
     for house in range(5) 
     if people / 2  != house and (people - 1) / 2  != house
        ]

# print(couples_not_meet_outside_house)
s = Solver()
s.add(one_or_zero)
s.add(only_one_house_at_the_same_time)
s.add(five_in_a_house)
s.add(two_rounds_per_house)
s.add(couples_at_their_house_are_there_together)
s.add(couple_must_be_in_their_house)
s.add(meet_at_most_four_times)
# Extra requirements
# s.add(meet_at_least_one_time)         # A
# print(meet_at_least_one_time)
s.add(meet_at_most_three_times)       # B
s.add(distinct_guests)                # C
s.add(couples_not_meet_outside_house)   # D

solved = s.check()

if solved == sat:
    model = s.model()
    for round in range(5):
        R = [[0 for x in range(10)] for y in range(5) ]

        for house in range(5):
            for person in range(10):
                R[house][person] = model.evaluate(x[round][person][house]).as_long()

        #define header names
        col_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]

        #display table
        print("Round", round+1)
        print(tabulate(R, headers=col_names, showindex="always"))
        print()
else:
    print("unsat")

