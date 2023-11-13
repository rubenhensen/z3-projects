from z3 import *

import sys
import csv

import timeit

start = timeit.default_timer()

########## read input ##########

inputfile = "seminar.csv"
if len(sys.argv) == 2:
  inputfile = sys.argv[1]

lines = []
with open(inputfile, newline='') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row in reader:
    lines.append(row)

books = [ x.strip().split(':')[0] for x in lines[1][1:] ]
students = [ line[0] for line in lines ]
professors = list(set([ p[0] for p in books ]))
preferences = [ [ int(x.strip().split(':')[1]) for x in line[1:] ] for line in lines ]

s = Solver()
######### some functions you may find useful ##########

def rank_by_id(student_id, book_id):
  return preferences[student_id][book_id]

def rank(student_name, book_name):
  sid = students.index(student_name)
  pid = books.index(book_name)
  return preferences[sid][pid]

def books_for(professor):
  return [ p for p in books if p[0] == professor ]

# matrix of boolean variables of books, professors and students
x = [[[Bool("%s_%s_%s" % (student, prof, book)) for book in books]
        for prof in professors ]
        for student in students]

# every_cell_one_or_zero = [Or(x[student][prof][book] == 0, x[student][prof][book] == 1) 
#                           for book in range(len(books)) 
#                           for prof in range(len(professors))
#                           for student in range(len(students))]
# print(every_cell_one_or_zero)
# s.add(every_cell_one_or_zero)

# Every student must have one book
one_book_per_student = [Sum([If(x[student][prof][book],1,0) for book in range(len(books)) for prof in range(len(professors))]) == 1
        for student in range(len(students))]
# print("one_book_per_student\n")
# print(one_book_per_student[0])
s.add(one_book_per_student)

# Every book can be used only once.
every_student_a_different_book = [Sum([If(x[student][prof][book],1,0)  for prof in range(len(professors)) for student in range(len(students))]) <= 1
        for book in range(len(books))]

# print("\nevery_student_a_different_book")
# print(every_student_a_different_book[0])
s.add(every_student_a_different_book)

########## print the solution ##########
if s.check() == sat:
  print("Found solution")
  m = s.model()
  
  result = [[m.evaluate(Sum([x[student][prof][book]  for prof in range(len(professors))]))
            for book in range(len(books))]
            for student in range(len(students))]
  for l in result:
    print(l)

else:
  print("No solution found")

stop = timeit.default_timer()

print('Time: ', stop - start)  