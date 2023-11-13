###############################################################################
# This is an example program handling the input and output for the Seminar    #
# assignment.  It is up to you to fill in the middle part. :)                 #
# (You may also use your own reading/writing if you prefer.)                  #
###############################################################################

from z3 import *

import sys
import csv
import helper
import numpy as np

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

# uncomment the following if you want to see the values of the main variables, and the output of the functions
print("students", students)
print("books", books)
print("professors", professors)
print(rank("Alice", "D1"))
print(books_for("C"))

########## finding a good assignment ##########

# matrix of boolean variables of books, professors and students
x = [[[Bool("%s_%s_%s" % (student, prof, book)) for book in books]
        for prof in professors ]
        for student in students]

# Every student has one book.
one_book_per_student = [Sum([If(x[student][prof][book], 1, 0) for book in range(len(books)) for prof in range(len(professors))]) == 1
        for student in range(len(students))]
# print("one_book_per_student\n")
# print(one_book_per_student[0])
s.add(one_book_per_student)

# Every book can be used only once.
every_student_a_different_book = [Sum([If(x[student][prof][book],1,0)  for prof in range(len(professors)) for student in range(len(students))]) <= 1
        for book in range(len(books))]
# print("every_student_a_different_book[0]\n")
# print(every_student_a_different_book)
s.add(every_student_a_different_book)

# Every professor can only be connected to their own books
professor_with_own_books = [x[student][prof][book] == False
                                 for prof in range(len(professors)) 
                                 for student in range(len(students))
                                 for book in range(len(books)) if books[book] not in books_for(professors[prof])]
# print("professor_with_own_books")
# print(professor_with_own_books)
s.add(professor_with_own_books)

# Every professor can supervise at most 2 books (optional)
professor_max_two_books = [Sum([x[student][prof][book] for student in range(len(students)) for book in range(len(books))])<= 2
                                for prof in range(len(professors))]
print(professor_max_two_books)
# s.add(professor_max_two_books)


# Maximize solution
res = s.check()
while (res == sat):
  m = s.model()
  print(m)
  block = []
  for var in m:
      block.append(var() != m[var])
  s.add(Or(block))
  res = s.check()



########## print the solution ##########
if s.check() == sat:
  print("Found solution")
  m = s.model()
  # result_3d = [m.evaluate(x[student][prof][book])
  #           for prof in range(len(professors))
  #           for book in range(len(books)) 
  #           for student in range(len(students))]
  
  print("student x book")
  result = [[m.evaluate(Sum([x[student][prof][book]  for prof in range(len(professors))]))
            for book in range(len(books))]
            for student in range(len(students))]
  result = helper.add_index(result, students, books)
  helper.print_table(result)

  print("prof x book")
  result = [[m.evaluate(Sum([x[student][prof][book]  for student in range(len(students))]))
            for book in range(len(books))]
            for prof in range(len(professors))
            ]
  result = helper.add_index(result, professors, books)
  helper.print_table(result)
  

else:
  print("No solution found")



# Alice : A1 (2)
# Bob : B1 (1)
# Carol : C1 (7)
# Dennis : C2 (10)
# Eliza : C3 (11)
# Fred : D1 (3)
# Georgianna : D2 (8)
# Harry : E1 (7)
# Ilse : F1 (14)
# John : F2 (7)
# Karen : G1 (9)
# Leo : G2 (9)
# Monica : G3 (5)
# Nico : H1 (3)
# Olivia : H2 (1)
# Patrick : I1 (7)