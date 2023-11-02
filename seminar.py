###############################################################################
# This is an example program handling the input and output for the Seminar    #
# assignment.  It is up to you to fill in the middle part. :)                 #
# (You may also use your own reading/writing if you prefer.)                  #
###############################################################################

from z3 import *

import sys
import csv

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
#print(students)
#print(books)
#print(professors)
#print(rank("Alice", "D1"))
#print(books_for("C"))

########## finding a good assignment ##########

# THIS IS WHERE YOUR CODE GOES

# remove this, since it's just to give an example for the output
solution = [ (students[i], books[i], rank_by_id(i, i)) for i in range(len(students)) ]

########## print the solution ##########

for s in solution:
  print(s[0] + " : " + s[1] + " (" + str(s[2]) + ")")

