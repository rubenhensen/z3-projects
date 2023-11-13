
# matrix of boolean variables of books, professors and students
x = [[[Int("%s_%s_%s" % (student, prof, book)) for book in books]
        for prof in professors ]
        for student in students]

every_cell_one_or_zero = [Or(x[student][prof][book] == 0, x[student][prof][book] == 1) 
                          for book in range(len(books)) 
                          for prof in range(len(professors))
                          for student in range(len(students))]
s.add(every_cell_one_or_zero)

one_book_per_student = [Sum([x[student][prof][book] for book in range(len(books)) for prof in range(len(professors))]) == 1
        for student in range(len(students))]
print(one_book_per_student[0])
s.add(one_book_per_student)

# Every book can be used only once.
every_student_a_different_book = [Sum([x[student][prof][book]  for prof in range(len(professors)) for student in range(len(students))]) <= 1
        for book in range(len(books))]
print(every_student_a_different_book[0])
s.add(every_student_a_different_book)

# professor_with_own_books = 





# remove this, since it's just to give an example for the output
# solution = [ (students[i], books[i], rank_by_id(i, i)) for i in range(len(students)) ]
# for s in solution:
#   print(s[0] + " : " + s[1] + " (" + str(s[2]) + ")")

########## print the solution ##########
if s.check() == sat:
  print("Found solution")
  m = s.model()
  # result_3d = [m.evaluate(x[student][prof][book])
  #           for prof in range(len(professors))
  #           for book in range(len(books)) 
  #           for student in range(len(students))]
  
  result = [[m.evaluate(Sum([x[student][prof][book]  for prof in range(len(professors))]))
            for book in range(len(books))]
            for student in range(len(students))]
  for l in result:
    print(l)

else:
  print("No solution found")
