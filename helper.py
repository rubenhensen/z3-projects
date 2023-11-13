def add_index(matrix, row, column):
    if len(matrix[0]) != len(column):
        raise ValueError("column index is not the right length")
    if len(matrix) != len(row):
        raise ValueError("row index is not the right length")
    matrix = [column] + matrix
    row = [""] + row
    result = []
    for i, l in enumerate(matrix):
        result.append([row[i]] + l)

    return result

def print_table(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '  '.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    ttable = '\n'.join(table)
    print(ttable)