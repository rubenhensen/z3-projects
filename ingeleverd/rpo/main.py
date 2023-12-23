import term

import sys
from term import Term, FnApp, FnSym, Var
from trs import Rule, Trs
import lpo_solver

def parse_term(desc, variables, sig, pos):
  i = pos + 1
  while i < len(desc) and desc[i].isalnum(): i += 1
  name = desc[pos:i]
  while i < len(desc) and desc[i] == ' ': i += 1
  if i == len(desc) or desc[i] == ',' or desc[i] == ')':
    x = Var(name)
    variables.add(x)
    return ( x, i )
  if desc[i] != '(': raise Error("Unexpected symbol at position " + str(i) + " of " + desc)
  i += 1
  args = []
  while i < len(desc) and desc[i] != ')':
    if desc[i] != ' ' and desc[i] != ',':
      (arg, i) = parse_term(desc, variables, sig, i)
      args += [ arg ]
    else:
      i += 1
  if i == len(desc): raise Error("Unexpected end of line (unclosed bracket) in " + desc)
  symbol = FnSym(name, len(args))
  sig.add(symbol)
  return ( FnApp((symbol, args)), i + 1 )

def parse(lines):
    # print("lines", lines)
    rules = [ line.split('->') for line in lines ]
    # print("rules", rules)
    rules = [ (r[0].strip(), r[1].strip()) for r in rules if len(r) == 2 ]
    # print("rules", rules)
    variables = set()
    sig = set()
    rules = [ ( parse_term(r[0], variables, sig, 0)[0], parse_term(r[1], variables, sig, 0)[0] ) for r in rules ]
    # print("rules", rules)
    rules = [ Rule(r[0], r[1]) for r in rules ]
    # print("vars", variables)
    # print("sig", sig)
    return Trs( list(variables), list(sig), rules )

def main(inputfile):
    inp = open(inputfile, 'r')
    lines = inp.readlines()
    trs = parse(lines)
    lpo_solver.prove_termination(trs)

if __name__ == '__main__':
    inputfile = "input.trs"
    if len(sys.argv) == 2: inputfile = sys.argv[1]
    main(inputfile)
