"""compare_engrad.py A.engrad B.engrad [dE_max dG_max]: compare two crest.engrad files.
Prints the energies, max |dE| and max |dG|; exits 0 if both are within the limits (default: identity tier,
1e-8 Eh and 1e-6 Eh/bohr), 1 otherwise. Byte-identical files report 'bitwise'."""
import sys


def read(path):
    lines = [l.strip() for l in open(path)]
    vals = [l for l in lines if l and not l.startswith('#')]
    n = int(vals[0])
    return float(vals[1]), [float(x) for x in vals[2:2 + 3 * n]]


a, b = sys.argv[1], sys.argv[2]
de_max = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
dg_max = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-6
if open(a, 'rb').read() == open(b, 'rb').read():
    print('bitwise identical')
    sys.exit(0)
ea, ga = read(a)
eb, gb = read(b)
de = abs(ea - eb)
dg = max(abs(x - y) for x, y in zip(ga, gb))
print(f'E_a {ea:.12f}  E_b {eb:.12f}  max|dE| {de:.2e} Eh  max|dG| {dg:.2e} Eh/bohr')
sys.exit(0 if (de <= de_max and dg <= dg_max and len(ga) == len(gb)) else 1)
