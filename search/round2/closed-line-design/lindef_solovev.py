"""TASK 2, first-order closed-line QS-MHS deformations of the Solov'ev (1:1:2) congruence by polynomial vector
fields xi of degree <= DEG (all s-Fourier modes at once).  See lindef_core.py for the linear system.
x0 = R_z(s) y0(lam,t), y0 = sqrt(2S)(cos lam cos t, -sin lam sin t, 0) + (kappa S/2) cos 2lam cos(2t+delta) e_z.
Expected trivial solutions: 6 Killing fields, (x,y,0) [S-scaling], (0,0,z) [kappa-scaling], (0,0,x^2+y^2) [delta-shift
+ translation: the family's z-shear].  Anything else is a genuine first-order deformation; a mode |n| >= 1 non-Killing
solution would be a non-axisymmetric one.
Run: python3 lindef_solovev.py [DEG] [case]   (case 0: S=2, kappa=3/2, cos delta=3/5;  case 1: S=8, kappa=1, delta=pi/2)
"""
import sys, time
import sympy as sp
from lindef_core import *
T0 = time.time()
DEG = int(sys.argv[1]) if len(sys.argv) > 1 else 3
case = int(sys.argv[2]) if len(sys.argv) > 2 else 0
if case == 0:
    S, kap, cd, sd = sp.Integer(2), sp.Rational(3, 2), sp.Rational(3, 5), sp.Rational(4, 5)
else:
    S, kap, cd, sd = sp.Integer(8), sp.Integer(1), sp.Integer(0), sp.Integer(1)
sq2S = sp.sqrt(2*S); assert sq2S.is_Rational
I = sp.I
cosL = lambda k: (L**k + L**(-k))/2; sinL = lambda k: (L**k - L**(-k))/(2*I)
cosE = lambda k: (E**k + E**(-k))/2; sinE = lambda k: (E**k - E**(-k))/(2*I)
cosT = lambda k: (T**k + T**(-k))/2; sinT = lambda k: (T**k - T**(-k))/(2*I)
dl = lambda f: sp.expand(I*L*sp.diff(f, L)); ds = lambda f: sp.expand(I*E*sp.diff(f, E)); dt = lambda f: sp.expand(I*T*sp.diff(f, T))
y0 = sp.Matrix([sq2S*cosL(1)*cosT(1), -sq2S*sinL(1)*sinT(1), kap*S/2*cosL(2)*(cosT(2)*cd - sinT(2)*sd)])
Rz = sp.Matrix([[cosE(1), -sinE(1), 0], [sinE(1), cosE(1), 0], [0, 0, 1]])
x0 = (Rz*y0).applyfunc(sp.expand)
fields = basis_fields(DEG)
print(f"Solov'ev congruence, DEG = {DEG} ({len(fields)} coefficients), case {case}: S={S}, kappa={kap}, (cos,sin)delta=({cd},{sd})")
cols = condition_columns(x0, dl, ds, dt, fields)
print(f"  columns built, {time.time()-T0:.0f} s")
M = assemble(cols, ['dJ_s', 'dJ_t', 'W', 'M1', 'M2_s', 'M2_t'], len(fields))
Ms = assemble(cols, ['SQ_ts', 'SQ_tl'], len(fields))
print(f"  system {M.shape[0]} x {M.shape[1]}; strong-QS block {Ms.shape[0]} rows; {time.time()-T0:.0f} s")
ns = M.nullspace()
print(f"  rank {len(fields)-len(ns)}, nullspace dim {len(ns)}, {time.time()-T0:.0f} s")
trivial = [sp.Matrix([X, Y, 0]), sp.Matrix([0, 0, Z]), sp.Matrix([0, 0, X**2 + Y**2])]
print("  solutions:")
nontriv = []
for v in ns:
    v = clean_vec(v); f = field_from(v, fields)
    kil = is_killing(f); strong_ok = all(r == 0 for r in (Ms*v).applyfunc(sp.expand)) if Ms.shape[0] else True
    modes = s_modes(f, x0)
    # membership in span(trivial)?
    a = sp.symbols('a0:3'); comb = f - sum((ai*tv for ai, tv in zip(a, trivial)), sp.zeros(3, 1))
    eqs = []
    for comp in comb: eqs += sp.Poly(sp.expand(comp), X, Y, Z).coeffs()
    triv = bool(sp.solve(eqs, a, dict=True)) if not kil else True
    print(f"   xi = {list(f)}  | Killing: {kil} | div = {divergence(f)} | s-modes {sorted(modes)} | strong-QS: {strong_ok} | trivial: {triv}")
    if not triv: nontriv.append(f)
print(f"\n  RESULT DEG={DEG}: nullspace {len(ns)} = 6 Killing + {len(ns)-6} others; non-trivial (beyond Killing, S-, kappa-scaling, z-shear): {len(nontriv)}")
for f in nontriv: print("   NON-TRIVIAL:", list(f))
print(f"done in {time.time()-T0:.0f} s")
