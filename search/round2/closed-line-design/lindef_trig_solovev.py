"""TASK 2 (complement to lindef_solovev.py): first-order closed-line QS-MHS deformations of the Solov'ev congruence
by TRIGONOMETRIC displacements in orbit coordinates, one s-mode n != 0 at a time.

x(lam,s,t) = R_z(s)[y0(lam,t) + eps Xi(lam,s,t)],   Xi = e^{ins} sum_{|j|<=Dl,|k|<=Dt} c_jk e^{i(j lam + k t)}  (c_jk in C^3).
With b = y0_t, w = e_z x y0, yl = y0_lam (all s-independent) the first-order conditions for mode n != 0 are
    (W)   b . Xi_t = 0
    (M1)  d_t[ b.(e_z x Xi + i n Xi) + w.Xi_t ] = 0
    (M2)  d_t[ b.Xi_lam + yl.Xi_t ] - 2 d_lam[ b.Xi_t ] = 0
    (dJ)  det(Xi_lam, w, b) + det(yl, e_z x Xi + i n Xi, b) + det(yl, w, Xi_t) = 0
(strong QS would add  b.(e_z x Xi + i n Xi) + w.Xi_t = 0  and  b.Xi_lam + yl.Xi_t = 0).
This class contains the mode-n Killing fields (n = 1: translations e_x, e_y and rotations about x, y) and is otherwise
independent of the polynomial class of lindef_solovev.py.  Exact rational linear algebra (Laurent polynomials in
L = e^{i lam}, T = e^{i t}).  Run: python3 lindef_trig_solovev.py n Dl Dt [case]
"""
import sys, time
import sympy as sp
T0 = time.time()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
Dl = int(sys.argv[2]) if len(sys.argv) > 2 else 2
Dt = int(sys.argv[3]) if len(sys.argv) > 3 else 3
case = int(sys.argv[4]) if len(sys.argv) > 4 else 0
if case == 0:
    S, kap, cd, sd = sp.Integer(2), sp.Rational(3, 2), sp.Rational(3, 5), sp.Rational(4, 5)
else:
    S, kap, cd, sd = sp.Integer(8), sp.Integer(1), sp.Integer(0), sp.Integer(1)
sq2S = sp.sqrt(2*S); assert sq2S.is_Rational
I = sp.I
L, T = sp.symbols('L T')
cosL = lambda k: (L**k + L**(-k))/2; sinL = lambda k: (L**k - L**(-k))/(2*I)
cosT = lambda k: (T**k + T**(-k))/2; sinT = lambda k: (T**k - T**(-k))/(2*I)
dl = lambda f: sp.expand(I*L*sp.diff(f, L)); dt = lambda f: sp.expand(I*T*sp.diff(f, T))
y0 = sp.Matrix([sq2S*cosL(1)*cosT(1), -sq2S*sinL(1)*sinT(1), kap*S/2*cosL(2)*(cosT(2)*cd - sinT(2)*sd)])
b = y0.applyfunc(dt); yl = y0.applyfunc(dl); ez = sp.Matrix([0, 0, 1]); w = ez.cross(y0)

# unknowns
cs = []; Xi = sp.zeros(3, 1)
for comp in range(3):
    for j in range(-Dl, Dl+1):
        for k in range(-Dt, Dt+1):
            c = sp.Symbol(f'c{comp}_{j+Dl}_{k+Dt}'); cs.append(c)
            Xi[comp] += c*L**j*T**k
Xi_t = Xi.applyfunc(dt); Xi_l = Xi.applyfunc(dl)
Xi_s = ez.cross(Xi) + I*n*Xi
g_tt = sp.expand(2*b.dot(Xi_t))
g_ts = sp.expand(b.dot(Xi_s) + w.dot(Xi_t))
g_tl = sp.expand(b.dot(Xi_l) + yl.dot(Xi_t))
dJ = sp.expand(sp.Matrix.hstack(Xi_l, w, b).det() + sp.Matrix.hstack(yl, Xi_s, b).det() + sp.Matrix.hstack(yl, w, Xi_t).det())
conds = {'W': g_tt, 'M1': dt(g_ts), 'M2': dt(g_tl) - dl(g_tt), 'dJ': dJ}
strong = {'SQ_ts': g_ts, 'SQ_tl': g_tl}
print(f"mode n = {n}, Dl = {Dl}, Dt = {Dt}, case {case}: {len(cs)} complex unknowns; expressions built {time.time()-T0:.0f} s")

def rows_of(expr):
    expr = sp.expand(expr)
    rows = {}
    for term in sp.Add.make_args(expr):
        coeff, cvar = term.as_independent(*cs, as_Add=False)
        assert cvar in cs
        mono = sp.Mul(*[f for f in sp.Mul.make_args(coeff) if f.has(L) or f.has(T)])
        num = sp.Mul(*[f for f in sp.Mul.make_args(coeff) if not (f.has(L) or f.has(T))])
        rows.setdefault(mono, {}); rows[mono][cvar] = rows[mono].get(cvar, 0) + num
    return [[r.get(c, 0) for c in cs] for r in rows.values()]

def matrix_of(d):
    rows = []
    for ex in d.values(): rows += rows_of(ex)
    rows = [r for r in rows if any(v != 0 for v in r)]
    return sp.Matrix(rows) if rows else sp.zeros(0, len(cs))
M = matrix_of(conds); Ms = matrix_of(strong)
print(f"  system {M.shape[0]} x {M.shape[1]}, {time.time()-T0:.0f} s")
ns = M.nullspace()
print(f"  rank {len(cs)-len(ns)}, nullspace dim {len(ns)}, {time.time()-T0:.0f} s")

# Killing fields of mode n in the rotating frame: R_z(-s) K(R_z(s) y): translations e_x, e_y -> (cos s, -sin s, 0) etc. (n = 1)
def rot_frame_killing():
    out = []
    if n == 1:
        # e^{is} components: R_z(-s) e_x = (cos s, -sin s, 0) = Re[(1, -i, 0)e^{is}] -> complex mode-1 vector (1, -i, 0)
        out.append(sp.Matrix([1, I, 0]))                        # translations e_x, e_y (complex combination)
        # rotation about x: K = e_x x y; in the rotating frame: R_z(-s) (e_x x R_z(s) y) = (R_z(-s)e_x) x y
        ex_rot = sp.Matrix([1, I, 0])
        out.append(ex_rot.cross(y0).applyfunc(sp.expand))       # rotations about x, y (complex combination)
    return out
kill = rot_frame_killing()
def in_span(f, basis):
    if not basis: return False
    a = sp.symbols(f'a0:{len(basis)}')
    comb = (f - sum((ai*bv for ai, bv in zip(a, basis)), sp.zeros(3, 1))).applyfunc(sp.expand)
    eqs = []
    for comp in comb:
        for term_mono, coeff in sp.Poly(comp*L**(Dl+3)*T**(Dt+3), L, T).as_dict().items():
            eqs.append(coeff)
    return bool(sp.solve(eqs, a, dict=True))
print("  solutions (as complex vector fields Xi(L,T) in the rotating frame):")
nontriv = 0
for v in ns:
    den = sp.lcm([sp.fraction(sp.nsimplify(a))[1] for a in v if a != 0]); v = v*den
    f = Xi.subs(dict(zip(cs, v))).applyfunc(sp.expand)
    strong_ok = all(sp.expand(r) == 0 for r in Ms*v)
    isk = in_span(f, kill)
    print(f"   Xi = {list(f)} | Killing: {isk} | strong-QS: {strong_ok}")
    if not isk: nontriv += 1
print(f"\n  RESULT mode n={n}, (Dl,Dt)=({Dl},{Dt}): nullspace {len(ns)}, non-Killing solutions: {nontriv}")
print(f"done in {time.time()-T0:.0f} s")
