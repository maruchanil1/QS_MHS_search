"""TASK 2 (design-first, decided at first order) -- polynomial deformations of the exact axisymmetric closed-line
Solov'ev congruence x0(lam,s,t) = R_z(s) y0(lam,t) (iota = 2, p' != 0), in orbit coordinates.

Every first-order deformation of the embedding is x = x0 + eps xi(x0) for a vector field xi on the tube (x0 is a
diffeomorphism), and then  delta g_ab = 2 x0_a . S(xi)(x0) . x0_b,  S = sym grad xi,  delta J = J0 div xi.
So the FIRST-ORDER closed-line weak-QS + MHS problem is the LINEAR system (see notes.md, Sec. 2):
    (dJ)  div xi restricted to each torus is constant (d_s = d_t = 0 of div xi o x0)
    (W)   d_s  delta g_tt = 0                       [u.grad|B| = 0,  u = d_s]
    (M1)  d_t  delta g_ts = d_s delta g_tt          [J^psi = 0]
    (M2)  d_t  delta g_tlam - d_lam delta g_tt = delta p'(lam)   (d_s, d_t of the left side vanish)
and STRONG QS would add  d_s delta g_ts = d_s delta g_tlam = 0  (reported separately, not imposed).
Equivalently: S(xi) B = c_lam(lam,s) grad lam + c_s(lam,s) grad s with c's constant along lines (strong: S(xi)B = 0).

Here xi = generic polynomial vector field of total degree <= DEG (all s-Fourier modes at once).  The identities are
Laurent polynomials in (e^{i lam}, e^{i s}, e^{i t}); coefficient extraction gives an exact rational linear system.
Trivial solutions expected: 6 Killing fields, the dilation xi = x, the z-scaling xi = z e_z (kappa is a free family
parameter).  Anything else = a genuine first-order closed-line QS-MHS deformation (mode n != 0 would be non-axisymmetric).
Run: python3 linear_deformation_solovev.py [DEG] [case]    (case 0: S=2, kappa=3/2, cos d=3/5; case 1: S=8, kappa=1, d=pi/2)
"""
import sys, time, itertools
import sympy as sp
T0 = time.time()
DEG = int(sys.argv[1]) if len(sys.argv) > 1 else 3
case = int(sys.argv[2]) if len(sys.argv) > 2 else 0
if case == 0:
    S, kap, cd, sd = sp.Integer(2), sp.Rational(3, 2), sp.Rational(3, 5), sp.Rational(4, 5)
else:
    S, kap, cd, sd = sp.Integer(8), sp.Integer(1), sp.Integer(0), sp.Integer(1)
sq2S = sp.sqrt(2*S)
assert sq2S.is_Rational

# exponential variables: L = e^{i lam}, E = e^{i s}, T = e^{i t}
L, E, T = sp.symbols('L E T')
I = sp.I
def cosL(k): return (L**k + L**(-k))/2
def sinL(k): return (L**k - L**(-k))/(2*I)
def cosE(k): return (E**k + E**(-k))/2
def sinE(k): return (E**k - E**(-k))/(2*I)
def cosT(k): return (T**k + T**(-k))/2
def sinT(k): return (T**k - T**(-k))/(2*I)
dl = lambda f: I*L*sp.diff(f, L)      # d/d lam
ds = lambda f: I*E*sp.diff(f, E)      # d/d s
dt = lambda f: I*T*sp.diff(f, T)      # d/d t

# congruence: y0 = sqrt(2S)(cos lam cos t, -sin lam sin t, 0) + (kappa S/2) cos 2lam cos(2t+delta) e_z
cos2td = cosT(2)*cd - sinT(2)*sd
y0 = sp.Matrix([sq2S*cosL(1)*cosT(1), -sq2S*sinL(1)*sinT(1), kap*S/2*cosL(2)*cos2td])
Rz = sp.Matrix([[cosE(1), -sinE(1), 0], [sinE(1), cosE(1), 0], [0, 0, 1]])
x0 = (Rz*y0).applyfunc(sp.expand)
x0_l, x0_s, x0_t = [x0.applyfunc(D) for D in (dl, ds, dt)]

# generic polynomial vector field
X, Y, Z = sp.symbols('x y z')
monos = [X**i*Y**j*Z**k for i in range(DEG+1) for j in range(DEG+1) for k in range(DEG+1) if i+j+k <= DEG]
cs = []
xi = sp.zeros(3, 1)
for comp in range(3):
    for mnm in monos:
        c = sp.Symbol(f'c{comp}_{len(cs)}'); cs.append(c)
        xi[comp] += c*mnm
print(f"DEG = {DEG}: {len(cs)} coefficients; case {case}: S={S}, kappa={kap}, (cos,sin)delta=({cd},{sd})")
Grad = sp.Matrix(3, 3, lambda i, j: sp.diff(xi[i], (X, Y, Z)[j]))
Sym = (Grad + Grad.T)/2
divxi = sum(Grad[i, i] for i in range(3))
sub = {X: x0[0], Y: x0[1], Z: x0[2]}
S0 = Sym.subs(sub, simultaneous=True)
div0 = sp.expand(divxi.subs(sub, simultaneous=True))
def dg(a, b): return sp.expand(2*(a.T*S0*b)[0, 0])
g_tt, g_ts, g_tl = dg(x0_t, x0_t), dg(x0_t, x0_s), dg(x0_t, x0_l)
print(f"  metric perturbations built, {time.time()-T0:.0f} s")

conds = {
    'dJ_s': ds(div0), 'dJ_t': dt(div0),
    'W': ds(g_tt),
    'M1': dt(g_ts) - ds(g_tt),
    'M2_s': ds(dt(g_tl) - dl(g_tt)), 'M2_t': dt(dt(g_tl) - dl(g_tt)),
}
strong = {'SQ_ts': ds(g_ts), 'SQ_tl': ds(g_tl)}

def linear_rows(expr):
    """expr linear homogeneous in cs, Laurent polynomial in L,E,T -> list of rows (dict c -> coefficient)."""
    expr = sp.expand(expr)
    rows = {}
    for term in sp.Add.make_args(expr):
        coeff, rest = term.as_independent(*cs, as_Add=False)
        # rest is one c symbol (linear)
        assert rest in cs, (term,)
        mono = sp.Mul(*[f for f in sp.Mul.make_args(coeff) if f.has(L) or f.has(E) or f.has(T)])
        num = sp.Mul(*[f for f in sp.Mul.make_args(coeff) if not (f.has(L) or f.has(E) or f.has(T))])
        rows.setdefault(mono, {}); rows[mono][rest] = rows[mono].get(rest, 0) + num
    return rows

def build_matrix(cond_dict):
    allrows = []
    for name, ex in cond_dict.items():
        rws = linear_rows(ex)
        for mono, rw in rws.items():
            allrows.append([sp.nsimplify(rw.get(c, 0)) for c in cs])
    M = sp.Matrix(allrows) if allrows else sp.zeros(0, len(cs))
    return M

M = build_matrix(conds)
print(f"  linear system: {M.shape[0]} equations x {M.shape[1]} unknowns, {time.time()-T0:.0f} s")
# remove zero rows, exact nullspace
M = sp.Matrix([r for r in M.tolist() if any(v != 0 for v in r)])
ns = M.nullspace()
print(f"  rank = {len(cs) - len(ns)}, nullspace dimension = {len(ns)}, {time.time()-T0:.0f} s")

# analyse each null vector: Killing? div? s-mode content of delta g_tt... and strong-QS residuals
def field_of(vec):
    return xi.subs(dict(zip(cs, vec)))
Mstrong = build_matrix(strong)
Mstrong = sp.Matrix([r for r in Mstrong.tolist() if any(v != 0 for v in r)]) if Mstrong.shape[0] else Mstrong
print("\n  basis of first-order solutions (each printed as the vector field xi):")
nonkilling = []
for v in ns:
    v = v*sp.lcm([sp.fraction(sp.nsimplify(a))[1] for a in v if a != 0])
    f = field_of(v).applyfunc(sp.factor)
    G_ = sp.Matrix(3, 3, lambda i, j: sp.diff(f[i], (X, Y, Z)[j]))
    killing = (G_ + G_.T).applyfunc(sp.expand) == sp.zeros(3, 3)
    dv = sp.expand(sum(G_[i, i] for i in range(3)))
    # s-modes present in xi(x0):
    fx0 = f.subs(sub, simultaneous=True).applyfunc(sp.expand)
    modes = set()
    for comp in fx0:
        for term in sp.Add.make_args(comp):
            modes.add(sp.degree(term.as_independent(L, T, as_Add=False)[1].subs(E, E), E) if term.has(E) else 0)
    strong_res = (Mstrong*v).applyfunc(sp.expand) if Mstrong.shape[0] else sp.zeros(1, 1)
    is_strong = all(r == 0 for r in strong_res)
    tag = "KILLING" if killing else ("div=%s" % dv)
    print(f"   xi = {list(f)}   [{tag}; strong-QS residual zero: {is_strong}]")
    if not killing:
        nonkilling.append(f)
print(f"\n  non-Killing solutions: {len(nonkilling)}  (expected trivial ones: dilation x, z-scaling z e_z)")
# check: are the non-Killing solutions spanned by x and z e_z?
if nonkilling:
    triv = [sp.Matrix([X, Y, Z]), sp.Matrix([0, 0, Z])]
    span_ok = True
    for f in nonkilling:
        a1, a2 = sp.symbols('a1 a2')
        sol = sp.solve(list((f - a1*triv[0] - a2*triv[1]).applyfunc(sp.expand)), [a1, a2], dict=True)
        # solve treats x,y,z as symbols: need identity in x,y,z -> compare coefficients
        eqs = []
        for comp in (f - a1*triv[0] - a2*triv[1]):
            eqs += sp.Poly(sp.expand(comp), X, Y, Z).coeffs()
        sol = sp.solve(eqs, [a1, a2], dict=True)
        if not sol:
            span_ok = False
            print("   NON-TRIVIAL first-order deformation:", list(f))
    print("  all non-Killing solutions are in span{x, z e_z} (trivial family deformations):", span_ok)
print(f"done in {time.time()-T0:.0f} s")
