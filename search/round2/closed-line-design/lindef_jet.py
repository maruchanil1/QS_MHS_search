"""TASK 2, strongest finite version: first-order closed-line QS-MHS deformations with ARBITRARY dependence on the
flux label, finite t-Fourier content, one s-mode n != 0, decided through the 1-jet in the label at a rational point.

x = R_z(s)[y0(lab,t) + eps Xi],  Xi = e^{ins}[C(t) + (lab - lab0) C'(t) + ...],  C = sum_{|k|<=Dt} c_k T^k, C' = sum d_k T^k.
The four first-order conditions (lindef_trig_solovev.py docstring) at lab = lab0 involve only (C, C'):
    (W)  b.C_t = 0
    (M1) d_t[ b.(e_z x C + i n C) + w.C_t ] = 0
    (M2) d_t[ b.C' + yl.C_t ] - 2[ b_lab.C_t + b.C'_t ] = 0
    (dJ) det(C', w, b) + det(yl, e_z x C + i n C, b) + det(yl, w, C_t) = 0
(with b = y0_t, w = e_z x y0, yl = y0_lab, all at lab0).  They are NECESSARY conditions on the jet of any solution with
t-degree <= Dt, whatever its label dependence.  If the nullspace in (c, d) equals the span of the jets of the mode-n
Killing fields, no non-isometric first-order deformation with t-degree <= Dt exists (for any label dependence).
Congruences: 'solovev' (label lam at cos lam = 4/5) and 'kepler' (label eps at (C, eps) = (5/4, 3/5); conditions are
rational in T with denominators (1 - eps cos eta)^k, cleared by multiplying with (1 - eps cos eta)^8; d_t = d_eta/tau_eta
and d_lab|_t = d_eps - (tau_eps/tau_eta) d_eta act on y0 before evaluation).
Run: python3 lindef_jet.py {solovev|kepler} n Dt
"""
import sys, time
import sympy as sp
T0 = time.time()
which = sys.argv[1] if len(sys.argv) > 1 else 'solovev'
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
Dt = int(sys.argv[3]) if len(sys.argv) > 3 else 3
I = sp.I
T, E = sp.symbols('T E')
lab = sp.symbols('lab', real=True)
cosT = lambda k: (T**k + T**(-k))/2; sinT = lambda k: (T**k - T**(-k))/(2*I)
deta = lambda f: I*T*sp.diff(f, T)

if which == 'solovev':
    S, kap, cd, sd = sp.Integer(2), sp.Rational(3, 2), sp.Rational(3, 5), sp.Rational(4, 5)
    cl, sl = sp.Rational(4, 5), sp.Rational(3, 5)           # cos lam0, sin lam0 ; cos 2lam0 = 7/25
    # y0 with symbolic lab (lam) for differentiation
    y0 = sp.Matrix([sp.sqrt(2*S)*sp.cos(lab)*cosT(1), -sp.sqrt(2*S)*sp.sin(lab)*sinT(1),
                    kap*S/2*sp.cos(2*lab)*(cosT(2)*cd - sinT(2)*sd)])
    dt = deta
    dlab = lambda f: sp.diff(f, lab)
    point = {sp.cos(lab): cl, sp.sin(lab): sl}
    def at(ex):
        ex = sp.expand(sp.expand_trig(ex)).subs(point)
        return sp.expand(ex)
    clear = sp.Integer(1)
else:
    C_, ev = sp.Rational(5, 4), sp.Rational(3, 5)
    a, mu = sp.Integer(1), sp.Integer(1)
    eps = lab; beta = sp.sqrt(1 - eps**2)
    sin_i = eps/(C_*beta); cos_i = sp.sqrt(1 - sin_i**2)
    Rx = sp.Matrix([[1, 0, 0], [0, cos_i, -sin_i], [0, sin_i, cos_i]])
    e = sp.Matrix([a*(cosT(1) - eps), a*beta*sinT(1), 0])
    y0 = Rx*e
    tau_eta = sp.sqrt(a**3/mu)*(1 - eps*cosT(1)); tau_lab = -sp.sqrt(a**3/mu)*sinT(1)
    dt = lambda f: deta(f)/tau_eta
    dlab = lambda f: sp.diff(f, lab) - (tau_lab/tau_eta)*deta(f)
    def at(ex):
        return sp.expand(sp.together(sp.expand(ex.subs(lab, ev))))
    clear = tau_eta.subs(lab, ev)**8

ez = sp.Matrix([0, 0, 1])
b = y0.applyfunc(dt); w = ez.cross(y0); yl = y0.applyfunc(dlab); bl = b.applyfunc(dlab)
# evaluate coefficient fields at the point
b0, w0, yl0, bl0 = [v.applyfunc(at) for v in (b, w, yl, bl)]
# unknown jets
cs, ds = [], []
Cv, Dv = sp.zeros(3, 1), sp.zeros(3, 1)
for comp in range(3):
    for k in range(-Dt, Dt+1):
        c = sp.Symbol(f'c{comp}_{k+Dt}'); d = sp.Symbol(f'd{comp}_{k+Dt}')
        cs.append(c); ds.append(d); Cv[comp] += c*T**k; Dv[comp] += d*T**k
unk = cs + ds
Cs = ez.cross(Cv) + I*n*Cv
# for kepler the d_t acting on unknown Laurent polynomials is deta/tau_eta with tau_eta at the point
if which == 'solovev':
    dtu = deta
else:
    dtu = lambda f: deta(f)/tau_eta.subs(lab, ev)
Ct = Cv.applyfunc(dtu); Dt_ = Dv.applyfunc(dtu)
g_tt = 2*b0.dot(Ct)
g_ts = b0.dot(Cs) + w0.dot(Ct)
g_tl = b0.dot(Dv) + yl0.dot(Ct)
M2 = dtu(g_tl) - 2*(bl0.dot(Ct) + b0.dot(Dt_))
dJ = sp.Matrix.hstack(Dv, w0, b0).det() + sp.Matrix.hstack(yl0, Cs, b0).det() + sp.Matrix.hstack(yl0, w0, Ct).det()
conds = {'W': g_tt, 'M1': dtu(g_ts), 'M2': M2, 'dJ': dJ}
strong = {'SQ_ts': g_ts, 'SQ_tl': g_tl}
print(f"{which}: mode n = {n}, Dt = {Dt}: {len(unk)} complex unknowns (c: values, d: label-derivatives); built {time.time()-T0:.0f} s")

def rows_of(expr):
    expr = sp.expand(sp.together(sp.expand(expr*clear))) if which == 'kepler' else sp.expand(expr)
    rows = {}
    for term in sp.Add.make_args(expr):
        coeff, cvar = term.as_independent(*unk, as_Add=False)
        assert cvar in unk, term
        mono = sp.Mul(*[f for f in sp.Mul.make_args(coeff) if f.has(T)])
        num = sp.Mul(*[f for f in sp.Mul.make_args(coeff) if not f.has(T)])
        rows.setdefault(mono, {}); rows[mono][cvar] = rows[mono].get(cvar, 0) + num
    out = [[r.get(u, 0) for u in unk] for r in rows.values()]
    return [r for r in out if any(v != 0 for v in r)]
def matrix_of(d):
    rows = []
    for ex in d.values(): rows += rows_of(ex)
    return sp.Matrix(rows) if rows else sp.zeros(0, len(unk))
M = matrix_of(conds); Ms = matrix_of(strong)
assert all(v.is_number for v in M)
print(f"  system {M.shape[0]} x {M.shape[1]}, {time.time()-T0:.0f} s")
ns = M.nullspace()
print(f"  rank {len(unk)-len(ns)}, nullspace dim {len(ns)}, {time.time()-T0:.0f} s")

# Killing jets of mode n (n = 1 only): translation (1,i,0) [C' = 0]; rotation (1,i,0) x y0 [C' = (1,i,0) x y0_lab]
kill = []
if n == 1:
    v1 = sp.Matrix([1, I, 0])
    kill.append((v1, sp.zeros(3, 1)))
    kill.append(((v1.cross(y0)).applyfunc(at), (v1.cross(yl)).applyfunc(at)))
def jet_vector(Cf, Df):
    vec = []
    for comp in range(3):
        for k in range(-Dt, Dt+1):
            vec.append(sp.expand(Cf[comp]*T**(Dt+5)).coeff(T, k + Dt + 5))
    for comp in range(3):
        for k in range(-Dt, Dt+1):
            vec.append(sp.expand(Df[comp]*T**(Dt+5)).coeff(T, k + Dt + 5))
    return sp.Matrix(vec)
K = [jet_vector(*kj) for kj in kill]
if K:
    Kmat = sp.Matrix.hstack(*K)
    print("  Killing jets satisfy the system:", all(sp.expand(v) == 0 for v in (M*Kmat)))
    joint = sp.Matrix.hstack(Kmat, *ns) if ns else Kmat
    print(f"  rank of [Killing jets | nullspace] = {joint.rank()}  (Killing jets: {Kmat.rank()}, nullspace: {len(ns)})")
    print("  => nullspace == span of Killing jets:", joint.rank() == Kmat.rank() == len(ns))
else:
    print("  no mode-n Killing fields; nullspace dim", len(ns), "=> non-isometric first-order deformations:", len(ns) > 0)
for v in ns:
    den = sp.lcm([sp.fraction(sp.nsimplify(x)) [1] for x in v if x != 0]); v = v*den
    Cf = Cv.subs(dict(zip(cs, v[:len(cs)]))).applyfunc(sp.expand); Df = Dv.subs(dict(zip(ds, v[len(cs):]))).applyfunc(sp.expand)
    strong_ok = all(sp.expand(r) == 0 for r in Ms*v)
    print(f"   C = {list(Cf)} ; C' = {list(Df)} | strong-QS: {strong_ok}")
print(f"done in {time.time()-T0:.0f} s")
