"""TASK 2, first-order closed-line QS-MHS deformations of the Kepler force-free congruence (iota = 1, p' = 0) by
polynomial vector fields xi of degree <= DEG.  Same linear system as lindef_solovev.py (lindef_core.py).
x0(psi,s,eta) = R_z(s) R_x(i(psi)) e(eta), e = a(cos eta - eps, sqrt(1-eps^2) sin eta, 0), eps = psi (label),
sin i = eps/(C sqrt(1-eps^2)), t = sqrt(a^3/mu)(eta - eps sin eta); d_t = d_eta/tau_eta, d_psi|_t = d_psi - (tau_psi/tau_eta) d_eta.
All conditions are Laurent polynomials in (E, T = e^{i eta}) divided by powers of (1 - eps cos eta): we multiply by
(1 - eps cos eta)^8 and evaluate the psi-dependent constants at the rational point (C, eps) = (5/4, 3/5) (sin i = 3/5,
cos i = 4/5) AFTER the psi-differentiation.  Identities in (s, eta) at one psi are necessary conditions, so the
nullspace found here CONTAINS the true one; a second point (25/36, 5/13) is intersected for safety.
Run: python3 lindef_kepler.py [DEG] [NPTS=2]
"""
import sys, time
import sympy as sp
from lindef_core import *
T0 = time.time()
DEG = int(sys.argv[1]) if len(sys.argv) > 1 else 2
NPTS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
I = sp.I
psi = sp.symbols('psi', real=True)
a, mu = sp.Integer(1), sp.Integer(1)
cosE = lambda k: (E**k + E**(-k))/2; sinE = lambda k: (E**k - E**(-k))/(2*I)
cosT = lambda k: (T**k + T**(-k))/2; sinT = lambda k: (T**k - T**(-k))/(2*I)

def setup(Cv):
    eps = psi; beta = sp.sqrt(1 - eps**2)
    sin_i = eps/(Cv*beta); cos_i = sp.sqrt(1 - sin_i**2)
    Rx = sp.Matrix([[1, 0, 0], [0, cos_i, -sin_i], [0, sin_i, cos_i]])
    Rz = sp.Matrix([[cosE(1), -sinE(1), 0], [sinE(1), cosE(1), 0], [0, 0, 1]])
    e = sp.Matrix([a*(cosT(1) - eps), a*beta*sinT(1), 0])
    x0 = Rz*Rx*e
    tau_eta = sp.sqrt(a**3/mu)*(1 - eps*cosT(1))
    tau_psi = -sp.sqrt(a**3/mu)*sinT(1)
    deta = lambda f: I*T*sp.diff(f, T)
    dt = lambda f: deta(f)/tau_eta
    ds = lambda f: I*E*sp.diff(f, E)
    dpsi = lambda f: sp.diff(f, psi) - (tau_psi/tau_eta)*deta(f)
    return x0, dpsi, ds, dt, tau_eta

fields = basis_fields(DEG)
print(f"Kepler force-free congruence, DEG = {DEG} ({len(fields)} coefficients)")
Mtot = None; Mstot = None
for (Cv, ev) in [(sp.Rational(5, 4), sp.Rational(3, 5)), (sp.Rational(25, 36), sp.Rational(5, 13))][:NPTS]:
    x0, dpsi, ds, dt, tau_eta = setup(Cv)
    clear = tau_eta**8
    post = lambda ex: sp.expand(sp.together(sp.expand(ex.subs(psi, ev))))
    cols = condition_columns(x0, dpsi, ds, dt, fields, clear=clear, post=post, gens=(E, T))
    M = assemble(cols, ['dJ_s', 'dJ_t', 'W', 'M1', 'M2_s', 'M2_t'], len(fields))
    Ms = assemble(cols, ['SQ_ts', 'SQ_tl'], len(fields))
    # sanity: all entries rational
    assert all((not v.has(psi)) and v.is_number for v in M) and all((not v.has(psi)) and v.is_number for v in Ms)
    print(f"  point (C,eps)=({Cv},{ev}): system {M.shape[0]} x {M.shape[1]}, rank {M.rank()}, {time.time()-T0:.0f} s")
    Mtot = M if Mtot is None else Mtot.col_join(M)
    Mstot = Ms if Mstot is None else Mstot.col_join(Ms)
ns = Mtot.nullspace()
print(f"  combined rank {len(fields)-len(ns)}, nullspace dim {len(ns)}, {time.time()-T0:.0f} s")
x0, _, _, _, _ = setup(sp.Rational(5, 4)); x0 = x0.subs(psi, sp.Rational(3, 5))
trivial = [sp.Matrix([X, Y, Z])]          # a-scaling (dilation) is the only free family parameter besides C
nontriv = []
for v in ns:
    v = clean_vec(v); f = field_from(v, fields)
    kil = is_killing(f); strong_ok = all(r == 0 for r in (Mstot*v).applyfunc(sp.expand)) if Mstot.shape[0] else True
    modes = s_modes(f, x0)
    a0 = sp.symbols('a0'); comb = f - a0*trivial[0]
    eqs = []
    for comp in comb: eqs += sp.Poly(sp.expand(comp), X, Y, Z).coeffs()
    triv = bool(sp.solve(eqs, [a0], dict=True)) if not kil else True
    print(f"   xi = {list(f)}  | Killing: {kil} | div = {divergence(f)} | s-modes {sorted(modes)} | strong-QS: {strong_ok} | trivial: {triv}")
    if not triv: nontriv.append(f)
print(f"\n  RESULT DEG={DEG}: nullspace {len(ns)}; non-trivial (beyond Killing and dilation): {len(nontriv)}")
for f in nontriv: print("   NON-TRIVIAL (at the two test points; must be re-verified for all psi):", list(f))
print(f"done in {time.time()-T0:.0f} s")
