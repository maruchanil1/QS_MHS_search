"""Linearisation of  xdd = -grad V  about a constant-speed closed orbit (the axis) and the first-order QS clock.

Units: B0 = 1 (V, Pi in units B0^2), s = arclength = time.  Frenet frame (T,N,Bn), kappa, tau functions of s.
On the axis: grad Pi = kappa N (Pi = -V).  Hess Pi . T = D(kappa N) = -kappa^2 T + kappa' N + kappa tau Bn.
Free on-axis data: P_NN, P_NB, P_BB (Frenet components of Hess Pi).

Results printed by this script (all symbolic):
  (L1) T-component of the linearised equation  <=>  (xi' - 2 kappa X)' = 0, xi' - 2 kappa X = delta E (energy variation)
  (L2) transverse system for dx = xi T + X N + Y Bn with delta E = 0:
         X'' + (3 kappa^2 - tau^2) X - 2 tau Y' - tau' Y = P_NN X + P_NB Y
         Y'' - tau^2 Y + 2 tau X' + tau' X          = P_NB X + P_BB Y
  (L3) first-order QS clock  <=>  kappa X is a pure harmonic e^{iks} on the field-line pair (k = 2 pi (iota0 - N)/L):
         X = (eta/kappa) e^{iks},  Y = X (kappa^2/eta^2)(sigma - i)   [flux conservation Im(conj(X) Y) = const fixes the -i]
       => closed forms for P_NN, P_NB, P_BB in terms of (kappa, tau, eta, sigma, k) and ONE compatibility ODE for sigma.
  (L4) the compatibility ODE is the s-derivative of a Riccati equation; its first integral is the on-axis parallel
       current density j = (curl B).T computed from the field-line linearisation dx' = (grad B) dx.
  (L5) numerical check of (L2) on explicit potentials with an exact constant-speed orbit (ellipse: kappa' != 0, tau = 0,
       P_NB != 0;  helix in V(r): tau != 0).

Run: python3 axis_linearization.py     (~1 min)
"""
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp

s = sp.symbols('s', real=True)
kap = sp.Function('kappa', real=True)(s); tau = sp.Function('tau', real=True)(s)
PNN = sp.Function('P_NN')(s); PNB = sp.Function('P_NB')(s); PBB = sp.Function('P_BB')(s)
xi = sp.Function('xi')(s); Xf = sp.Function('X')(s); Yf = sp.Function('Y')(s)

def D(f):
    """Covariant s-derivative of a vector given by Frenet components (f_T, f_N, f_B)."""
    fT, fN, fB = f
    return sp.Matrix([sp.diff(fT, s) - kap * fN,
                      sp.diff(fN, s) + kap * fT - tau * fB,
                      sp.diff(fB, s) + tau * fN])

# Hess Pi . T from grad Pi = kappa N along the axis
HT = D(sp.Matrix([0, kap, 0]))
print("Hess Pi . T =", list(sp.simplify(HT)), "  (expected [-kappa^2, kappa', kappa tau])")
HessPi = sp.Matrix([[HT[0], HT[1], HT[2]],
                    [HT[1], PNN, PNB],
                    [HT[2], PNB, PBB]])
dx = sp.Matrix([xi, Xf, Yf])
eq = sp.expand(D(D(dx)) - HessPi * dx)      # linearised: dx'' = Hess Pi dx   (V = -Pi)

# (L1)
eqT = sp.expand(eq[0])
print("(L1) T-equation:", eqT, "  == d/ds (xi' - 2 kappa X) ?",
      sp.simplify(eqT - sp.diff(sp.diff(xi, s) - 2 * kap * Xf, s)) == 0)

# (L2): substitute xi' = 2 kappa X (delta E = 0); xi itself drops out
dE = sp.symbols('dE')
sub = {sp.diff(xi, s, 2): sp.diff(2 * kap * Xf, s), sp.diff(xi, s): 2 * kap * Xf + dE}
eqN = sp.expand(eq[1].subs(sub)); eqB = sp.expand(eq[2].subs(sub))
print("(L2) xi drops out of N,B equations:", (eqN.has(xi), eqB.has(xi)))
eqN_claim = sp.diff(Xf, s, 2) + (3 * kap**2 - tau**2) * Xf - 2 * tau * sp.diff(Yf, s) - sp.diff(tau, s) * Yf - PNN * Xf - PNB * Yf
eqB_claim = sp.diff(Yf, s, 2) - tau**2 * Yf + 2 * tau * sp.diff(Xf, s) + sp.diff(tau, s) * Xf - PNB * Xf - PBB * Yf
print("     N-eq (dE=0) matches claim:", sp.simplify(eqN.subs(dE, 0) - eqN_claim) == 0,
      "| dE enters N-eq as:", sp.factor(eqN.coeff(dE)))
print("     B-eq (dE=0) matches claim:", sp.simplify(eqB.subs(dE, 0) - eqB_claim) == 0,
      "| dE enters B-eq as:", sp.factor(eqB.coeff(dE)))

# ---------------------------------------------------------------- (L3) first-order clock
eta, k = sp.symbols('eta k', positive=True)
sig = sp.Function('sigma', real=True)(s)
Xc = (eta / kap) * sp.exp(sp.I * k * s)
w = (kap**2 / eta**2) * (sig - sp.I)
Yc = Xc * w
# real symbols for the derivatives (sympy does not know that kappa', sigma'' ... are real)
k0, k1, k2, t0, t1, s0, s1, s2 = sp.symbols('kappa kappa1 kappa2 tau tau1 sigma sigma1 sigma2', real=True)
jr = sp.symbols('j', real=True)
realify = {sp.diff(kap, s, 2): k2, sp.diff(kap, s): k1, sp.diff(tau, s): t1,
           sp.diff(sig, s, 2): s2, sp.diff(sig, s): s1}
def R(e):
    e = e.subs(realify)   # (order matters: sympy substitutes higher derivatives first automatically for dict? do explicit)
    return e.subs({kap: k0, tau: t0, sig: s0})
def realify_expr(e):
    e = e.subs(sp.diff(kap, s, 2), k2).subs(sp.diff(sig, s, 2), s2).subs(sp.diff(kap, s), k1).subs(sp.diff(tau, s), t1).subs(sp.diff(sig, s), s1)
    return e.subs({kap: k0, tau: t0, sig: s0})
print("(L3) flux conservation Im(conj(X) Y) =", sp.simplify(sp.im(sp.conjugate(Xc) * Yc)), " (constant)")
RN = realify_expr(sp.expand((sp.diff(Xc, s, 2) + (3 * kap**2 - tau**2) * Xc - 2 * tau * sp.diff(Yc, s) - sp.diff(tau, s) * Yc) / Xc))
RB = realify_expr(sp.expand((sp.diff(Yc, s, 2) - tau**2 * Yc + 2 * tau * sp.diff(Xc, s) + sp.diff(tau, s) * Xc) / Xc))
RN = sp.expand(sp.simplify(RN)); RB = sp.expand(sp.simplify(RB))
u = realify_expr(sp.re(w)); v = realify_expr(sp.im(w))
RN_re, RN_im = sp.re(RN), sp.im(RN)
RB_re, RB_im = sp.re(RB), sp.im(RB)
# P_NN + P_NB w = RN ,  P_NB + P_BB w = RB   (P's real)
P_BB_expr = sp.simplify(RB_im / v)
P_NB_fromB = sp.simplify(RB_re - u * RB_im / v)
P_NB_fromN = sp.simplify(RN_im / v)
P_NN_expr = sp.simplify(RN_re - u * RN_im / v)
compat = sp.simplify(sp.expand(P_NB_fromB - P_NB_fromN))
print("     P_BB =", sp.collect(sp.expand(P_BB_expr), [s0, s1, s2]))
print("     P_NB (from N-eq) =", sp.collect(sp.expand(P_NB_fromN), [s0, s1]))
print("     P_NN =", sp.collect(sp.expand(P_NN_expr), [s0, s1]))
print("     compatibility (P_NB from B-eq minus from N-eq) = 0 :")
print("        ", sp.collect(sp.expand(compat * eta**2 / k0**2), [s2, s1]))

# ---------------------------------------------------------------- (L4) first integral: on-axis parallel current
# field-line linearisation dx' = G dx, with G T = kappa N (from B.grad B = grad Pi).  Transverse block:
# G (X N + Y Bn) = (X' - tau Y) N + (tau X + Y') Bn  (+ T-component).  j = G_BN - G_NB (Frenet components).
X1, X2 = sp.re(Xc), sp.im(Xc); Y1, Y2 = sp.re(Yc), sp.im(Yc)
M = sp.Matrix([[sp.diff(X1, s) - tau * Y1, sp.diff(X2, s) - tau * Y2],
               [tau * X1 + sp.diff(Y1, s), tau * X2 + sp.diff(Y2, s)]]) * sp.Matrix([[X1, X2], [Y1, Y2]]).inv()
j = sp.simplify(realify_expr(sp.simplify(M[1, 0] - M[0, 1])))
print("(L4) on-axis parallel current density j = (curl B).T = G_BN - G_NB =", sp.collect(sp.expand(j), [s0, s1]))
riccati = sp.solve(sp.Eq(j, jr), s1)[0]
print("     => Riccati:  sigma' =", sp.collect(sp.expand(riccati), [s0]))
# d/ds of j: chain rule with the symbol dictionary
dj = (sp.diff(j, k0) * k1 + sp.diff(j, k1) * k2 + sp.diff(j, t0) * t1 + sp.diff(j, s0) * s1 + sp.diff(j, s1) * s2)
ratio = sp.simplify(compat / dj)
print("     compatibility ODE / (dj/ds) =", ratio, "   (nonzero => compat <=> j = const)")
lap = sp.simplify(-k0**2 + P_NN_expr + P_BB_expr)
print("     Laplacian Pi on axis =", sp.collect(sp.expand(lap), [s0, s1]))

# closed forms with sigma' eliminated via the Riccati and sigma'' via its derivative
ric_s2 = (sp.diff(riccati, k0) * k1 + sp.diff(riccati, t0) * t1 + sp.diff(riccati, s0) * riccati)
sub_r = {s2: ric_s2, s1: riccati}
PNNf = sp.simplify(P_NN_expr.subs(s2, ric_s2).subs(s1, riccati))
PNBf = sp.simplify(P_NB_fromN.subs(s2, ric_s2).subs(s1, riccati))
PBBf = sp.simplify(P_BB_expr.subs(s2, ric_s2).subs(s1, riccati))
lapf = sp.simplify(lap.subs(s2, ric_s2).subs(s1, riccati))
print("\n==== EXACT on-axis Hessian of Pi for a first-order-QS field (units B0^2, arclength; k = 2pi(iota0-N)/L) ====")
for name, e in [("P_NN", PNNf), ("P_NB", PNBf), ("P_BB", PBBf), ("Lap Pi", lapf)]:
    print(f"  {name} =", sp.collect(sp.expand(e), [s0, jr, k]))
print("  P_TT = -kappa^2,  P_TN = kappa',  P_TB = kappa tau")
print("  Lap Pi (vacuum j=0) as sum of squares:", sp.simplify(lapf.subs(jr, 0) - (2*k0**2 + 2*(t0 + k*eta**2/k0**2)**2 + 2*(k*s0 - k1/k0)**2)) == 0)

# helical (screw-invariant) reference values from the handoff (§10.4): P_NB = -(l2/l3) kappa'/kappa, P_BB = (z_s - l2 tau)/l3
l2, l3, zs = sp.symbols('lambda2 lambda3 z_s')
print("\n  P_BB - P_BB^hel =", sp.collect(sp.expand(PBBf - (zs - l2 * t0) / l3), [s0, jr, k]))
print("  P_NB - P_NB^hel =", sp.collect(sp.expand(PNBf + (l2 / l3) * k1 / k0), [s0, jr, k]))

import pickle
with open('hessian_closed_forms.pkl', 'wb') as fh:
    pickle.dump({'P_NN': sp.srepr(PNNf), 'P_NB': sp.srepr(PNBf), 'P_BB': sp.srepr(PBBf), 'riccati': sp.srepr(riccati),
                 'j': sp.srepr(j), 'symbols': 'kappa kappa1 kappa2 tau tau1 sigma eta k j'}, fh)

# ---------------------------------------------------------------- (L5) numerical checks of (L2)
print("\n(L5) numerical checks of the transverse system (L2) on explicit potentials")
x, y, z = sp.symbols('x y z', real=True)

def check(Vexpr, gamma_s, L, label, nsteps=400):
    """Vexpr: sympy potential with gamma_s(s) an exact unit-speed orbit (sympy Matrix in s).  Compare the true
    linearised solution (integrated with the exact Hessian) with the Frenet-frame system (L2)."""
    g = gamma_s
    T = g.diff(s); Tp = T.diff(s)
    kappa_e = sp.sqrt(Tp.dot(Tp)); N = Tp / kappa_e; Bn = T.cross(N)
    tau_e = -Bn.diff(s).dot(N)
    # verify orbit: gamma'' = -grad V(gamma), |gamma'| = 1
    gV = sp.Matrix([sp.diff(Vexpr, v) for v in (x, y, z)])
    subg = {x: g[0], y: g[1], z: g[2]}
    res = sp.simplify(Tp + gV.subs(subg))
    print(f"  [{label}] orbit residual gamma''+gradV:", list(res), " |T|^2-1:", sp.simplify(T.dot(T) - 1))
    H = sp.hessian(Vexpr, (x, y, z)).subs(subg)
    fr = sp.Matrix.hstack(T, N, Bn)
    Hf = sp.simplify(fr.T * (-H) * fr)      # Hess Pi in Frenet components
    print(f"  [{label}] Hess Pi Frenet components: TT-(-k^2)={sp.simplify(Hf[0,0]+kappa_e**2)}, TN-k'={sp.simplify(Hf[0,1]-kappa_e.diff(s))}, TB-k tau={sp.simplify(Hf[0,2]-kappa_e*tau_e)}")
    fH = sp.lambdify(s, H, 'numpy'); ffr = sp.lambdify(s, fr, 'numpy')
    fk = sp.lambdify(s, kappa_e, 'numpy'); ft = sp.lambdify(s, tau_e, 'numpy')
    ftp = sp.lambdify(s, tau_e.diff(s), 'numpy')
    fP = sp.lambdify(s, [Hf[1, 1], Hf[1, 2], Hf[2, 2]], 'numpy')
    # true linearised flow
    def rhs_true(si, u_):
        d, dd = u_[:3], u_[3:]
        return np.concatenate([dd, -np.array(fH(si), float) @ d])
    # Frenet system: state (X, X', Y, Y', xi) with xi' = 2 kappa X
    def rhs_fr(si, u_):
        X_, Xp, Y_, Yp, xi_ = u_
        kk, tt, tp = fk(si), ft(si), ftp(si); pnn, pnb, pbb = fP(si)
        Xpp = pnn * X_ + pnb * Y_ - (3 * kk**2 - tt**2) * X_ + 2 * tt * Yp + tp * Y_
        Ypp = pnb * X_ + pbb * Y_ + tt**2 * Y_ - 2 * tt * Xp - tp * X_
        return [Xp, Xpp, Yp, Ypp, 2 * kk * X_]
    rng = np.random.default_rng(1)
    err = 0.0
    for trial in range(3):
        X0, Xp0, Y0, Yp0 = rng.normal(size=4)
        # initial dx, dx' in Cartesian with xi=0, delta E = 0: xi' = 2 kappa X
        F0 = np.array(ffr(0.0), float); kk0 = fk(0.0); tt0 = ft(0.0)
        comp = np.array([0.0, X0, Y0]); compp = np.array([2 * kk0 * X0 - kk0 * X0, kk0 * 0.0 + Xp0 - tt0 * Y0, tt0 * X0 + Yp0])
        u0 = np.concatenate([F0 @ comp, F0 @ compp])
        solT = solve_ivp(rhs_true, (0, L), u0, rtol=1e-11, atol=1e-13, dense_output=True)
        solF = solve_ivp(rhs_fr, (0, L), [X0, Xp0, Y0, Yp0, 0.0], rtol=1e-11, atol=1e-13, dense_output=True)
        for si in np.linspace(0, L, 7)[1:]:
            F = np.array(ffr(si), float)
            comps_true = F.T @ solT.sol(si)[:3]
            Xf_, _, Yf_, _, xif = solF.sol(si)
            err = max(err, abs(comps_true[1] - Xf_), abs(comps_true[2] - Yf_), abs(comps_true[0] - xif))
    print(f"  [{label}] max |true - Frenet-system| over 3 random ICs, 6 times, (xi,X,Y): {err:.2e}")

# Example 1: ellipse (a cos s-ish) — use angle parameter th with unit-speed reparametrisation via numeric? Use instead
# an exact unit-speed planar curve: gamma = (cos s, sin s, 0) is a circle (kappa'=0). For kappa' != 0 use a helix-free
# 3-D example: the ellipse in arclength is not elementary, so test with V having the *circle* as orbit plus z-couplings,
# and separately a 2-D non-circular exact unit-speed curve: gamma(s) = (s - sin s... ) not unit speed.  Use instead the
# curve gamma(s) = (sin s, s/2 ... ) no.  ==> Use a helix (tau != 0) and an 'ellipse' handled by numeric arclength below.
R0, bet = sp.Rational(1), sp.pi / 5
# Helix with unit speed: (R0 cos(c s), R0 sin(c s), sin(bet) s) with c = cos(bet)/R0
c = sp.cos(bet) / R0
helix = sp.Matrix([R0 * sp.cos(c * s), R0 * sp.sin(c * s), sp.sin(bet) * s])
r2 = x**2 + y**2
kap_h = sp.cos(bet)**2 / R0
# V(r) with V'(R0) = kappa (centripetal), plus a non-symmetric second-order term vanishing to 2nd order on the helix? keep V(r)+z-term
f1 = sp.sqrt(r2) - R0
f2 = sp.im(sp.expand((x + sp.I * y) * sp.exp(-sp.I * c * z / sp.sin(bet)), complex=True))
f2 = y * sp.cos(c * z / sp.sin(bet)) - x * sp.sin(c * z / sp.sin(bet))
Vhelix = kap_h * f1 + sp.Rational(3, 7) * f1**2 + sp.Rational(2, 5) * f1 * f2 + sp.Rational(1, 3) * f2**2 * (1 + z**2 / 4)
check(Vhelix, helix, 2.5, "helix, V(r,z)")

# Example 2: planar unit-speed curve with kappa' != 0: use the arclength-parametrised catenary-like curve
# gamma(s) = (asinh(s), sqrt(1+s^2), 0): |gamma'| = sqrt(1/(1+s^2) + s^2/(1+s^2)) = 1.  kappa = 1/(1+s^2).
cat = sp.Matrix([sp.asinh(s), sp.sqrt(1 + s**2), 0])
# potential with this curve as orbit: level set y = cosh(x) i.e. f = y - cosh(x) = 0; grad f = (-sinh x, 1, 0), |grad f| = cosh x
# need grad V = -kappa N on the curve; N = (-sinh x,1,0)/cosh x (pointing 'inward' i.e. toward +y), kappa = 1/cosh^2 x
# => grad V = -(1/cosh^3 x) grad f  on the curve.  V = -f/cosh(x)^3 + f^2 * h(x,y,z) + z^2 m(x,y) + z f n(x,y)
f = y - sp.cosh(x)
Vcat = -f / sp.cosh(x)**3 + f**2 * (sp.Rational(1, 2) + x**2 / 5) + z**2 * (sp.Rational(1, 3) + sp.sin(x) / 4) + z * f * (sp.Rational(2, 5) + y / 7)
check(Vcat, cat, 1.5, "catenary (kappa'!=0, P_NB!=0), V(x,y,z)")
