"""TASK 1: non-degenerate validation (iota, I, K, p' all nonzero) of the Boozer normal-flow reduction
(flow_core.py) on two EXACT axisymmetric closed-line MHS congruences with elementary Boozer angles
(boozer_axisym.py):
  (i)  Solov'ev via the 1:1:2 oscillator V = (x^2+y^2+4z^2)/2:  iota = 2, p' != 0, I != 0, K != 0
       x = R_z(s) ( 2 sqrt(m) cosh(sig) cos t, -2 sqrt(m) sinh(sig) sin t, kappa m cos(2t+delta) ),
       2 m cosh(2 sig) = S const; flux label lam = e^{sig}, m = S lam^2/(lam^4+1).
  (ii) Kepler force-free family V = -1/rho:  iota = 1, p' = 0, K != 0, j != 0
       x = R_z(s) R_x(i(eps)) (cos eta - eps, sqrt(1-eps^2) sin eta, 0), dt/deta = 1 - eps cos eta,
       sin i = eps/(C sqrt(1-eps^2)); flux label lam = eps.
Checks at exact rational points: (1),(2),(4'),(4), sqrt g = (G+iota I)/B^2, G,I flux functions, K = K(psi,theta),
C1 = C2 = 0, W.Da closed form, mu_alg = mu_true, COMPAT = 0.
Run: python3 task1_validate.py      (~1-2 min)
"""
import sympy as sp, time, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from boozer_axisym import AxisymCongruence, Rz
t0 = time.time()

lam, s, tau = sp.symbols('lam s tau', positive=True)


def run(name, cong, point):
    print(f"\n===== {name} =====")
    res = cong.check_identities(point)
    for k, v in res.items():
        print(f"  {k}: {v}")
    print(f"  ({time.time()-t0:.1f} s)")
    import flow_core as fc
    sub = cong.jets(point)
    print(f"  jets evaluated ({time.time()-t0:.1f} s); x_theta = {[sub[fc.Xs[(i,1,0)]] for i in range(3)]}, x_phi = {[sub[fc.Xs[(i,0,1)]] for i in range(3)]}")
    # true x_psi and mu_true
    xpsi_true = sp.Matrix([sp.nsimplify(sp.simplify(cong.x_ps[i].xreplace(point))) for i in range(3)])
    a_n = fc.a.xreplace(sub); Y_n = fc.Y.xreplace(sub)
    v_n = Y_n.cross(a_n)/a_n.dot(a_n)
    mu_true = sp.nsimplify(sp.simplify((xpsi_true - v_n).dot(a_n)/a_n.dot(a_n)))
    dec = (xpsi_true - v_n - mu_true*a_n).applyfunc(sp.simplify)
    print(f"  x_psi - (Y x a/|a|^2 + mu_true a) = {list(dec)},  mu_true = {mu_true}")
    C1_n = fc.ev(fc.C1, sub); C2_n = fc.ev(fc.uD(fc.f), sub)
    wda = fc.ev(fc.WDa - fc.WDa_claim, sub); wda_val = fc.ev(fc.WDa, sub)
    print(f"  C1 = {C1_n}, C2 = {C2_n}, W.Da - closed form = {wda}, W.Da = {wda_val}")
    comp, mu_alg_n = fc.COMPAT_at(sub)
    print(f"  mu_alg - mu_true = {sp.simplify(mu_alg_n - mu_true)};  COMPAT = {comp}   ({time.time()-t0:.1f} s)")
    return comp, sp.simplify(mu_alg_n - mu_true), C1_n, C2_n, wda


# ---------- (i) Solov'ev / 1:1:2 ----------
S, kap = sp.Rational(17, 4), sp.Integer(1)
delta = sp.atan(sp.Rational(4, 3))            # cos delta = 3/5, sin delta = 4/5
m = S*lam**2/(lam**4 + 1)
A = sp.sqrt(S)*(lam**2 + 1)/sp.sqrt(lam**4 + 1)     # 2 sqrt(m) cosh sig
Bc = sp.sqrt(S)*(lam**2 - 1)/sp.sqrt(lam**4 + 1)    # 2 sqrt(m) sinh sig
X0 = sp.Matrix([A*sp.cos(tau), -Bc*sp.sin(tau), kap*m*sp.cos(2*tau + delta)])
integrand = sp.expand(X0.diff(tau).dot(X0.diff(tau)))
Fint = sp.integrate(integrand, tau)
Vosc = lambda x: (x[0]**2 + x[1]**2 + 4*x[2]**2)/2
osc = AxisymCongruence(X0, lam, s, tau, sp.Integer(1), Fint, 2, Vosc)
pt = {lam: sp.Integer(2), s: sp.atan(sp.Rational(5, 12)), tau: sp.atan(sp.Rational(3, 4))}
print("Solov'ev: m =", m.subs(pt), " B^2(t) =", sp.simplify(osc.B2), " Lambda = G + iota I =", osc.Lam, " J =", osc.J)
r_osc = run("(i) Solov'ev via 1:1:2 (iota = 2)", osc, pt)
pt2 = {lam: sp.Integer(3), s: sp.atan(sp.Rational(3, 4)), tau: sp.atan(sp.Rational(5, 12))}
r_osc2 = run("(i) Solov'ev, 2nd point (lam = 3)", osc, pt2)

# ---------- (ii) Kepler force-free ----------
C = sp.Rational(5, 4)
eps = lam
beta = sp.sqrt(1 - eps**2)
sin_i = eps/(C*beta); cos_i = sp.sqrt(1 - sin_i**2)
Rx = sp.Matrix([[1, 0, 0], [0, cos_i, -sin_i], [0, sin_i, cos_i]])
e_vec = sp.Matrix([sp.cos(tau) - eps, beta*sp.sin(tau), 0])
X0k = Rx*e_vec
dt_deta = 1 - eps*sp.cos(tau)
Fint_k = tau + eps*sp.sin(tau)
Vkep = lambda x: -1/sp.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
kep = AxisymCongruence(X0k, lam, s, tau, dt_deta, Fint_k, 1, Vkep)
ptk = {lam: sp.Rational(3, 5), s: sp.atan(sp.Rational(5, 12)), tau: sp.atan(sp.Rational(3, 4))}
print("\nKepler: B^2 =", sp.simplify(kep.B2), " Lambda =", kep.Lam, " J =", kep.J)
r_kep = run("(ii) Kepler force-free (iota = 1)", kep, ptk)
ptk2 = {lam: sp.Rational(5, 13), s: sp.atan(sp.Rational(3, 4)), tau: sp.atan(sp.Rational(12, 5))}
r_kep2 = run("(ii) Kepler, 2nd point (eps = 5/13)", kep, ptk2)

print("\nSUMMARY (COMPAT, mu_alg-mu_true, C1, C2, W.Da-closedform):")
for nm, r in (("Solov'ev pt1", r_osc), ("Solov'ev pt2", r_osc2), ("Kepler pt1", r_kep), ("Kepler pt2", r_kep2)):
    print(f"  {nm}: {r}")
print(f"done in {time.time()-t0:.1f} s")
