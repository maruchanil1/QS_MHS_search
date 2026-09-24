"""TASK 1: non-degenerate validation (iota, I, K, p' all nonzero) of the Boozer normal-flow reduction
(flow_core.py) on two EXACT axisymmetric closed-line MHS congruences with elementary Boozer angles
(boozer_axisym.py, exact truncated-series jets from series_jets.py):
  (i)  Solov'ev via the 1:1:2 oscillator V = (x^2+y^2+4z^2)/2:  iota = 2, p' != 0, I != 0, K != 0
       x = R_z(s) ( 2 sqrt(m) cosh(sig) cos t, -2 sqrt(m) sinh(sig) sin t, kappa m cos(2t+delta) ),
       2 m cosh(2 sig) = S const; flux label lam = e^{sig}, m = S lam^2/(lam^4+1).
  (ii) Kepler force-free family V = -1/rho:  iota = 1, p' = 0, K != 0, j != 0
       x = R_z(s) R_x(i(eps)) (cos eta - eps, sqrt(1-eps^2) sin eta, 0), dt/deta = 1 - eps cos eta,
       sin i = eps/(C sqrt(1-eps^2)); flux label lam = eps.
Checks at exact rational points: (1),(2),(4'),(4), sqrt g = (G+iota I)/B^2, G,I,p flux functions, K = K(psi,theta),
C1 = C2 = 0, W.Da closed form, x_psi = Y x a/|a|^2 + mu a, mu_alg = mu_true, COMPAT = 0,
and the round-2 structural identities (task 2): A_phi = |a|^2 + I C1, A_theta = iota|a|^2 - G C1, A_0 = d_a C1.
Run: python3 task1_validate.py      (~2 min; log in task1_validate.log)
"""
import sympy as sp, time, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from boozer_axisym import AxisymCongruence, Rz
t0 = time.time()

lam, s, tau = sp.symbols('lam s tau', positive=True)
import flow_core as fc
print(f"flow_core built ({fc.BUILD_TIME:.1f} s)")
# structural identities (exact polynomial identities in the jets)
A_ph, A_th, A_0 = fc.A_ph, fc.A_th, fc.A_0
a2 = fc.a.dot(fc.a)
da = lambda E: fc.I*fc.dph(E) - fc.G*fc.dth(E)
print("[task 2 structure] A_phi   == |a|^2 + I C1      :", sp.expand(A_ph - a2 - fc.I*fc.C1) == 0)
print("[task 2 structure] A_theta == iota|a|^2 - G C1  :", sp.expand(A_th - fc.iota*a2 + fc.G*fc.C1) == 0)
print("[task 2 structure] A_0     == (I d_phi - G d_theta) C1 :", sp.expand(A_0 - da(fc.C1)) == 0)
print("   => on a surface with C1 == 0:  d_psi C1 = |a|^2 D(mu) + R   (magnetic differential equation for the Boozer gauge mu)")


def run(name, cong, point, direct=True):
    print(f"\n===== {name} =====")
    if direct:
        res = cong.check_identities(point)
        for k, v in res.items():
            print(f"  {k}: {v}")
        print(f"  (direct derivations done, {time.time()-t0:.1f} s)")
    sub, xpsi_true, chk = cong.jets(point)
    for k, v in chk.items():
        print(f"  {k}: {v}")
    print(f"  jets by exact series ({time.time()-t0:.1f} s); x_theta = {[sub[fc.Xs[(i,1,0)]] for i in range(3)]}, "
          f"x_phi = {[sub[fc.Xs[(i,0,1)]] for i in range(3)]}, x_psi = {list(xpsi_true)}")
    vals = {k: sub[getattr(fc, k)] for k in ('iota', 'G', 'I', 'Gp', 'Ip', 'pp')}
    vals['K'] = sub[fc.Ks[0]]; vals['Kp'] = sub[fc.Ks[1]]; vals['f_psi'] = sub[fc.Fp[0]]
    print(f"  flux data: { {k: sp.simplify(vv) for k, vv in vals.items()} }")
    # identities (1),(2),(4'),(4) from the jets
    x_t = sp.Matrix([sub[fc.Xs[(i, 1, 0)]] for i in range(3)]); x_p = sp.Matrix([sub[fc.Xs[(i, 0, 1)]] for i in range(3)])
    Wv = x_p + vals['iota']*x_t
    sg = xpsi_true.dot(x_t.cross(x_p))
    fval = Wv.dot(Wv)/(vals['G'] + vals['iota']*vals['I'])
    S_ = sp.simplify
    print(f"  sqrt g = {S_(sg)},  |W|^2/(G+iota I) = {S_(fval)},  (1): {S_(Wv.dot(x_p) - vals['G']*sg)}, (2): {S_(Wv.dot(x_t) - vals['I']*sg)}, "
          f"(4'): {S_(Wv.dot(xpsi_true) - vals['K']*sg)}, (4): {S_(vals['iota']*vals['Kp'] - (vals['Gp'] + vals['iota']*vals['Ip'] + vals['pp']*sg))}")
    # decomposition of x_psi
    a_n = fc.a.xreplace(sub); Y_n = fc.Y.xreplace(sub)
    v_n = Y_n.cross(a_n)/a_n.dot(a_n)
    mu_true = sp.simplify((xpsi_true - v_n).dot(a_n)/a_n.dot(a_n))
    dec = (xpsi_true - v_n - mu_true*a_n).applyfunc(sp.simplify)
    print(f"  x_psi - (Y x a/|a|^2 + mu_true a) = {list(dec)},  mu_true = {mu_true}")
    C1_n = fc.ev(fc.C1, sub); C2_n = fc.ev(fc.uD(fc.f), sub)
    wda = fc.ev(fc.WDa - fc.WDa_claim, sub); wda_val = fc.ev(fc.WDa, sub)
    print(f"  C1 = {C1_n}, C2 = {C2_n}, W.Da - closed form = {wda}, W.Da = {wda_val}")
    comp, mu_alg_n = fc.COMPAT_at(sub)
    print(f"  mu_alg - mu_true = {sp.simplify(mu_alg_n - mu_true)};  COMPAT = {comp}   ({time.time()-t0:.1f} s)")
    return dict(COMPAT=comp, dmu=sp.simplify(mu_alg_n - mu_true), C1=C1_n, C2=C2_n, WDa_res=wda, sub=sub, xpsi=xpsi_true)


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
print("\nSolov'ev: m =", m.subs(pt), " B^2(t) =", sp.simplify(osc.B2), "\n  Lambda = G + iota I =", osc.Lam, "  J =", osc.J)
r_osc = run("(i) Solov'ev via 1:1:2 (iota = 2), point 1", osc, pt, direct=True)
pt2 = {lam: sp.Integer(3), s: sp.atan(sp.Rational(3, 4)), tau: sp.atan(sp.Rational(5, 12))}
r_osc2 = run("(i) Solov'ev, point 2 (lam = 3)", osc, pt2, direct=False)

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
r_kep = run("(ii) Kepler force-free (iota = 1), point 1", kep, ptk, direct=True)
# second Kepler instance with C = 25/36 so that eps = 5/13 gives sin i = 3/5 (all jets rational)
C2_ = sp.Rational(25, 36)
sin_i2 = eps/(C2_*beta); cos_i2 = sp.sqrt(1 - sin_i2**2)
Rx2 = sp.Matrix([[1, 0, 0], [0, cos_i2, -sin_i2], [0, sin_i2, cos_i2]])
kep2 = AxisymCongruence(Rx2*e_vec, lam, s, tau, dt_deta, Fint_k, 1, Vkep)
ptk2 = {lam: sp.Rational(5, 13), s: sp.atan(sp.Rational(3, 4)), tau: sp.atan(sp.Rational(12, 5))}
r_kep2 = run("(ii) Kepler, point 2 (eps = 5/13, C = 25/36)", kep2, ptk2, direct=False)

print("\nSUMMARY (COMPAT, mu_alg-mu_true, C1, C2, W.Da-closedform):")
allok = True
for nm, r in (("Solov'ev pt1", r_osc), ("Solov'ev pt2", r_osc2), ("Kepler pt1", r_kep), ("Kepler pt2", r_kep2)):
    print(f"  {nm}: {r['COMPAT']}, {r['dmu']}, {r['C1']}, {r['C2']}, {r['WDa_res']}")
    allok &= all(v == 0 for v in (r['COMPAT'], r['dmu'], r['C1'], r['C2'], r['WDa_res']))
print("ALL RESIDUALS EXACTLY ZERO:", allok)
print(f"done in {time.time()-t0:.1f} s")
