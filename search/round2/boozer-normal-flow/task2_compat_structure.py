"""TASK 2: structure of the compatibility condition COMPAT = d_psi C1 |_(mu = mu_alg) and of its psi-chain.

Notation (flow_core): x_t, x_p tangents, W = x_p + iota x_t (= D x, D = d_phi + iota d_theta), a = I x_p - G x_t
(= d_a x, d_a = I d_phi - G d_theta), u = d_phi + N d_theta, chi = theta - N phi, f = |W|^2/(G+iota I) (= sqrt g),
n = x_t x x_p, P = |n|^2/f, K = B_psi = K(chi) with (iota - N) K' = G' + iota I' + p' f  [(4)],
x_psi = v + mu a = K/(G+iota I) W + mu a + (f/|n|) n/|n|.

Claims verified here (exact arithmetic):
 [A] polynomial identities:  coeff(mu_phi) = |a|^2 + I C1,  coeff(mu_theta) = iota|a|^2 - G C1,  coeff(mu) = d_a C1
     => on a surface with C1 == 0:   d_psi C1 = |a|^2 D(mu) + R          (magnetic differential equation for mu)
 [B] R = -(G+N I) f K' - 2 (d_a W . n)/P - iota' |a|^2/(G+iota I) + (I' G - G' I) f          (mod C1, dC1, C2)
 [C] mu_alg = -h(chi) + m0^chi(chi) + m0^h,    h := f_psi/((G+N I) f'),
     m0^chi = [(iota-N)(2 K' f + K f') + (iota' I - G' - iota I') f] / ((G+iota I)(G+N I) f'),
     m0^h   = -2 (D W . n) / (P (G+iota I)(G+N I) f')            (this is the Maupertuis normal-curvature law)
 [D] COMPAT = |a|^2 [ -(iota - N) hhat'(chi) + Qhat ],  hhat := h - m0^chi,
     Qhat := D(m0^h) - 2 (d_a W . n)/(P |a|^2) + f [ (I'G - G'I) - (G+N I) K' ]/|a|^2 - iota'/(G+iota I)
     => COMPAT = 0  <=>  u(Qhat) = 0  and  oint Qhat dchi = 0;  then f_psi is DETERMINED (up to the gauge constant).
 [E] COMPAT is affine in (f_psi, f_psi') with the coefficients of [D] and independent of f_psi^(n>=2).
 [F] Qhat is affine in the flux-constant derivatives:  Qhat = Q0 + iota' Q1 + G' Q2 + I' Q3 + p' Q4
     with u(Q1) = 0 (iota' drops out of the surface condition u(Qhat) = 0).
Tests: random rational jets constrained to C1 = dC1 = d^2C1 = 0, u(f) = d u(f) = 0 (N = 1, 2; iota generic), and the
exact axisymmetric congruences (Solov'ev iota = 2, Kepler iota = 1) where (iota - N) hhat'_true = Qhat is checked.
Run: python3 task2_compat_structure.py     (~1.5 min)
"""
import sympy as sp, time, sys, random
sys.path.insert(0, __file__.rsplit('/', 1)[0])
t0 = time.time()
import flow_core as fc
from flow_core import Xs, Ks, Fp, Ms, iota, iotap, G, Gp, I, Ip, pp, N, mu, x_t, x_p, W, a, C1, f, dth, dph, D, uD
print(f"flow_core built ({fc.BUILD_TIME:.1f} s)")

a2 = a.dot(a)
da = lambda E: I*dph(E) - G*dth(E)
vD = lambda V: V.applyfunc(D)
vda = lambda V: V.applyfunc(da)
n_vec = x_t.cross(x_p)
P = n_vec.dot(n_vec)/f
fp = dth(f)                                  # f'(chi) = d_theta f
fpp = dth(fp)
K0, K1, K2 = Ks[0], Ks[1], Ks[2]
DWn = vD(W).dot(n_vec)                       # (D W).n = h(W,W) |n|
daWn = vda(W).dot(n_vec)                     # (d_a W).n = h(W,a) |n|

# [A]
print("[A] coeff(mu_phi)   == |a|^2 + I C1        :", sp.expand(fc.A_ph - a2 - I*C1) == 0)
print("[A] coeff(mu_theta) == iota |a|^2 - G C1   :", sp.expand(fc.A_th - iota*a2 + G*C1) == 0)
print("[A] coeff(mu)       == (I d_phi - G d_theta) C1 :", sp.expand(fc.A_0 - da(C1)) == 0)

# explicit formulas
R_claim = -(G + N*I)*f*K1 - 2*daWn/P - iotap*a2/(G + iota*I) + (Ip*G - Gp*I)*f
h_sym = Fp[0]/((G + N*I)*fp)
m0chi = ((iota - N)*(2*K1*f + K0*fp) + (iotap*I - Gp - iota*Ip)*f)/((G + iota*I)*(G + N*I)*fp)
m0h = -2*DWn/(P*(G + iota*I)*(G + N*I)*fp)
mu_claim = -h_sym + m0chi + m0h
hhat = h_sym - m0chi
Qhat = D(m0h) - 2*daWn/(P*a2) + f*((Ip*G - Gp*I) - (G + N*I)*K1)/a2 - iotap/(G + iota*I)
COMPAT_claim = a2*(-(iota - N)*dth(hhat) + Qhat)
R_fc = fc.dpsi_C1.xreplace({Ms[(1, 0)]: 0, Ms[(0, 1)]: 0, mu: 0})


def constrained_point(seed, Nval, iota_val):
    """random rational jets with C1 = dC1 = d^2 C1 = 0 and u(f) = d u(f) = 0 imposed exactly (solved linearly)."""
    random.seed(seed)
    rq = lambda: sp.Rational(random.randint(-9, 9), random.randint(1, 4))
    sub = {s_: rq() for s_ in list(Xs.values()) + Ks + Fp}
    sub.update({iota: iota_val, iotap: rq(), I: rq(), Ip: rq(), Gp: rq(), pp: rq(), N: Nval})
    # order 1: C1 = 0 -> G
    Gval = sp.solve(C1.xreplace({k: v for k, v in sub.items() if k != G}), G)[0]
    sub[G] = Gval
    # order 2: d_theta C1, d_phi C1, u f  ->  x0_20, x0_02, x0_11
    unk2 = [Xs[(0, 2, 0)], Xs[(0, 0, 2)], Xs[(0, 1, 1)]]
    eqs2 = [dth(C1), dph(C1), uD(W.dot(W))]
    sol2 = sp.solve([e.xreplace({k: v for k, v in sub.items() if k not in unk2}) for e in eqs2], unk2, dict=True)[0]
    sub.update(sol2)
    # order 3: d^2 C1 (3), d u f (2)  ->  x0_30, x0_21, x0_12, x1_30, x1_03
    eqs3 = [dth(dth(C1)), dth(dph(C1)), dph(dph(C1)), dth(uD(W.dot(W))), dph(uD(W.dot(W)))]
    jets3 = [Xs[(i, a_, b_)] for i in range(3) for a_ in range(4) for b_ in range(4 - a_) if a_ + b_ == 3]
    eqs3n = [sp.expand(e.xreplace({k: v for k, v in sub.items() if k not in jets3})) for e in eqs3]
    Amat = sp.Matrix([[e.coeff(s_) for s_ in jets3] for e in eqs3n])          # linear in the 3-jets
    assert all(sp.expand(e - sum(Amat[r, c]*jets3[c] for c in range(len(jets3))) - e.subs({s_: 0 for s_ in jets3})) == 0
               for r, e in enumerate(eqs3n))
    piv = Amat.rref()[1]
    unk3 = [jets3[c] for c in piv]
    sol3 = sp.solve([e.xreplace({k: v for k, v in sub.items() if k not in unk3}) for e in eqs3n], unk3, dict=True)
    assert len(sol3) == 1, sol3
    sub.update(sol3[0])
    return sub


def ev(e, sub):
    return sp.simplify(sp.sympify(e).xreplace(sub))


print("\nRandom constrained rational points (C1, dC1, d^2C1, u f, d u f all imposed exactly):")
allok = True
for seed, Nval, io in ((1, 1, sp.Rational(2, 5)), (2, 1, sp.Rational(-3, 7)), (3, 2, sp.Rational(3, 4)), (4, 0, sp.Rational(5, 3))):
    sub = constrained_point(seed, Nval, io)
    chk = {k: ev(e, sub) for k, e in (('C1', C1), ('dth C1', dth(C1)), ('dph C1', dph(C1)), ('u f', uD(f)), ('dth u f', dth(uD(f))))}
    rB = ev(R_fc - R_claim, sub)
    rC = ev(fc.mu_alg - mu_claim, sub)
    comp, _ = fc.COMPAT_at(sub)
    rD = sp.simplify(comp - ev(COMPAT_claim, sub))
    # [E] affine structure in f_psi data
    cF0 = ev(sp.diff(COMPAT_claim, Fp[0]) - a2*(iota - N)*fpp/((G + N*I)*fp**2), sub)
    cF1 = ev(sp.diff(COMPAT_claim, Fp[1]) + a2*(iota - N)/((G + N*I)*fp), sub)
    dep_hi = [s_ for s_ in Fp[2:] if fc.dpsi_C1.has(s_) or fc.mu_alg.has(s_)]
    # the actual flow_core COMPAT: check affinity numerically by second differences in Fp[0], Fp[1]
    def comp_at(df0, df1):
        s2 = dict(sub); s2[Fp[0]] = sub[Fp[0]] + df0; s2[Fp[1]] = sub[Fp[1]] + df1
        return fc.COMPAT_at(s2)[0]
    c00, c10, c20, c01, c02, c11 = comp_at(0, 0), comp_at(1, 0), comp_at(2, 0), comp_at(0, 1), comp_at(0, 2), comp_at(1, 1)
    affine = sp.simplify(c20 - 2*c10 + c00) == 0 and sp.simplify(c02 - 2*c01 + c00) == 0 and sp.simplify(c11 - c10 - c01 + c00) == 0
    slopes_ok = sp.simplify((c10 - c00) - ev(a2*(iota - N)*fpp/((G + N*I)*fp**2), sub)) == 0 and \
        sp.simplify((c01 - c00) + ev(a2*(iota - N)/((G + N*I)*fp), sub)) == 0
    # [F] Qhat affine in (iotap, Gp, Ip, K1) and u(coefficient of iotap) = 0
    Qc = {s_: sp.diff(Qhat, s_) for s_ in (iotap, Gp, Ip)}
    lin = all(sp.diff(Qc[s_], s2) == 0 for s_ in Qc for s2 in (iotap, Gp, Ip))
    u_iotap = ev(uD(Qc[iotap]), sub)
    ok = all(v == 0 for v in chk.values()) and rB == 0 and rC == 0 and rD == 0 and cF0 == 0 and cF1 == 0 and affine and slopes_ok and lin and u_iotap == 0
    allok &= ok
    print(f"  seed {seed}, N = {Nval}, iota = {io}: constraints {list(chk.values())}; [B] R - R_claim = {rB}; [C] mu_alg - mu_claim = {rC}; "
          f"[D] COMPAT - COMPAT_claim = {rD}; [E] slope residuals ({cF0}, {cF1}), affine in (f_psi, f_psi') = {affine}, slopes = {slopes_ok}, "
          f"depends on f_psi^(n>=2): {dep_hi}; [F] Qhat affine in (iota',G',I') = {lin}, u(dQhat/diota') = {u_iotap}   ({time.time()-t0:.0f} s)")
print("ALL STRUCTURAL CLAIMS HOLD AT ALL CONSTRAINED POINTS:", allok)

# ---------------- exact axisymmetric congruences: f_psi is determined by the surface ----------------
print("\nExact congruences (N = 0): check (iota - N) hhat'_true = Qhat, i.e. the surface data determine f_psi.")
from boozer_axisym import AxisymCongruence
lam, s, tau = sp.symbols('lam s tau', positive=True)
S, kap = sp.Rational(17, 4), sp.Integer(1)
delta = sp.atan(sp.Rational(4, 3))
m = S*lam**2/(lam**4 + 1)
A = sp.sqrt(S)*(lam**2 + 1)/sp.sqrt(lam**4 + 1); Bc = sp.sqrt(S)*(lam**2 - 1)/sp.sqrt(lam**4 + 1)
X0 = sp.Matrix([A*sp.cos(tau), -Bc*sp.sin(tau), kap*m*sp.cos(2*tau + delta)])
Fint = sp.integrate(sp.expand(X0.diff(tau).dot(X0.diff(tau))), tau)
osc = AxisymCongruence(X0, lam, s, tau, sp.Integer(1), Fint, 2, lambda x: (x[0]**2 + x[1]**2 + 4*x[2]**2)/2)
C = sp.Rational(5, 4); beta = sp.sqrt(1 - lam**2); sin_i = lam/(C*beta); cos_i = sp.sqrt(1 - sin_i**2)
Rx = sp.Matrix([[1, 0, 0], [0, cos_i, -sin_i], [0, sin_i, cos_i]])
X0k = Rx*sp.Matrix([sp.cos(tau) - lam, beta*sp.sin(tau), 0])
kep = AxisymCongruence(X0k, lam, s, tau, 1 - lam*sp.cos(tau), tau + lam*sp.sin(tau), 1, lambda x: -1/sp.sqrt(x[0]**2 + x[1]**2 + x[2]**2))
allok2 = True
for name, cong, pt in (("Solov'ev iota=2", osc, {lam: sp.Integer(2), s: sp.atan(sp.Rational(5, 12)), tau: sp.atan(sp.Rational(3, 4))}),
                       ("Solov'ev iota=2 pt2", osc, {lam: sp.Integer(3), s: sp.atan(sp.Rational(3, 4)), tau: sp.atan(sp.Rational(5, 12))}),
                       ("Kepler iota=1", kep, {lam: sp.Rational(3, 5), s: sp.atan(sp.Rational(5, 12)), tau: sp.atan(sp.Rational(3, 4))}),
                       ("Kepler iota=1 pt2", kep, {lam: sp.Rational(5, 13), s: sp.atan(sp.Rational(3, 4)), tau: sp.atan(sp.Rational(12, 5))})):
    sub, xpsi_true, chk = cong.jets(pt)
    lhs = ev((iota - N)*dth(hhat), sub); rhs = ev(Qhat, sub)
    mu_true = ev(mu_claim, sub)
    a_n = a.xreplace(sub); v_n = fc.Y.xreplace(sub).cross(a_n)/a_n.dot(a_n)
    mu_dir = sp.simplify((xpsi_true - v_n).dot(a_n)/a_n.dot(a_n))
    comp, _ = fc.COMPAT_at(sub)
    ok = (lhs - rhs == 0) and (mu_true - mu_dir == 0) and comp == 0
    allok2 &= ok
    print(f"  {name}: (iota-N) hhat' = {lhs}, Qhat = {rhs}, difference = {lhs - rhs}; mu_claim - mu_true = {mu_true - mu_dir}; COMPAT = {comp}; "
          f"m0^h = {ev(m0h, sub)}, f' = {ev(fp, sub)}, K' = {sub[K1]}")
print("f_psi DETERMINED BY THE SURFACE ON BOTH EXACT CONGRUENCES:", allok2)
print(f"done in {time.time()-t0:.0f} s")
