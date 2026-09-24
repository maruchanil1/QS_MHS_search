"""flow_core.py -- reusable jet machinery for the Boozer normal flow (round 1, global-reformulations,
boozer_flow_reduction.py), packaged as a module.

Objects (all sympy, exact):
  Xs[(i,a,b)]  : symbol for d_theta^a d_phi^b x_i on the surface           (a+b <= ORD = 3)
  Ks[n], Fp[n] : K^{(n)}(chi), f_psi^{(n)}(chi)  (chi-derivatives of B_psi and of d_psi f, f = sqrt g)
  iota, iotap, G, Gp, I, Ip, pp, N : flux constants of the surface (iota', G', I', p' are psi-derivatives)
  W = x_p + iota x_t,  a = I x_p - G x_t,  Y = W - K x_t x x_p
  xpsi  = Y x a/|a|^2 + mu a                       (normal flow; mu = tangential freedom)
  C1    = I (W.x_p) - G (W.x_t)                    (surface condition 1)
  f     = |W|^2/(G + iota I)                       (= sqrt g on C1 = 0)
  mu_alg: solution of  d_psi f = f_psi(chi)  for mu (algebraic; valid where W.Da != 0)
  dpsi_C1: d_psi C1 as a linear expression in (mu, mu_theta, mu_phi)
  COMPAT_at(sub): d_psi C1 with mu -> mu_alg and its derivatives, evaluated at a point (dict sub of all symbols)

Usage:  from flow_core import *   (build time ~15 s)
"""
import sympy as sp
import time

_t0 = time.time()
ORD = 3
Xs = {(i, a, b): sp.Symbol(f'x{i}_{a}{b}') for i in range(3) for a in range(ORD + 1) for b in range(ORD + 1 - a)}
Ms = {(a, b): sp.Symbol(f'mu_{a}{b}') for a in range(3) for b in range(3 - a)}
Ks = [sp.Symbol(f'K{n}') for n in range(ORD + 2)]
Fp = [sp.Symbol(f'fpsi{n}') for n in range(ORD + 2)]
iota, iotap, G, Gp, I, Ip, pp, N = sp.symbols('iota iotap G Gp I Ip pp N')
FLUX = [iota, iotap, G, Gp, I, Ip, pp, N]
mu = Ms[(0, 0)]


def vec(a, b):
    return sp.Matrix([Xs[(i, a, b)] for i in range(3)])


x_t, x_p = vec(1, 0), vec(0, 1)


def derivation(E, rule):
    out = 0
    for s in E.free_symbols:
        r = rule(s)
        if r is not None and r != 0:
            out += sp.diff(E, s)*r
    return out


inv_X = {v: k for k, v in Xs.items()}
inv_M = {v: k for k, v in Ms.items()}


def rule_theta(s):
    if s in inv_X:
        i, a, b = inv_X[s]
        return Xs.get((i, a + 1, b))
    if s in inv_M:
        a, b = inv_M[s]
        return Ms.get((a + 1, b))
    if s in Ks:
        return Ks[Ks.index(s) + 1] if Ks.index(s) + 1 < len(Ks) else None
    if s in Fp:
        return Fp[Fp.index(s) + 1] if Fp.index(s) + 1 < len(Fp) else None
    return 0


def rule_phi(s):
    if s in inv_X:
        i, a, b = inv_X[s]
        return Xs.get((i, a, b + 1))
    if s in inv_M:
        a, b = inv_M[s]
        return Ms.get((a, b + 1))
    if s in Ks:
        return -N*Ks[Ks.index(s) + 1] if Ks.index(s) + 1 < len(Ks) else None
    if s in Fp:
        return -N*Fp[Fp.index(s) + 1] if Fp.index(s) + 1 < len(Fp) else None
    return 0


dth = lambda E: derivation(E, rule_theta)
dph = lambda E: derivation(E, rule_phi)
D = lambda E: dph(E) + iota*dth(E)
uD = lambda E: dph(E) + N*dth(E)
vD = lambda V: V.applyfunc(D)

W = x_p + iota*x_t
a = I*x_p - G*x_t
Y = W - Ks[0]*x_t.cross(x_p)
xpsi = Y.cross(a)/a.dot(a) + mu*a
sqrtg = xpsi.dot(x_t.cross(x_p))
C1 = sp.expand(I*W.dot(x_p) - G*W.dot(x_t))
f = W.dot(W)/(G + iota*I)
WDa = W.dot(vD(a))
WDa_claim = -(G + iota*I)*(G + N*I)*dth(f)/2

_xpsi_jets = {}


def xpsi_jet(i, a_, b_):
    key = (i, a_, b_)
    if key not in _xpsi_jets:
        e = xpsi[i]
        for _ in range(a_):
            e = dth(e)
        for _ in range(b_):
            e = dph(e)
        _xpsi_jets[key] = e
    return _xpsi_jets[key]


def rule_psi(s):
    if s in inv_X:
        i, a_, b_ = inv_X[s]
        return xpsi_jet(i, a_, b_)
    return {iota: iotap, G: Gp, I: Ip}.get(s, 0)


dpsi = lambda E: derivation(E, rule_psi)

dpsi_f = dpsi(f)
dpsi_f0 = dpsi_f.subs({Ms[(1, 0)]: 0, Ms[(0, 1)]: 0})
cmu = sp.diff(dpsi_f0, mu)
rest = dpsi_f0.subs(mu, 0)
mu_alg = (Fp[0] - rest)/cmu
dpsi_C1 = dpsi(C1)
A_th, A_ph, A_0 = (sp.diff(dpsi_C1, s) for s in (Ms[(1, 0)], Ms[(0, 1)], mu))
BUILD_TIME = time.time() - _t0


def ev(expr, sub):
    """exact evaluation of expr at the point sub (all symbols numeric)."""
    return sp.nsimplify(sp.simplify(expr.xreplace(sub)))


def mu_alg_jets(sub):
    """mu_alg and its first theta, phi derivatives at the point sub (chain rule on the symbols)."""
    mu_v = mu_alg.xreplace(sub)
    mu_th = 0
    mu_ph = 0
    for s in mu_alg.free_symbols:
        d = sp.diff(mu_alg, s).xreplace(sub)
        rt, rp = rule_theta(s), rule_phi(s)
        if rt not in (None, 0):
            mu_th += d*rt.xreplace(sub)
        if rp not in (None, 0):
            mu_ph += d*rp.xreplace(sub)
    return sp.nsimplify(sp.simplify(mu_v)), sp.nsimplify(sp.simplify(mu_th)), sp.nsimplify(sp.simplify(mu_ph))


def COMPAT_at(sub):
    """(COMPAT, mu_alg) at the point sub: d_psi C1 with mu -> mu_alg, mu_theta -> d_theta mu_alg, mu_phi -> d_phi mu_alg."""
    mu_v, mu_th, mu_ph = mu_alg_jets(sub)
    val = dpsi_C1.xreplace(sub).xreplace({Ms[(1, 0)]: mu_th, Ms[(0, 1)]: mu_ph, mu: mu_v})
    return sp.nsimplify(sp.simplify(val)), mu_v


if __name__ == '__main__':
    print(f"flow_core built in {BUILD_TIME:.1f} s")
    G_on_C1 = I*W.dot(x_p)/W.dot(x_t)
    div = lambda e: sp.cancel(sp.together(e.subs(G, G_on_C1))) == 0
    print("(1),(2),(4'),|W|^2 identities modulo C1:", div(W.dot(x_p) - G*sqrtg), div(W.dot(x_t) - I*sqrtg),
          div(W.dot(xpsi) - Ks[0]*sqrtg), div(W.dot(W) - (G + iota*I)*sqrtg))
    print("dpsi_C1 linear in (mu, mu_t, mu_p):", all(sp.diff(A, s) == 0 for A in (A_th, A_ph, A_0) for s in (Ms[(1, 0)], Ms[(0, 1)], mu)))
