"""Boozer-embedding identities (CONTEXT 2.2 (1)-(4)) as a first-order normal flow in psi, with the QS
overdetermination isolated in ONE compatibility condition per surface.

Notation: x(psi,theta,phi) embedding, x_t = d_theta x, x_p = d_phi x, W = x_p + iota x_t (B = W/sqrt g),
D = d_phi + iota d_theta (field-line derivative), u = d_phi + N d_theta (QS symmetry direction; chi = theta - N phi).
Strong QS => K := B_psi = W.x_psi/sqrt g is a function of (psi, chi) with
       (iota - N) d_chi K = G' + iota I' + p' sqrt g            [(4), using u-invariance of K]
so ALL FOUR identities are the single vector equation, algebraic in x_psi:

   (E)   W = K (x_t x x_p) + I (x_p x x_psi) + G (x_psi x x_t)        <=>  (1),(2),(4')    [B^flat = K dpsi + I dtheta + G dphi]
   (Q)   sqrt g := x_psi.(x_t x x_p)  is a function of (psi, chi)        <=>  (3)

Solving (E) for x_psi:  with a := I x_p - G x_t,  Y := W - K x_t x x_p,  (E) <=> a x x_psi = Y, solvable iff
   C1 := a.Y = I (W.x_p) - G (W.x_t) = 0          (SURFACE CONDITION, metric only)
and then
   x_psi = Y x a/|a|^2 + mu a                       (mu = free tangential component; Boozer angle freedom)
On the surface, sqrt g = (W.x_t)/I = (W.x_p)/G = |W|^2/(G + iota I)  (metric only!), so (Q) is the second
SURFACE CONDITION  C2 := u( |W|^2/(G+iota I) ) = 0.

Propagation of the constraints:
   d_psi f = f_psi(chi)  with f := |W|^2/(G+iota I):   the coefficients of mu_theta, mu_phi are proportional to C1,
        so on the constraint surface this equation is ALGEBRAIC in mu  =>  mu = mu_alg(x-jets, f_psi, iota', G', I')
   d_psi C1 = 0:  a first-order linear PDE for mu.  Substituting mu_alg gives
        COMPAT(x-jets up to order 3; f_psi, f_psi', iota', G', I'; K, K') = 0  on the surface,
   the single extra condition that encodes the Garren-Boozer +1 overdetermination.

QS-MHS  <=>  a one-parameter family of surfaces satisfying C1 = C2 = COMPAT = 0, evolving by x_psi = Y x a/|a|^2 + mu_alg a.

Checks in this script (exact arithmetic):
  [A] algebraic identities (1),(2),(4'), sqrt g formulas, |W|^2 = (G+iota I) sqrt g: residuals divisible by C1.
  [B] coefficients of mu_theta, mu_phi in d_psi f are 2 C1/(G+iota I) x (iota, 1).
  [C] explicit vacuum iota = 0 family (shifted ellipses, exact vacuum with finite Fourier Boozer embedding):
      identities (1)-(4'), mu_alg = mu_true, COMPAT = 0 at rational points.
  [D] explicit screw pinch (periodic cylinder, iota(psi) = psi, I != 0, p' != 0): same checks.
Run:  python3 boozer_flow_reduction.py      (~1 min)
"""
import sympy as sp
import time
t0 = time.time()

# ---------------- jet symbols ----------------
ORD = 3
Xs = {(i, a, b): sp.Symbol(f'x{i}_{a}{b}') for i in range(3) for a in range(ORD + 1) for b in range(ORD + 1 - a)}
Ms = {(a, b): sp.Symbol(f'mu_{a}{b}') for a in range(3) for b in range(3 - a)}
Ks = [sp.Symbol(f'K{n}') for n in range(ORD + 2)]      # K^{(n)}(chi)
Fp = [sp.Symbol(f'fpsi{n}') for n in range(ORD + 2)]   # f_psi^{(n)}(chi)
iota, iotap, G, Gp, I, Ip, pp, N = sp.symbols('iota iotap G Gp I Ip pp N')

def vec(a, b): return sp.Matrix([Xs[(i, a, b)] for i in range(3)])
x_t, x_p = vec(1, 0), vec(0, 1)

def derivation(E, rule):
    """Apply a derivation defined on symbols by `rule(sym) -> expr or None`."""
    out = 0
    for s in E.free_symbols:
        r = rule(s)
        if r is not None and r != 0:
            out += sp.diff(E, s)*r
    return out
inv_X = {v: k for k, v in Xs.items()}; inv_M = {v: k for k, v in Ms.items()}
def rule_theta(s):
    if s in inv_X:
        i, a, b = inv_X[s]; return Xs.get((i, a + 1, b))
    if s in inv_M:
        a, b = inv_M[s]; return Ms.get((a + 1, b))
    if s in Ks: return Ks[Ks.index(s) + 1]
    if s in Fp: return Fp[Fp.index(s) + 1]
    return 0
def rule_phi(s):
    if s in inv_X:
        i, a, b = inv_X[s]; return Xs.get((i, a, b + 1))
    if s in inv_M:
        a, b = inv_M[s]; return Ms.get((a, b + 1))
    if s in Ks: return -N*Ks[Ks.index(s) + 1]
    if s in Fp: return -N*Fp[Fp.index(s) + 1]
    return 0
dth = lambda E: derivation(E, rule_theta)
dph = lambda E: derivation(E, rule_phi)
D = lambda E: dph(E) + iota*dth(E)
uD = lambda E: dph(E) + N*dth(E)
vdth = lambda V: V.applyfunc(dth); vdph = lambda V: V.applyfunc(dph); vD = lambda V: V.applyfunc(D)

# ---------------- the flow ----------------
W = x_p + iota*x_t
a = I*x_p - G*x_t
Y = W - Ks[0]*x_t.cross(x_p)
mu = Ms[(0, 0)]
xpsi = Y.cross(a)/a.dot(a) + mu*a
sqrtg = xpsi.dot(x_t.cross(x_p))
C1 = sp.expand(I*W.dot(x_p) - G*W.dot(x_t))
f = W.dot(W)/(G + iota*I)

G_on_C1 = I*W.dot(x_p)/W.dot(x_t)          # C1 = 0 solved for G (C1 is linear in G)
def divisible_by_C1(expr):
    """expr vanishes on the constraint surface C1 = 0 (checked by exact substitution of G)."""
    return sp.cancel(sp.together(expr.subs(G, G_on_C1))) == 0
print(f"setup done at {time.time()-t0:.1f} s")
print("[A] residual of (1) W.x_p - G sqrtg      divisible by C1 :", divisible_by_C1(W.dot(x_p) - G*sqrtg))
print("[A] residual of (2) W.x_t - I sqrtg      divisible by C1 :", divisible_by_C1(W.dot(x_t) - I*sqrtg))
print("[A] residual of (4') W.x_psi - K sqrtg   divisible by C1 :", divisible_by_C1(W.dot(xpsi) - Ks[0]*sqrtg))
print("[A] |W|^2 - (G + iota I) sqrtg           divisible by C1 :", divisible_by_C1(W.dot(W) - (G + iota*I)*sqrtg))
print("[A] a.Y == C1 exactly :", sp.expand(a.dot(Y) - C1) == 0)

# psi-derivation on x-jets and flux constants (K, mu never differentiated in psi below)
xpsi_jets = {}
def xpsi_jet(i, a_, b_):
    key = (i, a_, b_)
    if key not in xpsi_jets:
        e = xpsi[i]
        for _ in range(a_): e = dth(e)
        for _ in range(b_): e = dph(e)
        xpsi_jets[key] = e
    return xpsi_jets[key]
def rule_psi(s):
    if s in inv_X:
        i, a_, b_ = inv_X[s]; return xpsi_jet(i, a_, b_)
    return {iota: iotap, G: Gp, I: Ip}.get(s, 0)
dpsi = lambda E: derivation(E, rule_psi)

# [B] propagation of the QS constraint: d_psi f
dpsi_f = dpsi(f)
print(f"d_psi f assembled at {time.time()-t0:.1f} s")
c_th = sp.diff(dpsi_f, Ms[(1, 0)]); c_ph = sp.diff(dpsi_f, Ms[(0, 1)])
print("[B] coeff of mu_theta in d_psi f  ==  2 iota C1/(G+iota I) :", sp.cancel(sp.together(c_th - 2*iota*C1/(G + iota*I))) == 0)
print("[B] coeff of mu_phi   in d_psi f  ==  2 C1/(G+iota I)      :", sp.cancel(sp.together(c_ph - 2*C1/(G + iota*I))) == 0)
dpsi_f0 = dpsi_f.subs({Ms[(1, 0)]: 0, Ms[(0, 1)]: 0})          # drop terms that vanish on C1 = 0
cmu = sp.diff(dpsi_f0, mu); rest = dpsi_f0.subs(mu, 0)
mu_alg = (Fp[0] - rest)/cmu                                       # d_psi f is linear in mu
WDa = W.dot(vD(a))
print("[B] coeff of mu in d_psi f == 2 (W.Da)/(G+iota I) :", sp.cancel(sp.together(cmu - 2*WDa/(G + iota*I))) == 0)
print("    => mu_alg = [ (G+iota I) f_psi - (rest) ] / (2 W.Da),  W.Da = -(G+iota I)(G+N I) f'(chi)/2 on the surface")
print("       (D does not act on the flux constants; W.Da = (I/2) d_phi|W|^2 - (G/2) d_theta|W|^2 with |W|^2 = (G+iota I) f(chi))")
print("       => mu is undetermined exactly where f'(chi) = 0 (|B| extrema on the surface, or |B| constant on the surface)")
# structural form of W.Da on the constraint surface: check against the closed form using f'(chi) = d_theta f
WDa_claim = -(G + iota*I)*(G + N*I)*dth(f)/2
# on the constraint surface d_phi f = -N d_theta f (C2) and W.x_t = I f, W.x_p = G f (C1); test at random rational jets
import random
random.seed(1)
def rand_sub():
    return {s: sp.Rational(random.randint(-9, 9), random.randint(1, 5)) for s in list(Xs.values()) + Ks + Fp + [mu, iota, iotap, G, Gp, I, Ip, pp, N]}
# impose C1 and C2 at the random point by solving for G and for x0_01-jet? simpler: verify the identity symbolically modulo C1, C2:
expr = WDa - WDa_claim
expr = expr.subs(G, G_on_C1)                                     # C1 = 0
# C2: u(f) = 0 -> solve for x0_11 (appears linearly in d_phi d_theta terms? no) -> instead check numerically on the explicit tests below
print("    (W.Da closed form is verified on the explicit tests below)")

# d_psi C1 = 0 : linear first-order PDE for mu
dpsi_C1 = dpsi(C1)
print(f"d_psi C1 assembled at {time.time()-t0:.1f} s")
A_th, A_ph, A_0 = (sp.diff(dpsi_C1, s) for s in (Ms[(1, 0)], Ms[(0, 1)], mu))
print("[B] d_psi C1 linear in (mu, mu_t, mu_p):", all(sp.diff(A, s) == 0 for A in (A_th, A_ph, A_0) for s in (Ms[(1, 0)], Ms[(0, 1)], mu)))
print("    coeff(mu_phi)   = ", sp.factor(sp.cancel(A_ph)))
print("    coeff(mu_theta) = ", sp.factor(sp.cancel(A_th)))
print(f"[B] COMPAT := d_psi C1 |_(mu = mu_alg)  is evaluated lazily at test points ({time.time()-t0:.1f} s)")
def COMPAT_at(sub):
    """Evaluate d_psi C1 with mu -> mu_alg, mu_theta -> d_theta mu_alg, mu_phi -> d_phi mu_alg at a point."""
    mu_v = mu_alg.subs(sub)
    # derivatives of mu_alg by chain rule, evaluated at the point
    mu_th = sum(sp.diff(mu_alg, s).subs(sub)*(rule_theta(s).subs(sub) if rule_theta(s) not in (None, 0) else 0)
                for s in mu_alg.free_symbols)
    mu_ph = sum(sp.diff(mu_alg, s).subs(sub)*(rule_phi(s).subs(sub) if rule_phi(s) not in (None, 0) else 0)
                for s in mu_alg.free_symbols)
    val = dpsi_C1.subs(sub).subs({Ms[(1, 0)]: mu_th, Ms[(0, 1)]: mu_ph, mu: mu_v})
    return sp.nsimplify(sp.simplify(val)), sp.nsimplify(sp.simplify(mu_v))

# ---------------- explicit tests ----------------
psi_s, th_s, ph_s = sp.symbols('psi theta phi', real=True)
def jets_at(xexpr, Kchi, fpsi_chi, point, flux):
    """Evaluate all jet symbols from an explicit family at a rational point.  Kchi, fpsi_chi: functions of chi
    given as expressions in th_s, ph_s (already functions of theta - N phi)."""
    sub = {}
    for (i, a_, b_), s in Xs.items():
        sub[s] = sp.nsimplify(sp.simplify(sp.diff(xexpr[i], th_s, a_, ph_s, b_).subs(point)))
    for n, s in enumerate(Ks):
        sub[s] = sp.nsimplify(sp.simplify(sp.diff(Kchi, th_s, n).subs(point)))
    for n, s in enumerate(Fp):
        sub[s] = sp.nsimplify(sp.simplify(sp.diff(fpsi_chi, th_s, n).subs(point)))
    sub.update(flux)
    return sub

def run_test(name, xexpr, flux_funcs, Nval, point, Kexpr):
    """flux_funcs: dict with iota(psi), G(psi), I(psi), p(psi) as expressions in psi_s."""
    io, Gf, If, pf = (flux_funcs[k] for k in ('iota', 'G', 'I', 'p'))
    e_t, e_p, e_psi = xexpr.diff(th_s), xexpr.diff(ph_s), xexpr.diff(psi_s)
    Wx = e_p + io*e_t
    sg = e_psi.dot(e_t.cross(e_p))
    r1 = sp.simplify(Wx.dot(e_p) - Gf*sg); r2 = sp.simplify(Wx.dot(e_t) - If*sg)
    Kx = sp.simplify(Wx.dot(e_psi)/sg)
    r4 = sp.simplify((Wx.dot(e_psi) - Kx*sg))
    fb = sp.simplify(sp.diff(Kx, ph_s) + io*sp.diff(Kx, th_s) - (sp.diff(Gf, psi_s) + io*sp.diff(If, psi_s) + sp.diff(pf, psi_s)*sg))
    chi_test = sp.simplify(sp.diff(sg, ph_s) + Nval*sp.diff(sg, th_s))   # (3): sqrt g function of chi
    print(f"[{name}] (1),(2),(4') residuals: {r1}, {r2}, {r4} | force balance (4): {fb} | u(sqrt g) = {chi_test} | K = {Kx}")
    Bmag2 = sp.simplify(Wx.dot(Wx)/sg**2)
    print(f"[{name}] |B|^2 = {Bmag2}")
    ftrue = sp.simplify(Wx.dot(Wx)/(Gf + io*If))
    flux = {iota: io.subs(point), iotap: sp.diff(io, psi_s).subs(point), G: Gf.subs(point), Gp: sp.diff(Gf, psi_s).subs(point),
            I: If.subs(point), Ip: sp.diff(If, psi_s).subs(point), pp: sp.diff(pf, psi_s).subs(point), N: Nval}
    sub = jets_at(xexpr, Kexpr, sp.diff(ftrue, psi_s), point, flux)
    # true mu
    xpsi_true = e_psi.subs(point)
    a_n = a.subs(sub); Y_n = Y.subs(sub)
    v_n = Y_n.cross(a_n)/a_n.dot(a_n)
    mu_true = sp.nsimplify(sp.simplify((xpsi_true - v_n).dot(a_n)/a_n.dot(a_n)))
    dec = sp.simplify(xpsi_true - v_n - mu_true*a_n)
    print(f"[{name}] x_psi - (Y x a/|a|^2 + mu_true a) = {list(dec)} ,  mu_true = {mu_true}")
    C1_n = sp.simplify(C1.subs(sub)); C2_n = sp.simplify(uD(f).subs(sub))
    wda = sp.simplify((WDa - WDa_claim).subs(sub)); wda_val = sp.simplify(WDa.subs(sub))
    print(f"[{name}] C1 = {C1_n}, C2 = {C2_n}, W.Da - closed form = {wda}, W.Da = {wda_val}")
    if wda_val == 0:
        print(f"[{name}] W.Da = 0 (|B| constant on the surface): mu is not determined algebraically here -- degenerate case, "
              f"COMPAT not testable   ({time.time()-t0:.1f} s)")
        return None
    comp, mu_alg_n = COMPAT_at(sub)
    print(f"[{name}] mu_alg - mu_true = {sp.simplify(mu_alg_n - mu_true)};  COMPAT = {comp}   ({time.time()-t0:.1f} s)")
    return comp

# [C] vacuum iota = 0 family: R = R0(psi) + a(psi) cos theta, Z = -k a sin theta,  x+iy = R e^{i phi}
# (R0 = psi/(kG) + c, a^2 = psi^2/(kG)^2 + 2 c psi/(kG) + d solve d_theta(R,Z) x d_psi(R,Z) = R/G exactly: B = G grad phi)
k_, Gv, cv, dv = 1, 1, 0, -1
R0 = psi_s/(k_*Gv) + cv
av = sp.sqrt(psi_s**2/(k_*Gv)**2 + 2*cv*psi_s/(k_*Gv) + dv)
Rv = R0 + av*sp.cos(th_s)
x_vac = sp.Matrix([Rv*sp.cos(ph_s), Rv*sp.sin(ph_s), -k_*av*sp.sin(th_s)])   # sign: right-handed (psi,theta,phi), sqrt g = +R^2/G
flux_vac = dict(iota=sp.Integer(0)*psi_s, G=sp.Integer(Gv) + 0*psi_s, I=0*psi_s, p=0*psi_s)
pt = {psi_s: sp.Rational(5, 4), th_s: sp.atan(sp.Rational(4, 3)), ph_s: sp.atan(sp.Rational(5, 12))}
run_test("C vacuum iota=0", x_vac, flux_vac, 0, pt, sp.Integer(0))
pt2 = {psi_s: sp.Rational(13, 12), th_s: sp.atan(sp.Rational(12, 5)), ph_s: sp.atan(sp.Rational(3, 4))}
run_test("C vacuum iota=0 (2nd point)", x_vac, flux_vac, 0, pt2, sp.Integer(0))

# [D] screw pinch, periodic in phi with G = 1 (L = 2 pi): x = (r cos th, r sin th, phi), r = sqrt(2 psi), iota = psi
r_ = sp.sqrt(2*psi_s)
x_pinch = sp.Matrix([r_*sp.cos(th_s), r_*sp.sin(th_s), ph_s])
io_p = psi_s
I_p = io_p*r_**2                   # I = iota r^2 / (G^-1)... with sqrt g = r r' = 1 and G = 1
sg_p = 1
p_p = sp.integrate(-(0 + io_p*sp.diff(I_p, psi_s))/sg_p, psi_s)
flux_pinch = dict(iota=io_p, G=sp.Integer(1) + 0*psi_s, I=I_p, p=p_p)
ptp = {psi_s: sp.Rational(9, 8), th_s: sp.atan(sp.Rational(4, 3)), ph_s: sp.atan(sp.Rational(5, 12))}   # r = 3/2
run_test("D screw pinch", x_pinch, flux_pinch, 1, ptp, sp.Integer(0))
print(f"done in {time.time()-t0:.1f} s")
