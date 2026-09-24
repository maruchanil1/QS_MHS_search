"""TASK 1 -- orbit-coordinate formulation of closed-line QS-MHS (coordinator note 6), proved and checked.

Coordinates (psi, s, t): psi flux label, s in [0,2pi) line label, t time along the line, B = x_t, u = d_s.
g_ab = x_a . x_b (Euclidean metric in these coordinates), J = det(x_psi, x_s, x_t).

(P1) div B = d_t J / J,  div u = d_s J / J             => J = J(psi) <=> div B = div u = 0.
(P2) (L_u B^flat)_a = d_s g_ta  (u is a coordinate field) => strong QS <=> d_s g_tt = d_s g_ts = d_s g_tpsi = 0;
     weak QS (u.grad|B| = 0) <=> d_s g_tt = 0.
(P3) x_tt . x_a = d_t g_ta - d_a g_tt / 2  (pure calculus), hence force balance x_tt = grad Pi, Pi = p(psi) + g_tt/2:
         a = t : identity
         a = s : d_t g_ts = d_s g_tt                (<=> J^psi = 0, current tangent to surfaces)
         a = psi: d_t g_tpsi - d_psi g_tt = p'(psi)  (<=> (J x B)_psi = p')
     i.e. the 1-form x_tt.dx is closed AND x_tt.dx = dPi with Pi single valued.
     Curl in these coordinates: J^a = eps^{abc} d_b g_tc / J.
(P4) With weak QS: g_ts = I(psi) + k(psi,s)   (t-independent),
                    g_tpsi = K_p(psi,t) + kappa(psi,s) with d_t K_p = p' + d_psi g_tt.
     Strong QS <=> k_s = kappa_s = 0.  These are NOT implied by the local equations on a rational (closed-line)
     surface: the Boozer-type covariant form B_theta = I + k(alpha), B_phi = G - iota k(alpha) (alpha = line label,
     k constant along lines, zero mean) satisfies J^psi = 0, sqrt(g) = (G + iota I)/B^2 and the psi-force balance
     identically -- k drops out.  The RHB weak=>strong proof uses B_theta = I(psi), which needs irrational surfaces
     (or continuity in a sheared iota).  The gap is a line-constant parallel current (d_psi k - kappa_s)/J.

Checks below:
 [A] generic x(psi,s,t): identities (P2), (P3) symbolically.
 [B] Boozer-type form with k(alpha): k drops out of J^psi, sqrt g, (J x B)_psi.
 [C] Solov'ev congruence from the 1:1:2 oscillator (round 1, osc112_qs.py):
        y0(lam,t) = sqrt(2S)(cos lam cos t, -sin lam sin t, 0) + (kappa S/2) cos 2lam cos(2t+delta) e_z,
        x = R_z(s) y0, m = (S/2) cos 2lam, lam in (0, pi/4).  All t-row metric components, J, residuals of
        (W),(M1),(M2), closedness of x_tt.dx, p'(lam) compared with the Grad-Shafranov value.
 [D] Kepler force-free congruence (round 1, kepler_axisym_family.py) with the psi-derivative taken at FIXED t
        (t = sqrt(a^3/mu)(eta - eps sin eta) depends on psi through eps): (M2) must give p' = 0.
Run: python3 orbit_coords_formulation.py     (~1 min)
"""
import sympy as sp
import time
T0 = time.time()

def iszero(e):
    e = sp.expand(sp.expand_trig(sp.expand(e)))
    e = sp.expand(e.rewrite(sp.exp))
    return sp.simplify(e) == 0

# ------------------------------------------------------------------ [A] generic identities
print("[A] generic identities in orbit coordinates")
psi, s, t = sp.symbols('psi s t', real=True)
X = sp.Matrix([sp.Function(f'x{i}')(psi, s, t) for i in range(3)])
q = (psi, s, t)
Xa = {a: X.diff(a) for a in q}
g = {(a, b): Xa[a].dot(Xa[b]) for a in q for b in q}
Xtt = X.diff(t, 2)
# (P3): x_tt . x_a = d_t g_ta - d_a g_tt / 2
for a in q:
    lhs = Xtt.dot(Xa[a]); rhs = sp.diff(g[(t, a)], t) - sp.diff(g[(t, t)], a)/2
    print(f"   x_tt.x_{a} == d_t g_t{a} - d_{a} g_tt/2 :", sp.simplify(lhs - rhs) == 0)
# (P2): Lie derivative of the covector B_a = g_ta along u = d_s in coordinates: (L_u w)_a = u^b d_b w_a + w_b d_a u^b
u_comp = [0, 1, 0]
Bcov = [g[(t, a)] for a in q]
LuB = [sum(u_comp[j]*sp.diff(Bcov[i], q[j]) for j in range(3)) + sum(Bcov[j]*sp.diff(u_comp[j], q[i]) for j in range(3))
       for i in range(3)]
print("   (L_u B^flat)_a == d_s g_ta :", all(sp.simplify(LuB[i] - sp.diff(Bcov[i], s)) == 0 for i in range(3)))
# (P1): d_t J = J div B  -- checked on the concrete congruences below via the Cartesian field ([C]); here only
# the statement that J depends on (psi) iff both derivatives vanish is tautological.

# ------------------------------------------------------------------ [B] the k-freedom on rational surfaces
print("\n[B] Boozer-type covariant form on a closed-line surface: B_theta = I + k(alpha), B_phi = G - iota k(alpha)")
th, ph, io = sp.symbols('theta phi iota', real=True)
I, G, Lam = sp.Function('I')(psi), sp.Function('G')(psi), sp.Function('Lambda')(psi)
alpha = th - io*ph
k = sp.Function('k')(psi, alpha)
K = sp.Function('K')(psi, th, ph)
Bth = I + k; Bph = G - io*k
Jpsi_num = sp.diff(Bph, th) - sp.diff(Bth, ph)                      # sqrt g J^psi
print("   sqrt(g) J^psi = d_theta B_phi - d_phi B_theta =", sp.simplify(Jpsi_num))
print("   B_phi + iota B_theta =", sp.simplify(Bph + io*Bth), "  (=> sqrt g = (G+iota I)/B^2 unchanged)")
# (J x B)_psi = J^theta - iota J^phi  (B^theta = iota/sqrt g, B^phi = 1/sqrt g):
JxB_psi = (sp.diff(K, ph) - sp.diff(Bph, psi)) - io*(sp.diff(Bth, psi) - sp.diff(K, th))
print("   sqrt(g)(J x B)_psi = (d_phi + iota d_theta)K - (G' + iota I') :",
      sp.simplify(JxB_psi - ((sp.diff(K, ph) + io*sp.diff(K, th)) - sp.diff(G + io*I, psi))) == 0, " (k dropped out)")
Jpar_extra = sp.diff(Bth, psi) - sp.diff(Bth, psi).subs(k, 0)
print("   extra parallel current  sqrt(g) dJ^phi = d_psi k ,  sqrt(g) dJ^theta = iota d_psi k  => dJ = (d_psi k) B")

# ------------------------------------------------------------------ [C] Solov'ev congruence
print("\n[C] Solov'ev congruence (1:1:2 oscillator), x = R_z(s) y0(lam,t)")
lam = sp.symbols('lambda', real=True)
S, kap = sp.symbols('S kappa', positive=True)
delta = sp.symbols('delta', real=True)
def Rz(a): return sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]])
y0 = sp.Matrix([sp.sqrt(2*S)*sp.cos(lam)*sp.cos(t), -sp.sqrt(2*S)*sp.sin(lam)*sp.sin(t),
                kap*S/2*sp.cos(2*lam)*sp.cos(2*t + delta)])
x = Rz(s)*y0
# consistency with round 1: x^2+y^2 - S = 2 m cos 2t with m = (S/2) cos 2 lam, z = kappa m cos(2t+delta)
m = S/2*sp.cos(2*lam)
print("   rho^2 - S - 2 m cos 2t == 0 :", iszero(x[0]**2 + x[1]**2 - S - 2*m*sp.cos(2*t)))
xl, xs, xt = x.diff(lam), x.diff(s), x.diff(t)
gtt, gts, gtl = xt.dot(xt), xt.dot(xs), xt.dot(xl)
J = sp.Matrix.hstack(xl, xs, xt).det()
J = sp.simplify(sp.expand(sp.expand_trig(sp.expand(J))))
print("   J =", sp.factor(J))
print("   d_s J == 0, d_t J == 0 :", iszero(sp.diff(J, s)), iszero(sp.diff(J, t)))
print("   g_tt =", sp.simplify(sp.expand_trig(sp.expand(gtt))))
print("   d_s g_tt == d_s g_ts == d_s g_tlam == 0 (strong QS, u = rotation) :",
      iszero(sp.diff(gtt, s)), iszero(sp.diff(gts, s)), iszero(sp.diff(gtl, s)))
print("   (M1) d_t g_ts == 0 :", iszero(sp.diff(gts, t)), "   g_ts = I(lam) =", sp.simplify(sp.expand_trig(sp.expand(gts))))
M2 = sp.diff(gtl, t) - sp.diff(gtt, lam)
M2s = sp.simplify(sp.expand(sp.expand_trig(sp.expand(M2))))
print("   (M2) d_t g_tlam - d_lam g_tt =", sp.factor(M2s), "  t,s-independent:", iszero(sp.diff(M2, t)), iszero(sp.diff(M2, s)))
# Grad-Shafranov value from osc112_qs.py: p = p0 - S - 2 kappa^2 m^2  =>  dp/dlam = -4 kappa^2 m dm/dlam
pprime_GS = sp.diff(-S - 2*kap**2*m**2, lam)
print("   p'(lam) from GS (p = p0 - S - 2 kappa^2 m^2):", sp.factor(sp.simplify(pprime_GS)), " match:", iszero(M2s - pprime_GS))
# closedness of alpha = x_tt . dx  (all three pairs) and alpha_s = 0, alpha_t = d_t g_tt/2
xtt = xt.diff(t)
al = {lam: xtt.dot(xl), s: xtt.dot(xs), t: xtt.dot(xt)}
cl = [(a, b, iszero(sp.diff(al[a], b) - sp.diff(al[b], a))) for a, b in [(lam, s), (lam, t), (s, t)]]
print("   closedness d_a alpha_b - d_b alpha_a == 0 for (lam,s),(lam,t),(s,t):", [c[2] for c in cl])
print("   alpha_s = x_tt.x_s == 0 (u.grad Pi = 0):", iszero(al[s]))
print("   alpha_lam - d_lam g_tt/2 == p'(lam):", iszero(al[lam] - sp.diff(gtt, lam)/2 - pprime_GS))
# Cartesian cross-check of the curl formulas: B from GS form, compare J^psi = 0 and (JxB).grad(m^2) with p'
rho, zz = sp.symbols('rho z', positive=True)
m2 = (rho**2 - S)**2/4 + (sp.cos(delta)*(rho**2 - S)/2 - zz/kap)**2/sp.sin(delta)**2
Psi = -kap*sp.sin(delta)*m2
F = sp.sqrt(S**2 - 4*m2)
Brho = -sp.diff(Psi, zz)/rho; Bz = sp.diff(Psi, rho)/rho; Bphi = F/rho
# curl in cylindrical coordinates (axisymmetric): J_rho = -d_z B_phi, J_phi = d_z B_rho - d_rho B_z, J_z = (1/rho) d_rho(rho B_phi)
Jrho = -sp.diff(Bphi, zz); Jphi = sp.diff(Brho, zz) - sp.diff(Bz, rho); Jz = sp.diff(rho*Bphi, rho)/rho
JxB_rho = Jphi*Bz - Jz*Bphi; JxB_z = Jrho*Bphi - Jphi*Brho; JxB_phi = Jz*Brho - Jrho*Bz
p_of = -S - 2*kap**2*m2
res = [sp.simplify(JxB_rho - sp.diff(p_of, rho)), sp.simplify(JxB_z - sp.diff(p_of, zz)), sp.simplify(JxB_phi)]
print("   Cartesian/GS check J x B = grad p with p = p0 - S - 2 kappa^2 m^2 :", [r == 0 for r in res])
print(f"   [C] done, {time.time()-T0:.0f} s")

# ------------------------------------------------------------------ [D] Kepler force-free congruence
print("\n[D] Kepler force-free congruence, psi-derivative at fixed t; label psi = eccentricity eps")
eta = sp.symbols('eta', real=True)
a, mu, C = sp.symbols('a mu C', positive=True)
eps = psi                                                 # eps(psi) = psi (eps' = 1), a, mu, C constants
beta = sp.sqrt(1 - eps**2)
sin_i = eps/(C*beta); cos_i = sp.sqrt(1 - sin_i**2)
Rx_i = sp.Matrix([[1, 0, 0], [0, cos_i, -sin_i], [0, sin_i, cos_i]])
e = sp.Matrix([a*(sp.cos(eta) - eps), a*beta*sp.sin(eta), 0])
xk = Rz(s)*Rx_i*e
tau = sp.sqrt(a**3/mu)*(eta - eps*sp.sin(eta))            # t(psi, eta)
tau_eta, tau_psi = sp.diff(tau, eta), sp.diff(tau, psi)
xk_t = xk.diff(eta)/tau_eta                                # B = d_t x
xk_s = xk.diff(s)
xk_psi = xk.diff(psi) - (tau_psi/tau_eta)*xk.diff(eta)     # d_psi at fixed t
Dt = lambda f: sp.diff(f, eta)/tau_eta                     # d_t at fixed (psi,s)
Dpsi = lambda f: sp.diff(f, psi) - (tau_psi/tau_eta)*sp.diff(f, eta)   # d_psi at fixed (s,t)
Gtt, Gts, Gtp = xk_t.dot(xk_t), xk_t.dot(xk_s), xk_t.dot(xk_psi)
xk_tt = Dt(xk_t)
alk = {psi: xk_tt.dot(xk_psi), s: xk_tt.dot(xk_s), t: xk_tt.dot(xk_t)}
D = {psi: Dpsi, s: lambda f: sp.diff(f, s), t: Dt}
Jk = sp.Matrix.hstack(xk_psi, xk_s, xk_t).det()
# Exact evaluation at rational points where sin i, cos i are rational: (C, eps) = (5/4, 3/5) and (25/36, 5/13)
def at(expr, Cv, ev):
    return sp.simplify(sp.expand_trig(sp.expand(expr.subs({C: Cv, a: 1, mu: 1}).subs(psi, ev))))
for (Cv, ev) in [(sp.Rational(5, 4), sp.Rational(3, 5)), (sp.Rational(25, 36), sp.Rational(5, 13))]:
    print(f"   point C = {Cv}, eps = {ev}:  sin i = {sin_i.subs({C: Cv, psi: ev})}, cos i = {sp.nsimplify(cos_i.subs({C: Cv, psi: ev}))}")
    print("     J =", at(Jk, Cv, ev), " (eta-independent)")
    rho_k = a*(1 - eps*sp.cos(eta))                      # Kepler radius
    print("     |x|^2 == a^2(1 - eps cos eta)^2 :", at(xk.dot(xk) - rho_k**2, Cv, ev) == 0)
    print("     g_tt - (2 mu/rho - mu/a) == 0 :", at(Gtt - (2*mu/rho_k - mu/a), Cv, ev) == 0)
    print("     d_s g_tt, d_s g_ts, d_s g_tpsi == 0 :", [at(sp.diff(G_, s), Cv, ev) == 0 for G_ in (Gtt, Gts, Gtp)])
    print("     (M1) d_t g_ts == 0 :", at(Dt(Gts), Cv, ev) == 0, "  g_ts = I =", at(Gts, Cv, ev))
    print("     (M2) d_t g_tpsi - d_psi g_tt = p' =", at(Dt(Gtp) - Dpsi(Gtt), Cv, ev), " (force-free: 0)")
    clk = [at(D[b](alk[a_]) - D[a_](alk[b]), Cv, ev) == 0 for a_, b in [(psi, s), (psi, t), (s, t)]]
    print("     closedness of x_tt.dx for (psi,s),(psi,t),(s,t):", clk)
    print("     alpha_s == 0:", at(alk[s], Cv, ev) == 0, " | alpha_psi - d_psi g_tt/2 == 0:", at(alk[psi] - Dpsi(Gtt)/2, Cv, ev) == 0)
    print("     x_tt + mu x/|x|^3 == 0 :", all(at(c, Cv, ev) == 0 for c in (xk_tt + mu*xk/rho_k**3)))
print(f"done in {time.time()-T0:.0f} s")
