"""TASK A(i)-(ii): Jacobian condition for tori swept by congruent Kepler ellipses.

x(psi,s,eta) = g(s,psi) e(eta),  e = a(cos eta - eps, beta sin eta, 0), beta = sqrt(1-eps^2),
t = sqrt(a^3/mu)(eta - eps sin eta).  d_s g = g Om_s^x, d_psi g = g Om_psi^x (body frame).
J = det(d_psi x, d_s x, d_t x) = D(eta) / (sqrt(a^3/mu)(1 - eps cos eta)),
D := det(Om_psi x e + e_psi, Om_s x e, e')   (the secular eta_psi term drops out of J).
Jacobian condition d_t J = 0  <=>  D(eta) = c0 (1 - eps cos eta) for all eta.

This script: expands D in harmonics of eta, prints the 6 coefficient equations, and solves them
for Om_s = (p1,p2,p3), Om_psi = (q1,q2,q3) given a, a', eps, eps'.
Run: python3 kepler_jacobian.py
"""
import sympy as sp

eta = sp.symbols('eta', real=True)
a, ap, eps, epsp = sp.symbols('a a_p epsilon epsilon_p', real=True)   # a' = a_p, eps' = epsilon_p
beta = sp.sqrt(1 - eps**2)
p1, p2, p3, q1, q2, q3, c0 = sp.symbols('p1 p2 p3 q1 q2 q3 c0', real=True)
Om_s = sp.Matrix([p1, p2, p3])
Om_psi = sp.Matrix([q1, q2, q3])

e = sp.Matrix([a*(sp.cos(eta) - eps), a*beta*sp.sin(eta), 0])
e_eta = e.diff(eta)
# psi-derivative at fixed eta (a -> a + a_p dpsi, eps -> eps + eps_p dpsi)
e_psi = e.diff(a)*ap + e.diff(eps)*epsp

A = Om_psi.cross(e) + e_psi
B = Om_s.cross(e)
D = sp.expand(sp.Matrix.hstack(A, B, e_eta).det())

# structural factorisation claimed in notes:  D = (P.w)(q3 m - n) - p3 m (Q.w)
w = sp.Matrix([e[1], -e[0]])
m = (e.T*e_eta)[0]
n = e_psi[0]*e_eta[1] - e_psi[1]*e_eta[0]
D_struct = (p1*w[0] + p2*w[1])*(q3*m - n) - p3*m*(q1*w[0] + q2*w[1])
print("structural formula D = (P.w)(q3 m - n) - p3 m (Q.w) holds:",
      sp.simplify(sp.expand(D - D_struct)) == 0)
print("m = e.e' =", sp.factor(sp.trigsimp(m)))
print("n = (e_psi x e')_3 =", sp.factor(sp.trigsimp(n)))

# Fourier coefficients of the target identity  D - c0 (1 - eps cos eta) = 0
target = sp.expand(D - c0*(1 - eps*sp.cos(eta)))
target = sp.expand(sp.expand_trig(target))
# rewrite in Fourier basis
target_f = sp.fu(target)  # not needed; do explicit projection instead
def coeff(f, k, kind):
    if k == 0:
        return sp.simplify(sp.integrate(f, (eta, 0, 2*sp.pi))/(2*sp.pi))
    basis = sp.cos(k*eta) if kind == 'c' else sp.sin(k*eta)
    return sp.simplify(sp.integrate(f*basis, (eta, 0, 2*sp.pi))/sp.pi)

eqs = {}
for k, kind in [(0, 'c'), (1, 'c'), (1, 's'), (2, 'c'), (2, 's'), (3, 'c'), (3, 's')]:
    eqs[(k, kind)] = coeff(target, k, kind)
    print(f"harmonic {kind}{k}: ", sp.factor(eqs[(k, kind)]))

# --- solve ---
E = [sp.factor(v) for v in eqs.values()]
sols = sp.solve(E, [p1, p2, p3, q1, q2, q3, c0], dict=True)
print("\nnumber of solution branches:", len(sols))
for S in sols:
    print(S)
    print("   c0 =", sp.simplify(S.get(c0, c0)))

# --- also solve treating c0 as given nonzero, eliminate ---
print("\nGroebner-type analysis: eliminate to find constraints with c0 != 0")
G = sp.groebner(E, p1, p2, p3, q1, q2, q3, c0, order='lex')
for g in G.exprs:
    print("  ", sp.factor(g))
