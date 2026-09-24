"""TASK 4b: the sign law  sign j = sign k  on the exactly solvable branches -- what it reduces to, and why it cannot follow
from the local rod identities alone.

S=1 branch (round 1):  k^2 = -p2/4,  sigma = -kappa'/(k kappa),  eta^2 in {C/k (A), -3C/k (B)},  j = p1/(2 k eta^2) + 2 tau0.
Rod moduli (rod_exact.py):  tau0 = l2/(2 l3),  C = (l2 c - alpha)/l3^2,  p1 = 4(1-c^2)/l3^2 + 4 l2 C/l3,  p0 = -4C^2.
This script verifies symbolically
   (A)  j k = (k/C) f_A / l3^2,  f_A = 2(1-c^2) + 3 l2 l3 C = l3^2 (p1/2 + 2 tau0 C)      (k/C > 0 on branch A)
   (B)  j k = (-k/C) f_B / (3 l3^2),  f_B = 2(1-c^2) - l2 l3 C = l3^2 (p1/2 - 6 tau0 C)   (-k/C > 0 on branch B)
so that  sign j = sign k  <=>  p1 + 4 tau0 C > 0 (A),  p1 - 12 tau0 C > 0 (B),
and shows that the rod moduli (q1, q2, q3, tau0) are independent (tau0 is not fixed by the cubic), so that for every
cubic with q3 < 0 < q1 < q2 there are rods (non-closed) with j k < 0 and rods with j = 0 exactly: the sign law observed on
the closed (1,2), (1,3) families is a property of CLOSURE, not of the rod identities.  The same is shown for the deg S = 0
branches K = -q3/4 and K = -(q1+q2+4q3)/4 (j is affine in tau0 with a nonzero slope).
Run: python3 sign_j_reduction.py   (~10 s)
"""
import sympy as sp, numpy as np

l2, l3, c, alpha, k = sp.symbols('lambda2 lambda3 c alpha k', real=True)
tau0 = l2 / (2 * l3); C = (l2 * c - alpha) / l3**2
p1 = 4 * (1 - c**2) / l3**2 + 4 * l2 * C / l3
fA = 2 * (1 - c**2) + 3 * l2 * l3 * C; fB = 2 * (1 - c**2) - l2 * l3 * C
for lab, eta2, f, pref in (("A", C / k, fA, k / C / l3**2), ("B", -3 * C / k, fB, -k / C / (3 * l3**2))):
    j = p1 / (2 * k * eta2) + 2 * tau0
    print(f"branch {lab}: j k - ({pref}) f_{lab} =", sp.simplify(j * k - pref * f))
print("f_A - l3^2 (p1/2 + 2 tau0 C) =", sp.simplify(fA - l3**2 * (p1 / 2 + 2 * tau0 * C)))
print("f_B - l3^2 (p1/2 - 6 tau0 C) =", sp.simplify(fB - l3**2 * (p1 / 2 - 6 * tau0 * C)))

# independence of tau0 from the cubic: the Frenet data of a rod are (kappa^2 elliptic from (p2,p1,p0)) and tau = tau0 + C/q with
# C^2 = -p0/4; the map (l2, l3, c, alpha) -> (p2, p1, p0, tau0) has full rank 4 (generic point):
p2 = 4 * c / l3 - l2**2 / l3**2; p0 = -4 * C**2
J = sp.Matrix([p2, p1, p0, tau0]).jacobian([l2, l3, c, alpha])
print("rank of d(p2,p1,p0,tau0)/d(l2,l3,c,alpha) at a generic point:", J.subs({l2: 0.95, l3: 0.63, c: 0.36, alpha: 0.56}).rank())

# explicit non-closed rods with the QA cubic but j k < 0 or j = 0 on branch A:
q1, q2, q3 = 0.222842, 2.212881, -2.448875           # QA-matched rod (round 1, R_max = 1.2126 units)
P2 = q1 + q2 + q3; P1 = -(q1 * q2 + q1 * q3 + q2 * q3); P0 = q1 * q2 * q3     # P = -(q-q1)(q-q2)(q-q3) = -q^3 + P2 q^2 + P1 q + P0
Cv = -np.sqrt(-P0) / 2                                # sign of C as on the QA rod (C = -0.549)
kv = -np.sqrt(-P2) / 2                                # branch A needs C/k > 0
print(f"\nQA cubic: p2 = {P2:+.5f}, p1 = {P1:.4f}, p0 = {P0:+.4f}; branch A: k = {kv:+.5f}, eta^2 = C/k = {Cv/kv:.4f}")
for t0v in (0.760091, 3.0, -P1 / (4 * Cv)):
    jv = P1 / (2 * kv * (Cv / kv)) + 2 * t0v
    print(f"   tau0 = {t0v:+.5f}: j = {jv:+.5f}, sign(j) = sign(k): {np.sign(jv) == np.sign(kv)}" + ("   (QA rod's tau0)" if abs(t0v - 0.760091) < 1e-6 else "") + ("   (j = 0 exactly: tau0 = -p1/(4C))" if abs(t0v + P1 / (4 * Cv)) < 1e-9 else ""))
print("   => branch A: sign j = sign k  <=>  p1 + 4 tau0 C > 0, i.e. (tau0 C < 0 on the closed rods) 4|tau0 C| < p1;  j = 0 iff tau0 = -p1/(4C).")
print("   margin 4|tau0 C|/p1 along the closed families (sign law <=> ratio < 1 on branch A; branch B: 12 tau0 C < p1 automatically when tau0 C < 0):")
import sys; sys.path.insert(0, '/home/user/QS_MHS_search/search/round2/rod-axis-elliptic')
from rods import family12, family13
for lab, fam in (("(1,2)", family12(0.016)), ("(1,3)", family13(0.04))):
    rat = [4 * abs(r['tau0'] * r['C']) / r['p1'] for r in fam]; sgn = [np.sign(r['tau0'] * r['C']) for r in fam]
    print(f"      {lab}: {len(fam)} rods, lamODE in [{fam[0]['lam']:.3f},{fam[-1]['lam']:.3f}], sign(tau0 C) = {set(sgn)}, 4|tau0 C|/p1 in [{min(rat):.3f}, {max(rat):.3f}], p1 in [{min(r['p1'] for r in fam):.2f},{max(r['p1'] for r in fam):.2f}]")

# deg S = 0 branches: j affine in tau0 with slope 2 -> same conclusion
q, K, a, b = sp.symbols('q K a b'); Q1, Q2, Q3 = sp.symbols('q1 q2 q3')
P = -(q - Q1) * (q - Q2) * (q - Q3); Qf = k + b / q**2 - a / q
print("\ndeg S = 0 branches (k a, k b as functions of the roots):")
for e3, m in ((0, -sp.Rational(1, 2)), (sp.Rational(1, 2), 0), (sp.Rational(1, 2), -sp.Rational(1, 2))):
    e0 = m - e3
    lp = e0 / q + e3 / (q - Q3)
    expr = P * (sp.diff(lp, q) + lp**2) + sp.diff(P, q) / 2 * lp + k * Qf
    eqs = [sp.expand(cf) for cf in sp.Poly(sp.expand(sp.numer(sp.together(expr))), q).all_coeffs()]
    eqs = [e for e in eqs if e != 0]
    sol = sp.solve(eqs, [a, b, k], dict=True)
    seen = set()
    for s_ in sol:
        if s_[k] == 0 or sp.factor(s_[k]**2) in seen: continue
        seen.add(sp.factor(s_[k]**2))
        A_ = sp.simplify(k * s_[a]).subs(k, s_[k]); B_ = sp.simplify(k * s_[b]).subs(k, s_[k])
        print(f"   e3={e3}, e0={e0}: k^2 = {sp.factor(s_[k]**2)},  k a = {sp.factor(A_)},  k b = {sp.factor(B_)}")
print("   j = (k a)/(k eta^2) + 2 tau0 with eta^2 from k^2 eta^4 + 2 C k eta^2 - k b = 0: tau0 enters j only through the additive 2 tau0"
      " => for fixed cubic, j runs over all reals as tau0 varies; a j = 0 rod exists for every branch (non-closed in general).")
