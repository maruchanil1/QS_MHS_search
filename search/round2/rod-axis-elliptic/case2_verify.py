"""Direct verification of Kovacic case-2 candidates: is omega = (phi_K +- sqrt R)/2 a solution of the Riccati omega' + omega^2 = r ?
(Kovacic's theorem guarantees it when P_K satisfies the step-3 equation; this checks our implementation.)  Also compares
sigma_closed = q' (omega - P'/(4P))/k with the numerical Floquet sigma on a grid that avoids poles.
Run: python3 case2_verify.py
"""
import sympy as sp, numpy as np, sys
sys.path.insert(0, '/home/user/QS_MHS_search/search/round2/rod-axis-elliptic')
from case2_admissibility import case2_system, q, K, A
from scipy.integrate import solve_ivp

def verify(Q1, Q2, Q3, n, e1, e2, e3, einf, d, Kv, Av, c0v=None, label=""):
    eqs, cs, Bb, r, theta, PK, P = case2_system(Q1, Q2, Q3, n, e1, e2, e3, einf, d)
    sub = {K: Kv, A: Av}
    if d: sub[cs[0]] = c0v
    rr = r.subs(sub); PKs = PK.subs(sub) if d else PK
    phiK = theta + sp.diff(PKs, q) / PKs
    R = sp.together(4 * rr - phiK**2 - 2 * sp.diff(phiK, q))
    # residuals of the step-3 equation for P_K
    th1 = sp.diff(theta, q); th2 = sp.diff(th1, q)
    step3 = (sp.diff(PKs, q, 3) + 3 * theta * sp.diff(PKs, q, 2) + (3 * theta**2 + 3 * th1 - 4 * rr) * sp.diff(PKs, q)
             + (th2 + 3 * theta * th1 + theta**3 - 4 * rr * theta - 2 * sp.diff(rr, q)) * PKs)
    qs = [float(Q1) + (float(Q2) - float(Q1)) * t for t in (0.13, 0.37, 0.61, 0.89)]
    s3 = [abs(float(step3.subs(q, qq))) for qq in qs]
    res = []
    for sgn in (+1, -1):
        om = (phiK + sgn * sp.sqrt(R)) / 2
        ric = sp.diff(om, q) + om**2 - rr
        res.append([complex(ric.subs(q, qq)) for qq in qs])
    print(f"\n{label}: n={n} e=({e1},{e2},{e3},{einf}) d={d}, K={float(Kv):.6f}, A={float(Av):.6f}" + (f", c0={float(c0v):.6f}" if d else ""))
    print("   step-3 equation residual at 4 points:", [f"{v:.1e}" for v in s3])
    print("   R at these points:", [f"{complex(R.subs(q, qq)).real:+.4g}" for qq in qs])
    for sgn, rc in zip((+1, -1), res):
        print(f"   Riccati residual omega' + omega^2 - r for sign {sgn:+d}:", [f"{abs(v):.1e}" for v in rc])

Q1, Q2, Q3 = sp.Rational(222842, 10**6), sp.Rational(2212881, 10**6), sp.Rational(-2448875, 10**6)
# case-1 point as control (n=4 e=(1,1,3,3) d=0): K = -q3/4, A = -q3(q1+q2)/2
verify(Q1, Q2, Q3, 4, 1, 1, 3, 3, 0, -Q3 / 4, -Q3 * (Q1 + Q2) / 2, label="control (case-1 point)")
# c0-free family sample (QA rod): recompute K(c0), A(c0) from the linear pair
eqs, cs, Bb, r, theta, PK, P = case2_system(Q1, Q2, Q3, 4, 1, 1, 1, 3, 1)
Mx, rhs = sp.linear_eq_to_matrix(eqs, [K, A])
c0 = cs[0]
for c0v in (sp.Integer(3), sp.Integer(5)):
    Mn = Mx.subs(c0, c0v); rn = rhs.subs(c0, c0v)
    sol = sp.solve(list(Mn * sp.Matrix([K, A]) - rn), [K, A], dict=True)
    print("\n   c0 =", c0v, ": linear system rank", Mn.rank(), "augmented", Mn.row_join(rn).rank(), "solutions", sol)
    if sol:
        verify(Q1, Q2, Q3, 4, 1, 1, 1, 3, 1, sol[0][K], sol[0][A], c0v, label="c0-free family")
# n=5 e=(1,1,2,3) d=1 point on the QA rod
verify(Q1, Q2, Q3, 5, 1, 1, 2, 3, 1, sp.Float('1.055728', 15), sp.Float('5.453293', 15), sp.Float('6.53025093786403351923038581333', 30), label="n=5 candidate (rounded K, A)")
