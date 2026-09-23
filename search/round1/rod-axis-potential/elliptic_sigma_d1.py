"""Kovacic case-1 branches with deg S = 1 for Hill's equation phi'' + [k^2 + k b/q^2 - k a/q] phi = 0 on a rod
(see elliptic_sigma.py).  With S = q + s0 the coefficient identities are overdetermined; a, b are eliminated linearly and
the remaining polynomial system in (K = k^2, s0) is reduced by resultants to a constraint polynomial in (q1, q2, q3):
rods admitting such a branch form a codimension-1 subfamily.  Prints the factored constraint(s).
Run: python3 elliptic_sigma_d1.py    (each case is time-boxed to ~2 min)
"""
import sympy as sp, time, signal

q = sp.symbols('q'); q1, q2, q3, k, a, b, s0 = sp.symbols('q1 q2 q3 k a b s0')
P = -(q - q1) * (q - q2) * (q - q3); Q = k + b / q**2 - a / q

class TO(Exception): pass
def handler(sig, frm): raise TO()
signal.signal(signal.SIGALRM, handler)

for e3 in (sp.Integer(0), sp.Rational(1, 2)):
    for m in (sp.Integer(0), -sp.Rational(1, 2)):
        d = 1; e0 = m - e3 - d
        S = q + s0
        lp = e0 / q + e3 / (q - q3) + sp.diff(S, q) / S
        expr = P * (sp.diff(lp, q) + lp**2) + sp.diff(P, q) / 2 * lp + k * Q
        eqs = [sp.expand(c) for c in sp.Poly(sp.expand(sp.numer(sp.together(expr))), q).all_coeffs()]
        eqs = [e for e in eqs if e != 0]
        lin = [e for e in eqs if e.has(a) or e.has(b)]
        sol_ab = sp.solve(lin[-2:], [a, b], dict=True)[0]
        rest = [sp.factor(sp.numer(sp.together(e.subs(sol_ab)))) for e in eqs]
        rest = [r for r in rest if r != 0]
        print(f"\n=== d=1, e3={e3}, m={m} (e0={e0}): {len(rest)} equations in (k, s0) after eliminating a,b; degrees in k: {[sp.degree(r, k) for r in rest]}, in s0: {[sp.degree(r, s0) for r in rest]}")
        signal.alarm(150)
        t0 = time.time()
        try:
            K = sp.symbols('K')
            restK = [sp.expand(r.subs(k**2, K)) for r in rest]
            if any(r.has(k) for r in restK):
                # odd powers of k remain: multiply/rationalise — substitute k = sqrt(K) and clear
                restK = [sp.expand(sp.numer(sp.together(r.subs(k, sp.sqrt(K))))) for r in rest]
            print("    still contains k (odd powers)?", any(r.has(k) for r in restK))
            # eliminate s0 by resultants against the first equation
            R = [sp.factor(sp.resultant(restK[0], r, s0)) for r in restK[1:]]
            print("    resultants w.r.t. s0 (factored):")
            for r in R:
                print("      ", r)
            if len(R) >= 2:
                RR = sp.factor(sp.resultant(sp.Poly(R[0], K).as_expr(), sp.Poly(R[1], K).as_expr(), K)) if (R[0].has(K) and R[1].has(K)) else sp.factor(sp.gcd(R[0], R[1]))
                print("    => rod constraint (resultant in K, factored):", RR)
            elif len(R) == 1:
                sols = sp.solve(R[0], K)
                print("    => K = k^2 solutions:", sols)
        except TO:
            print("    [time-boxed out]")
        finally:
            signal.alarm(0)
        print(f"    [{time.time()-t0:.1f}s]")
