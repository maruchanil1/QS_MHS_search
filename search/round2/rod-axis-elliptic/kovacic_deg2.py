"""TASK 4a: Kovacic case-1 branches with deg S = 2 for Hill's equation on a rod (round-1 elliptic_sigma_d1.py pattern).
   phi'' + [k^2 + k b/q^2 - k a/q] phi = 0  in the variable q = kappa^2:  P phi_qq + (P'/2) phi_q + k Q phi = 0,
   P = -(q-q1)(q-q2)(q-q3),  Q = k + b/q^2 - a/q,  phi = q^e0 (q-q3)^e3 S(q),  S = q^2 + s1 q + s0,  e0 = m - e3 - 2.
Unknowns: (k, a, b, s0, s1) with (q1, q2, q3) free rod moduli.  a, b eliminated linearly; then K = k^2; s0 eliminated by
resultants against the equation linear in s0 (if any) or pairwise; then s1; the gcd of the resulting polynomials in K is
the K-polynomial of the branch (a rod-independent branch exists iff the final resultants have a common factor in K).
Run: python3 kovacic_deg2.py     (time-boxed per case)
"""
import sympy as sp, time, signal

q = sp.symbols('q'); q1, q2, q3, k, a, b, s0, s1, K = sp.symbols('q1 q2 q3 k a b s0 s1 K')
P = -(q - q1) * (q - q2) * (q - q3); Q = k + b / q**2 - a / q

class TO(Exception): pass
def handler(sig, frm): raise TO()
signal.signal(signal.SIGALRM, handler)

summary = []
for e3 in (sp.Integer(0), sp.Rational(1, 2)):
    for m in (sp.Integer(0), -sp.Rational(1, 2)):
        d = 2; e0 = m - e3 - d
        S = q**2 + s1 * q + s0
        lp = e0 / q + e3 / (q - q3) + sp.diff(S, q) / S
        expr = P * (sp.diff(lp, q) + lp**2) + sp.diff(P, q) / 2 * lp + k * Q
        eqs = [sp.expand(c) for c in sp.Poly(sp.expand(sp.numer(sp.together(expr))), q).all_coeffs()]
        eqs = [e for e in eqs if e != 0]
        lin = [e for e in eqs if e.has(a) or e.has(b)]
        # pick two equations that determine a, b (lowest powers of q carry b, a)
        sol_ab = None
        for i in range(len(lin)):
            for jj in range(i + 1, len(lin)):
                try:
                    s_ = sp.solve([lin[i], lin[jj]], [a, b], dict=True)
                except Exception:
                    s_ = []
                if s_ and set(s_[0].keys()) == {a, b}:
                    sol_ab = s_[0]; break
            if sol_ab: break
        rest = [sp.numer(sp.together(e.subs(sol_ab))) for e in eqs]
        rest = [sp.expand(r) for r in rest if sp.expand(r) != 0]
        rest = [sp.expand(sp.numer(sp.together(r.subs(k, sp.sqrt(K))))) for r in rest]
        tag = f"d=2, e3={e3}, m={m} (e0={e0})"
        print(f"\n=== {tag}: {len(rest)} equations in (K, s0, s1); degrees in s0 {[sp.degree(r, s0) for r in rest]}, in s1 {[sp.degree(r, s1) for r in rest]}, in K {[sp.degree(r, K) for r in rest]}; contains sqrt(K)? {any(r.has(sp.sqrt(K)) for r in rest)}")
        signal.alarm(110); t0 = time.time()
        try:
            # step 1: the top coefficient equation is independent of s0 and linear in s1 -> solve for s1
            lin_s1 = [r for r in rest if sp.degree(r, s0) == 0 and sp.degree(r, s1) == 1]
            assert lin_s1, "no equation linear in s1 and free of s0"
            s1_sol = sp.solve(lin_s1[0], s1)[0]
            print("   s1 =", sp.factor(s1_sol))
            rest1 = [sp.factor(sp.numer(sp.together(r.subs(s1, s1_sol)))) for r in rest if r is not lin_s1[0]]
            rest1 = [r for r in rest1 if r != 0]
            print(f"   after substituting s1: {len(rest1)} polynomials in (K, s0), degrees in s0 {[sp.degree(r, s0) for r in rest1]}, in K {[sp.degree(r, K) for r in rest1]}")
            # step 2: eliminate s0 by resultants against the lowest-degree pivot
            rest1.sort(key=lambda r: (sp.degree(r, s0), sp.degree(r, K)))
            piv = rest1[0]
            R2 = []
            for r in rest1[1:]:
                res = sp.factor(sp.resultant(piv, r, s0))
                R2.append(res)
            R2 = [r for r in R2 if r != 0]
            print(f"   after eliminating s0 (pivot degree {sp.degree(piv, s0)} in s0): {len(R2)} polynomials in K, degrees {[sp.degree(r, K) for r in R2]}")
            g = None
            for r in R2:
                pr = sp.Poly(r, K)
                g = pr if g is None else sp.gcd(g, pr)
            print("   gcd over K of the final resultants (factored):", sp.factor(g.as_expr()) if g is not None else None)
            # also list factors of the first resultant, each possible K-branch, and check which factors are common
            facs = sp.factor_list(R2[0])[1]
            common = []
            for fac, mult in facs:
                if fac.has(K):
                    ok = all(sp.rem(sp.Poly(r, K), sp.Poly(fac, K)).is_zero if sp.Poly(r, K).degree() >= sp.Poly(fac, K).degree() else False for r in R2[1:]) if len(R2) > 1 else True
                    common.append((fac, ok))
            for fac, ok in common:
                print(f"      factor {fac}  -> common to all resultants: {ok}")
            summary.append((tag, [f for f, ok in common if ok]))
        except TO:
            print("   [time-boxed out]"); summary.append((tag, 'timeout'))
        finally:
            signal.alarm(0)
        print(f"   [{time.time()-t0:.1f}s]")

print("\n\n#### SUMMARY deg S = 2 ####")
for tag, fs in summary:
    print(tag, "->", fs)
