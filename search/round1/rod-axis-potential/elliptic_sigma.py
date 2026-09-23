"""Can the first-order QS solution around a Kirchhoff-rod axis be an ELLIPTIC function of arclength?  (exact, sympy)

From axis_linearization.py: the first-order clock around a constant-speed closed orbit holds iff
   X = (eta/kappa) e^{iks},  Y = X (kappa^2/eta^2)(sigma - i),   sigma' = -k(1+sigma^2) - k eta^4/kappa^4 + eta^2 (j - 2 tau)/kappa^2,
j = on-axis (curl B).T = const, k = 2 pi (iota0 - N)/L.  On a rod (rod_exact.py) q = kappa^2 is elliptic,
q'^2 = P(q) = -(q-q1)(q-q2)(q-q3), q3 < 0 < q1 < q2, tau = tau0 + C/q, C^2 = -q1 q2 q3/4.

sigma = phi'/(k phi) linearises the Riccati to HILL'S EQUATION with an elliptic potential
        phi'' + [k^2 + k b/q^2 - k a/q] phi = 0,     a = eta^2 (j - 2 tau0),  b = k eta^4 + 2 eta^2 C      (H)
(in the variable q:  P phi_qq + (P'/2) phi_q + k Q phi = 0, Q = k + b/q^2 - a/q).
sigma elliptic (rational in q, q')  <=>  phi'/phi rational in q  <=>  (Kovacic case 1)
        phi = q^e0 (q - q3)^e3 S(q),  S polynomial of degree d,  e3 in {0, 1/2},  e0 + e3 + d in {0, -1/2}
(exponents: q1,q2,q3 -> {0,1/2}; q=0 -> e(1-e) = k b/(q1 q2 q3); infinity -> {0,-1/2}; e1 = e2 = 0 because sigma must
be finite at the curvature extrema).  Stellarator symmetry (sigma odd) is built in (sigma = R(q) q').
This script solves the resulting polynomial identities exactly for d <= 3 and both e3, and post-processes eta^2, j.

Run: python3 elliptic_sigma.py        (~1-2 min)
"""
import sympy as sp, sys, time

q = sp.symbols('q')
q1, q2, q3, k, a, b = sp.symbols('q1 q2 q3 k a b')
P = -(q - q1) * (q - q2) * (q - q3)
Q = k + b / q**2 - a / q
DMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 3

def coefficient_equations(d, e3, m):
    e0 = m - e3 - d
    cs = list(sp.symbols(f's0:{d}'))
    S = q**d + sum(cf * q**i for i, cf in enumerate(cs))
    lp = e0 / q + e3 / (q - q3) + sp.diff(S, q) / S
    expr = P * (sp.diff(lp, q) + lp**2) + sp.diff(P, q) / 2 * lp + k * Q
    num = sp.numer(sp.together(expr))
    poly = sp.Poly(sp.expand(num), q)
    eqs = [sp.expand(cf) for cf in poly.all_coeffs()]
    eqs = [e for e in eqs if e != 0]
    return e0, cs, S, eqs

def solve_case(d, e3, m):
    t0 = time.time()
    e0, cs, S, eqs = coefficient_equations(d, e3, m)
    tag = f"d={d}, e3={e3}, m={m} (e0={e0})"
    # a, b enter linearly: eliminate them with two equations that involve them
    lin = [e for e in eqs if e.has(a) or e.has(b)]
    sol_ab = None
    for i in range(len(lin)):
        for jj in range(i + 1, len(lin)):
            try:
                s_ = sp.solve([lin[i], lin[jj]], [a, b], dict=True)
            except Exception:
                s_ = []
            if s_ and all(v.has(k) or True for v in s_[0].values()) and set(s_[0].keys()) == {a, b}:
                sol_ab = s_[0]; break
        if sol_ab: break
    if sol_ab is None:
        print(f"\n=== {tag}: could not eliminate a,b; equations:", eqs); return []
    rest = [sp.factor(sp.together(e.subs(sol_ab))) for e in eqs]
    rest = [sp.numer(r) for r in rest if r != 0]
    rest = [r for r in rest if r != 0]
    unknowns = [k] + cs
    print(f"\n=== {tag}: {len(eqs)} coefficient eqs; after eliminating (a,b): {len(rest)} eqs for {unknowns} (+ q1,q2,q3 free)")
    out = []
    if not rest:
        print("    identically satisfied -> k free?!  (check)"); return out
    # solve remaining polynomial system; allow q3 to be solved for if overdetermined
    try:
        sols = sp.solve(rest, unknowns, dict=True)
    except Exception as ex:
        sols = []
        print("    sp.solve failed:", ex)
    if not sols:
        try:
            sols = sp.solve(rest, unknowns + [q3], dict=True)
            if sols: print("    (needed to solve for q3 as well: constraint on the rod)")
        except Exception as ex:
            print("    sp.solve (with q3) failed:", ex)
    for so in sols:
        kk = so.get(k, k)
        if kk == 0 or kk.has(sp.I) and sp.im(kk) != 0:
            continue
        full = dict(so)
        full[a] = sp.simplify(sol_ab[a].subs(so)); full[b] = sp.simplify(sol_ab[b].subs(so))
        resid = [sp.simplify(e.subs(full)) for e in eqs]
        if any(r != 0 for r in resid):
            print("    (rejected: residuals", resid, ")"); continue
        print("    SOLUTION:", {str(kk_): sp.simplify(vv) for kk_, vv in full.items()})
        Rq = sp.simplify((e0 / q + e3 / (q - q3) + sp.diff(S, q).subs(so) / S.subs(so)) / kk)
        print("      R(q) =", Rq, "  (sigma = R(kappa^2) (kappa^2)')")
        out.append((tag, full, Rq))
    if not out:
        print("    no solution with k != 0")
    print(f"    [{time.time()-t0:.1f}s]")
    return out

results = []
for d in range(0, DMAX + 1):
    for e3 in (sp.Integer(0), sp.Rational(1, 2)):
        for m in (sp.Integer(0), -sp.Rational(1, 2)):
            results += solve_case(d, e3, m)

print("\n\n############ post-processing: eta^2 and j for each solution ############")
eta2 = sp.symbols('eta2')
for tag, full, Rq in results:
    kk, aa, bb = full[k], full[a], full[b]
    print(f"\n--- {tag} ---  k = {kk},  a = {aa},  b = {bb}")
    for sgn in (+1, -1):
        Cexpr = sgn * sp.sqrt(-q1 * q2 * q3) / 2
        if q3 in full: Cexpr = Cexpr.subs(q3, full[q3])
        sol_eta = sp.solve(sp.Eq(kk * eta2**2 + 2 * eta2 * Cexpr, bb), eta2)
        print(f"   C = {sgn:+d} sqrt(-q1 q2 q3)/2:  eta^2 in", [sp.simplify(e) for e in sol_eta],
              ";  j = a/eta^2 + 2 tau0  (vacuum j=0  <=>  a = -2 eta^2 tau0)")
