"""All exactly-solvable (Kovacic case-1, deg S <= 1) first-order QS branches on a rod, and a scan of the closed (1,2)
Kirchhoff-rod family for CURRENT-FREE (j = 0) branches.

For each (d, e3, m) with d <= 1:  K = k^2 is a root of an explicit polynomial in K with coefficients in (q1,q2,q3);
s0 (d=1) is rational in (K, q's);  A := k a and Bt := k b are polynomial in (K, s0, q's).  Then
   eta^2 solves  K eta^4 + 2 C k eta^2 - Bt = 0  (k = +-sqrt K, C = (l2 c - alpha)/l3^2),   j = A/(k eta^2) + 2 tau0,
   sigma = R(q) q',  R = (1/k)[e0/q + e3/(q-q3) + S'/S],  admissible iff eta^2 > 0 and S has no zero in [q1,q2].
Run: python3 branches_scan.py
"""
import sympy as sp, numpy as np, sys
sys.path.insert(0, '.')
from vacuum_S1_search import rod_data   # closed (1,2) rod family (R_max = 1), prints its own scan on import? no: guarded below

q = sp.symbols('q'); q1, q2, q3, k, a, b, s0, K = sp.symbols('q1 q2 q3 k a b s0 K')
P = -(q - q1) * (q - q2) * (q - q3); Q = k + b / q**2 - a / q

branches = []
for d in (0, 1):
    for e3 in (sp.Integer(0), sp.Rational(1, 2)):
        for m in (sp.Integer(0), -sp.Rational(1, 2)):
            e0 = m - e3 - d
            S = q + s0 if d == 1 else sp.Integer(1)
            lp = e0 / q + e3 / (q - q3) + sp.diff(S, q) / S
            expr = P * (sp.diff(lp, q) + lp**2) + sp.diff(P, q) / 2 * lp + k * Q
            eqs = [sp.expand(c) for c in sp.Poly(sp.expand(sp.numer(sp.together(expr))), q).all_coeffs()]
            eqs = [e for e in eqs if e != 0]
            lin = [e for e in eqs if e.has(a) or e.has(b)]
            sol_ab = sp.solve(lin[-2:], [a, b], dict=True)[0]
            A_ = sp.expand(sp.simplify(k * sol_ab[a]).subs(k**2, K)); Bt_ = sp.expand(sp.simplify(k * sol_ab[b]).subs(k**2, K))
            rest = [sp.numer(sp.together(e.subs(sol_ab))) for e in eqs]
            rest = [sp.expand(r.subs(k**2, K)) for r in rest if sp.expand(r) != 0]
            rest = [sp.expand(sp.numer(sp.together(r.subs(k, sp.sqrt(K))))) for r in rest]
            if d == 1:
                lin_s0 = [r for r in rest if sp.degree(r, s0) == 1][0]
                s0_sol = sp.solve(lin_s0, s0)[0]
                polys = [sp.factor(sp.numer(sp.together(r.subs(s0, s0_sol)))) for r in rest if r is not lin_s0]
                polyK = sp.Poly(sp.gcd(*[sp.Poly(p_, K) for p_ in polys]) if len(polys) > 1 else polys[0], K)
            else:
                s0_sol = sp.Integer(0)
                polyK = sp.Poly(sp.gcd(*[sp.Poly(r, K) for r in rest]) if len(rest) > 1 else rest[0], K)
            tag = f"d={d},e3={e3},e0={e0}"
            print(f"[{tag}]  K-polynomial: {sp.factor(polyK.as_expr())}" + (f"   s0 = {sp.simplify(s0_sol)}" if d == 1 else ""))
            fK = sp.lambdify((q1, q2, q3), polyK.all_coeffs(), 'numpy')
            fs0 = sp.lambdify((K, q1, q2, q3), s0_sol, 'numpy')
            fA = sp.lambdify((K, s0, q1, q2, q3), A_, 'numpy'); fB = sp.lambdify((K, s0, q1, q2, q3), Bt_, 'numpy')
            branches.append(dict(tag=tag, d=d, e0=float(e0), e3=float(e3), fK=fK, fs0=fs0, fA=fA, fB=fB))

def evaluate(rod):
    """all admissible exactly-solvable points for one rod: list of dicts"""
    P_ = np.poly1d([-1, rod['p2'], rod['p1'], rod['p0']]); rts = np.sort(P_.roots.real)
    qq3, qq1, qq2 = rts
    C = rod['C']; tau0 = rod['l2'] / (2 * rod['l3']); L = rod['L']
    out = []
    for br in branches:
        coeffs = np.atleast_1d(np.array(br['fK'](qq1, qq2, qq3), float))
        Kroots = sorted([r.real for r in (np.roots(coeffs) if len(coeffs) > 1 else []) if abs(r.imag) < 1e-9 and r.real > 1e-12])
        for ir, Kr in enumerate(Kroots):
            s0v = float(br['fs0'](Kr, qq1, qq2, qq3)) if br['d'] == 1 else 0.0
            if br['d'] == 1 and (-s0v >= qq1 - 1e-12 and -s0v <= qq2 + 1e-12):
                continue          # S has a zero in the physical range -> sigma has a pole
            Av = float(br['fA'](Kr, s0v, qq1, qq2, qq3)); Bv = float(br['fB'](Kr, s0v, qq1, qq2, qq3))
            for ksign in (+1, -1):
                kv = ksign * np.sqrt(Kr)
                e2 = np.roots([Kr, 2 * C * kv, -Bv])
                for idx, ev in enumerate(sorted(e2.real)):
                    if abs(ev) < 1e-12 or ev <= 0 or abs(e2[0].imag) > 1e-9:
                        continue
                    jv = Av / (kv * ev) + 2 * tau0
                    out.append(dict(tag=br['tag'], K=Kr, k=kv, s0=s0v, eta2=ev, j=jv, iota=kv * L / (2 * np.pi), key=(br['tag'] + f'#K{ir}', ksign, idx)))
    return out

if __name__ == '__main__':
    print("\n#### exactly-solvable points on the QA-matched rod (lamODE = -0.4273, R_max = 1) ####")
    d0 = rod_data(-0.4273, [0.4632, -0.3408])
    for pt in evaluate(d0):
        print(f"  {pt['tag']:22s} k={pt['k']:+.4f} iota0-N={pt['iota']:+.4f} s0={pt['s0']:+.4f} eta^2={pt['eta2']:.4f} j={pt['j']:+.4f}")
    print("\n#### scan of the closed (1,2) family for j = 0 (continuation from the QA rod in both directions; circles excluded) ####")
    table = {}; rods = []
    for lams in (np.arange(-0.4273, -0.3095, 0.004), np.arange(-0.4273, -0.578, -0.004)):
        guess = [0.4632, -0.3408]
        for lam in lams:
            dd = rod_data(lam, guess)
            if dd is None or abs(dd['beta']) < 1e-3:      # None or degenerate circle
                print(f"   (stop at lamODE={lam:.4f}: {'no rod' if dd is None else 'circle'})"); break
            guess = dd['p']; rods.append(dd)
            for pt in evaluate(dd):
                table.setdefault(pt['key'], []).append((lam, pt['j'], pt['iota'], pt['eta2']))
    lam_all = np.array(sorted(r['lam'] for r in rods)); dlam = 0.004
    print(f"   {len(rods)} genuine closed (1,2) rods, lamODE in [{lam_all.min():.4f}, {lam_all.max():.4f}]; p2 sign change (birth of the S=1 branch) near lamODE = "
          + str([f"{r['lam']:.4f}" for r in sorted(rods, key=lambda r: r['lam']) if abs(r['p2']) < 0.06]))
    lams = [0, dlam]
    from scipy.optimize import brentq
    found = 0
    for key, rows in table.items():
        rows.sort()
        lam_arr = np.array([r[0] for r in rows]); j_arr = np.array([r[1] for r in rows])
        print(f"  branch {key[0]:22s} sign(k)={key[1]:+d} eta-root#{key[2]}: present on {len(rows)} rods, j range [{j_arr.min():+.3f},{j_arr.max():+.3f}], iota0-N range [{min(r[2] for r in rows):+.3f},{max(r[2] for r in rows):+.3f}], eta^2 range [{min(r[3] for r in rows):.3f},{max(r[3] for r in rows):.3f}]")
        for i in range(len(rows) - 1):
            if j_arr[i] * j_arr[i + 1] < 0 and abs(lam_arr[i + 1] - lam_arr[i]) < 1.5 * dlam:
                g0 = [r['p'] for r in rods if abs(r['lam'] - lam_arr[i]) < 1e-9][0]
                def F(lam, key=key):
                    dd = rod_data(lam, g0)
                    pts = [p for p in evaluate(dd) if p['key'] == key]
                    return pts[0]['j'] if pts else np.nan
                try:
                    lam0 = brentq(F, lam_arr[i], lam_arr[i + 1], xtol=1e-10)
                except Exception as ex:
                    print("     (root refinement failed:", ex, ")"); continue
                dd = rod_data(lam0, g0); pt = [p for p in evaluate(dd) if p['key'] == key][0]
                found += 1
                print(f"   *** j = 0 at lamODE = {lam0:.7f}: rod (R_max=1) alpha={dd['alpha']:.6f} beta={dd['beta']:.6f} L={dd['L']:.6f} l2={dd['l2']:.6f} l3={dd['l3']:.6f} c={dd['c']:.6f} C={dd['C']:.6f}")
                print(f"       k = {pt['k']:+.6f}, iota0 - N = kL/2pi = {pt['iota']:+.6f}, eta^2 = {pt['eta2']:.6f} (eta={np.sqrt(pt['eta2']):.4f}), s0 = {pt['s0']:+.6f}, K-branch {pt['tag']}")
                np.save(f"vacuum_rod_{key[0].replace(',', '_').replace('=', '').replace('#','_')}_{key[1]:+d}_{key[2]}.npy", dict(rod=dd, point=pt), allow_pickle=True)
    print(f"\n{found} current-free exactly-solvable point(s) found in the closed (1,2) family (deg S <= 1).")
