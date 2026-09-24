"""TASK 4c: other closed Kirchhoff-rod families and the sign of the on-axis current on the exactly solvable branches.

Rod ODE (rod_exact.py convention): gamma'' = T x I / lamODE, I = alpha e_z + e_z x x, start at (R_max=1, 0, 0) with
T(0) = (0, cos beta, sin beta) (stellarator-symmetric point, curvature extremum).  Closure of a (1, nfp) rod: the next
curvature extremum lies at phi = pi/nfp with Z = 0 (two conditions on (alpha, beta) at fixed lamODE -> 1-parameter families),
L = 2 nfp t_q; closure |x(L) - x(0)| is checked a posteriori.  alpha = 0 rods ("Kida rods", N meridional): two conditions
on (beta, lamODE) -> isolated rods.
For every rod all Kovacic case-1 branches with deg S <= 1 are evaluated with round 1's branches_scan.evaluate (K-polynomials,
eta^2, j); we record sign(j) sign(k), min |j|, and for the S=1 branch (needs p2 < 0) the vacuum functions
f_A = 2(1-c^2) + 3 l2 l3 C  (branch eta^2 = C/k),  f_B = 2(1-c^2) - l2 l3 C  (branch eta^2 = -3C/k):
sign j = sign k on branch A <=> f_A > 0, on branch B <=> f_B > 0 (derivation in notes.md).
Run: python3 rod_families_scan.py     (~1-2 min)
"""
import numpy as np, sys, warnings
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/QS_MHS_search/search/round1/rod-axis-potential')
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from branches_scan import evaluate      # builds the deg<=1 branches on import

Rmax = 1.0

def rod_rhs(t, y, al, lam):
    x, T = y[:3], y[3:]
    I = np.array([-x[1], x[0], al])
    return np.concatenate([T, np.cross(T, I) / lam])

def kappa_of(y, al, lam):
    x, T = y[:3], y[3:]
    return np.linalg.norm(np.cross(T, np.array([-x[1], x[0], al])) / lam)

def shoot(al, bet, lam, nfp, want=False):
    y0 = np.array([Rmax, 0, 0, 0, np.cos(bet), np.sin(bet)])
    ev = lambda t, y, al, lam: np.arctan2(y[1], y[0]) - np.pi / nfp
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(rod_rhs, (0, 40), y0, args=(al, lam), events=ev, rtol=1e-11, atol=1e-13, dense_output=True)
    if len(sol.t_events[0]) == 0:
        return (None, None) if want else [1e3, 1e3]
    t1 = sol.t_events[0][0]; y1 = sol.sol(t1); h = 1e-5
    dk = (kappa_of(sol.sol(t1 + h), al, lam) - kappa_of(sol.sol(t1 - h), al, lam)) / (2 * h)
    if want: return sol, t1
    return [dk, y1[2]]

def rod_moduli(al, bet, lam, nfp):
    sol, tq = shoot(al, bet, lam, nfp, True)
    if sol is None: return None
    L = 2 * nfp * tq
    full = solve_ivp(rod_rhs, (0, L), np.array([Rmax, 0, 0, 0, np.cos(bet), np.sin(bet)]), args=(al, lam), rtol=1e-11, atol=1e-13, dense_output=True)
    yL = full.sol(L); y0 = full.sol(0.0)
    closure = np.linalg.norm(yL[:3] - y0[:3]) + np.linalg.norm(yL[3:] - y0[3:])
    T0 = y0[3:]; x0 = y0[:3]; I0 = np.array([-x0[1], x0[0], al])
    acc0 = np.cross(T0, I0) / lam; k0 = np.linalg.norm(acc0); N0 = acc0 / k0; B0 = np.cross(T0, N0)
    l2 = T0 @ I0; l3 = (B0 @ I0) / k0
    c = T0[2] + l3 * k0**2 / 2
    C = (l2 * c - al) / l3**2
    p2 = 4 * c / l3 - l2**2 / l3**2; p1 = 4 * (1 - c**2) / l3**2 + 4 * l2 * C / l3; p0 = -4 * C**2
    fA = 2 * (1 - c**2) + 3 * l2 * l3 * C; fB = 2 * (1 - c**2) - l2 * l3 * C
    # sanity: kappa range along the rod and the roots of the cubic
    ss = np.linspace(0, L, 400); Y = full.sol(ss); kap = np.array([kappa_of(Y[:, i], al, lam) for i in range(len(ss))])
    rts = np.sort(np.roots([-1, p2, p1, p0]).real)
    return dict(lam=lam, alpha=al, beta=bet, L=L, l2=l2, l3=l3, c=c, C=C, p2=p2, p1=p1, p0=p0, fA=fA, fB=fB, closure=closure,
                kmin=kap.min(), kmax=kap.max(), roots=rts, nfp=nfp, tau0=l2 / (2 * l3))

def find_rod(lam, guess, nfp):
    p, info, ier, msg = fsolve(lambda p: shoot(p[0], p[1], lam, nfp), guess, xtol=1e-12, full_output=True)
    if ier != 1 or np.sum(np.abs(shoot(p[0], p[1], lam, nfp))) > 1e-7: return None
    return rod_moduli(p[0], p[1], lam, nfp)

def report(rods, label):
    print(f"\n#### {label}: {len(rods)} rods ####")
    print("  lamODE   alpha    beta     L      l2      l3      c       C      tau0     p2      fA      fB   closure  q1-kmin^2 q2-kmax^2")
    for r in rods:
        print(f"  {r['lam']:7.4f} {r['alpha']:7.4f} {r['beta']:7.4f} {r['L']:6.3f} {r['l2']:7.4f} {r['l3']:7.4f} {r['c']:7.4f} {r['C']:7.4f} {r['tau0']:7.4f} {r['p2']:7.4f} {r['fA']:7.4f} {r['fB']:7.4f}  {r['closure']:.0e}  {r['roots'][1]-r['kmin']**2:+.1e} {r['roots'][2]-r['kmax']**2:+.1e}")
    table = {}
    for r in rods:
        for pt in evaluate(r):
            table.setdefault(pt['key'], []).append((r['lam'], pt['j'], pt['k'], pt['iota'], pt['eta2']))
    viol = 0
    for key, rows in sorted(table.items()):
        j = np.array([x[1] for x in rows]); kk = np.array([x[2] for x in rows]); io_ = np.array([x[3] for x in rows])
        same = np.all(np.sign(j) == np.sign(kk))
        if not same: viol += 1
        print(f"   branch {key[0]:22s} sign k={key[1]:+d} eta-root#{key[2]}: {len(rows):3d} rods, sign(j)=sign(k) on all: {same}, min|j| = {np.min(np.abs(j)):.3f}, j in [{j.min():+.2f},{j.max():+.2f}], iota0-N in [{io_.min():+.3f},{io_.max():+.3f}]")
    print(f"   => branches violating sign j = sign k: {viol};  min f_A = {min(r['fA'] for r in rods):.4f}, min f_B = {min(r['fB'] for r in rods):.4f}, p2 range [{min(r['p2'] for r in rods):.4f}, {max(r['p2'] for r in rods):.4f}]")
    return table

# ---------------- (1,3) family: locate by scanning lamODE with several beta guesses ----------------
rods13 = []
found = {}
for lam in np.arange(-1.6, -0.15, 0.025):
    for bg in (0.15, 0.35, -0.15, -0.35, 0.6):
        r = find_rod(lam, [-lam * 0.9, bg], 3)
        if r is not None and abs(r['beta']) > 2e-2 and r['closure'] < 1e-6:
            key = round(r['beta'], 3)
            if not any(abs(x['lam'] - lam) < 1e-9 and abs(abs(x['beta']) - abs(r['beta'])) < 1e-6 for x in rods13):
                rods13.append(r)
rods13.sort(key=lambda r: (r['lam'], r['beta']))
if rods13:
    # continuation to fill the family densely
    dense = []
    lams = sorted(set(round(r['lam'], 6) for r in rods13))
    r0 = min(rods13, key=lambda r: abs(r['beta'] - 0.3)) if rods13 else None
    for direction in (+1, -1):
        guess = [r0['alpha'], r0['beta']]; lam = r0['lam']
        for _ in range(80):
            lam += direction * 0.01
            r = find_rod(lam, guess, 3)
            if r is None or abs(r['beta']) < 2e-2 or r['closure'] > 1e-6: break
            guess = [r['alpha'], r['beta']]; dense.append(r)
    dense.append(r0); dense.sort(key=lambda r: r['lam'])
    tab13 = report(dense, "closed (1,3) rods, R_max = 1")
else:
    print("no (1,3) rods found in the scanned lamODE range")

# ---------------- alpha = 0 (Kida) rods: solve for (beta, lam) at fixed alpha = 0 ----------------
for nfp in (2, 3):
    sols = []
    for bg in (0.1, 0.25, 0.4, 0.6, -0.25, -0.4):
        for lg in (-0.3, -0.5, -0.8, -1.2, -2.0, 0.5, 1.0):
            p, info, ier, msg = fsolve(lambda p: shoot(0.0, p[0], p[1], nfp), [bg, lg], xtol=1e-12, full_output=True)
            if ier == 1 and np.sum(np.abs(shoot(0.0, p[0], p[1], nfp))) < 1e-7 and abs(p[0]) > 2e-2:
                r = rod_moduli(0.0, p[0], p[1], nfp)
                if r is not None and r['closure'] < 1e-6 and not any(abs(x['beta'] - r['beta']) < 1e-5 and abs(x['lam'] - r['lam']) < 1e-5 for x in sols):
                    sols.append(r)
    if sols:
        report(sols, f"alpha = 0 (Kida) closed (1,{nfp}) rods, R_max = 1")
    else:
        print(f"\n#### alpha = 0 (Kida) (1,{nfp}) rods: none found from the guess grid ####")

# ---------------- for reference: the (1,2) family end points and QA rod ----------------
r = find_rod(-0.4273, [0.4632, -0.3408], 2)
print("\nQA-matched (1,2) rod check: ", {k: round(float(v), 5) for k, v in r.items() if k in ('alpha', 'beta', 'L', 'l2', 'l3', 'c', 'C', 'p2', 'fA', 'fB', 'closure')})
