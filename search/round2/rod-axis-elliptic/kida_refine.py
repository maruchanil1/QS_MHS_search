"""TASK 4c (follow-up): high-precision refinement of the alpha = 0 (Kida) closed (1,2) rod at lamODE ~ -0.5242 (R_max = 1) and
of the on-axis current j on its deg-0 branch (e3 = 1/2, e0 = -1/2): K = -q3/4, k a = -q3 (q1+q2)/2, k b = -3 q1 q2 q3/4,
   eta^2 = [ -C k + sqrt(C^2 K + K k b) ] / K,   j = (k a)/(k eta^2) + 2 tau0,
for which kida_rods.py found j = -0.0065 (k < 0), a 0.5 % cancellation.  Question: is j = 0 exactly (a VACUUM exactly solvable
first-order QS point on a closed rod)?  We (i) re-solve the closure conditions with tight tolerances, (ii) evaluate the moduli in
several equivalent ways (from the start point; from the Frenet data along the rod: l2, l3, c, C as means with stds), (iii) print
j with an error estimate from the spread, (iv) print the exact j = 0 condition  F := -q3 (q1+q2)/2 - 2 tau0 (C + sqrt(C^2 + k b)) = 0
(k < 0 branch) and its value.
Run: python3 kida_refine.py   (~1 min)
"""
import numpy as np, sys
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
sys.path.insert(0, '/home/user/QS_MHS_search/search/round2/rod-axis-elliptic')
from rods import rod_rhs, kappa_of

def shoot(bet, lam, nfp, rtol, want=False):
    y0 = np.array([1.0, 0, 0, 0, np.cos(bet), np.sin(bet)])
    ev = lambda t, y, al, lam: np.arctan2(y[1], y[0]) - np.pi / nfp
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(rod_rhs, (0, 60), y0, args=(0.0, lam), events=ev, rtol=rtol, atol=rtol * 1e-2, dense_output=True)
    if len(sol.t_events[0]) == 0:
        return (None, None) if want else [1e3, 1e3]
    t1 = sol.t_events[0][0]; y1 = sol.sol(t1)
    # kappa' at t1 analytically: kappa^2 = |T x I|^2/lam^2, d/ds via the ODE (finite difference with tiny h on the dense output)
    h = 1e-6
    dk = (kappa_of(sol.sol(t1 + h), 0.0, lam) - kappa_of(sol.sol(t1 - h), 0.0, lam)) / (2 * h)
    if want: return sol, t1
    return [dk, y1[2]]

def analyse(bet, lam, nfp, rtol, npts):
    sol, tq = shoot(bet, lam, nfp, rtol, True)
    L = 2 * nfp * tq
    full = solve_ivp(rod_rhs, (0, L), np.array([1.0, 0, 0, 0, np.cos(bet), np.sin(bet)]), args=(0.0, lam), rtol=rtol, atol=rtol * 1e-2, dense_output=True)
    ss = np.linspace(0, L, npts, endpoint=False); Y = full.sol(ss)
    x = Y[:3].T; T = Y[3:].T
    I = np.stack([-x[:, 1], x[:, 0], np.zeros(len(ss))], 1)
    acc = np.cross(T, I) / lam; kap = np.linalg.norm(acc, axis=1); N = acc / kap[:, None]; Bn = np.cross(T, N)
    l2s = np.einsum('ij,ij->i', T, I); l3s = np.einsum('ij,ij->i', Bn, I) / kap
    cs = T[:, 2] + l3s * kap**2 / 2
    l2, l3, c = l2s.mean(), l3s.mean(), cs.mean()
    C = l2 * c / l3**2; tau0 = l2 / (2 * l3)
    p2 = 4 * c / l3 - l2**2 / l3**2; p1 = 4 * (1 - c**2) / l3**2 + 4 * l2 * C / l3; p0 = -4 * C**2
    rts = np.sort(np.roots([-1, p2, p1, p0]).real); q3, q1, q2 = rts
    closure = np.linalg.norm(full.sol(L) - full.sol(0.0))
    # branch e3 = 1/2, e0 = -1/2:  K = -q3/4, ka = -q3 (q1+q2)/2, kb = -3 q1 q2 q3 /4
    K = -q3 / 4; ka = -q3 * (q1 + q2) / 2; kb = -3 * q1 * q2 * q3 / 4
    out = {}
    for sk in (+1, -1):
        k = sk * np.sqrt(K)
        e2 = np.roots([K, 2 * C * k, -kb]); e2 = [v.real for v in e2 if abs(v.imag) < 1e-12 and v.real > 0]
        out[sk] = [(ev, ka / (k * ev) + 2 * tau0) for ev in e2]
    F = -q3 * (q1 + q2) / 2 - 2 * tau0 * (C + np.sqrt(C**2 + kb))
    return dict(L=L, l2=l2, l3=l3, c=c, C=C, tau0=tau0, p2=p2, p1=p1, p0=p0, q=(q1, q2, q3), closure=closure, std=(l2s.std(), l3s.std(), cs.std()),
                kmin2=kap.min()**2, kmax2=kap.max()**2, j=out, F=F, K=K)

print("Kida (alpha = 0) closed (1,2) rod near beta = 0.81614, lamODE = -0.52420 (R_max = 1)")
for rtol in (1e-10, 1e-12, 1e-13):
    p, info, ier, msg = fsolve(lambda p: shoot(p[0], p[1], 2, rtol), [0.81614, -0.52420], xtol=1e-14, full_output=True)
    res = shoot(p[0], p[1], 2, rtol)
    r = analyse(p[0], p[1], 2, rtol, 20000)
    print(f"\n rtol = {rtol:.0e}: beta = {p[0]:.12f}, lamODE = {p[1]:.12f}, closure residuals {res}, |x(L)-x(0)| = {r['closure']:.1e}, L = {r['L']:.10f}")
    print(f"   l2 = {r['l2']:.10f} (std {r['std'][0]:.1e}), l3 = {r['l3']:.10f} (std {r['std'][1]:.1e}), c = {r['c']:.10f} (std {r['std'][2]:.1e}), C = {r['C']:.10f}, tau0 = {r['tau0']:.10f}")
    print(f"   p2 = {r['p2']:.10f}, p1 = {r['p1']:.10f}, p0 = {r['p0']:.10f}; roots q1 = {r['q'][0]:.10f} (kmin^2 = {r['kmin2']:.10f}), q2 = {r['q'][1]:.10f} (kmax^2 = {r['kmax2']:.10f}), q3 = {r['q'][2]:.10f}")
    print(f"   branch K = -q3/4 = {r['K']:.10f}:  k > 0: (eta^2, j) = {[(round(e,8), round(j,8)) for e, j in r['j'][+1]]};  k < 0: (eta^2, j) = {[(round(e,8), round(j,8)) for e, j in r['j'][-1]]}")
    print(f"   exact j = 0 condition (k < 0 branch): F = -q3(q1+q2)/2 - 2 tau0 (C + sqrt(C^2 + kb)) = {r['F']:+.10f}   (j = F/(C + sqrt(C^2+kb)) up to sign)")
