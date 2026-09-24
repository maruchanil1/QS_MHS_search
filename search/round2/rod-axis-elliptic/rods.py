"""Shared helpers: closed Kirchhoff rods by shooting (rod_exact.py / rod_families_scan.py conventions) and their moduli.
Rod ODE: gamma'' = T x I / lamODE, I = alpha e_z + e_z x x; start at (R_max = 1, 0, 0), T(0) = (0, cos beta, sin beta).
A (1, nfp) rod closes iff the next curvature extremum lies at phi = pi/nfp with Z = 0 (two conditions on (alpha, beta)).
Moduli (rod_exact.py): l2 = T.I, l3 = Bn.I/kappa, c = z_s + l3 kappa^2/2, C = (l2 c - alpha)/l3^2, tau0 = l2/(2 l3),
  (kappa^2)'^2 = -q^3 + p2 q^2 + p1 q + p0,  p2 = 4c/l3 - l2^2/l3^2,  p1 = 4(1-c^2)/l3^2 + 4 l2 C/l3,  p0 = -4 C^2.
"""
import numpy as np, warnings
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
warnings.filterwarnings('ignore')
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
    ss = np.linspace(0, L, 400); Y = full.sol(ss); kap = np.array([kappa_of(Y[:, i], al, lam) for i in range(len(ss))])
    rts = np.sort(np.roots([-1, p2, p1, p0]).real)
    return dict(lam=lam, alpha=al, beta=bet, L=L, l2=l2, l3=l3, c=c, C=C, p2=p2, p1=p1, p0=p0, fA=fA, fB=fB, closure=closure,
                kmin=kap.min(), kmax=kap.max(), roots=rts, nfp=nfp, tau0=l2 / (2 * l3))

def find_rod(lam, guess, nfp):
    p, info, ier, msg = fsolve(lambda p: shoot(p[0], p[1], lam, nfp), guess, xtol=1e-12, full_output=True)
    if ier != 1 or np.sum(np.abs(shoot(p[0], p[1], lam, nfp))) > 1e-7: return None
    return rod_moduli(p[0], p[1], lam, nfp)

def family12(step=0.008):
    """closed (1,2) rods by continuation from the QA-matched rod (lamODE = -0.4273) in both directions; circles rejected"""
    rods = []
    for direction in (+1, -1):
        guess = [0.4632, -0.3408]; lam = -0.4273
        for _ in range(60):
            r = find_rod(lam, guess, 2)
            if r is None or abs(r['beta']) < 1e-3 or r['closure'] > 1e-6: break
            rods.append(r); guess = [r['alpha'], r['beta']]
            lam += direction * step
            if lam > -0.30 or lam < -0.59: break
    rods.sort(key=lambda r: r['lam'])
    return rods

def family13(step=0.02):
    """closed (1,3) rods by continuation from lamODE = -0.6 (rod_families_scan.out)"""
    rods = []
    for direction in (+1, -1):
        guess = [0.3785, 0.4413]; lam = -0.595
        for _ in range(60):
            r = find_rod(lam, guess, 3)
            if r is None or abs(r['beta']) < 2e-2 or r['closure'] > 1e-6: break
            rods.append(r); guess = [r['alpha'], r['beta']]
            lam += direction * step
            if lam > -0.40 or lam < -0.77: break
    rods.sort(key=lambda r: r['lam'])
    return rods
