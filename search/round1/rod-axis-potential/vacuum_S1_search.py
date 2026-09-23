"""Search the closed (1,2) Kirchhoff-rod family for rods on which the exact elliptic first-order QS branch (S=1)
     k = sqrt(-p2)/2,  sigma = -kappa'/(k kappa),  eta^2 in {C/k, -3C/k},  j = p1/(2 k eta^2) + 2 tau0
is CURRENT-FREE on axis (j = 0), i.e. (rod_exact.py notation, tau0 = l2/(2 l3), C = (l2 c - alpha)/l3^2):
     branch A (eta^2 = C/k):    p1 = -4 C tau0   <=>  f_A := 2(1-c^2) + 3 l2 l3 C = 0
     branch B (eta^2 = -3C/k):  p1 = 12 C tau0   <=>  f_B := 2(1-c^2) -   l2 l3 C = 0
together with p2 = q1+q2+q3 < 0 (k real) and eta^2 > 0.  Closed (1,2) rods: shooting as in rod_exact.py with R_max = 1
(scale) and the ODE parameter lamODE = -lambda3 varied.  Prints f_A, f_B, p2 along the family and roots.
Run: python3 vacuum_S1_search.py
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq

Rmax = 1.0

def rod_rhs(t, y, al, lam):
    x, T = y[:3], y[3:]
    I = np.array([-x[1], x[0], al])
    return np.concatenate([T, np.cross(T, I) / lam])

def kappa_of(y, al, lam):
    x, T = y[:3], y[3:]
    return np.linalg.norm(np.cross(T, np.array([-x[1], x[0], al])) / lam)

def shoot(p, lam, want=False):
    al, bet = p
    y0 = np.array([Rmax, 0, 0, 0, np.cos(bet), np.sin(bet)])
    ev = lambda t, y, al, lam: np.arctan2(y[1], y[0]) - np.pi / 2
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(rod_rhs, (0, 30), y0, args=(al, lam), events=ev, rtol=1e-11, atol=1e-13, dense_output=True)
    if len(sol.t_events[0]) == 0:
        return [1e3, 1e3] if not want else (None, None)
    t1 = sol.t_events[0][0]; y1 = sol.sol(t1); h = 1e-5
    dk = (kappa_of(sol.sol(t1 + h), al, lam) - kappa_of(sol.sol(t1 - h), al, lam)) / (2 * h)
    if want: return sol, t1
    return [dk, y1[2]]

def rod_data(lam, guess):
    p = fsolve(shoot, guess, args=(lam,), xtol=1e-12)
    res = shoot(p, lam)
    if abs(res[0]) + abs(res[1]) > 1e-6:
        return None
    al, bet = p
    sol, tq = shoot(p, lam, True)
    L = 4 * tq
    y0 = sol.sol(0.0); T0 = y0[3:]; x0 = y0[:3]
    I0 = np.array([-x0[1], x0[0], al])
    acc0 = np.cross(T0, I0) / lam; k0 = np.linalg.norm(acc0); N0 = acc0 / k0; B0 = np.cross(T0, N0)
    l2 = T0 @ I0; l3 = (B0 @ I0) / k0
    zs0 = T0[2]; c = zs0 + l3 * k0**2 / 2
    C = (l2 * c - al) / l3**2
    p2 = 4 * c / l3 - l2**2 / l3**2; p1 = 4 * (1 - c**2) / l3**2 + 4 * l2 * C / l3; p0 = -4 * C**2
    fA = 2 * (1 - c**2) + 3 * l2 * l3 * C; fB = 2 * (1 - c**2) - l2 * l3 * C
    return dict(lam=lam, alpha=al, beta=bet, L=L, l2=l2, l3=l3, c=c, C=C, p2=p2, p1=p1, p0=p0, fA=fA, fB=fB, kmax=k0, p=p)

if __name__ == "__main__":
    print(" lamODE   alpha    beta      L      l2      l3       c        C       p2       fA       fB   iota0(S1)=sqrt(-p2)L/(4pi)")
    rows = []
    guess = [0.561636 / 1.2126, -0.340808]
    for lam in np.concatenate([np.linspace(-0.4273, -0.30, 14), np.linspace(-0.44, -0.75, 20)]):
        d = rod_data(lam, guess)
        if d is None:
            print(f"{lam:8.4f}  no closed (1,2) rod found from guess"); continue
        guess = d['p']
        iota = np.sqrt(-d['p2']) * d['L'] / (4 * np.pi) if d['p2'] < 0 else float('nan')
        rows.append(d)
        print(f"{lam:8.4f} {d['alpha']:8.4f} {d['beta']:8.4f} {d['L']:7.3f} {d['l2']:7.4f} {d['l3']:7.4f} {d['c']:8.4f} {d['C']:8.4f} {d['p2']:8.4f} {d['fA']:8.4f} {d['fB']:8.4f}   {iota:.4f}")
    rows.sort(key=lambda d: d['lam'])
    lams = np.array([d['lam'] for d in rows]); fA = np.array([d['fA'] for d in rows]); fB = np.array([d['fB'] for d in rows]); p2 = np.array([d['p2'] for d in rows])
    def refine(fname):
        f = fA if fname == 'A' else fB
        for i in range(len(lams) - 1):
            if f[i] * f[i + 1] < 0:
                g = rows[i]['p']
                def F(lam):
                    d = rod_data(lam, g)
                    return d['f' + fname] if d else np.nan
                lam0 = brentq(F, lams[i], lams[i + 1], xtol=1e-10)
                d = rod_data(lam0, g)
                k = np.sqrt(-d['p2']) / 2 if d['p2'] < 0 else float('nan')
                eta2 = d['C'] / k if fname == 'A' else -3 * d['C'] / k
                print(f"\nBranch {fname} root: lamODE = {lam0:.6f}, alpha={d['alpha']:.6f}, beta={d['beta']:.6f}, L={d['L']:.5f}, l2={d['l2']:.6f}, l3={d['l3']:.6f}, c={d['c']:.6f}, C={d['C']:.6f}")
                print(f"   p2 = {d['p2']:.6f} (need <0), k = {k:.6f}, iota0 - N = kL/2pi = {k*d['L']/(2*np.pi):.5f}, eta^2 = {eta2:.5f} (need >0; sign of k chosen = sign(C) for A, -sign(C) for B)")
                print(f"   kappa_max = {d['kmax']:.4f}; check j = p1/(2 k eta^2) + 2 tau0 = {d['p1']/(2*k*eta2) + d['l2']/d['l3']:.2e}")
    refine('A'); refine('B')
