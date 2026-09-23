"""CHECK ONLY (first-order data at definition level, from my own Floquet/clock derivation, not NAE machinery):
on the exact (1,2) rod, find the stellarator-symmetric periodic sigma for vacuum (j = 0) and the QA value
iota0 = 0.4232 (k = 2 pi iota0 / L), by shooting on eta.  Then:
  (a) evaluate the exact on-axis Hessian P_NN, P_NB, P_BB and compare with the handoff §6.4 VMEC fits
      P_NN = 3.19 kappa^2 - 0.86 tau + 0.93,  P_BB = -1.56 z_s + 0.86 tau - 0.11,  P_NB = 0.78 kappa'/kappa + 0.29 kappa'
      (same functional forms, refit), and Lap Pi = 1.71 kappa^2 + 1.36;
  (b) test whether sigma q' is a low-degree Laurent polynomial in q = kappa^2 (the 'algebraic first-order conjecture',
      handoff §9.1) -> decides whether the §6.4 fits reflect an exact elliptic structure.
Run: python3 numeric_sigma_QArod.py
"""
import numpy as np, pickle, sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

rod = np.load('rod12.npy', allow_pickle=True).item()
s = rod['s']; kap = rod['kappa']; tau = rod['tau']; L = rod['L']; lam2 = rod['lambda2']; lam3 = rod['lambda3']; zs = rod['zs']
C = rod['C']; q3, q1, q2 = rod['roots']; tau0 = lam2 / (2 * lam3)
n = len(s); ds = s[1] - s[0]; kk = 2 * np.pi * np.fft.rfftfreq(n, d=ds)
def deriv(f, m=1): return np.fft.irfft((1j * kk)**m * np.fft.rfft(f), n=n)
def interp(f):
    fh = np.fft.rfft(f) / n
    def g(x): return np.real(fh[0]) + 2 * np.real(np.sum(fh[1:] * np.exp(1j * kk[1:] * x)))
    return g
fk, ft = interp(kap), interp(tau)
kap1 = deriv(kap); kap2 = deriv(kap, 2); tau1 = deriv(tau); q = kap**2

import sys
iota0 = 0.4232 * (float(sys.argv[1]) if len(sys.argv) > 1 else 1.0); k = 2 * np.pi * iota0 / L; j = 0.0
print(f"target iota0 = {iota0} (NOTE: this rod is the MIRROR IMAGE of the VMEC axis: tau, z_s have opposite signs; so mirror the fits: tau->-tau, z_s->-z_s, P_NB->-P_NB), k = {k:.6f}; vacuum j = 0; rod quarter length L/4 = {L/4:.6f} (kappa max at 0, min at L/4)")

def riccati_rhs(x, y, eta):
    sg = y[0]; kp = fk(x); tt = ft(x)
    return [-k * (1 + sg**2) - k * eta**4 / kp**4 + eta**2 * (j - 2 * tt) / kp**2]

def shoot(eta):
    # sigma(0) = 0 -> sigma(L/4) must be 0 (odd about both symmetry points); integrate the Riccati (may blow up -> use
    # the projective form via the linear Hill system to be safe): phi'' + [k^2 + k b/q^2 - k a/q] phi = 0, sigma = phi'/(k phi)
    a = eta**2 * (j - 2 * tau0); b = k * eta**4 + 2 * eta**2 * C
    def hill(x, y):
        qq = fk(x)**2
        return [y[1], -(k**2 + k * b / qq**2 - k * a / qq) * y[0]]
    sol = solve_ivp(hill, (0, L / 4), [1.0, 0.0], rtol=1e-11, atol=1e-13)   # sigma(0)=0 <=> phi'(0)=0
    return sol.y[1, -1] / (k * sol.y[0, -1])    # = sigma(L/4)

etas = np.linspace(0.3, 2.5, 89)
vals = np.array([shoot(e) for e in etas])
roots = []
for i in range(len(etas) - 1):
    if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0 and abs(vals[i]) + abs(vals[i + 1]) < 50:
        try:
            roots.append(brentq(shoot, etas[i], etas[i + 1], xtol=1e-12))
        except Exception:
            pass
print("eta values giving a stellarator-symmetric periodic sigma with iota0 = 0.4232:", np.array2string(np.array(roots), precision=6))

cf = pickle.load(open('hessian_closed_forms.pkl', 'rb'))
E = {name: sp.sympify(cf[name]) for name in ('P_NN', 'P_NB', 'P_BB', 'riccati')}
syms = {str(x): x for e in E.values() for x in e.free_symbols}
args = [syms[nm] for nm in ['kappa', 'kappa1', 'kappa2', 'tau', 'tau1', 'sigma', 'eta', 'k', 'j']]
F = {name: sp.lambdify(args, E[name], 'numpy') for name in E}

for eta in roots:
    a = eta**2 * (j - 2 * tau0); b = k * eta**4 + 2 * eta**2 * C
    def hill(x, y):
        qq = fk(x)**2
        return [y[1], -(k**2 + k * b / qq**2 - k * a / qq) * y[0]]
    sol = solve_ivp(hill, (0, L), [1.0, 0.0], rtol=1e-11, atol=1e-13, dense_output=True)
    ph = sol.sol(s)
    sigma = ph[1] / (k * ph[0])
    print(f"\n=== eta = {eta:.6f}: sigma range [{sigma.min():.3f}, {sigma.max():.3f}], sigma(L/2) = {sigma[n//2]:.2e} (periodicity), Hill multiplier phi(L)/phi(0) = {sol.y[0,-1]:.4f} ===")
    if np.max(np.abs(sigma)) > 50:
        print("   (sigma has a pole: discard)"); continue
    PNN = F['P_NN'](kap, kap1, kap2, tau, tau1, sigma, eta, k, j)
    PNB = F['P_NB'](kap, kap1, kap2, tau, tau1, sigma, eta, k, j)
    PBB = F['P_BB'](kap, kap1, kap2, tau, tau1, sigma, eta, k, j)
    lap = -kap**2 + PNN + PBB
    def fit(name, val, basis, labels):
        A = np.stack(basis, 1); coef, *_ = np.linalg.lstsq(A, val, rcond=None)
        res = np.sqrt(np.mean((A @ coef - val)**2)) / (val.max() - val.min())
        print(f"   {name} = " + " + ".join(f"{c:+.3f} {l}" for c, l in zip(coef, labels)) + f"   (rel. rms residual {res:.1e})")
    print("   (a) fits in the functional forms of handoff §6.4 [VMEC QA: P_NN = 3.19 k^2 - 0.86 tau + 0.93; P_BB = -1.56 z_s + 0.86 tau - 0.11; P_NB = 0.78 k'/k + 0.29 k'; Lap = 1.71 k^2 + 1.36]:")
    fit("P_NN", PNN, [kap**2, tau, np.ones(n)], ["kappa^2", "tau", "1"])
    fit("P_BB", PBB, [zs, tau, np.ones(n)], ["z_s", "tau", "1"])
    fit("P_NB", PNB, [kap1 / kap, kap1], ["kappa'/kappa", "kappa'"])
    fit("LapPi", lap, [kap**2, np.ones(n)], ["kappa^2", "1"])
    print(f"   ranges: P_NN [{PNN.min():.3f},{PNN.max():.3f}] P_BB [{PBB.min():.3f},{PBB.max():.3f}] P_NB [{PNB.min():.3f},{PNB.max():.3f}]  (VMEC table: P_NN 0.30..8.26, P_BB -1.03..1.69, P_NB -1.05..0)")
    # helical comparison
    PBB_hel = (zs - lam2 * tau) / lam3; PNB_hel = -(lam2 / lam3) * kap1 / kap
    print(f"   helical predictions: P_BB^hel range [{PBB_hel.min():.3f},{PBB_hel.max():.3f}], P_NB^hel range [{PNB_hel.min():.3f},{PNB_hel.max():.3f}]")
    # (b) rationality test: sigma q' = R(q) with R a Laurent polynomial in q
    qp = 2 * kap * kap1
    for deg in (2, 3, 4):
        A = np.stack([q**i for i in range(-deg, deg + 1)], 1)
        coef, *_ = np.linalg.lstsq(A, sigma * qp, rcond=None)
        res = np.max(np.abs(A @ coef - sigma * qp)) / np.max(np.abs(sigma * qp))
        print(f"   (b) sigma*q' ~ Laurent poly in q, i=-{deg}..{deg}: rel. max residual {res:.1e}")
    # compare: the exact elliptic S=1 solution has sigma*q' = -q'^2/(k q) = -P(q)/(k q): exactly Laurent of degree (-1..2)
    # also print sigma at a few points for the record
    print("   sigma samples at s/L = 0, .05, .1, .15, .2, .25:", np.array2string(sigma[[0, n//20, n//10, 3*n//20, n//5, n//4]], precision=4))
