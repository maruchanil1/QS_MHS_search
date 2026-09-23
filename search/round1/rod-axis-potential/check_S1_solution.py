"""Direct numerical verification of the exact elliptic first-order QS solution on the (1,2) rod.

Claim (from elliptic_sigma.py, case S=1, e3=0, e0=-1/2):  for a Kirchhoff rod with p2 := q1+q2+q3 < 0,
     k = +-sqrt(-p2)/2,   sigma = -kappa'/(k kappa),   eta^2 in {C/k, -3C/k} (the positive one),
     j = p1/(2 k eta^2) + 2 tau0
solves the first-order QS Riccati exactly.  Equivalently  X = (eta/kappa) e^{iks},  Y = -(kappa e^{iks})'/(k eta).
Checks: (1) Riccati residual along the rod (FFT derivatives);  (2) with the closed-form on-axis Hessian (P_NN, P_NB, P_BB)
from axis_linearization.py, integrate the transverse linear system (L2) from the claimed initial data and compare with the
claimed (X, Y) over one full length L;  (3) 4x4 transverse monodromy: eigenvalues e^{+-ikL} and the second pair;
(4) rationality of P_NN, P_NB, P_BB in q = kappa^2 (least-squares fit to a Laurent polynomial in q).

Run: python3 check_S1_solution.py   (needs rod12.npy from rod_exact.py and hessian_closed_forms.pkl)
"""
import numpy as np, pickle, sympy as sp
from scipy.integrate import solve_ivp

rod = np.load('rod12.npy', allow_pickle=True).item()
s = rod['s']; kap = rod['kappa']; tau = rod['tau']; L = rod['L']; lam2 = rod['lambda2']; lam3 = rod['lambda3']
C = rod['C']; q3, q1, q2 = rod['roots']; tau0 = lam2 / (2 * lam3)
p2 = q1 + q2 + q3; p1 = -(q1 * q2 + q1 * q3 + q2 * q3); p0 = q1 * q2 * q3
print(f"rod: q1={q1:.6f} q2={q2:.6f} q3={q3:.6f}  p2={p2:.6f} p1={p1:.6f} p0={p0:.6f}  C={C:.6f} tau0={tau0:.6f} L={L:.5f}")
assert p2 < 0, "S=1 solution needs p2 < 0"
n = len(s); ds = s[1] - s[0]
kk = 2 * np.pi * np.fft.rfftfreq(n, d=ds)
def deriv(f, m=1):
    return np.fft.irfft((1j * kk)**m * np.fft.rfft(f), n=n)
kap1 = deriv(kap); kap2 = deriv(kap, 2); tau1 = deriv(tau)
q = kap**2

# closed forms
cf = pickle.load(open('hessian_closed_forms.pkl', 'rb'))
E = {name: sp.sympify(cf[name]) for name in ('P_NN', 'P_NB', 'P_BB', 'riccati')}
syms = {str(x): x for e in E.values() for x in e.free_symbols}
order = ['kappa', 'kappa1', 'kappa2', 'tau', 'tau1', 'sigma', 'eta', 'k', 'j']
args = [syms[nm] for nm in order]
F = {name: sp.lambdify(args, E[name], 'numpy') for name in E}

k0 = np.sqrt(-p2) / 2
for ksign in (+1, -1):
    k = ksign * k0
    for eta2 in (C / k, -3 * C / k):
        if eta2 <= 0:
            continue
        eta = np.sqrt(eta2)
        j = p1 / (2 * k * eta2) + 2 * tau0
        sigma = -kap1 / (k * kap)
        sigma1 = deriv(sigma)
        print(f"\n=== k = {k:+.6f} (iota0 - N = kL/2pi = {k*L/(2*np.pi):+.5f}), eta^2 = {eta2:.5f} (eta = {eta:.4f}), j = {j:.5f} ===")
        ric = F['riccati'](kap, kap1, kap2, tau, tau1, sigma, eta, k, j)
        print(f"  (1) Riccati residual max |sigma' - RHS| = {np.max(np.abs(sigma1 - ric)):.2e}   (scale |sigma'| max {np.max(np.abs(sigma1)):.3f})")
        PNN = F['P_NN'](kap, kap1, kap2, tau, tau1, sigma, eta, k, j)
        PNB = F['P_NB'](kap, kap1, kap2, tau, tau1, sigma, eta, k, j)
        PBB = F['P_BB'](kap, kap1, kap2, tau, tau1, sigma, eta, k, j)
        # interpolants (periodic, via FFT series) for the ODE integration
        def interp(f):
            fh = np.fft.rfft(f) / n
            def g(x):
                ph = np.exp(1j * kk * x)
                return np.real(fh[0]) + 2 * np.real(np.sum(fh[1:] * ph[1:]))
            return g
        fk, ft, ftp = interp(kap), interp(tau), interp(tau1)
        fNN, fNB, fBB = interp(PNN), interp(PNB), interp(PBB)
        def rhs(x, u):
            X_, Xp, Y_, Yp = u
            kk_, tt, tp = fk(x), ft(x), ftp(x)
            pnn, pnb, pbb = fNN(x), fNB(x), fBB(x)
            Xpp = pnn * X_ + pnb * Y_ - (3 * kk_**2 - tt**2) * X_ + 2 * tt * Yp + tp * Y_
            Ypp = pnb * X_ + pbb * Y_ + tt**2 * Y_ - 2 * tt * Xp - tp * X_
            return [Xp, Xpp, Yp, Ypp]
        # claimed complex solution: X = eta e^{iks} g, g = 1/kappa ;  Y = -(1/(k eta)) e^{iks} h, h = kappa' + i k kappa
        # (e^{iks} is NOT L-periodic, so differentiate the periodic factors g, h by FFT and the phase analytically)
        ph = np.exp(1j * k * s)
        g = 1 / kap; g1 = deriv(g); g2 = deriv(g, 2)
        h = kap1 + 1j * k * kap; h1 = kap2 + 1j * k * kap1; h2 = deriv(kap, 3) + 1j * k * kap2
        Xc = eta * ph * g; Xcp = eta * ph * (g1 + 1j * k * g); Xcpp = eta * ph * (g2 + 2j * k * g1 - k**2 * g)
        Yc = -ph * h / (k * eta); Ycp = -ph * (h1 + 1j * k * h) / (k * eta); Ycpp = -ph * (h2 + 2j * k * h1 - k**2 * h) / (k * eta)
        resN = Xcpp + (3 * kap**2 - tau**2) * Xc - 2 * tau * Ycp - tau1 * Yc - PNN * Xc - PNB * Yc
        resB = Ycpp - tau**2 * Yc + 2 * tau * Xcp + tau1 * Xc - PNB * Xc - PBB * Yc
        scaleN = np.max(np.abs(PNN * Xc)); scaleB = np.max(np.abs(PBB * Yc))
        print(f"  (2) pointwise residual of (L2) for the claimed (X,Y): N-eq {np.max(np.abs(resN)):.2e} (scale {scaleN:.1f}), B-eq {np.max(np.abs(resB)):.2e} (scale {scaleB:.1f})")
        print(f"      kappa*X = eta e^(iks) exactly harmonic by construction; |X| range [{np.min(np.abs(Xc)):.3f},{np.max(np.abs(Xc)):.3f}]; flux Im(conj(X)Y) = {np.mean(np.imag(np.conj(Xc)*Yc)):.6f} +- {np.std(np.imag(np.conj(Xc)*Yc)):.1e}")
        # (3) monodromy of the 4x4 transverse system
        M = np.zeros((4, 4))
        for i in range(4):
            u0 = np.zeros(4); u0[i] = 1
            M[:, i] = solve_ivp(rhs, (0, L), u0, rtol=1e-11, atol=1e-13).y[:, -1]
        ev = np.linalg.eigvals(M)
        ang = np.angle(ev) / (2 * np.pi)
        print(f"  (3) transverse monodromy over L: |ev| = {np.array2string(np.abs(ev), precision=5)}, rotation numbers = {np.array2string(ang, precision=5)}; expected +-{k*L/(2*np.pi):.5f} for the field-line pair; det M = {np.linalg.det(M):.6f}")
        # (4) rationality in q: fit each P to sum_{i=-2..2} r_i q^i
        A = np.stack([q**i for i in range(-2, 3)], 1)
        for name, val in (('P_NN', PNN), ('P_NB / (kappa kappa1)', PNB / (kap * kap1 + 1e-300)), ('P_BB', PBB)):
            if 'NB' in name:
                # P_NB is odd: divide by q' = 2 kappa kappa'  ->  even, rational in q
                val = PNB / (2 * kap * kap1)
                mask = np.abs(kap1) > 0.05 * np.max(np.abs(kap1))
                coef, *_ = np.linalg.lstsq(A[mask], val[mask], rcond=None)
                res = np.max(np.abs(A[mask] @ coef - val[mask])) / np.max(np.abs(val[mask]))
                print(f"  (4) P_NB/q' as Laurent poly in q (i=-2..2): coeffs {np.array2string(coef, precision=4)}, rel. residual {res:.1e}")
            else:
                coef, *_ = np.linalg.lstsq(A, val, rcond=None)
                res = np.max(np.abs(A @ coef - val)) / np.max(np.abs(val))
                print(f"  (4) {name} as Laurent poly in q (i=-2..2): coeffs {np.array2string(coef, precision=4)}, rel. residual {res:.1e}")
        print(f"      P_NN range [{PNN.min():.3f},{PNN.max():.3f}], P_BB range [{PBB.min():.3f},{PBB.max():.3f}], P_NB range [{PNB.min():.3f},{PNB.max():.3f}]")
