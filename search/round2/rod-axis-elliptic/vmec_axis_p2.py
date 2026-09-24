"""TASK 1: test of the conjecture  p2 := q1 + q2 + q3 = 0  (kappa^2 a pure Weierstrass p, 4 c l3 = l2^2) on the
precise-QA VMEC magnetic axis (Landreman-Paul 2021 QA, nfp = 2).

Axis: R(phi) = sum_n R_n cos(n phi), Z(phi) = -sum_n Zc_n sin(n phi) (handoff 10.3 sign; the mirror image has the same
rod data).  Two data sets: (A) the 7 harmonics printed in handoff section 6 (n <= 6); (B) all 9 harmonics (n <= 16) read
from wout_LandremanPaul2021_QA_lowres.nc (downloaded from the simsopt test files, handoff 10.1) if present.
The Fourier series are finite, so all derivatives are exact (sympy -> lambdify).

Rod identities (rod_exact.py, Frenet convention N = T'/kappa, Bn = T x N, tau = -Bn'.N):
   I = alpha e_z + e_z x x,  l2 = T.I,  l3 = Bn.I/kappa  (constants on a rod),
   z_s = c - l3 kappa^2/2,  tau = tau0 + C/kappa^2  (C = (l2 c - alpha)/l3^2, tau0 = l2/(2 l3)),
   (kappa^2)'^2 = -q^3 + p2 q^2 + p1 q + p0,  p2 = 4c/l3 - l2^2/l3^2,  p0 = -4 C^2 = q1 q2 q3.
Three estimators of p2:
   (E1) moduli formula p2 = 4c/l3 - l2^2/l3^2 with (alpha, l2, l3, c) fitted (screw fit);
   (E2) Frenet-only: q1 = kappa_min^2, q2 = kappa_max^2, C from the fit tau = tau0 + C/q, q3 = -4C^2/(q1 q2);
   (E3) Frenet-only: least-squares fit of q'^2 + q^3 to p2 q^2 + p1 q + p0.
Run: python3 vmec_axis_p2.py
"""
import numpy as np, sympy as sp, os

def frenet_data(Rn, Zc, n=4096):
    ph = sp.symbols('phi', real=True)
    R = sum(v * sp.cos(k * ph) for k, v in Rn.items())
    Z = -sum(v * sp.sin(k * ph) for k, v in Zc.items())
    x = sp.Matrix([R * sp.cos(ph), R * sp.sin(ph), Z])
    d1 = x.diff(ph); d2 = d1.diff(ph); d3 = d2.diff(ph)
    f = sp.lambdify(ph, [x, d1, d2, d3], 'numpy')
    phis = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.zeros((n, 3)); D1 = np.zeros((n, 3)); D2 = np.zeros((n, 3)); D3 = np.zeros((n, 3))
    for i, p in enumerate(phis):
        a, b, c_, d = f(p)
        X[i] = np.ravel(a); D1[i] = np.ravel(b); D2[i] = np.ravel(c_); D3[i] = np.ravel(d)
    sp_ = np.linalg.norm(D1, axis=1)
    T = D1 / sp_[:, None]
    cr = np.cross(D1, D2)
    kap = np.linalg.norm(cr, axis=1) / sp_**3
    tau = np.einsum('ij,ij->i', cr, D3) / np.linalg.norm(cr, axis=1)**2
    Bn = cr / np.linalg.norm(cr, axis=1)[:, None]
    N = np.cross(Bn, T)
    return dict(phis=phis, X=X, T=T, N=N, Bn=Bn, kap=kap, tau=tau, sp=sp_, n=n)

def analyze(Rn, Zc, label):
    d = frenet_data(Rn, Zc)
    X, T, N, Bn, kap, tau, sp_, phis, n = d['X'], d['T'], d['N'], d['Bn'], d['kap'], d['tau'], d['sp'], d['phis'], d['n']
    L = np.sum(sp_) * (2 * np.pi / n)
    w = sp_ / np.sum(sp_)
    mean = lambda g: np.sum(w * g)
    std = lambda g: np.sqrt(mean((g - mean(g))**2))
    print(f"\n################ {label} ################")
    print(f"axis: L = {L:.5f}, kappa in [{kap.min():.4f}, {kap.max():.4f}], tau in [{tau.min():.4f}, {tau.max():.4f}]  (handoff: 6.6776, [0.488,1.468], [-0.513,1.565])")
    h = X[:, 0] * T[:, 1] - X[:, 1] * T[:, 0]
    Tz = T[:, 2]
    alpha = -(mean(Tz * h) - mean(Tz) * mean(h)) / (mean(Tz**2) - mean(Tz)**2)
    Ivec = np.stack([-X[:, 1], X[:, 0], alpha * np.ones(n)], 1)
    l2s = np.einsum('ij,ij->i', T, Ivec); l3s = np.einsum('ij,ij->i', Bn, Ivec) / kap; NIs = np.einsum('ij,ij->i', N, Ivec)
    l2, l3 = mean(l2s), mean(l3s)
    print(f"screw fit: alpha = {alpha:.5f}, l2 = T.I = {l2:.5f} (std {std(l2s):.1e}), l3 = Bn.I/kappa = {l3:.5f} (std {std(l3s):.1e}), N.I rms {np.sqrt(mean(NIs**2)):.1e}")
    print(f"   companion check kappa(l2 - l3 tau) = e_z.Bn: rms mismatch {np.sqrt(mean((kap*(l2 - l3*tau) - Bn[:,2])**2)):.1e} (scale {np.sqrt(mean(Bn[:,2]**2)):.2f})")
    q = kap**2
    cs = T[:, 2] + l3 * q / 2; c = mean(cs)
    print(f"(R2) c = z_s + l3 kappa^2/2 = {c:.5f} (std {std(cs):.1e})")
    C_mod = (l2 * c - alpha) / l3**2; tau0_mod = l2 / (2 * l3)
    print(f"(R3) moduli: tau0 = l2/(2 l3) = {tau0_mod:.5f}, C = (l2 c - alpha)/l3^2 = {C_mod:.5f};  tau - (tau0 + C/q): rms {np.sqrt(mean((tau - tau0_mod - C_mod/q)**2)):.1e} (tau scale {std(tau):.2f})")
    # (E1)
    p2_E1 = 4 * c / l3 - l2**2 / l3**2
    p1_E1 = 4 * (1 - c**2) / l3**2 + 4 * l2 * C_mod / l3; p0_E1 = -4 * C_mod**2
    dp2_dl3 = -4 * c / l3**2 + 2 * l2**2 / l3**3; dp2_dc = 4 / l3; dp2_dl2 = -2 * l2 / l3**2
    sig_E1 = np.sqrt((dp2_dl3 * std(l3s))**2 + (dp2_dc * std(cs))**2 + (dp2_dl2 * std(l2s))**2)
    p2_pw = 4 * cs / l3s - l2s**2 / l3s**2
    print(f"(E1) p2 = 4c/l3 - l2^2/l3^2 = {p2_E1:+.4f} +- {sig_E1:.4f} (linear propagation of the pointwise stds; dp2/dl3 = {dp2_dl3:.2f}, dp2/dc = {dp2_dc:.2f});  pointwise p2: mean {mean(p2_pw):+.4f}, std {std(p2_pw):.4f}")
    print(f"     4 c l3 - l2^2 = {4*c*l3 - l2**2:+.5f} (+- {abs(l3**2)*sig_E1:.5f});   p1 = {p1_E1:.4f}, p0 = {p0_E1:.4f}")
    rts = np.sort(np.roots([-1, p2_E1, p1_E1, p0_E1]).real)
    print(f"     roots of the cubic: q3 = {rts[0]:.4f}, q1 = {rts[1]:.4f} (kappa_min^2 = {q.min():.4f}), q2 = {rts[2]:.4f} (kappa_max^2 = {q.max():.4f})")
    # (E2)
    A = np.stack([np.ones(n), 1 / q], 1) * np.sqrt(w)[:, None]
    (tau0_fit, C_fit), *_ = np.linalg.lstsq(A, tau * np.sqrt(w), rcond=None)
    res_tau = tau - tau0_fit - C_fit / q
    q1, q2 = q.min(), q.max()
    p2_E2 = q1 + q2 - 4 * C_fit**2 / (q1 * q2)
    # uncertainty: refit C on the two halves of the curve (kappa above/below its mean) and take the difference
    m1 = q > mean(q); m2 = ~m1
    Cs = []
    for m in (m1, m2):
        Am = np.stack([np.ones(m.sum()), 1 / q[m]], 1) * np.sqrt(w[m])[:, None]
        Cs.append(np.linalg.lstsq(Am, tau[m] * np.sqrt(w[m]), rcond=None)[0][1])
    sig_C = abs(Cs[0] - Cs[1]) / 2
    sig_E2 = abs(8 * C_fit / (q1 * q2)) * sig_C
    print(f"(E2) tau = tau0 + C/q fit: tau0 = {tau0_fit:.5f}, C = {C_fit:.5f} (rms resid {np.sqrt(mean(res_tau**2)):.1e}; C from high-/low-kappa halves {Cs[0]:.4f}/{Cs[1]:.4f});  q1 = {q1:.5f}, q2 = {q2:.5f}")
    print(f"     p2 = q1 + q2 - 4C^2/(q1 q2) = {p2_E2:+.4f} +- {sig_E2:.4f}    (q3 = -4C^2/(q1 q2) = {-4*C_fit**2/(q1*q2):.4f})")
    # (E3)
    qh = np.fft.rfft(q); kk = np.fft.rfftfreq(n, d=phis[1] - phis[0]) * 2 * np.pi
    qp = np.fft.irfft(1j * kk * qh, n=n) / sp_
    y = qp**2 + q**3
    A3 = np.stack([q**2, q, np.ones(n)], 1) * np.sqrt(w)[:, None]
    coef3, *_ = np.linalg.lstsq(A3, y * np.sqrt(w), rcond=None)
    res3 = y - A3 @ coef3 / np.sqrt(w)
    p2_E3 = coef3[0]
    # split-sample uncertainty
    p2s = []
    for m in (m1, m2):
        Am = np.stack([q[m]**2, q[m], np.ones(m.sum())], 1) * np.sqrt(w[m])[:, None]
        p2s.append(np.linalg.lstsq(Am, y[m] * np.sqrt(w[m]), rcond=None)[0][0])
    sig_E3 = abs(p2s[0] - p2s[1]) / 2
    print(f"(E3) q'^2 + q^3 = p2 q^2 + p1 q + p0 fit: p2 = {p2_E3:+.4f} +- {sig_E3:.4f} (halves {p2s[0]:+.3f}/{p2s[1]:+.3f}), p1 = {coef3[1]:.4f}, p0 = {coef3[2]:.4f} (rms resid {np.sqrt(mean(res3**2)):.1e}, scale max q'^2 = {np.max(qp**2):.3f})")
    A4 = np.stack([q**3, q**2, q, np.ones(n)], 1) * np.sqrt(w)[:, None]
    coef4, *_ = np.linalg.lstsq(A4, qp**2 * np.sqrt(w), rcond=None)
    print(f"     free-leading-coefficient fit (rod test T1): q'^2 = {coef4[0]:.4f} q^3 {coef4[1]:+.4f} q^2 {coef4[2]:+.4f} q {coef4[3]:+.4f}   (rod: leading coefficient -1)")
    # (E4) exact curvature extrema + moduli p0 = -4C^2:  q3 = p0/(q1 q2)
    p2_E4 = q1 + q2 + p0_E1 / (q1 * q2)
    sig_E4 = abs(2 * p0_E1 / (q1 * q2) / C_mod) * abs(C_mod - C_fit)      # C from moduli vs from the tau fit
    print(f"(E4) p2 = q1 + q2 - 4 C_mod^2/(q1 q2) (exact extrema, moduli C) = {p2_E4:+.4f} +- {sig_E4:.4f}")
    ests = np.array([p2_E1, p2_E2, p2_E3, p2_E4]); sigs = np.array([sig_E1, sig_E2, sig_E3, sig_E4])
    print(f"==== {label}: p2 estimates E1 {p2_E1:+.4f} (+-{sig_E1:.3f}), E2 {p2_E2:+.4f} (+-{sig_E2:.3f}), E3 {p2_E3:+.4f} (+-{sig_E3:.3f}), E4 {p2_E4:+.4f} (+-{sig_E4:.3f}); scale q1+q2 = {q1+q2:.3f}; p2/(q1+q2): {np.array2string(ests/(q1+q2), precision=3)} ====")
    if 0 < -p2_E1:
        kS1 = np.sqrt(-p2_E1) / 2
        print(f"     S=1 branch on this axis (k^2 = -p2/4, E1): k = {kS1:.4f}, iota0 - N = kL/2pi = +-{kS1*L/(2*np.pi):.4f}  (QA: iota0 = 0.4232)")
    return dict(p2=ests, sig=sigs, l2=l2, l3=l3, c=c, alpha=alpha, C=C_mod, L=L, q1=q1, q2=q2, NI=np.sqrt(mean(NIs**2)), lam3_rel=std(l3s)/abs(l3))

# (A) handoff table (n <= 6)
RA = {0: 1.00396, 2: 0.18397, 4: 0.02169, 6: 0.00259}
ZA = {2: -0.15804, 4: -0.02057, 6: -0.00255}
resA = analyze(RA, ZA, "(A) handoff 7-harmonic axis (n <= 6)")

# (B) full harmonics from the wout file if available
if os.path.exists('wout_QA.nc'):
    from scipy.io import netcdf_file
    f = netcdf_file('wout_QA.nc', 'r', mmap=False)
    nfp = int(f.variables['nfp'].data)
    rcc = np.array(f.variables['raxis_cc'].data, float); zcs = np.array(f.variables['zaxis_cs'].data, float)
    RB = {i * nfp: float(v) for i, v in enumerate(rcc) if v != 0}
    ZB = {i * nfp: float(v) for i, v in enumerate(zcs) if v != 0}
    print("\nfull axis harmonics from wout: R_n =", {k: f"{v:.3e}" for k, v in RB.items()}, "\n                              Zc_n =", {k: f"{v:.3e}" for k, v in ZB.items()})
    resB = analyze(RB, ZB, "(B) full 9-harmonic wout axis (n <= 16)")
    # (C) truncation sensitivity: drop the last two harmonics
    RC = {k: v for k, v in RB.items() if k <= 12}; ZC = {k: v for k, v in ZB.items() if k <= 12}
    resC = analyze(RC, ZC, "(C) wout axis truncated at n <= 12 (truncation sensitivity)")
    rows = [("A n<=6", resA), ("B n<=16", resB), ("C n<=12", resC)]
    # (D) reactor-scale QA (R0 = 10.17): scale-free ratio p2/(q1+q2) tests universality
    if os.path.exists('wout_QA_reactor.nc'):
        g = netcdf_file('wout_QA_reactor.nc', 'r', mmap=False)
        nfp = int(g.variables['nfp'].data); R0 = float(g.variables['raxis_cc'].data[0])
        rcc = np.array(g.variables['raxis_cc'].data, float) / R0 * RB[0]; zcs = np.array(g.variables['zaxis_cs'].data, float) / R0 * RB[0]
        RD = {i * nfp: float(v) for i, v in enumerate(rcc) if v != 0}; ZD = {i * nfp: float(v) for i, v in enumerate(zcs) if v != 0}
        resD = analyze(RD, ZD, f"(D) reactor-scale QA axis rescaled by {RB[0]/R0:.5f} to R_00 = {RB[0]:.5f} (ns=50, independent VMEC run)")
        rows.append(("D reactor", resD))
    print("\n\n==================== SUMMARY ====================")
    for lab, r in rows:
        print(f"{lab:9s}: p2 (E1,E2,E3,E4) = {np.array2string(r['p2'], precision=4)}, sigmas {np.array2string(r['sig'], precision=3)}; p2/(q1+q2) = {np.array2string(r['p2']/(r['q1']+r['q2']), precision=3)}; N.I rms {r['NI']:.1e}, std(l3)/l3 {r['lam3_rel']:.1e}; l2={r['l2']:.4f} l3={r['l3']:.4f} c={r['c']:.4f} alpha={r['alpha']:.4f} C={r['C']:.4f}; 4cl3-l2^2 = {4*r['c']*r['l3']-r['l2']**2:+.4f}")
    r = resB
    allB = r['p2']
    print(f"\nFull lowres axis: E1 (moduli formula) p2 = {allB[0]:+.4f} +- {r['sig'][0]:.3f} (formal); all estimators in [{allB.min():+.4f}, {allB.max():+.4f}], mean {allB.mean():+.4f}, half-spread {np.ptp(allB)/2:.3f}; scale q1+q2 = {r['q1']+r['q2']:.3f}")
    print(f"   => p2/(q1+q2) = {allB.mean()/(r['q1']+r['q2']):+.3f} +- {np.ptp(allB)/2/(r['q1']+r['q2']):.3f} (systematic: estimator spread reflects the 0.1-0.3% non-rod-ness of the axis)")
    print("   Exact zero is disfavoured (all four estimators negative; E1 5 sigma formal, ~2 sigma against the systematic spread) but the lowres axis cannot exclude |p2| < 0.03.")
    print("Reference: exact (1,2) rod matched to this axis (rod_exact.py): p2 = -0.0131, l2 = 0.9551, l3 = 0.6283 (mirror), c = 0.3609, C = -0.5495 (its l2 differs by 1% from the fit -> shifts l2^2/l3^2 by 0.05).")
