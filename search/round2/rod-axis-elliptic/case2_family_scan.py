"""The one-parameter Kovacic case-2 family (n = 4, e = (1,1,1,3), d = 1; k b = -3 p0/4 as on the S=1 branch): for every c0 the
step-3 equation is satisfied with K(c0), A(c0) rational (case2_verify.py: Riccati residual 1e-14), so Hill's equation has the
Liouvillian solutions y = exp int (phi_K +- sqrt R)/2, phi_K = theta + 1/(q + c0).  The family ends at the S=1 point as c0 -> inf.
Along the rod sqrt R flips sign at the turning points q1, q2 where R has odd zeros, so
   sigma(s) = [ q' (phi_K - P'/(2P)) +- |q'| sqrt(R) ] / (2k)  (real iff R >= 0 on (q1,q2)),
whose even part +-|q'| sqrt(R)/(2k) makes the point NON-stellarator-symmetric (hyperbolic Floquet multipliers).
This script scans c0 on two rods and reports where the point is physical (K > 0, R >= 0, |tr M| >= 2, eta^2 > 0), the
current j = A/(k eta^2) + 2 tau0 for both signs of k, the closed-form-vs-Floquet sigma check with the correct sign flip,
and whether j crosses zero (a vacuum exactly solvable point would be a seed for the finite ansatz).
Run: python3 case2_family_scan.py   (~1 min).  Result: sqrt(R) = c q/((q+c0) q') so sigma is elliptic (in C(q,q')) with an even part;
the closed form agrees with the numerical Floquet sigma to ~1e-5 away from the turning points; |j| >= 1.7 on the physical part.
"""
import sympy as sp, numpy as np, sys
sys.path.insert(0, '/home/user/QS_MHS_search/search/round2/rod-axis-elliptic')
from case2_admissibility import case2_system, q, K, A
from scipy.integrate import solve_ivp

def family(Q1, Q2, Q3):
    eqs, cs, Bb, r, theta, PK, P = case2_system(Q1, Q2, Q3, 4, 1, 1, 1, 3, 1)
    c0 = cs[0]
    Mx, rhs = sp.linear_eq_to_matrix(eqs, [K, A])
    pair = None
    for i in range(len(eqs)):
        for jj in range(i + 1, len(eqs)):
            det = sp.expand(Mx[i, 0] * Mx[jj, 1] - Mx[i, 1] * Mx[jj, 0])
            if det != 0: pair = (i, jj, det); break
        if pair: break
    i, jj, det = pair
    Ksol = sp.factor(sp.together((rhs[i] * Mx[jj, 1] - rhs[jj] * Mx[i, 1]) / det))
    Asol = sp.factor(sp.together((Mx[i, 0] * rhs[jj] - Mx[jj, 0] * rhs[i]) / det))
    rr = r.subs({K: Ksol, A: Asol}); phiK = theta + 1 / (q + c0)
    R = sp.together(4 * rr - phiK**2 - 2 * sp.diff(phiK, q))
    return c0, Ksol, Asol, Bb, R, phiK, P

def floquet(Kf, Af, Bf, q1f, q2f, q3f):
    Pc = np.poly1d([-1, q1f + q2f + q3f, -(q1f * q2f + q1f * q3f + q2f * q3f), q1f * q2f * q3f]); dPc = Pc.deriv()
    def rhs(s, y):
        qq, qd, f1, f1d, f2, f2d = y
        U = Kf + Bf / qq**2 - Af / qq
        return [qd, dPc(qq) / 2, f1d, -U * f1, f2d, -U * f2]
    ev = lambda s, y: y[1]; ev.terminal = True; ev.direction = -1
    y0 = [q1f + 1e-13, 0.0, 1, 0, 0, 1]
    sol = solve_ivp(rhs, (0, 200), y0, events=ev, rtol=1e-12, atol=1e-14)
    t1 = sol.t_events[0][0]
    sol2 = solve_ivp(rhs, (0, 2 * t1), y0, rtol=1e-12, atol=1e-14, dense_output=True)
    yL = sol2.y[:, -1]; M = np.array([[yL[2], yL[4]], [yL[3], yL[5]]])
    return M, sol2, 2 * t1

def scan(label, Q1, Q2, Q3, C, tau0, L, c0grid):
    c0, Ksol, Asol, Bb, R, phiK, P = family(Q1, Q2, Q3)
    print(f"\n######## {label} ########\n   K(c0) = {Ksol}\n   A(c0) = {Asol}\n   k b = {Bb} = {float(Bb):.6f};  K -> {float(sp.limit(Ksol, c0, sp.oo)):.6f} = -p2/4 = {float(-(Q1+Q2+Q3)/4):.6f} as c0 -> inf (S=1 point)")
    Rstruct = sp.factor(sp.simplify(R * P * (q + c0)**2 / q**2))
    print(f"   structure of the radical: R P (q+c0)^2 / q^2 = {Rstruct}  (independent of q => sqrt(R) = c q/((q+c0) q'), the radical is q' itself: sigma is in C(q, q'))")
    print( "   => sigma(s) = q' (theta + 1/(q+c0) - P'/(2P))/(2k)  +  c q/(2k (q+c0)),   c^2 = the constant above (>= 0 needed for real sigma); the second (even) term breaks stellarator symmetry")
    q1f, q2f, q3f = float(Q1), float(Q2), float(Q3)
    Rf = sp.lambdify((q, c0), R, 'numpy'); phf = sp.lambdify((q, c0), phiK, 'numpy')
    Pf = sp.lambdify(q, P, 'numpy'); dPf = sp.lambdify(q, sp.diff(P, q), 'numpy')
    qs = np.linspace(q1f + 1e-7, q2f - 1e-7, 4001)
    print("     c0        K        k(+)   iota0-N     Rmin      tr M    eta^2(k>0)  j(k>0)   eta^2(k<0)  j(k<0)   sigma dev  mean(sigma)")
    rows = []
    for c0v in c0grid:
        Kf = float(Ksol.subs(c0, c0v)); Af = float(Asol.subs(c0, c0v)); Bf = float(Bb)
        if Kf <= 0:
            print(f"   {c0v:8.3f}  K = {Kf:+.4f} <= 0"); continue
        Rv = Rf(qs, c0v); Rmin = np.min(Rv)
        M, sol2, Tq = floquet(Kf, Af, Bf, q1f, q2f, q3f); tr = np.trace(M)
        kk = np.sqrt(Kf)
        e2 = {}
        for sk in (+1, -1):
            rts = np.roots([Kf, 2 * C * sk * kk, -Bf]); e2[sk] = [v.real for v in rts if abs(v.imag) < 1e-12 and v.real > 0]
        js = {sk: [Af / (sk * kk * ev) + 2 * tau0 for ev in e2[sk]] for sk in (+1, -1)}
        dev = np.nan; mean_sig = np.nan
        if abs(tr) >= 2 - 1e-9 and Rmin >= 0:
            w, V = np.linalg.eig(M); idx = 0
            v = np.real(V[:, idx]); s_grid = np.linspace(0, Tq, 1601); Y = sol2.sol(s_grid)
            phi = v[0] * Y[2] + v[1] * Y[4]; dphi = v[0] * Y[3] + v[1] * Y[5]; qq = Y[0]; qd = Y[1]
            sig_num = dphi / (kk * phi)
            sr = np.sqrt(np.maximum(Rf(qq, c0v), 0))
            base = qd * (phf(qq, c0v) - dPf(qq) / (2 * Pf(qq))) / (2 * kk)
            mask = np.abs(qd) > 0.3 * np.max(np.abs(qd))          # away from the turning points (cancellation 1/P in the closed form)
            best = min(np.max(np.abs((sig_num - (base + sgn * np.abs(qd) * sr / (2 * kk)))[mask])) for sgn in (+1, -1)) / (1 + np.max(np.abs(sig_num)))
            dev = best; mean_sig = np.mean(sig_num)
        flag = "physical (real sigma)" if (abs(tr) >= 2 - 1e-9 and Rmin >= 0) else "complex sigma"
        print(f"   {c0v:8.3f}  {Kf:8.5f}  {kk:7.4f}  {kk*L/(2*np.pi):7.4f}  {Rmin:+9.3g}  {tr:+8.4f}  {str([round(x,4) for x in e2[+1]]):11s} {str([round(x,3) for x in js[+1]]):8s} {str([round(x,4) for x in e2[-1]]):11s} {str([round(x,3) for x in js[-1]]):8s}  {dev:9.1e}  {mean_sig:+9.4f}   {flag}")
        rows.append((c0v, Kf, tr, Rmin, js))
    phys = [r_ for r_ in rows if abs(r_[2]) >= 2 - 1e-9 and r_[3] >= 0]
    if phys:
        allj = [jv for r_ in phys for sk in (+1, -1) for jv in r_[4][sk]]
        print(f"   => physical part of the family: c0 in [{min(r_[0] for r_ in phys):.3f}, inf); |j| in [{min(abs(x) for x in allj):.3f}, {max(abs(x) for x in allj):.3f}]; sign changes of j(k>0): {any(np.sign(a)!=np.sign(b) for a in [r_[4][+1][0] for r_ in phys if r_[4][+1]] for b in [r_[4][+1][0] for r_ in phys if r_[4][+1]])}, of j(k<0): {any(np.sign(a)!=np.sign(b) for a in [r_[4][-1][0] for r_ in phys if r_[4][-1]] for b in [r_[4][-1][0] for r_ in phys if r_[4][-1]])}")

grid = [1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 50.0, 200.0]
scan("QA-matched (1,2) rod", sp.Rational(222842, 10**6), sp.Rational(2212881, 10**6), sp.Rational(-2448875, 10**6), -0.549454, 0.760091, 6.65696, grid)
scan("lamODE = -0.5 (1,2) rod", sp.Rational(684788, 10**6), sp.Rational(1769765, 10**6), sp.Rational(-3388971, 10**6), -1.013304, 0.916102, 5.864041, [3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 12.0, 30.0, 200.0])
