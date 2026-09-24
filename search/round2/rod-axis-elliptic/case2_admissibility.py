"""TASK 4b (follow-up): physical admissibility of the Kovacic CASE-2 points found by kovacic_case2.py.
For a case-2 point (K, A, Bb; theta, P_K) the Liouvillian solutions are y = exp int omega_y, omega_y = (phi_K +- sqrt(R))/2,
phi_K = theta + P_K'/P_K, R = 4r - phi_K^2 - 2 phi_K', and sigma = q' (omega_y - P'/(4P)) / k  (phi = y P^{-1/4}).
Checks along the real rod (q in [q1, q2]):
  (i)  Hill's equation phi'' + (K + Bb/q^2 - A/q) phi = 0 integrated in s over one period of q; Floquet multipliers rho, 1/rho:
       real (|tr M| >= 2) <=> a real periodic sigma exists;  the Floquet eigenvector gives sigma_num(s) = phi'/(k phi);
  (ii) sign of R on [q1, q2] (R >= 0 <=> the two Liouvillian solutions are real);  zeros of phi (poles of sigma);
  (iii) sigma_num versus the closed form q'(omega_y - P'/(4P))/k (max relative deviation): certifies that the numerical
       Floquet solution IS the Liouvillian one;
  (iv) sigma's period (L/2 = period of kappa^2, or L) and parity about the curvature extrema (stellarator symmetry: odd).
Also samples the one-parameter family reported as 'c0 FREE' (n=4, e=(1,1,1,3), d=1).
Run: python3 case2_admissibility.py    (~1 min)
"""
import sympy as sp, numpy as np, itertools, sys
from scipy.integrate import solve_ivp
q = sp.symbols('q'); K, A = sp.symbols('K A')

def case2_system(Q1, Q2, Q3, n, e1, e2, e3, einf, d):
    P = -(q - Q1) * (q - Q2) * (q - Q3)
    p0 = Q1 * Q2 * Q3
    Bb = p0 * (4 - n**2) / sp.Integer(16)
    kQ = K + Bb / q**2 - A / q
    p1 = sp.diff(P, q) / (2 * P)
    r = p1**2 / 4 + sp.diff(p1, q) / 2 - kQ / P
    theta = sp.Rational(1, 2) * (e1 / (q - Q1) + e2 / (q - Q2) + e3 / (q - Q3) + sp.Integer(2 - n) / q)
    cs = sp.symbols(f'c0:{d}') if d > 0 else ()
    PK = q**d + sum(cs[i] * q**i for i in range(d))
    th1 = sp.diff(theta, q); th2 = sp.diff(th1, q)
    expr = (sp.diff(PK, q, 3) + 3 * theta * sp.diff(PK, q, 2) + (3 * theta**2 + 3 * th1 - 4 * r) * sp.diff(PK, q)
            + (th2 + 3 * theta * th1 + theta**3 - 4 * r * theta - 2 * sp.diff(r, q)) * PK)
    num = sp.numer(sp.together(expr))
    eqs = [sp.expand(c) for c in sp.Poly(sp.expand(num), q).all_coeffs()]
    eqs = [e for e in eqs if e != 0]
    return eqs, list(cs), Bb, r, theta, PK, P

def solve_pattern(Q1, Q2, Q3, n, e1, e2, e3, einf, d):
    eqs, cs, Bb, r, theta, PK, P = case2_system(Q1, Q2, Q3, n, e1, e2, e3, einf, d)
    out = []
    if d == 0:
        s_ = sp.solve(eqs, [K, A], dict=True)
        for s in s_:
            if K in s: out.append((s[K], s[A], {}))
    else:
        c0 = cs[0]
        Mx, rhs = sp.linear_eq_to_matrix(eqs, [K, A])
        pair = None
        for i in range(len(eqs)):
            for jj in range(i + 1, len(eqs)):
                det = sp.expand(Mx[i, 0] * Mx[jj, 1] - Mx[i, 1] * Mx[jj, 0])
                if det != 0: pair = (i, jj, det); break
            if pair: break
        i, jj, det = pair
        Ksol = sp.together((rhs[i] * Mx[jj, 1] - rhs[jj] * Mx[i, 1]) / det)
        Asol = sp.together((Mx[i, 0] * rhs[jj] - Mx[jj, 0] * rhs[i]) / det)
        rest = [sp.expand(sp.numer(sp.together(Mx[m, 0] * Ksol + Mx[m, 1] * Asol - rhs[m]))) for m in range(len(eqs)) if m not in (i, jj)]
        rest = [p for p in rest if p != 0]
        if not rest:
            for c0v in (sp.Rational(-3), sp.Rational(-1), sp.Rational(1, 2), sp.Rational(1), sp.Rational(3)):
                out.append((Ksol.subs(c0, c0v), Asol.subs(c0, c0v), {c0: c0v, 'free': True}))
        else:
            g = None
            for p in rest:
                g = sp.Poly(p, c0) if g is None else sp.gcd(g, sp.Poly(p, c0))
            if g.degree() > 0:
                for root in g.nroots(n=30):
                    if abs(sp.im(root)) > 1e-12: continue
                    c0v = sp.re(root)
                    out.append((Ksol.subs(c0, c0v), Asol.subs(c0, c0v), {c0: c0v}))
    return out, Bb, r, theta, PK, P, cs

def check_point(Q1, Q2, Q3, Kv, Av, Bbv, r, theta, PK, P, csub, C, tau0, L, tag):
    Kf, Af, Bf = float(Kv), float(Av), float(Bbv)
    if Kf <= 0:
        print(f"   {tag}: K = {Kf:+.6g} <= 0: unphysical"); return
    q1f, q2f, q3f = float(Q1), float(Q2), float(Q3)
    Pc = np.poly1d([-1, q1f + q2f + q3f, -(q1f * q2f + q1f * q3f + q2f * q3f), q1f * q2f * q3f]); dPc = Pc.deriv()
    def rhs(s, y):
        qq, qd, f1, f1d, f2, f2d = y
        U = Kf + Bf / qq**2 - Af / qq
        return [qd, dPc(qq) / 2, f1d, -U * f1, f2d, -U * f2]
    ev = lambda s, y: y[1]; ev.terminal = True; ev.direction = -1
    y0 = [q1f + 1e-13, 0.0, 1, 0, 0, 1]
    sol = solve_ivp(rhs, (0, 200), y0, events=ev, rtol=1e-12, atol=1e-14)
    t1 = sol.t_events[0][0]; Tq = 2 * t1                    # period of q (= L/2 for a (1,2) rod)
    sol2 = solve_ivp(rhs, (0, Tq), y0, rtol=1e-12, atol=1e-14, dense_output=True)
    yL = sol2.y[:, -1]; M = np.array([[yL[2], yL[4]], [yL[3], yL[5]]]); tr = np.trace(M)
    # closed-form R and omega
    rr = r.subs({K: Kv, A: Av}); phiK = theta + sp.diff(PK, q) / PK
    phiK = phiK.subs(csub) if csub else phiK
    R = sp.simplify(4 * rr - phiK**2 - 2 * sp.diff(phiK, q))
    Rf = sp.lambdify(q, R, 'numpy'); phf = sp.lambdify(q, phiK, 'numpy'); Pf = sp.lambdify(q, P, 'numpy'); dPf = sp.lambdify(q, sp.diff(P, q), 'numpy')
    qs = np.linspace(q1f + 1e-6, q2f - 1e-6, 2001); Rv = Rf(qs)
    Rmin, Rmax = np.min(Rv), np.max(Rv)
    msg = f"   {tag}: K = {Kf:.6f} (k = +-{np.sqrt(Kf):.5f}, iota0-N = +-{np.sqrt(Kf)*L/(2*np.pi):.4f}), A = {Af:+.6f}, Bb = {Bf:+.6f}, P_K = {PK.subs(csub) if csub else PK}; tr M = {tr:+.6f}; R on (q1,q2): [{Rmin:+.4g}, {Rmax:+.4g}]"
    if abs(tr) < 2 - 1e-9:
        print(msg + "  -> complex Floquet multipliers: sigma complex, UNPHYSICAL"); return
    # real Floquet eigenvectors
    w, V = np.linalg.eig(M)
    res = []
    for idx in range(2):
        v = np.real(V[:, idx]); rho = np.real(w[idx])
        # integrate phi with IC v over one period and sample sigma = phi'/(k phi) for k = +sqrt K (sign k only rescales sigma by sign)
        s_grid = np.linspace(0, Tq, 801)
        Y = sol2.sol(s_grid)
        phi = v[0] * Y[2] + v[1] * Y[4]; dphi = v[0] * Y[3] + v[1] * Y[5]
        qq = Y[0]; qd = Y[1]
        nzero = np.sum(np.sign(phi[:-1]) != np.sign(phi[1:]))
        kk = np.sqrt(Kf)
        sig_num = dphi / (kk * phi)
        # closed form: sigma = q' (omega_y - P'/(4P))/k with omega_y = (phi_K +- sqrt R)/2 ; choose the sign that matches
        with np.errstate(invalid='ignore'):
            sr = np.sqrt(np.maximum(Rf(qq), 0))
        best = None
        for sgn in (+1, -1):
            om = (phf(qq) + sgn * sr) / 2 - dPf(qq) / (4 * Pf(qq))
            sig_cl = qd * om / kk
            dev = np.max(np.abs(sig_num - sig_cl)) / (1 + np.max(np.abs(sig_num)))
            if best is None or dev < best[0]: best = (dev, sgn)
        # parity about s = 0 (q = q1, curvature minimum) and s = t1 (q = q2): sigma odd?
        s_sym = np.linspace(0.01, min(0.5, Tq / 4), 50)
        sig_p = np.interp(s_sym, s_grid, sig_num); sig_m = np.interp(Tq - s_sym, s_grid, sig_num)
        odd = np.max(np.abs(sig_p + sig_m)) / (1 + np.max(np.abs(sig_p)))
        res.append((rho, nzero, np.max(np.abs(sig_num)), best[0], best[1], odd))
    out = []
    for rho, nzero, smax, dev, sgn, odd in res:
        out.append(f"rho={rho:+.5f}: zeros of phi per period={nzero}, max|sigma|={smax:.3g}, dev(sigma_num, closed form sign {sgn:+d})={dev:.1e}, oddness residual={odd:.1e}")
    e2p = np.roots([Kf, 2 * C * np.sqrt(Kf), -Bf]); e2m = np.roots([Kf, -2 * C * np.sqrt(Kf), -Bf])
    etas = [(+1, ev.real) for ev in e2p if abs(ev.imag) < 1e-9 and ev.real > 0] + [(-1, ev.real) for ev in e2m if abs(ev.imag) < 1e-9 and ev.real > 0]
    js = [(sk, ev, Af / (sk * np.sqrt(Kf) * ev) + 2 * tau0) for sk, ev in etas]
    print(msg + "  -> REAL Floquet multipliers")
    for o in out: print("        " + o)
    print("        eta^2, j for (sign k):", [(sk, round(ev, 5), round(jv, 4)) for sk, ev, jv in js], "  sign j = sign k:", all(np.sign(jv) == sk for sk, ev, jv in js))

def run(label, Q1, Q2, Q3, C, tau0, L, patterns):
    print(f"\n######## {label} ########")
    for (n, e1, e2, e3, einf, d) in patterns:
        sols, Bb, r, theta, PK, P, cs = solve_pattern(Q1, Q2, Q3, n, e1, e2, e3, einf, d)
        for Kv, Av, csub in sols:
            tag = f"n={n} e=({e1},{e2},{e3},{einf}) d={d}" + (" [c0 free family, sample]" if csub.get('free') else "")
            csub2 = {k_: v for k_, v in csub.items() if k_ != 'free'}
            check_point(Q1, Q2, Q3, Kv, Av, Bb, r, theta, PK, P, csub2, C, tau0, L, tag)

if __name__ == '__main__':
    pats = [(3, 1, 1, 1, 2, 0), (4, 1, 1, 1, 1, 0), (4, 1, 1, 3, 3, 0), (4, 1, 1, 1, 3, 1), (5, 1, 1, 1, 2, 1), (5, 1, 1, 2, 3, 1), (5, 1, 2, 1, 3, 1), (5, 2, 1, 1, 3, 1)]
    extra = [tuple(int(x) for x in a.split(',')) for a in sys.argv[1:]]
    run("QA-matched (1,2) rod", sp.Rational(222842, 10**6), sp.Rational(2212881, 10**6), sp.Rational(-2448875, 10**6), -0.549454, 0.760091, 6.65696, pats + extra)
    run("lamODE = -0.5 (1,2) rod", sp.Rational(684788, 10**6), sp.Rational(1769765, 10**6), sp.Rational(-3388971, 10**6), -1.013304, 0.916102, 5.864041, pats + extra)
