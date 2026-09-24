"""TASK 4b: Kovacic CASE 2 (dihedral / "sigma quadratic over C(q)") for the first-order QS Hill equation on a rod.

Hill's equation for phi (sigma = phi'/(k phi)) in q = kappa^2:   P phi_qq + (P'/2) phi_q + k Q phi = 0,
   P = -(q-q1)(q-q2)(q-q3),  k Q = K + Bb/q^2 - A/q   (K = k^2, A = k a, Bb = k b;  a = eta^2 (j - 2 tau0),  b = k eta^4 + 2 eta^2 C).
Normal form y'' = r y, y = phi P^{1/4}:  r = p1^2/4 + p1'/2 - kQ/P,  p1 = P'/(2P).  r has double poles at q1,q2,q3 (coefficient
-3/16, exponent difference 1/2), at 0 (coefficient b0 = -Bb/p0, p0 = q1 q2 q3) and at infinity (coefficient -3/16).
Case 1 (sigma in q' C(q)) was classified in round 1 (deg S <= 1) and kovacic_deg2.py (deg S = 2).  Case 2 = solutions whose
log-derivative is algebraic of degree 2 over C(q): Kovacic's algorithm (1986):
   E_c = {2 + m sqrt(1+4 b_c) : m = 0, +-2} cap Z for each double pole c;  E_{q_i} = E_inf = {1,2,3};
   E_0 = {2, 2 +- n} needs sqrt(1 + 4 b0) = n/2 in Z/2, i.e. Bb = p0 (4 - n^2)/16;
   d = (e_inf - e_1 - e_2 - e_3 - e_0)/2 in Z_{>=0}  (forces e_0 = 2 - n <= 0, n >= 2),
   theta = (1/2) sum e_c/(q - c);  monic P_K of degree d with
   P''' + 3 theta P'' + (3 theta^2 + 3 theta' - 4 r) P' + (theta'' + 3 theta theta' + theta^3 - 4 r theta - 2 r') P = 0,
   then omega = (phi_K +- sqrt(4r - phi_K^2 - 2 phi_K'))/2, phi_K = theta + P'/P, and y = exp(int omega).
For each admissible (n, e_1, e_2, e_3, e_inf) the condition is a polynomial identity in q, i.e. a polynomial system in
(K, A, coefficients of P_K).  We solve it EXACTLY (Groebner) on two rational rods (QA-matched (1,2) rod, lamODE = -0.5 rod)
for n <= NMAX and d <= DMAX, and, for d = 0 (linear in (K, A)), also with symbolic (q1,q2,q3).
Physical filters on any solution: K > 0 real; eta^2 > 0 from K eta^4 + 2 C k eta^2 - Bb = 0; the Hill equation must have a
REAL Floquet solution (sigma real): |tr M| >= 2 for the monodromy over one period of kappa^2.
Run: python3 kovacic_case2.py [NMAX] [DMAX]     (default 8, 1; ~1-2 min)
"""
import sympy as sp, numpy as np, itertools, time, sys
from scipy.integrate import solve_ivp

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 8
DMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 1
q = sp.symbols('q')
K, A = sp.symbols('K A')
t_start = time.time()

def combos(nmax, dmax):
    out = []
    for n in range(2, nmax + 1):
        e0 = 2 - n
        for e1, e2, e3, einf in itertools.product((1, 2, 3), repeat=4):
            two_d = einf - e1 - e2 - e3 - e0
            if two_d >= 0 and two_d % 2 == 0 and two_d // 2 <= dmax:
                out.append((n, e1, e2, e3, einf, two_d // 2))
    return out

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
    return eqs, list(cs), Bb, r, theta, PK

def floquet_trace(Kv, Av, Bv, Q1, Q2, Q3):
    """monodromy trace of phi'' + (K + Bb/q^2 - A/q) phi = 0 over one period of q(s), q'^2 = P(q), starting at q = q1"""
    Pc = np.poly1d([-1, Q1 + Q2 + Q3, -(Q1 * Q2 + Q1 * Q3 + Q2 * Q3), Q1 * Q2 * Q3]); dPc = Pc.deriv()
    def rhs(s, y):
        qq, qd, f1, f1d, f2, f2d = y
        U = Kv + Bv / qq**2 - Av / qq
        return [qd, dPc(qq) / 2, f1d, -U * f1, f2d, -U * f2]
    # half period: time from q1 to q2
    ev = lambda s, y: y[1]; ev.terminal = True; ev.direction = -1
    y0 = [Q1 + 1e-12, 0.0, 1, 0, 0, 1]
    sol = solve_ivp(rhs, (0, 200), y0, events=ev, rtol=1e-11, atol=1e-13, dense_output=True)
    if len(sol.t_events[0]) == 0: return np.nan
    # the event at q' = 0 crossing downward: first is at q2 (q' goes + -> -) -> half period; full period by symmetry of q: integrate to 2 t1
    t1 = sol.t_events[0][0]
    sol2 = solve_ivp(rhs, (0, 2 * t1), y0, rtol=1e-11, atol=1e-13)
    yL = sol2.y[:, -1]
    M = np.array([[yL[2], yL[4]], [yL[3], yL[5]]])
    return np.trace(M), 2 * t1

def analyse_rod(label, Q1, Q2, Q3, C, tau0, L):
    """exact rational rod: d = 0 patterns by linear algebra in (K, A); d = 1 patterns by eliminating (K, A) linearly from two
    equations (the system is bilinear: linear in (K, A) with coefficients polynomial in c0) and taking the gcd of the
    remaining polynomials in c0."""
    print(f"\n######## rod {label}: q1 = {Q1}, q2 = {Q2}, q3 = {Q3}  (C = {C:.4f}, tau0 = {tau0:.4f}, L = {L:.4f}) ########")
    found = 0
    for (n, e1, e2, e3, einf, d) in combos(NMAX, DMAX):
        t0 = time.time()
        eqs, cs, Bb, r, theta, PK = case2_system(Q1, Q2, Q3, n, e1, e2, e3, einf, d)
        tag = f"n={n} e=({e1},{e2},{e3},{einf}) d={d}"
        sols = []
        if d == 0:
            Mx, rhs = sp.linear_eq_to_matrix(eqs, [K, A])
            Aug = Mx.row_join(rhs)
            if Aug.rank() > Mx.rank():
                continue
            s_ = sp.solve(eqs, [K, A], dict=True)
            sols = s_ if s_ else []
        else:
            c0 = cs[0]
            Mx, rhs = sp.linear_eq_to_matrix(eqs, [K, A])
            pair = None
            for i in range(len(eqs)):
                for jj in range(i + 1, len(eqs)):
                    det = sp.expand(Mx[i, 0] * Mx[jj, 1] - Mx[i, 1] * Mx[jj, 0])
                    if det != 0:
                        pair = (i, jj, det); break
                if pair: break
            if pair is None:
                continue
            i, jj, det = pair
            Ksol = sp.together((rhs[i] * Mx[jj, 1] - rhs[jj] * Mx[i, 1]) / det)
            Asol = sp.together((Mx[i, 0] * rhs[jj] - Mx[jj, 0] * rhs[i]) / det)
            rest = [sp.expand(sp.numer(sp.together(Mx[m, 0] * Ksol + Mx[m, 1] * Asol - rhs[m]))) for m in range(len(eqs)) if m not in (i, jj)]
            rest = [p for p in rest if p != 0]
            g = None
            for p in rest:
                g = sp.Poly(p, c0) if g is None else sp.gcd(g, sp.Poly(p, c0))
            if g is None:
                print(f"   {tag}: c0 FREE, K = {Ksol}, A = {Asol}")
                continue
            if g.degree() <= 0:
                continue
            for root in g.nroots():
                if abs(sp.im(root)) > 1e-9: continue
                c0v = sp.re(root)
                if abs(det.subs(c0, c0v)) < 1e-12: continue
                sols.append({K: Ksol.subs(c0, c0v), A: Asol.subs(c0, c0v), c0: c0v})
        for s_ in sols:
            if K not in s_:
                print(f"   {tag}: K undetermined (A = {s_.get(A)}): the trivial a = b = 0 family, sigma = +-i, unphysical")
                continue
            Kv = complex(s_[K]); Av = complex(s_[A]); Bv = float(Bb)
            if abs(Kv.imag) > 1e-9 or Kv.real <= 1e-12:
                print(f"   {tag}: K = {Kv.real:+.6g} (not positive: unphysical), A = {Av.real:+.6g}, P_K = {PK.subs(s_) if d else 1}")
                continue
            Kv = Kv.real; Av = Av.real
            found += 1
            trM, per = floquet_trace(Kv, Av, Bv, float(Q1), float(Q2), float(Q3))
            for ksign in (+1, -1):
                kv = ksign * np.sqrt(Kv)
                e2r = np.roots([Kv, 2 * C * kv, -Bv])
                etas = [ev.real for ev in e2r if abs(ev.imag) < 1e-9 and ev.real > 0]
                js = [Av / (kv * ev) + 2 * tau0 for ev in etas]
                print(f"   {tag}: K = {Kv:.6f} (k = {kv:+.5f}, iota0-N = {kv*L/(2*np.pi):+.4f}), A = ka = {Av:+.6f}, Bb = kb = {Bv:+.6f}, P_K = {PK.subs(s_) if d else 1};  Floquet tr M = {trM:+.6f} (|tr| >= 2 needed for real sigma; = 2 for case-1 points);  eta^2 = {np.round(etas,5)}, j = {np.round(js,4)}")
        if time.time() - t0 > 20:
            print(f"   [{tag} took {time.time()-t0:.0f}s]")
    print(f"   => case-2 patterns with a solution K > 0 on this rod: {found}   [{time.time()-t_start:.0f}s]")

# ---------------- symbolic d = 0 analysis
SYM = len(sys.argv) > 3 and sys.argv[3] == "sym"
print("==== d = 0 patterns with symbolic (q1,q2,q3): linear system for (K, A); consistency constraints on the rod ====" if SYM else "(symbolic d = 0 part skipped; run with third argument sym; saved results in kovacic_case2_sym.out)")
Q1s, Q2s, Q3s = sp.symbols('q1 q2 q3')
for (n, e1, e2, e3, einf, d) in (combos(NMAX, 0) if SYM else []):
    t0 = time.time()
    eqs, cs, Bb, r, theta, PK = case2_system(Q1s, Q2s, Q3s, n, e1, e2, e3, einf, 0)
    Mx, rhs = sp.linear_eq_to_matrix(eqs, [K, A])
    Aug = Mx.row_join(rhs)
    # rank over the rational function field: generic rank of the coefficient matrix and of the augmented one
    rk = Mx.rank(simplify=True); rk_aug = Aug.rank(simplify=True)
    if rk_aug > rk:
        verdict = "INCONSISTENT for generic rods"
        # constraint polynomials: solve two independent equations and substitute
        try:
            sol = sp.solve(eqs[:2], [K, A], dict=True)
            cons = [sp.factor(sp.numer(sp.together(e.subs(sol[0])))) for e in eqs[2:]] if sol else []
            g = cons[0]
            for c_ in cons[1:]:
                g = sp.gcd(g, c_)
            verdict += f"; gcd of constraints = {sp.factor(g)}"
        except Exception as ex:
            verdict += f" (constraint extraction failed: {ex})"
    else:
        sol = sp.solve(eqs, [K, A], dict=True)
        verdict = f"CONSISTENT for all rods: {[{kk: sp.factor(v) for kk, v in s_.items()} for s_ in sol]}"
    print(f"   n={n} e=({e1},{e2},{e3},{einf}): {len(eqs)} equations, rank {rk} / augmented {rk_aug}: {verdict}   [{time.time()-t0:.1f}s]")

# ---------------- exact rational rods
# QA-matched (1,2) rod (round 1, R_max = 1.2126 units): q1 = 0.222842, q2 = 2.212881, q3 = -2.448875, C = -0.549454, tau0 = 0.760091, L = 6.65696
analyse_rod("QA-matched (1,2) rod", sp.Rational(222842, 10**6), sp.Rational(2212881, 10**6), sp.Rational(-2448875, 10**6), -0.549454, 0.760091, 6.65696)
# lamODE = -0.5 (1,2) rod (R_max = 1): roots -3.388971, 0.684788, 1.769765, C = -1.013304, tau0 = 0.916102, L = 5.864041
analyse_rod("lamODE = -0.5 (1,2) rod", sp.Rational(684788, 10**6), sp.Rational(1769765, 10**6), sp.Rational(-3388971, 10**6), -1.013304, 0.916102, 5.864041)
print(f"\n[total {time.time()-t_start:.0f}s]")
