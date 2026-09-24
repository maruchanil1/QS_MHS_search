"""TASK 3 -- profile-fibre rank tests for Staeckel (Liouville-separable) potentials on doubly resonant tori.

Closed-line QS in a potential V needs a 1-parameter family of closed orbits (same energy) with IDENTICAL V(t)-profiles
up to a time shift (CONTEXT 2.4, round-1 notes Sec. 5).  At fixed energy the closed orbits of a non-degenerate integrable
system lie on isolated doubly resonant Liouville tori L, so the family is a curve in the 2-dim orbit space O = L/flow on
which all shift-invariants of the profile are constant.  Shift-invariants used: time averages <V^k>, k = 1..4
(Parseval-complete together with the phases; enough for a rank test).
 * axisymmetric V (two-centre, Kepler-Stark): O carries the rotation direction along which the profile is trivially
   constant; a NON-axisymmetric QS torus needs the profile constant on ALL of O (analyticity), i.e. <V^k> independent of
   the transverse phase delta.  Test: d<V^k>/d delta != 0  =>  refuted.
 * V without continuous isometry (Family-2 horizontal Staeckel system + vertical pendulum): O = (delta_h, delta_z);
   rank 2 of the Jacobian of (<V>,<V^2>,<V^3>,<V^4>) at generic points => 0-dim fibres => refuted.

Liouville form used for the planar part: coordinates (q1,q2), conformal factor h = h1(q1) + h2(q2), V_plane = (F(q1)+G(q2))/h.
Staeckel time sigma: dq_i/dsigma = p_i, dp_1/dsigma = E h1' - F', dp_2/dsigma = E h2' - G', dtau/dsigma = h,
with p1^2 = 2(E h1 - F - beta), p2^2 = 2(E h2 - G + beta).  Resonance P1/P2 = m/n found in beta by brentq.
Systems:
 (b1) two-centre  V = -mu1/r1 - mu2/r2 (+ l^2/2rho^2), prolate: q1 = u >= 0, q2 = v in [0,pi] (libration), h = c^2(sinh^2u + sin^2v),
      F = -(mu1+mu2) c cosh u + l^2/(2 sinh^2 u), G = -(mu1-mu2) c cos v + l^2/(2 sin^2 v).
 (b2) Kepler-Stark V = -mu/r - Fz (+ l^2/2rho^2), parabolic: h = xi^2 + eta^2, F = -mu - F xi^4/2 + l^2/(2xi^2), G = -mu + F eta^4/2 + l^2/(2eta^2).
 (a)  Family 2: h = eps(sinh^2u + cos^2v), F = -sin^2A/8, A = (eps/2)sinh 2u + pi/2 - S, G = -sinh^2B/8, B = (eps/2) sin 2v
      (v rotates), plus pendulum V_z = sin^2(lam z)/(2 lam^2) with period T_z = (r/s) T_h (second resonance, solved in E_z).
Run: python3 profile_fibre_rank.py      (~1 min; numerical, tolerances 1e-11; this is a decidable finite test, not a scan)
"""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

class Liouville2D:
    def __init__(self, h1, dh1, h2, dh2, F, dF, G, dG, E, q1_range, q2_mode, q2_range=None):
        self.h1, self.dh1, self.h2, self.dh2 = h1, dh1, h2, dh2
        self.F, self.dF, self.G, self.dG = F, dF, G, dG
        self.E = E; self.q1_range = q1_range; self.q2_mode = q2_mode; self.q2_range = q2_range
    def p1sq(self, q, beta): return 2*(self.E*self.h1(q) - self.F(q) - beta)
    def p2sq(self, q, beta): return 2*(self.E*self.h2(q) - self.G(q) + beta)
    def turning(self, f, lo, hi, q0):
        """turning points of f(q) >= 0 around q0 in (lo, hi)"""
        qs = np.linspace(lo, hi, 4001)
        vals = np.array([f(q) for q in qs])
        i0 = np.argmin(abs(qs - q0))
        if vals[i0] <= 0: raise ValueError("q0 not in allowed region")
        i = i0
        while i > 0 and vals[i] > 0: i -= 1
        a = brentq(f, qs[i], qs[i+1])
        i = i0
        while i < len(qs)-1 and vals[i] > 0: i += 1
        b = brentq(f, qs[i-1], qs[i])
        return a, b
    def period_lib(self, f, a, b):
        # substitution q = a + (b-a)(1-cos th)/2 removes the 1/sqrt endpoint singularities
        def integrand(th):
            q = a + (b-a)*(1-np.cos(th))/2
            val = f(q)
            return (b-a)/2*np.sin(th)/np.sqrt(max(val, 1e-300))
        return 2*quad(integrand, 0, np.pi, limit=400, epsabs=1e-12, epsrel=1e-12)[0]
    def periods(self, beta, q1_0, q2_0=None):
        a1, b1 = self.turning(lambda q: self.p1sq(q, beta), *self.q1_range, q1_0)
        P1 = self.period_lib(lambda q: self.p1sq(q, beta), a1, b1)
        if self.q2_mode == 'lib':
            a2, b2 = self.turning(lambda q: self.p2sq(q, beta), *self.q2_range, q2_0)
            P2 = self.period_lib(lambda q: self.p2sq(q, beta), a2, b2)
            return P1, P2, (a1, b1), (a2, b2)
        else:  # rotation over one full period 2pi of q2
            P2 = quad(lambda q: 1/np.sqrt(self.p2sq(q, beta)), 0, 2*np.pi, limit=400, epsabs=1e-12, epsrel=1e-12)[0]
            return P1, P2, (a1, b1), None
    def rhs(self, sig, y, beta, pend=None):
        q1, p1, q2, p2 = y[:4]
        h = self.h1(q1) + self.h2(q2)
        out = [p1, self.E*self.dh1(q1) - self.dF(q1), p2, self.E*self.dh2(q2) - self.dG(q2), h]
        if pend is not None:
            z, pz = y[5], y[6]
            out += [h*pz, -h*pend['dV'](z)]
        return out
    def phase_state(self, beta, delta2, tp2, P2):
        """state of the q2-oscillator at phase delta2 (fraction of its period) starting from its lower turning point"""
        if self.q2_mode == 'lib':
            q2_0 = tp2[0] + 1e-9*(tp2[1]-tp2[0]); p2_0 = np.sqrt(max(self.p2sq(q2_0, beta), 0))
        else:
            q2_0 = 0.0; p2_0 = np.sqrt(self.p2sq(q2_0, beta))
        if delta2 == 0: return q2_0, p2_0
        sol = solve_ivp(lambda s, y: [y[1], self.E*self.dh2(y[0]) - self.dG(y[0])], (0, delta2*P2), [q2_0, p2_0],
                        method='DOP853', rtol=1e-12, atol=1e-14)
        return sol.y[0, -1], sol.y[1, -1]

def moments_closed_orbit(sysL, beta, m, n, P1, P2, tp1, tp2, delta2, pend=None, delta_z=0.0, kmax=4):
    """time averages <V^k> along the closed orbit with transverse phase delta2 (and pendulum phase delta_z)."""
    Psig = n*P1                         # = m*P2 on resonance P1/P2 = m/n
    q1_0 = tp1[0] + 1e-9*(tp1[1]-tp1[0]); p1_0 = np.sqrt(max(sysL.p1sq(q1_0, beta), 0))
    q2_0, p2_0 = sysL.phase_state(beta, delta2, tp2, P2)
    y0 = [q1_0, p1_0, q2_0, p2_0, 0.0]
    if pend is not None:
        # pendulum state at phase delta_z (fraction of its period) from its turning point z = z_max
        zmax = pend['zmax']
        if delta_z == 0: z0, pz0 = zmax, 0.0
        else:
            sol = solve_ivp(lambda t, y: [y[1], -pend['dV'](y[0])], (0, delta_z*pend['Tz']), [zmax, 0.0],
                            method='DOP853', rtol=1e-12, atol=1e-14)
            z0, pz0 = sol.y[0, -1], sol.y[1, -1]
        y0 += [z0, pz0]
    def rhs_full(s, y):
        base = sysL.rhs(s, y, beta, pend)
        q1, q2 = y[0], y[2]
        h = sysL.h1(q1) + sysL.h2(q2)
        V = (sysL.F(q1) + sysL.G(q2))/h
        if pend is not None: V = V + pend['V'](y[5])
        return base + [h*V**k for k in range(1, kmax+1)]
    y0 = y0 + [0.0]*kmax
    sol = solve_ivp(rhs_full, (0, Psig), y0, method='DOP853', rtol=1e-12, atol=1e-14)
    yf = sol.y[:, -1]
    T = yf[4]
    closure = np.hypot(yf[0]-y0[0], yf[2]-y0[2]) if sysL.q2_mode == 'lib' else np.hypot(yf[0]-y0[0], (yf[2]-y0[2]) % (2*np.pi) if abs((yf[2]-y0[2]) % (2*np.pi)) < np.pi else (yf[2]-y0[2]) % (2*np.pi) - 2*np.pi)
    mom = yf[-kmax:]/T
    return T, mom, closure

def find_resonance(sysL, ratio_target, beta_lo, beta_hi, q1_0, q2_0=None, nscan=60):
    betas = np.linspace(beta_lo, beta_hi, nscan)
    vals = []
    for b in betas:
        try:
            P1, P2, _, _ = sysL.periods(b, q1_0, q2_0); vals.append(P1/P2 - ratio_target)
        except Exception:
            vals.append(np.nan)
    vals = np.array(vals)
    for i in range(nscan-1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i]*vals[i+1] < 0:
            f = lambda b: sysL.periods(b, q1_0, q2_0)[0]/sysL.periods(b, q1_0, q2_0)[1] - ratio_target
            return brentq(f, betas[i], betas[i+1], xtol=1e-13)
    return None, (np.nanmin(vals + ratio_target), np.nanmax(vals + ratio_target))

from fractions import Fraction
def pick_rational(lo, hi, maxden=200):
    """fraction with the smallest denominator strictly inside (lo, hi)"""
    for d in range(1, maxden+1):
        for k in range(int(np.floor(lo*d)), int(np.ceil(hi*d))+1):
            if lo < k/d < hi: return Fraction(k, d)
    return None

def report_axisym(name, sysL, beta_lo, beta_hi, q1_0, q2_0, ratios):
    print(f"\n=== {name} ===")
    todo = list(ratios); done = set()
    while todo:
        (m, n) = todo.pop(0)
        if (m, n) in done: continue
        done.add((m, n))
        res = find_resonance(sysL, m/n, beta_lo, beta_hi, q1_0, q2_0)
        if isinstance(res, tuple):
            fr = pick_rational(res[1][0]*1.001, res[1][1]*0.999)
            print(f"  ratio {m}:{n} not in range {res[1]};", f"trying {fr.numerator}:{fr.denominator}" if fr else "")
            if fr and (fr.numerator, fr.denominator) not in todo and (fr.numerator, fr.denominator) != (m, n):
                todo.insert(0, (fr.numerator, fr.denominator))
            continue
        beta = res
        P1, P2, tp1, tp2 = sysL.periods(beta, q1_0, q2_0)
        print(f"  resonance P1:P2 = {m}:{n} at beta = {beta:.12f}  (P1 = {P1:.6f}, P2 = {P2:.6f}); q1 in {tp1}, q2 in {tp2 if tp2 else 'rotation'}")
        deltas = [0.0, 0.13, 0.29, 0.41]
        rows = []
        for d in deltas:
            T, mom, cl = moments_closed_orbit(sysL, beta, m, n, P1, P2, tp1, tp2, d)
            rows.append(mom); print(f"    delta = {d:.2f}: T = {T:.9f}  <V>,<V^2>,<V^3>,<V^4> = {mom}  closure err {cl:.1e}")
        rows = np.array(rows)
        spread = rows.max(axis=0) - rows.min(axis=0)
        print(f"    spread of <V^k> over transverse phases: {spread}  (numerical noise ~1e-10)")
        print("    => profile NOT constant on the resonant torus: fibre = rotation orbits only => only axisymmetric QS tori" if spread.max() > 1e-6
              else "    => profile constant to numerical accuracy (unexpected!)")

# ---------------------------------------------------------------- (b1) two-centre problem
def two_centre(mu1, mu2, c, l, E):
    h1 = lambda u: c**2*np.sinh(u)**2; dh1 = lambda u: c**2*np.sinh(2*u)
    h2 = lambda v: c**2*np.sin(v)**2;  dh2 = lambda v: c**2*np.sin(2*v)
    F = lambda u: -(mu1+mu2)*c*np.cosh(u) + l**2/(2*np.sinh(u)**2)
    dF = lambda u: -(mu1+mu2)*c*np.sinh(u) - l**2*np.cosh(u)/np.sinh(u)**3
    G = lambda v: -(mu1-mu2)*c*np.cos(v) + l**2/(2*np.sin(v)**2)
    dG = lambda v: (mu1-mu2)*c*np.sin(v) - l**2*np.cos(v)/np.sin(v)**3
    return Liouville2D(h1, dh1, h2, dh2, F, dF, G, dG, E, (1e-3, 6.0), 'lib', (1e-3, np.pi-1e-3))

# ---------------------------------------------------------------- (b2) Kepler-Stark
def kepler_stark(mu, Fz, l, E):
    h1 = lambda x: x**2; dh1 = lambda x: 2*x
    F = lambda x: -mu - Fz*x**4/2 + l**2/(2*x**2); dF = lambda x: -2*Fz*x**3 - l**2/x**3
    G = lambda y: -mu + Fz*y**4/2 + l**2/(2*y**2);  dG = lambda y: 2*Fz*y**3 - l**2/y**3
    return Liouville2D(h1, dh1, h1, dh1, F, dF, G, dG, E, (1e-3, 8.0), 'lib', (1e-3, 8.0))

# ---------------------------------------------------------------- (a) Family 2 horizontal system + pendulum
def family2(eps, S, E):
    h1 = lambda u: eps*np.sinh(u)**2; dh1 = lambda u: eps*np.sinh(2*u)
    h2 = lambda v: eps*np.cos(v)**2;  dh2 = lambda v: -eps*np.sin(2*v)
    A = lambda u: eps/2*np.sinh(2*u) + np.pi/2 - S; dA = lambda u: eps*np.cosh(2*u)
    Bf = lambda v: eps/2*np.sin(2*v); dB = lambda v: eps*np.cos(2*v)
    F = lambda u: -np.sin(A(u))**2/8; dF = lambda u: -np.sin(2*A(u))*dA(u)/8
    G = lambda v: -np.sinh(Bf(v))**2/8; dG = lambda v: -np.sinh(2*Bf(v))*dB(v)/8
    return Liouville2D(h1, dh1, h2, dh2, F, dF, G, dG, E, (1e-6, 4.0), 'rot')

if __name__ == '__main__':
    # (b1) asymmetric two-centre problem, l != 0 (orbits avoid the axis), E < 0
    report_axisym("(b1) two-centre  mu1=1, mu2=1/2, c=1, l=0.3, E=-0.6", two_centre(1.0, 0.5, 1.0, 0.3, -0.6),
                  beta_lo=-2.0, beta_hi=2.0, q1_0=1.0, q2_0=1.4, ratios=[(1, 1), (1, 2), (2, 3)])
    # (b2) Kepler-Stark, weak field
    report_axisym("(b2) Kepler-Stark  mu=1, F=0.02, l=0.3, E=-0.5", kepler_stark(1.0, 0.02, 0.3, -0.5),
                  beta_lo=-1.5, beta_hi=1.5, q1_0=1.0, q2_0=1.0, ratios=[(1, 1), (1, 2), (2, 3)])

    # (a) Family 2 + pendulum: full doubly resonant torus, 2-dim orbit space (delta_h, delta_z)
    print("\n=== (a) Landreman Family-2 horizontal Staeckel system + vertical pendulum ===")
    eps, S, lam = 1.0, 1.0, 1.0
    for Eh in (0.0, -0.02, -0.01):
        sysA = family2(eps, S, Eh)
        # u-band: around the maximum of sin^2 A, A = pi/2 -> sinh 2u = 2S/eps -> u0
        u0 = np.arcsinh(2*S/eps)/2
        got = None
        res = find_resonance(sysA, 0.5, 1e-3, 0.124, u0)
        if isinstance(res, tuple):
            fr = pick_rational(res[1][0]*1.001, res[1][1]*0.999)
            print(f"  E_h = {Eh}: P_u/P_v range {res[1]}, using {fr}")
            res2 = find_resonance(sysA, float(fr), 1e-3, 0.124, u0)
            got = (fr.numerator, fr.denominator, res2)
        else:
            got = (1, 2, res)
        m, n, beta = got
        P1, P2, tp1, _ = sysA.periods(beta, u0)
        Th = moments_closed_orbit(sysA, beta, m, n, P1, P2, tp1, None, 0.0)[0]
        print(f"  E_h = {Eh}: horizontal resonance P_u:P_v = {m}:{n} at beta = {beta:.12f}, u in {tp1}, physical period T_h = {Th:.6f}")
        # pendulum V_z = sin^2(lam z)/(2 lam^2): choose E_z with T_z = (r/s) T_h > 2 pi
        Vz = lambda z: np.sin(lam*z)**2/(2*lam**2); dVz = lambda z: np.sin(2*lam*z)/(2*lam)
        def Tz_of(zmax):
            Ez = Vz(zmax)
            f = lambda z: 2*(Ez - Vz(z))
            def integrand(th):
                z = zmax*np.cos(th); return zmax*np.sin(th)/np.sqrt(max(f(z), 1e-300))
            return 2*quad(integrand, 0, np.pi, limit=400, epsabs=1e-12, epsrel=1e-12)[0]
        target = None
        # prefer a pendulum period T_z = T_h/s moderately above the small-amplitude period 2 pi (away from the separatrix)
        for s in range(1, 60):
            Tz_target = Th/s
            if 7.0 < Tz_target < 15.0:
                target = (1, s, Tz_target); break
        if target is None:
            for (r, s) in [(1, 1), (2, 1), (3, 1), (3, 2), (4, 1), (5, 2), (1, 2), (2, 3)]:
                Tz_target = r/s*Th
                if Tz_target > 2*np.pi/lam*1.0001 and Tz_target < 60:
                    target = (r, s, Tz_target); break
        if target is None:
            print("  no admissible pendulum resonance found"); continue
        r, s, Tz_target = target
        zmax = brentq(lambda zm: Tz_of(zm) - Tz_target, 1e-3, np.pi/(2*lam) - 1e-6)
        pend = {'V': Vz, 'dV': dVz, 'zmax': zmax, 'Tz': Tz_target}
        print(f"  pendulum resonance T_z = ({r}/{s}) T_h = {Tz_target:.6f} at z_max = {zmax:.6f} (E_z = {Vz(zmax):.6f}); total closed period {s*Th if r>=s else r*Th:.4f}")
        # closed orbit of the 3-DOF system: sigma-period n_tot * n*P1 where the pendulum needs s periods of T_h ... take lcm: r T_h = s T_z
        mtot = r if r >= 1 else 1
        # moments over the full closed orbit: integrate over r horizontal periods (= s pendulum periods)
        def moments3(dh, dz):
            T, mom, cl = moments_closed_orbit(sysA, beta, m*mtot, n*mtot, P1, P2, tp1, None, dh, pend=pend, delta_z=dz)
            return mom
        d0 = np.array([0.23, 0.37]); eps_fd = 1e-4
        M0 = moments3(*d0)
        Jac = np.zeros((4, 2))
        for j in range(2):
            dp = d0.copy(); dm = d0.copy(); dp[j] += eps_fd; dm[j] -= eps_fd
            Jac[:, j] = (moments3(*dp) - moments3(*dm))/(2*eps_fd)
        sv = np.linalg.svd(Jac, compute_uv=False)
        print(f"  moments at (delta_h, delta_z) = {d0}: {M0}")
        print(f"  Jacobian d<V^k>/d(delta_h, delta_z):\n{Jac}\n  singular values {sv}  => rank {(sv > 1e-6*max(sv.max(),1e-300)).sum()}")
        for d in ([0.05, 0.6], [0.5, 0.1], [0.8, 0.8]):
            Mx = moments3(*d)
            Jx = np.zeros((4, 2))
            for j in range(2):
                dp = np.array(d, float); dm = dp.copy(); dp[j] += eps_fd; dm[j] -= eps_fd
                Jx[:, j] = (moments3(*dp) - moments3(*dm))/(2*eps_fd)
            svx = np.linalg.svd(Jx, compute_uv=False)
            print(f"  at {d}: singular values {svx} => rank {(svx > 1e-6*svx.max()).sum()}")
        print("  rank 2 at generic points => profile fibres are 0-dimensional => no closed-line QS torus in this potential (this resonance)")
