"""boozer_axisym.py -- exact Boozer angles for an axisymmetric closed-line congruence given as
        x(lam, s, tau) = R_z(s) X0(lam, tau),      B = d_t x = x_tau / (dt/dtau),
with lam a flux label, s the rotation angle (orbit label), tau the orbit parameter (t itself, or the eccentric
anomaly), one toroidal transit per tau-period 2 pi, and the orbit's poloidal winding iota (integer).

Construction (elementary; coordinator note 4 of BULLETIN round 1):
    Lambda(lam) := (1/2pi) oint B^2 dt  = G + iota I,
    F(lam, tau)  := int_0^tau B^2 dt / Lambda          (B . grad phi_B = B^2/(G+iota I)  =>  d_t phi_B = F_t)
    phi_B = s + F,   theta_B = iota F        (rotation covariance fixes the s-dependence: R_c: s -> s + c must be
                                              phi_B -> phi_B + c at fixed theta_B, and theta_B - iota phi_B = -iota s)
    sqrt g_B = det(x_lam, x_s, x_tau)/(dt/dtau) / (-iota F_t) = (G+iota I)/B^2  requires
    det(x_psi, x_s, x_t) = -iota  =>  dpsi/dlam = -J/iota,  J := det(x_lam, x_s, x_t)  (J is lam-only: div B = 0).
Jets: theta, phi, psi act as derivations on functions of (lam, s, tau):
    d_phi = d_s,     d_theta = -(1/iota) d_s + (1/(iota F_tau)) d_tau,
    d_psi  = (dlam/dpsi) [ d_lam - (F_lam/F_tau) d_tau ]        (theta, phi fixed; iota constant).
Then G = B.x_phi = B.x_s (= angular momentum), I = (Lambda - G)/iota, K = B.x_psi, f = sqrt g = Lambda/B^2.

Caveat (1:1:2): the orbit is invariant under rotation by pi, x(s, t+pi) = x(s+pi, t); the label s is defined mod pi
on the quotient torus, and (theta_B, phi_B) -> (theta_B + 2 pi, phi_B + 2 pi) under (s,t) -> (s+pi, t+pi) since
F(pi) = pi (B^2 is pi-periodic).  All local (jet) statements are unaffected.

class AxisymCongruence(X0, lam, s, tau, dt_dtau, Fint, iota, Vpot)
    X0      : sympy 3-vector in (lam, tau) (before the rotation R_z(s))
    dt_dtau : dt/dtau (expression in lam, tau)
    Fint    : an antiderivative in tau of B^2 dt/dtau = |x_tau|^2/(dt/dtau)   (must be exact; checked)
    Vpot(x) : potential V (for p = -E = -(B^2/2 + V))
Methods: check_identities(point)  (direct symbolic derivations, ~30 s),
         jets(point, ORD)         (exact truncated-series jets in (psi, theta, phi); seconds)  -> dict for flow_core.
"""
import sympy as sp
from series_jets import TSeries, series_of_ad, invert_map, const_eval, dP, dT, dF, dl, ds, dt


def Rz(s):
    return sp.Matrix([[sp.cos(s), -sp.sin(s), 0], [sp.sin(s), sp.cos(s), 0], [0, 0, 1]])


class AxisymCongruence:
    def __init__(self, X0, lam, s, tau, dt_dtau, Fint, iota_val, Vpot, simp=sp.simplify):
        self.lam, self.s, self.tau, self.iota = lam, s, tau, sp.Integer(iota_val)
        self.simp = simp
        self.X0 = X0
        self.x = Rz(s)*X0
        self.dt = dt_dtau
        self.B = self.x.diff(tau)/dt_dtau
        self.B2 = simp(self.B.dot(self.B))
        integrand = simp(X0.diff(tau).dot(X0.diff(tau))/dt_dtau)
        self.Fint_ok = simp(sp.diff(Fint, tau) - integrand) == 0
        Lam = simp((Fint.subs(tau, 2*sp.pi) - Fint.subs(tau, 0))/(2*sp.pi))
        self.Lam = Lam                                     # G + iota I
        self.F = (Fint - Fint.subs(tau, 0))/Lam            # phi_B = s + F, theta_B = iota F
        self.Ft = simp(sp.diff(self.F, tau))
        self.Flam = simp(sp.diff(self.F, lam))
        # Jacobian in (lam, s, t) and the flux label
        J = simp(sp.Matrix.hstack(self.x.diff(lam), self.x.diff(s), self.B).det())
        self.J = J
        self.J_const = simp(sp.diff(J, tau)) == 0 and simp(sp.diff(J, s)) == 0
        self.dpsi_dlam = simp(-J/self.iota)
        self.dlam_dpsi = 1/self.dpsi_dlam
        self.V = Vpot(self.x)
        self.E = simp(self.B2/2 + self.V)
        self.p = -self.E

    # ---------------- direct derivations (slow but independent) ----------------
    def d_phi(self, E):
        return sp.diff(E, self.s)

    def d_theta(self, E):
        return -sp.diff(E, self.s)/self.iota + sp.diff(E, self.tau)/(self.iota*self.Ft)

    def d_psi(self, E):
        return self.dlam_dpsi*(sp.diff(E, self.lam) - self.Flam/self.Ft*sp.diff(E, self.tau))

    def boozer_data(self):
        x, B = self.x, self.B
        self.x_th = x.applyfunc(self.d_theta)
        self.x_ph = x.applyfunc(self.d_phi)
        self.x_ps = x.applyfunc(self.d_psi)
        self.Gf = self.simp(B.dot(self.x_ph))
        self.If = self.simp((self.Lam - self.Gf)/self.iota)
        self.Kf = B.dot(self.x_ps)
        self.f = self.Lam/self.B2
        return self.Gf, self.If, self.Kf

    def check_identities(self, point):
        """residuals of (1),(2),(4'),(4), sqrt g, flux-function property of G, I, K at a point (exact)."""
        self.boozer_data()
        x_th, x_ph, x_ps = self.x_th, self.x_ph, self.x_ps
        Wv = x_ph + self.iota*x_th
        sg = sp.Matrix.hstack(x_ps, x_th, x_ph).det()
        ev = lambda e: self.simp(sp.sympify(e).xreplace(point))
        out = {}
        out['Fint exact'] = self.Fint_ok
        out['J const in (s,tau)'] = self.J_const
        out['G flux function (d_s, d_tau)'] = (self.simp(sp.diff(self.Gf, self.s)) == 0, self.simp(sp.diff(self.Gf, self.tau)) == 0)
        out['K s-independent'] = self.simp(sp.diff(self.Kf, self.s)) == 0
        out['sqrt g - Lam/B^2'] = ev(sg - self.f)
        out['(1) W.x_phi - G sqrt g'] = ev(Wv.dot(x_ph) - self.Gf*sg)
        out['(2) W.x_theta - I sqrt g'] = ev(Wv.dot(x_th) - self.If*sg)
        out["(4') W.x_psi - K sqrt g"] = ev(Wv.dot(x_ps) - self.Kf*sg)
        Gp, Ip, pp = (self.d_psi(q) for q in (self.Gf, self.If, self.p))
        out['(4) iota d_theta K - (G\'+iota I\'+p\' sqrt g)'] = ev(self.iota*self.d_theta(self.Kf) - (Gp + self.iota*Ip + pp*sg))
        out['B = W/sqrt g'] = list((self.B - Wv/sg).applyfunc(ev))
        out['values'] = dict(G=ev(self.Gf), I=ev(self.If), K=ev(self.Kf), iota=self.iota, Gp=ev(Gp), Ip=ev(Ip), pp=ev(pp),
                             sqrtg=ev(sg), B2=ev(self.B2), psi_prime=ev(self.dpsi_dlam))
        return out

    # ---------------- exact series jets ----------------
    def series_data(self, point, order=4):
        """Taylor series (exact) of x, B, f, p, G in the Boozer displacements (dP, dT, dF) at the point."""
        lam, s, tau = self.lam, self.s, self.tau
        vars_ = (lam, s, tau)
        ser = lambda e: series_of_ad(e, vars_, point, order)
        # forward map
        Fser = ser(self.F)
        Fser = Fser - Fser.const()           # removes the only irrational constant (c*tau0)
        assert all(v.is_Rational for v in Fser.d.values()), 'non-rational F coefficients'
        # psi(lam): Taylor coefficients from derivatives of psi' = dpsi/dlam (no antiderivative needed)
        dpsi_d = {}
        dk = self.dpsi_dlam
        for k in range(1, order + 1):
            dpsi_d[(k, 0, 0)] = const_eval(dk.subs(point))/sp.factorial(k)
            dk = sp.diff(dk, lam)
        psi_ser = TSeries((dl, ds, dt), order, dpsi_d)
        theta_ser = Fser*self.iota
        phi_ser = TSeries((dl, ds, dt), order, {(0, 1, 0): 1}) + Fser
        inv = invert_map([psi_ser, theta_ser, phi_ser])
        comp = lambda S: S.compose(inv)
        X = [comp(ser(self.x[i])) for i in range(3)]
        Bv = [comp(ser(self.B[i])) for i in range(3)]
        f_ser = comp(ser(self.Lam/self.B2))
        p_ser = comp(ser(self.p))
        Lam_ser = comp(ser(self.Lam))
        # derived series
        Xpsi = [X[i].deriv(0) for i in range(3)]
        Xphi = [X[i].deriv(2) for i in range(3)]
        K_ser = sum((Bv[i]*Xpsi[i] for i in range(3)), TSeries((dP, dT, dF), order - 1, 0))
        G_ser = sum((Bv[i]*Xphi[i] for i in range(3)), TSeries((dP, dT, dF), order - 1, 0))
        I_ser = (Lam_ser.truncate(order - 1) - G_ser)*(1/self.iota)
        self.series = dict(x=X, B=Bv, f=f_ser, p=p_ser, K=K_ser, G=G_ser, I=I_ser, inv=inv, order=order)
        return self.series

    def jets(self, point, ORD=3, order=4):
        """dict {flow_core symbol: exact value} at the point (lam, s, tau values), for N = 0 (chi = theta),
        plus the true x_psi (key 'xpsi_true') and the flux-function checks (key 'checks')."""
        from flow_core import Xs, Ks, Fp, iota, iotap, G, Gp, I, Ip, pp, N
        S = self.series_data(point, order)
        X, K_ser, f_ser, p_ser, G_ser, I_ser = S['x'], S['K'], S['f'], S['p'], S['G'], S['I']
        sub = {}
        for (i, a_, b_), sym in Xs.items():
            sub[sym] = X[i].jet((0, a_, b_))
        for n, sym in enumerate(Ks):
            sub[sym] = K_ser.jet((0, n, 0)) if n <= K_ser.order else sp.Symbol(f'K{n}_unavailable')
        fpsi = f_ser.deriv(0)
        for n, sym in enumerate(Fp):
            sub[sym] = fpsi.jet((0, n, 0)) if n <= fpsi.order else sp.Symbol(f'fpsi{n}_unavailable')
        sub.update({iota: self.iota, iotap: 0, G: G_ser.const(), Gp: G_ser.jet((1, 0, 0)), I: I_ser.const(),
                    Ip: I_ser.jet((1, 0, 0)), pp: p_ser.jet((1, 0, 0)), N: 0})
        xpsi_true = sp.Matrix([X[i].jet((1, 0, 0)) for i in range(3)])
        # checks: G, I, p, K have no (theta, phi) dependence at the orders available; K is theta-only
        chk = {}
        z_ = lambda e: sp.simplify(sp.expand(e)) == 0
        chk['G flux function (all theta,phi jets vanish)'] = all(z_(v) for k, v in G_ser.d.items() if k[1] + k[2] > 0)
        chk['p flux function'] = all(z_(v) for k, v in p_ser.d.items() if k[1] + k[2] > 0)
        chk['K phi-independent (K = K(psi, theta))'] = all(z_(v) for k, v in K_ser.d.items() if k[2] > 0)
        chk['f = sqrt g phi-independent'] = all(z_(v) for k, v in f_ser.d.items() if k[2] > 0)
        return sub, xpsi_true, chk
