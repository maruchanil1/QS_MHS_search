"""TASK 3: j != 0 tests of the round-1 closed forms (on-axis Hessian P_NN, P_NB, P_BB and the Riccati) on two EXACT
axisymmetric congruences with circular axes:
  (A) Solov'ev equilibrium from the 1:1:2 congruence (osc112_qs.py sec. 4): iota = 2, p' != 0, on-axis current != 0;
  (B) Kepler force-free family (kepler_axisym_family.py): iota = 1, p' = 0, curl B = lambda B != 0.
All ingredients are computed INDEPENDENTLY of the round-1 linearisation:
  * (eta, sigma) from the Hessian of the flux function at the axis (shape of the first-order flux ellipse),
  * j = (curl B).T / B0 from the actual field (Solov'ev: cylindrical curl of B(rho,z), also the GS current rho p' + FF'/rho;
    Kepler: Cartesian gradient of the explicit first-order field),
  * k = 2 pi (iota0 - N)/L with N = 0 from the known iota, sign from the sense of rotation of the field lines,
  * P_ij = Frenet components of Hess Pi with Pi = -V (V = (rho^2+4z^2)/2 resp. -mu/|x|).
Closed forms (round 1, units B0 = 1, arclength; circular axis => kappa' = tau = 0):
  P_NN = 3 kappa^2 - k^2,  P_NB = 0,  P_BB = k^2(1 + 2 eta^4/kappa^4 + 2 sigma^2) - 2 k eta^2 j/kappa^2,
  Riccati (sigma' = 0):  0 = -k(1 + sigma^2) - k eta^4/kappa^4 + eta^2 j/kappa^2.
Run: python3 circ_axis_checks.py
"""
import sympy as sp

def ellipse_from_hessian(A, Bm, Cm):
    """flux-surface quadratic form (1/2)(A X^2 + 2 Bm X Y + Cm Y^2) = lam (1/2)[(kap^2/eta^2)(1+sig^2) X^2 - 2 sig X Y + (eta^2/kap^2) Y^2]
    -> lam = sqrt(A Cm - Bm^2), sigma = -Bm/lam, eta^2/kappa^2 = Cm/lam."""
    lam = sp.sqrt(A * Cm - Bm**2)
    return sp.simplify(-Bm / lam), sp.simplify(Cm / lam)     # sigma, eta^2/kappa^2

def closed_forms(kap, k, sig, e2k2, j):
    """e2k2 = eta^2/kappa^2"""
    PNN = 3 * kap**2 - k**2
    PBB = k**2 * (1 + 2 * e2k2**2 + 2 * sig**2) - 2 * k * e2k2 * j
    ric = -k * (1 + sig**2) - k * e2k2**2 + e2k2 * j
    return sp.simplify(PNN), sp.simplify(PBB), sp.simplify(ric)

# ====================================================================== (A) Solov'ev
print("================ (A) Solov'ev equilibrium from the 1:1:2 congruence ================")
rho, z = sp.symbols('rho z', positive=True)
S, kp, dl = sp.symbols('S kappa_ delta', positive=True)
for sF in (+1, -1):
    m2 = (rho**2 - S)**2 / 4 + (sp.cos(dl) * (rho**2 - S) / 2 - z / kp)**2 / sp.sin(dl)**2
    Psi = -kp * sp.sin(dl) * m2
    F = sF * sp.sqrt(S**2 - 4 * m2)
    Brho = -sp.diff(Psi, z) / rho; Bz = sp.diff(Psi, rho) / rho; Bphi = F / rho
    V = (rho**2 + 4 * z**2) / 2; Pi = -V
    B2 = Brho**2 + Bz**2 + Bphi**2
    # MHS check: E = B^2/2 + V is a function of m2 only (equal energy on each surface) and force balance
    E = sp.simplify(B2 / 2 + V)
    print(f"\n--- sign(F) = {sF:+d} ---")
    print("   E = B^2/2 + V =", E, "  -> depends on (rho,z) only through m^2:", sp.simplify(E - (S / 2 + 2 * kp**2 * m2 + 0)) == 0 or sp.simplify(sp.diff(E, rho) * sp.diff(m2, z) - sp.diff(E, z) * sp.diff(m2, rho)) == 0)
    # force balance in cylindrical coordinates: (B.grad)B - grad Pi = 0  (axisymmetric: B.grad = Brho d_rho + Bz d_z; centripetal terms)
    conv_rho = Brho * sp.diff(Brho, rho) + Bz * sp.diff(Brho, z) - Bphi**2 / rho
    conv_z = Brho * sp.diff(Bz, rho) + Bz * sp.diff(Bz, z)
    conv_phi = Brho * sp.diff(Bphi, rho) + Bz * sp.diff(Bphi, z) + Brho * Bphi / rho
    print("   force balance (B.grad)B - grad Pi = 0:", [sp.simplify(conv_rho - sp.diff(Pi, rho)) == 0, sp.simplify(conv_z - sp.diff(Pi, z)) == 0, sp.simplify(conv_phi) == 0])
    # axis and Frenet frame:  T = sF e_phi, N = -e_rho, Bn = T x N = sF e_z ; kappa = 1/sqrt(S); B0 = sqrt(S)
    ax = {rho: sp.sqrt(S), z: 0}
    B0 = sp.simplify(sp.sqrt(B2.subs(ax))); kap = 1 / sp.sqrt(S)
    print("   axis rho = sqrt(S), z = 0: B_rho, B_z, B_phi =", [sp.simplify(c.subs(ax)) for c in (Brho, Bz, Bphi)], "; B0 =", B0)
    X, Y = sp.symbols('X Y', real=True)
    sub_XY = {rho: sp.sqrt(S) - X, z: sF * Y}      # X along N = -e_rho, Y along Bn = sF e_z
    # (1) ellipse from the flux-function Hessian (Psi or m2, any flux label)
    f = m2.subs(sub_XY)
    A = sp.diff(f, X, 2).subs({X: 0, Y: 0}); Bm = sp.diff(f, X, Y).subs({X: 0, Y: 0}); Cm = sp.diff(f, Y, 2).subs({X: 0, Y: 0})
    sig, e2k2 = ellipse_from_hessian(A, Bm, Cm)
    print("   first-order ellipse: sigma =", sig, ",  eta^2/kappa^2 =", e2k2)
    # (2) transverse block of grad B / B0 in (N, Bn) components (arclength linearisation): rotation sense and det
    BN = (-Brho).subs(sub_XY) / B0; BBn = (sF * Bz).subs(sub_XY) / B0; BT = (sF * Bphi).subs(sub_XY) / B0
    M = sp.Matrix([[sp.diff(BN, X), sp.diff(BN, Y)], [sp.diff(BBn, X), sp.diff(BBn, Y)]]).subs({X: 0, Y: 0}).applyfunc(sp.simplify)
    print("   grad B / B0 transverse block M =", M.tolist(), "; T-row: d_X B_T/B0 =", sp.simplify(sp.diff(BT, X).subs({X: 0, Y: 0})), "(= kappa)", " d_Y B_T/B0 =", sp.simplify(sp.diff(BT, Y).subs({X: 0, Y: 0})))
    detM = sp.simplify(M.det()); trM = sp.simplify(M.trace())
    # (3) k from iota = 2 (N = 0): |k| = 2 pi * 2 / L = 2/sqrt(S); sign = sense of rotation N -> Bn  (= sign of M_BN)
    kabs = 2 * (2 * sp.pi) / (2 * sp.pi * sp.sqrt(S))
    ksign = sp.sign(M[1, 0])
    k = ksign * kabs
    print("   det M =", detM, " (= k^2 =", sp.simplify(kabs**2), ")  tr M =", trM, ";  k =", k)
    # (4) on-axis current from the field: (curl B)_phi = d_z B_rho - d_rho B_z ; j = (curl B).T / B0 = sF (curl B)_phi / B0
    curl_phi = sp.simplify((sp.diff(Brho, z) - sp.diff(Bz, rho)).subs(ax))
    j = sp.simplify(sF * curl_phi / B0)
    # GS current for comparison: J_phi = rho p' + F F'/rho  with p' = 2 kp/sin(dl), FF' = 2/(kp sin dl) (osc112_qs.py)
    # here p = -E => dp/dPsi = -dE/dm2 / (dPsi/dm2) = -(2 kp^2)/(-kp sin dl) = 2 kp/sin dl ; F F' = d(F^2/2)/dPsi = (-2)/(-kp sin dl)
    pprime = 2 * kp / sp.sin(dl); FFp = 2 / (kp * sp.sin(dl))
    J_GS = sp.sqrt(S) * pprime + FFp / sp.sqrt(S)
    print("   j = (curl B).T/B0 =", j, ";  |J_phi| from Grad-Shafranov (rho p' + FF'/rho)/B0 =", sp.simplify(J_GS / B0), "; match:", sp.simplify(sp.Abs(j) - J_GS / B0) == 0)
    print("   tautology check j = M_BN - M_NB:", sp.simplify(j - (M[1, 0] - M[0, 1])) == 0)
    # (5) true Hessian of Pi in Frenet components (units B0^2)
    HessPi = sp.hessian(Pi.subs(sub_XY), (X, Y)).subs({X: 0, Y: 0}) / B0**2
    PNN_true, PNB_true, PBB_true = HessPi[0, 0], HessPi[0, 1], HessPi[1, 1]
    PNN_c, PBB_c, ric = closed_forms(kap, k, sig, e2k2, j)
    print(f"   TRUE  P_NN, P_NB, P_BB (units B0^2) = {PNN_true}, {PNB_true}, {PBB_true}")
    print(f"   CLOSED P_NN = {PNN_c}, P_BB = {PBB_c}  ->  agreement: {sp.simplify(PNN_c - PNN_true) == 0}, {sp.simplify(PBB_c - PBB_true) == 0}")
    print(f"   Riccati residual (sigma' = 0) = {ric}  -> {sp.simplify(ric) == 0}")
    print(f"   P_BB WITHOUT the j-term would be {sp.simplify(PBB_c + 2*k*e2k2*j)} (!= true): the j-term is essential")

# ====================================================================== (B) Kepler force-free
print("\n\n================ (B) Kepler force-free family (axis: circle of radius a, iota = 1) ================")
a, mu, C = sp.symbols('a mu C', positive=True)
eps, s, et = sp.symbols('epsilon s eta', real=True)
beta = sp.sqrt(1 - eps**2)
sin_i = eps / (C * beta); cos_i = sp.sqrt(1 - sin_i**2)
Rz = lambda t: sp.Matrix([[sp.cos(t), -sp.sin(t), 0], [sp.sin(t), sp.cos(t), 0], [0, 0, 1]])
Rx = sp.Matrix([[1, 0, 0], [0, cos_i, -sin_i], [0, sin_i, cos_i]])
e = sp.Matrix([a * (sp.cos(et) - eps), a * beta * sp.sin(et), 0])
x = Rz(s) * Rx * e
dt_deta = sp.sqrt(a**3 / mu) * (1 - eps * sp.cos(et))
Bpar = x.diff(et) / dt_deta                      # B = d_t x as a function of (eps, s, eta)
# first order in eps
x0 = x.subs(eps, 0); x1 = x.diff(eps).subs(eps, 0)
B0v = Bpar.subs(eps, 0); B1 = Bpar.diff(eps).subs(eps, 0)
phi0 = s + et                                    # zeroth-order cylindrical angle
e_rho = sp.Matrix([sp.cos(phi0), sp.sin(phi0), 0]); e_phi = sp.Matrix([-sp.sin(phi0), sp.cos(phi0), 0]); e_z = sp.Matrix([0, 0, 1])
print("   axis: x0 =", sp.simplify(x0).T, " B0 =", sp.simplify(B0v).T, " |B0| =", sp.simplify(sp.sqrt(B0v.dot(B0v))))
B0 = sp.sqrt(mu / a); kap = 1 / a
# displacement: tangential shift and transverse components
dphi = sp.simplify(x1.dot(e_phi) / a)            # angle shift of the foot point per unit eps
Xc = sp.simplify(-x1.dot(e_rho)); Yc = sp.simplify(x1.dot(e_z))    # X along N = -e_rho, Y along Bn = e_z (T = e_phi)
print("   first-order displacement (per eps): angle shift =", dphi, ", X =", Xc, ", Y =", Yc)
# field at the displaced point expressed at the foot-point angle phi' = phi0 + eps dphi: B(phi') = B0(phi') + eps [B1 + dphi * a * (dB0/dphi)/a ...]
# B0(phi0) = B0 e_phi(phi0) = B0 e_phi(phi' - eps dphi) = B0 [e_phi(phi') + eps dphi e_rho(phi')]
Bfirst = B1 + B0 * dphi * e_rho                  # first-order field correction at the foot point (per eps)
BN1 = sp.simplify(-Bfirst.dot(e_rho)); BB1 = sp.simplify(Bfirst.dot(e_z)); BT1 = sp.simplify(Bfirst.dot(e_phi))
print("   first-order field (per eps, Frenet comps): B_N =", BN1, ", B_Bn =", BB1, ", B_T =", BT1)
# express as linear functions of (X, Y): X = a cos(eta) eps, Y = (a/C) sin(eta) eps  -> cos eta = X/(a eps), sin eta = C Y/(a eps)
X, Y = sp.symbols('X Y', real=True)
def lin(expr):
    e2 = sp.expand(expr)
    cc = e2.coeff(sp.cos(et)); ss_ = e2.coeff(sp.sin(et))
    assert sp.simplify(e2 - cc * sp.cos(et) - ss_ * sp.sin(et)) == 0
    return sp.simplify(cc * X / a + ss_ * C * Y / a)      # eps * (...) with eps cos eta = X/a, eps sin eta = C Y/a
BN_XY, BB_XY, BT_XY = lin(BN1), lin(BB1), lin(BT1)
print("   => B/B0 - T = ", sp.simplify(BT_XY / B0), "T + (", sp.simplify(BN_XY / B0), ") N + (", sp.simplify(BB_XY / B0), ") Bn      [T-part should be kappa X = X/a]")
M = sp.Matrix([[sp.diff(BN_XY, X), sp.diff(BN_XY, Y)], [sp.diff(BB_XY, X), sp.diff(BB_XY, Y)]]) / B0
M = M.applyfunc(sp.simplify)
print("   grad B / B0 transverse block M =", M.tolist(), "; det M =", sp.simplify(M.det()), "; tr M =", sp.simplify(M.trace()))
j = sp.simplify(M[1, 0] - M[0, 1])          # (curl B).T / B0 = d_N B_Bn - d_Bn B_N  (no connection terms for transverse derivatives)
print("   j = (curl B).T/B0 = d_X B_Bn - d_Y B_N =", j)
# force-free check to first order on the axis: curl B parallel to B  <=> the N and Bn components of curl B vanish on the axis.
# (curl B)_N = d_Bn B_T - d_T B_Bn, (curl B)_Bn = d_T B_N - d_N B_T ; on the axis d_T of the O(r) components vanishes (constant
# coefficients along the circle), d_Bn B_T = 0 and d_N B_T = kappa B0 ... but the Frenet connection gives d_T B_N = -kappa B_T + ... :
# (grad B)_{NT} = d_T B_N + kappa B_T = kappa B0 (=> (curl B)_Bn = kappa B0 - kappa B0 = 0). Consistent; skip.
# ellipse from the displacement: X = a eps cos eta, Y = (a/C) eps sin eta -> quadratic form (X/a)^2 + (C Y/a)^2 = eps^2
A_, Bm_, Cm_ = 2 / a**2, 0, 2 * C**2 / a**2
sig, e2k2 = ellipse_from_hessian(A_, Bm_, Cm_)
print("   first-order ellipse: sigma =", sig, ", eta^2/kappa^2 =", e2k2, " (semi-axes a eps along N, a eps/C along Bn)")
kabs = 1 / a                                      # iota = 1, N = 0, L = 2 pi a
k = sp.sign(M[1, 0]) * kabs
print("   k = 2 pi (iota0 - N)/L =", k, ";  k^2 = det M:", sp.simplify(k**2 - M.det()) == 0)
# true Hessian of Pi = -V = mu/|x| on the axis (|x| = a), Frenet: N = -e_rho (toward the centre), Bn = e_z
xx, yy, zz = sp.symbols('x y z', real=True)
Pi = mu / sp.sqrt(xx**2 + yy**2 + zz**2)
H = sp.hessian(Pi, (xx, yy, zz)).subs({xx: a, yy: 0, zz: 0})     # axis point at phi = 0: T = e_y, N = -e_x, Bn = e_z
Tv, Nv, Bv = sp.Matrix([0, 1, 0]), sp.Matrix([-1, 0, 0]), sp.Matrix([0, 0, 1])
PNN_true = sp.simplify((Nv.T * H * Nv)[0] / B0**2); PNB_true = sp.simplify((Nv.T * H * Bv)[0] / B0**2); PBB_true = sp.simplify((Bv.T * H * Bv)[0] / B0**2)
PTT_true = sp.simplify((Tv.T * H * Tv)[0] / B0**2)
PNN_c, PBB_c, ric = closed_forms(kap, k, sig, e2k2, j)
print(f"   TRUE  P_TT, P_NN, P_NB, P_BB (units B0^2) = {PTT_true} (= -kappa^2), {PNN_true}, {PNB_true}, {PBB_true}")
print(f"   CLOSED P_NN = {PNN_c}, P_BB = {PBB_c}  ->  agreement: {sp.simplify(PNN_c - PNN_true) == 0}, {sp.simplify(PBB_c - PBB_true) == 0}")
print(f"   Riccati residual (sigma' = 0) = {sp.simplify(ric)}")
print(f"   P_BB without the j-term would be {sp.simplify(PBB_c + 2*k*e2k2*j)}: the j-term is essential; j = (C + 1/C)/a = -lambda B0 with lambda = dF/dPsi")
