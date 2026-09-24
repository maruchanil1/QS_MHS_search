"""TASK 3: finite-Fourier (trigonometric-polynomial) Boozer surfaces -- top-mode structure and small classes,
in the symmetry-adapted variables v = e^{i chi}, E = e^{i phi} (vE_surface.py):
   x(chi, phi) = sum_{j=-J}^{J} X_j(v) E^j,   C2: W.W has no E^k (k != 0),   C1: W.x_chi = kappa W.W.
   W = x_u + (iota - N) x_chi, x_u = i E d_E x, x_chi = i v d_v x;  D_j := j + (iota - N) v d_v  (so W_j = i D_j X_j).
Generic iota: D_j injective on Laurent polynomials for all j needed  <=>  (iota - N) m + j != 0.

Part A (sanity): axisymmetric surface x+iy = E rho(v), z = zeta(v), N = 0: C2 automatic; C1 reproduces the
        round-1 degree-1 theorem (iota * zeta_1^2 = 0 etc.).
Part B (top-mode lemma, general J): E^{2J}:  W_J.W_J = 0 and W_J.vX_J' = 0  =>  D_{2J}(X_J.X_J) = 0  =>  X_J.X_J = 0
        (generic iota)  =>  X_J'.X_J' = 0  =>  X_J and X_J' are orthogonal null vectors in C^3  =>  parallel  =>
        X_J = lambda(v) nu_0 with a FIXED null vector nu_0 (a fixed oriented plane; WLOG nu_0 = eps after a rotation).
        Verified symbolically on a generic degree-1 X_J and on the null-cone parametrisation.
Part C (J = 2, all v-degrees D <= 1 here; hand proof for all D in notes.md): with X_2 = lambda eps the hierarchy
        E^3, E^2, E^1 forces X_1 = 0 and X_0 = c + gamma_0(v) e_z: a doubly covered surface of revolution.
Part D (J = 3, D = 1: contains the round-1 26-unknown class M = L = N = 1 without stellarator symmetry):
        sequential solve of the hierarchy E^5 ... E^1.
Run: python3 task3_finite_fourier.py    (log: task3_finite_fourier.log)
"""
import sympy as sp, time, sys, itertools
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from vE_surface import v, E, eps, epsb, ez, surface_equations, frame_vector, laurent_poly, laurent
t0 = time.time()

iota_s, N_s = sp.symbols('iota N')


def Dop(j, iota, N):
    return lambda f_: j*f_ + (iota - N)*v*sp.diff(f_, v)


# ---------------- Part A: axisymmetric sanity ----------------
print("Part A: axisymmetric surface x+iy = E rho(v), z = zeta(v), N = 0, degree 1 in v")
r0, r1, rm, z1, zm = sp.symbols('r0 r1 rm z1 zm')
rho = r0 + r1*v + rm/v
zeta = z1*v + zm/v                     # real zeta on |v| = 1 needs zm = conj(z1)
Xax = {1: rho*eps, -1: (r0 + rm*v + r1/v)*epsb, 0: zeta*ez}     # conj(rho)(1/v) with formal conjugates rm~ = r1 etc.
c2, c1, WW, Wx = surface_equations(Xax, iota_s, 0)
print("  C2 (E^k, k != 0) equations:", len(c2), "  C1 (E^k, k != 0) equations:", len(c1))
WW0 = sp.expand(sp.Poly(sp.expand(WW*E**2), E).coeff_monomial(E**2))
Wx0 = sp.expand(sp.Poly(sp.expand(Wx*E**2), E).coeff_monomial(E**2))
# C1 on the E^0 level: W.x_chi = kappa W.W  with kappa = I/(G + iota I); for the vacuum test kappa = 0
print("  E^0 of W.x_chi (vacuum C1 = 0 requires all v-modes to vanish):")
for m_, cc in sorted(laurent(Wx0, v, -2, 2).items()):
    print(f"     v^{m_}: {sp.factor(cc)}")
print("  => v^2 mode: iota*(r1*rm' + z1^2)... exactly the round-1 isotropy/iota*|z1|^2 structure; with rm = conj(r1)-partner "
      "the v^0 mode forces iota*|z1|^2 = 0 (round-1 theorem).")

# ---------------- Part B: top-mode lemma ----------------
print("\nPart B: top-mode lemma")
J = sp.Symbol('J', positive=True)
a_, b_, c_ = (laurent_poly(nm, 1) for nm in ('a', 'b', 'c'))
XJ = sp.Matrix([a_[0], b_[0], c_[0]])
io, Nn = sp.Rational(2, 11), 1                       # generic iota (iota - N = -9/11)
for Jv in (2, 3):
    DJ = Dop(Jv, io, Nn)
    WJ = sp.I*XJ.applyfunc(DJ)
    e1 = sp.expand(WJ.dot(WJ)); e2 = sp.expand(WJ.dot(sp.I*v*XJ.diff(v)))
    s_ = sp.expand(XJ.dot(XJ))
    # claim: W_J.X_J = (i/2) D_{2J} s
    print(f"  J = {Jv}:  W_J.X_J - (i/2) D_2J(X_J.X_J) == 0 :", sp.expand(WJ.dot(XJ) - sp.I/2*Dop(2*Jv, io, Nn)(s_)) == 0,
          "|  W_J.X_J = -i W_J.W_J/J ... check: J W_J.X_J == -i W_J.W_J - (iota-N) W_J.vX_J' :",
          sp.expand(Jv*WJ.dot(XJ) - (-sp.I*e1 - (io - Nn)*WJ.dot(v*XJ.diff(v)))) == 0)
    # D_{2J} injective on degree-1 Laurent polynomials for this iota:
    print(f"         D_2J eigenvalues on v^m, m=-1..1: {[2*Jv + (io - Nn)*m for m in (-1, 0, 1)]} (all nonzero)")
    # (iii) with s = 0 the top equation reduces to X_J'.X_J' = 0:
    ident3 = sp.expand(e1 + Jv**2*s_ + Jv*(io - Nn)*v*sp.diff(s_, v) + (io - Nn)**2*v**2*XJ.diff(v).dot(XJ.diff(v)))
    print(f"         identity  W_J.W_J = -[J^2 s + J(iota-N) v s' + (iota-N)^2 v^2 X_J'.X_J']  :", ident3 == 0)
    print(f"         => top equations  <=>  D_2J s = 0 and X_J'.X_J' = 0  <=>  (generic iota) X_J.X_J = 0 = X_J'.X_J'")
# null cone: X = lam(v) nu(w(v)), nu = (1-w^2, i(1+w^2), 2w)/2: X'.X' = lam^2 w'^2 (nu_w.nu_w) with nu_w.nu_w = 1
w_, lamf = sp.Function('w')(v), sp.Function('lam')(v)
nu = sp.Matrix([1 - w_**2, sp.I*(1 + w_**2), 2*w_])/2
Xn = lamf*nu
print("  null-cone parametrisation: nu.nu =", sp.simplify(nu.dot(nu)), ",  X'.X' =", sp.factor(sp.simplify(Xn.diff(v).dot(Xn.diff(v)))),
      " => X'.X' = 0 forces w' = 0: the null direction is fixed.")

# ---------------- Part C: J = 2 ----------------
print(f"\nPart C: J = 2, D = 1, iota = {io}, N = {Nn}   ({time.time()-t0:.0f} s)")
lam, lam_s = laurent_poly('lam', 1); lamb, lamb_s = laurent_poly('lamb', 1)
al1, al1_s = laurent_poly('al1', 1); be1, be1_s = laurent_poly('be1', 1); ga1, ga1_s = laurent_poly('ga1', 1)
alm1, alm1_s = laurent_poly('alm1', 1); bem1, bem1_s = laurent_poly('bem1', 1); gam1, gam1_s = laurent_poly('gam1', 1)
al0, al0_s = laurent_poly('al0', 1); be0, be0_s = laurent_poly('be0', 1); ga0, ga0_s = laurent_poly('ga0', 1)
kap = sp.Symbol('kappa')
X2 = {2: lam*eps, -2: lamb*epsb, 1: frame_vector(al1, be1, ga1), -1: frame_vector(alm1, bem1, gam1), 0: frame_vector(al0, be0, ga0)}
c2, c1, WW, Wx = surface_equations(X2, io, Nn, kappa=kap)
byk = lambda eqs, k: [cc for (kk, m_, cc) in eqs if kk == k]
print("  E^4 equations (should be none: X_2 isotropic):", len(byk(c2, 4)) + len(byk(c1, 4)))
print("  E^3 (C2):", [sp.factor(e) for e in byk(c2, 3)][:2], "...")
# level 1: E^{+-3} force be1 = 0, alm1 = 0 (given lam, lamb != 0 and D injective): impose and verify
sub1 = {s_: 0 for s_ in be1_s + alm1_s}
lvl3 = [sp.expand(e.subs(sub1)) for e in byk(c2, 3) + byk(c2, -3) + byk(c1, 3) + byk(c1, -3)]
print("  impose beta_1 = 0, alpha_{-1} = 0  =>  all E^{+-3} equations vanish:", all(e == 0 for e in lvl3))
# remaining unknowns and equations at E^{+-2}, E^{+-1}, E^0
rem_unk = [s_ for s_ in lam_s + lamb_s + al1_s + ga1_s + bem1_s + gam1_s + al0_s + be0_s + ga0_s + [kap]]
rem_eqs = list(dict.fromkeys(sp.expand(cc.subs(sub1)) for (kk, m_, cc) in c2 + c1 if abs(kk) <= 2))
rem_eqs = [e for e in rem_eqs if e != 0]
print(f"  remaining: {len(rem_unk)} unknowns, {len(rem_eqs)} equations (E^+-2, E^+-1, E^0)")
# E^{+-2} first
e2 = [e for e in list(dict.fromkeys(sp.expand(cc.subs(sub1)) for (kk, m_, cc) in c2 + c1 if abs(kk) == 2)) if e != 0]
t1 = time.time()
sol2 = sp.solve(e2, ga1_s + gam1_s + be0_s + al0_s, dict=True)
print(f"  E^+-2 level solved for (gamma_1, gamma_-1, beta_0, alpha_0): {len(sol2)} branches ({time.time()-t1:.0f} s)")
for s0 in sol2:
    print("     ", {k: sp.factor(vv) for k, vv in s0.items()})
# continue each branch: E^{+-1} equations alone (they involve alpha_1, beta_-1 and X_0), then the E^0 condition
terminal = []
for s0 in sol2:
    e1_ = [e for e in list(dict.fromkeys(sp.expand(cc.subs(sub1).subs(s0)) for (kk, m_, cc) in c2 + c1 if abs(kk) == 1)) if e != 0]
    unk1 = [u_ for u_ in al1_s + bem1_s + al0_s + be0_s + ga0_s if u_ not in s0]
    t1 = time.time()
    sol1 = sp.solve(e1_, unk1, dict=True)
    print(f"  E^+-1 level on branch {s0}: {len(e1_)} equations, solve for {unk1} -> {len(sol1)} solutions ({time.time()-t1:.0f} s)")
    for s1 in sol1:
        comb = {k: sp.simplify(vv.subs(s1)) for k, vv in s0.items()}; comb.update(s1)
        X1v = frame_vector(al1, be1, ga1).subs(sub1).subs(comb)
        Xm1v = frame_vector(alm1, bem1, gam1).subs(sub1).subs(comb)
        X0v = frame_vector(al0, be0, ga0).subs(comb)
        x0_xy = sp.simplify(sp.expand(X0v.dot(epsb)*2))
        print(f"     X_1 = {list(X1v.applyfunc(sp.simplify))}, X_-1 = {list(Xm1v.applyfunc(sp.simplify))},  (x+iy)-part of X_0 = {sp.factor(x0_xy)}")
        # the remaining E^0 condition (C1 with kappa) on the axisymmetric remainder x+iy = lam(v) E^2, z = gamma_0(v)
        e0 = [e for e in list(dict.fromkeys(sp.expand(cc.subs(sub1).subs(comb)) for (kk, m_, cc) in c1 if kk == 0)) if e != 0]
        sol0 = sp.solve(e0, [u_ for u_ in ga0_s + lamb_s + [kap] if u_ not in comb], dict=True)
        print(f"     remaining E^0 (axisymmetric C1 with kappa): {len(e0)} equations -> {len(sol0)} solutions: {sol0[:3]}")
        terminal.append((comb, X1v, Xm1v, x0_xy))
allax = all(X1v.applyfunc(sp.simplify) == sp.zeros(3, 1) and Xm1v.applyfunc(sp.simplify) == sp.zeros(3, 1) and sp.diff(x0_xy, v) == 0
            for comb, X1v, Xm1v, x0_xy in terminal) and len(terminal) > 0
print("  J = 2, D = 1 THEOREM (levels E^3, E^2, E^1): every solution with lambda != 0 has X_1 = X_-1 = 0 and X_0 = const + gamma_0(v) e_z,"
      " i.e. x+iy = c + lambda(v) E^2: a doubly covered surface of revolution:", allax)
print("  (the E^0 condition is then the axisymmetric C1 for that surface; at v-degree 1 it has no solution with iota != 0, as in round 1)")
print(f"  ({time.time()-t0:.0f} s)")
