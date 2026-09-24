"""TASK 2: strain-kernel condition S(u) B = 0 (strong QS, global-reformulations' lemma) on and near the axis, with the
CORRECTLY normalised symmetry vector (coordinator note 1)
        u = e_phi + N e_theta = [(G + N I) B + (N - iota) B x grad psi] / B^2 .
Question: which identities on the first-order data (eta, sigma(s), k, j; kappa, tau) does S(u)B = 0 impose to first
order in the distance r from the axis, beyond the Riccati / j = const of axis_linearization.py?

Setting (units B0 = 1, s = arclength).  Frenet tube coordinates x = gamma(s) + X N(s) + Y Bn(s); orthonormal frame
e = (T, N, Bn); directional derivatives  d_N = d_X,  d_Bn = d_Y,  d_T = [d_s + tau Y d_X - tau X d_Y]/(1 - kappa X);
frame derivatives  d_T e_j = Gamma_{ij} e_i /(1 - kappa X) with Gamma = e_i . e_j'  (Frenet), d_N e_j = d_Bn e_j = 0.
Gradient tensor (grad u)_{ia} = e_i . d_a u = d_a u_i + Gamma_{ij} u_j delta_{aT}/(1 - kappa X);  S = sym(grad u).

Inputs to first order (all at definition level, from the exact first-order QS solution of round 1):
  B = T + B1,  B1 = kappa X T + (M_NN X + M_NB Y) N + (M_BN X + M_BB Y) Bn,  M := on-axis transverse block of grad B,
      obtained from the complex first-order solution X_c = (eta/kappa) e^{iks}, Y_c = X_c (kappa^2/eta^2)(sigma - i):
      (X_c' - tau Y_c, Y_c' + tau X_c) = M (X_c, Y_c);  j = M_BN - M_NB.
  psi = (1/2)[(kappa^2/eta^2)(1 + sigma^2) X^2 - 2 sigma X Y + (eta^2/kappa^2) Y^2]  (toroidal flux / 2 pi; the first-order
      ellipse), grad psi = psi_X N + psi_Y Bn + O(r^2).
  G = L/(2 pi) = G0 on axis, I = O(r^2), N - iota0 = -k L/(2 pi) = -k G0  (k = 2 pi (iota0 - N)/L).
  => u/G0 = [B - k B x grad psi]/B^2 + O(r^2) = T + u1 + u2,  u1 known,  u2 = GENERAL unknown quadratic in (X, Y)
     (it contains G'(0), I'(0), iota'(0), the second-order field and psi_3, none of which is first-order data).
Output: S(u) B truncated at O(r): the O(1) part, the O(r) part; elimination of u2; the residual identities.
Run: python3 strain_kernel_axis.py   (~20 s)
"""
import sympy as sp

s = sp.symbols('s', real=True)
X, Y = sp.symbols('X Y', real=True)
eta, k = sp.symbols('eta k', positive=True)
kap = sp.Function('kappa', real=True)(s); tau = sp.Function('tau', real=True)(s); sig = sp.Function('sigma', real=True)(s)

# ---------- first-order data: transverse block M of grad B on the axis from the complex solution ----------
Xc = (eta / kap) * sp.exp(sp.I * k * s)
Yc = Xc * (kap**2 / eta**2) * (sig - sp.I)
# real basis: (Re, Im) of the complex solution give two real solutions
X1, X2 = sp.re(Xc), sp.im(Xc); Y1, Y2 = sp.re(Yc), sp.im(Yc)
Sol = sp.Matrix([[X1, X2], [Y1, Y2]])
DSol = sp.Matrix([[sp.diff(X1, s) - tau * Y1, sp.diff(X2, s) - tau * Y2],
                  [tau * X1 + sp.diff(Y1, s), tau * X2 + sp.diff(Y2, s)]])
M = sp.simplify(DSol * Sol.inv())
M = M.applyfunc(lambda e: sp.simplify(sp.expand(e)))
print("M (transverse block of grad B on axis, rows N,Bn; columns d_N, d_Bn):")
print("   M_NN =", M[0, 0]); print("   M_NB =", M[0, 1]); print("   M_BN =", M[1, 0]); print("   M_BB =", M[1, 1])
jcur = sp.simplify(M[1, 0] - M[0, 1])
print("   j = M_BN - M_NB =", jcur, "   tr M =", sp.simplify(M[0, 0] + M[1, 1]))

# ---------- fields ----------
psi = sp.Rational(1, 2) * ((kap**2 / eta**2) * (1 + sig**2) * X**2 - 2 * sig * X * Y + (eta**2 / kap**2) * Y**2)
psiX, psiY = sp.diff(psi, X), sp.diff(psi, Y)
# B to first order (Frenet components)
B1 = sp.Matrix([kap * X, M[0, 0] * X + M[0, 1] * Y, M[1, 0] * X + M[1, 1] * Y])
Bvec = sp.Matrix([1, 0, 0]) + B1
B2 = 1 + 2 * kap * X                       # |B|^2 to first order
# B x grad psi to first order: T x (psiX N + psiY Bn) = psiX Bn - psiY N
BxG = sp.Matrix([0, -psiY, psiX])
# u / G0 = (B - k B x grad psi)/B^2, expanded to first order, plus general quadratic u2
u1 = (Bvec - k * BxG) * (1 - 2 * kap * X)
def trunc(e, deg):
    e = sp.expand(e)
    return sum(t for t in sp.Add.make_args(e) if sp.Poly(t, X, Y).total_degree() <= deg) if e != 0 else sp.Integer(0)
u1 = u1.applyfunc(lambda e: trunc(e, 1))
print("\nu/G0 to first order (Frenet components):")
for nm, e in zip("T N B".split(), u1):
    print(f"   u_{nm} =", sp.collect(sp.expand(e), [X, Y]))
# general quadratic part
a = [sp.Function(f'a{i}')(s) for i in range(3)]; b = [sp.Function(f'b{i}')(s) for i in range(3)]; c = [sp.Function(f'c{i}')(s) for i in range(3)]
u2 = sp.Matrix([sp.Rational(1, 2) * (a[i] * X**2 + 2 * b[i] * X * Y + c[i] * Y**2) for i in range(3)])
u = u1 + u2

# ---------- gradient tensor in the Frenet frame, truncated at O(r) ----------
Gam = sp.Matrix([[0, -kap, 0], [kap, 0, -tau], [0, tau, 0]])      # Gamma_ij = e_i . e_j'
inv1 = 1 + kap * X + kap**2 * X**2                                # 1/(1 - kappa X) truncated
def dT(f):
    return trunc((sp.diff(f, s) + tau * Y * sp.diff(f, X) - tau * X * sp.diff(f, Y)) * inv1, 1)
gradu = sp.zeros(3, 3)     # gradu[i, a] = e_i . d_a u
for i in range(3):
    gradu[i, 0] = trunc(dT(u[i]) + sum(Gam[i, j] * u[j] for j in range(3)) * inv1, 1)
    gradu[i, 1] = trunc(sp.diff(u[i], X), 1)
    gradu[i, 2] = trunc(sp.diff(u[i], Y), 1)
S = (gradu + gradu.T) / 2
divu = sp.simplify(trunc(gradu.trace(), 0))
print("\ndiv u on axis (must vanish):", divu)
SB = (S * Bvec).applyfunc(lambda e: trunc(e, 1))

# ---------- O(1) and O(r) parts ----------
print("\nS(u) B on the axis (O(1)):", [sp.simplify(e.subs({X: 0, Y: 0})) for e in SB])
eqs = {}
for i, nm in enumerate("T N B".split()):
    e = sp.expand(SB[i])
    for var in (X, Y):
        eqs[(nm, var)] = sp.simplify(e.coeff(var))
    print(f"   O(r) coefficient of X in (S B)_{nm}:", sp.collect(eqs[(nm, X)], [a[0], b[0], c[0]]))
    print(f"   O(r) coefficient of Y in (S B)_{nm}:", sp.collect(eqs[(nm, Y)], [a[0], b[0], c[0]]))

# T-component: no u2 -> identity on first-order data?
print("\nT-component O(r) equations (contain no u2):", [sp.simplify(eqs[('T', X)]), sp.simplify(eqs[('T', Y)])])
# N-, B-components: solve for the transverse gradient of u2_T: unknowns a0, b0, c0 (4 equations, 3 unknowns)
unk = [a[0], b[0], c[0]]
lin = [eqs[('N', X)], eqs[('N', Y)], eqs[('B', X)], eqs[('B', Y)]]
print("\nN,B-components: do they involve u2_N, u2_B (a1,b1,c1,a2,b2,c2)?", any(e.has(*(a[1:] + b[1:] + c[1:])) for e in lin))
solNX = sp.solve(lin[0], a[0], dict=True)[0]     # a0 from (N, X)
solNY = sp.solve(lin[1], b[0], dict=True)[0]     # b0 from (N, Y)
solBY = sp.solve(lin[3], c[0], dict=True)[0]     # c0 from (B, Y)
constraint = sp.simplify(lin[2].subs(solNY))     # (B, X) must give the same b0
print("   a0 =", sp.simplify(solNX[a[0]])); print("   b0 =", sp.simplify(solNY[b[0]])); print("   c0 =", sp.simplify(solBY[c[0]]))
print("   consistency (B,X) after substituting b0 from (N,Y):", constraint)
# is the constraint implied by the Riccati (j = const)?  Compare with d/ds of j = M_BN - M_NB
sig1, sig2, k0, k1, k2, t0, t1, s0 = sp.symbols('sigma1 sigma2 kappa kappa1 kappa2 tau tau1 sigma0', real=True)
def real_sub(e):
    # substitute highest derivatives first (otherwise Derivative(const) -> 0 silently)
    e = e.subs(sp.diff(sig, s, 2), sig2).subs(sp.diff(kap, s, 2), k2)
    e = e.subs(sp.diff(sig, s), sig1).subs(sp.diff(kap, s), k1).subs(sp.diff(tau, s), t1)
    return e.subs({kap: k0, tau: t0, sig: s0})
cons_r = sp.factor(real_sub(constraint))
print("   constraint in jet symbols:", cons_r)
djds = real_sub(sp.diff(jcur, s))
print("   dj/ds in jet symbols:", sp.factor(djds))
print("   constraint / (dj/ds) =", sp.simplify(cons_r / djds), "   (a nonzero function => constraint <=> j = const, i.e. the Riccati compatibility; NO new identity)")
# the Y-part of (S B)_B alone fixes c0; the X-part of (S B)_N fixes a0: no further conditions. Count: 4 eqs, 3 unknowns, 1 constraint.
# Also report the full transverse (N,Bn) block of S on the axis: it must be nonzero (u is not Killing) but traceless
S0 = S.applyfunc(lambda e: sp.simplify(e.subs({X: 0, Y: 0})))
print("\nS(u)/G0 on the axis (Frenet components):"); sp.pprint(S0.applyfunc(sp.factor))
print("   det S0 =", sp.simplify(S0.det()), "  (must be 0: B in ker S)")
print("   S0_NN + S0_BB =", sp.simplify(S0[1, 1] + S0[2, 2]))
print("   eigenvalues of the transverse block (+-mu):  mu^2 =", sp.factor(sp.simplify(S0[1, 1]**2 + S0[1, 2]**2)))
