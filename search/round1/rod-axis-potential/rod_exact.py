"""Exact Kirchhoff rod: symbolic first integrals in the Frenet frame + numerically exact closed (1,2) rod.

Symbolic (Frenet-component calculus, D = covariant s-derivative):  Killing screw I = alpha e_z + e_z x x,
on a unit-speed rod I = l2 T + l3 kappa Bn (l2, l3 const).  With e_z = (zT, zN, zB) in Frenet components,
D e_z = 0 and D I = e_z x T give
   (R1) zN = -l3 kappa',            zB = kappa (l2 - l3 tau)
   (R2) z_s := zT = c - l3 kappa^2/2
   (R3) tau = l2/(2 l3) + C/kappa^2,      C = (l2 c - alpha)/l3^2
   (R4) r^2 = l2^2 - alpha^2 + l3^2 kappa^2
   (R5) q := kappa^2 satisfies  q'^2 = P(q) = -q^3 + p2 q^2 + p1 q + p0,
        p2 = 4c/l3 - l2^2/l3^2,  p1 = 4(1-c^2)/l3^2 + 4 l2 C/l3,  p0 = -4 C^2   (so P(0) < 0: third root q3 < 0)
   => every Frenet quantity of the rod is rational in (q, q'); kappa^2 is an elliptic function of s.
Numerical: the (1,2) rod of handoff §7.1 (shooting), all identities checked to ~1e-9; rank test showing the rod
admits a unique Killing field with N.xi = 0 (=> a potential with a Killing symmetry having the rod as orbit must be
screw-invariant with THIS pitch); z_s != 0 at the curvature extrema (=> no separable V_h(x,y)+V_z(z) has the rod as orbit).

Run: python3 rod_exact.py
"""
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ------------------------------------------------------------------ symbolic
s = sp.symbols('s', real=True)
l2, l3, alpha, c = sp.symbols('lambda2 lambda3 alpha c', real=True)
kap = sp.Function('kappa', real=True)(s)
tau = sp.Function('tau', real=True)(s)
def D(f):
    fT, fN, fB = f
    return sp.Matrix([sp.diff(fT, s) - kap * fN, sp.diff(fN, s) + kap * fT - tau * fB, sp.diff(fB, s) + tau * fN])
I = sp.Matrix([l2, 0, l3 * kap])
ez = sp.Matrix([sp.Function('zT')(s), sp.Function('zN')(s), sp.Function('zB')(s)])
# D I = e_z x T  (Frenet components of e_z x T = (0, zB, -zN))
eqI = D(I) - sp.Matrix([0, ez[2], -ez[1]])
solZ = sp.solve([eqI[1], eqI[2]], [ez[1], ez[2]], dict=True)[0]
print("(R1) zN =", solZ[ez[1]], ",  zB =", sp.factor(solZ[ez[2]]))
zN, zB = solZ[ez[1]], solZ[ez[2]]
# D e_z = 0, T-component: zT' = kappa zN  =>  zT = c - l3 kappa^2/2
zT = c - l3 * kap**2 / 2
print("(R2) zT' - kappa zN =", sp.simplify(sp.diff(zT, s) - kap * zN))
# e_z . I = alpha  =>  tau
tau_sol = sp.solve(sp.Eq(l2 * zT + l3 * kap * zB, alpha), tau)[0]
C = (l2 * c - alpha) / l3**2
print("(R3) tau =", sp.simplify(tau_sol), " == l2/(2 l3) + C/kappa^2 :", sp.simplify(tau_sol - (l2 / (2 * l3) + C / kap**2)) == 0)
# remaining Frenet equations for e_z: N- and B-components of D e_z = 0 must hold given (R5)
q = sp.symbols('q', positive=True)
p2 = 4 * c / l3 - l2**2 / l3**2; p1 = 4 * (1 - c**2) / l3**2 + 4 * l2 * C / l3; p0 = -4 * C**2
P = -q**3 + p2 * q**2 + p1 * q + p0
# |e_z|^2 = 1  <=>  zT^2 + zN^2 + zB^2 = 1 ; with kappa'^2 = q'^2/(4q) this is q'^2 = P(q):
norm = (zT**2 + zN**2 + zB**2 - 1).subs(tau, tau_sol)
qp = sp.symbols('qp')   # q'
norm_q = sp.simplify(norm.subs(sp.diff(kap, s), qp / (2 * sp.sqrt(q))).subs(kap, sp.sqrt(q)))
print("(R5) |e_z|^2 = 1  <=>  q'^2 = P(q):", sp.simplify(sp.solve(norm_q, qp**2)[0] - P) == 0)
# B-component of D e_z = 0 : zB' + tau zN = 0  identically (given tau of R3)
eB = sp.simplify((sp.diff(zB, s) + tau * zN).subs(tau, tau_sol).doit())
print("     zB' + tau zN = 0 identically:", sp.simplify(eB) == 0)
# N-component: zN' + kappa zT - tau zB = 0  <=> kappa'' = ... ; check it is implied by q'^2 = P (differentiate)
eN = (sp.diff(zN, s) + kap * zT - tau * zB).subs(tau, tau_sol)
# express with q: kappa = sqrt(q), kappa' = q'/(2 sqrt q), kappa'' = q''/(2 sqrt q) - q'^2/(4 q^{3/2}),  q'' = P'(q)/2
qpp = sp.diff(P, q) / 2
eN_q = eN.subs(sp.diff(kap, s, 2), qpp / (2 * sp.sqrt(q)) - P / (4 * q**sp.Rational(3, 2))).subs(sp.diff(kap, s), sp.sqrt(P) / (2 * sp.sqrt(q))).subs(kap, sp.sqrt(q))
print("     zN' + kappa zT - tau zB = 0 given q'^2 = P(q):", sp.simplify(eN_q) == 0)
print("(R4) r^2 = |I|^2 - alpha^2 = l2^2 + l3^2 kappa^2 - alpha^2  (|I|^2 = alpha^2 + r^2 for the screw field)")
print("     P(q) = ", sp.collect(sp.expand(P), q))

# ------------------------------------------------------------------ numerical exact (1,2) rod (handoff §7.1 recipe)
lam3 = -0.6283; Rmax = 1.2126

def rod_rhs(t, y, al):
    x, T = y[:3], y[3:]
    Ivec = np.array([-x[1], x[0], al])          # alpha e_z + e_z x x
    return np.concatenate([T, np.cross(T, Ivec) / lam3])

def kappa_of(y, al):
    x, T = y[:3], y[3:]
    Ivec = np.array([-x[1], x[0], al])
    acc = np.cross(T, Ivec) / lam3
    return np.linalg.norm(acc)

def shoot(pars, want_solution=False):
    al, bet = pars
    y0 = np.array([Rmax, 0, 0, 0, np.cos(bet), np.sin(bet)])
    # integrate until phi = pi/2 (event), then evaluate d(kappa)/ds and Z there
    ev = lambda t, y, al: np.arctan2(y[1], y[0]) - np.pi / 2
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(rod_rhs, (0, 10), y0, args=(al,), events=ev, rtol=1e-12, atol=1e-13, dense_output=True)
    t1 = sol.t_events[0][0]
    y1 = sol.sol(t1)
    h = 1e-5
    dk = (kappa_of(sol.sol(t1 + h), al) - kappa_of(sol.sol(t1 - h), al)) / (2 * h)
    if want_solution:
        return sol, t1
    return [dk, y1[2]]

pars = fsolve(shoot, [0.56, -0.34], xtol=1e-13)
al, bet = pars
sol, tq = shoot(pars, True)
L = 4 * tq
print(f"\nExact (1,2) rod: alpha = {al:.6f}, beta = {bet:.6f}, quarter length = {tq:.6f}, L = {L:.5f} (handoff: 0.561636, -0.340808, L=6.65696)")
full = solve_ivp(rod_rhs, (0, L), np.array([Rmax, 0, 0, 0, np.cos(bet), np.sin(bet)]), args=(al,), rtol=1e-12, atol=1e-13, dense_output=True)
yL = full.sol(L); print("closure |x(L)-x(0)|, |T(L)-T(0)|:", np.linalg.norm(yL[:3] - full.sol(0)[:3]), np.linalg.norm(yL[3:] - full.sol(0)[3:]))

ss = np.linspace(0, L, 2001)[:-1]
Y = full.sol(ss)
X_, T_ = Y[:3].T, Y[3:].T
Ivec = np.stack([-X_[:, 1], X_[:, 0], al * np.ones(len(ss))], 1)
acc = np.cross(T_, Ivec) / lam3
kap_n = np.linalg.norm(acc, axis=1)
N_ = acc / kap_n[:, None]
Bn_ = np.cross(T_, N_)
# tau from the ODE: acc' = d/ds (T x I)/l3 = (acc x I + T x (e_z x T))/l3 ; tau = Bn . N' /1 = Bn.(acc'/kappa)  (N' = -kappa T + tau Bn)
ez = np.array([0, 0, 1.0])
accp = (np.cross(acc, Ivec) + np.cross(T_, np.cross(ez, T_))) / lam3
tau_n = np.einsum('ij,ij->i', Bn_, accp) / kap_n
lam2_n = np.einsum('ij,ij->i', T_, Ivec)
lam3_n = np.einsum('ij,ij->i', Bn_, Ivec) / kap_n
print(f"T.I = lambda2: mean {lam2_n.mean():.9f}, std {lam2_n.std():.1e};  Bn.I/kappa = lambda3: mean {lam3_n.mean():.6f}, std {lam3_n.std():.1e};  N.I rms {np.sqrt((np.einsum('ij,ij->i', N_, Ivec)**2).mean()):.1e}")
lam2v = lam2_n.mean()
lam3F = lam3_n.mean()     # Frenet-relation lambda3 = Bn.I/kappa (= -ODE parameter, since T x I = -lambda3 kappa N)
print(f"NOTE: T x I = -lambda3 kappa N, so the ODE parameter {lam3} is -lambda3; Frenet lambda3 = {lam3F:.6f} is used below.")
lam3 = lam3F
qn = kap_n**2
zs = T_[:, 2]
# (R2): z_s = c - l3 q/2
cfit = np.mean(zs + lam3 * qn / 2); print(f"(R2) z_s - (c - l3 q/2): c = {cfit:.9f}, residual max {np.max(np.abs(zs - (cfit - lam3 * qn / 2))):.1e}")
# (R3): tau = l2/(2 l3) + C/q,  C = (l2 c - alpha)/l3^2
Cn = (lam2v * cfit - al) / lam3**2
print(f"(R3) tau - (l2/(2l3) + C/q): C = {Cn:.9f}, residual max {np.max(np.abs(tau_n - (lam2v / (2 * lam3) + Cn / qn))):.1e}")
# (R4)
print(f"(R4) r^2 - (l2^2 - alpha^2 + l3^2 q): residual max {np.max(np.abs(X_[:,0]**2 + X_[:,1]**2 - (lam2v**2 - al**2 + lam3**2 * qn))):.1e}")
# (R5)
p2n = 4 * cfit / lam3 - lam2v**2 / lam3**2; p1n = 4 * (1 - cfit**2) / lam3**2 + 4 * lam2v * Cn / lam3; p0n = -4 * Cn**2
qp_n = np.gradient(qn, ss)   # spectral would be better; use FFT derivative
qhat = np.fft.rfft(qn); kk = 2 * np.pi * np.fft.rfftfreq(len(ss), d=ss[1] - ss[0])
qp_n = np.fft.irfft(1j * kk * qhat, n=len(ss))
Pn = -qn**3 + p2n * qn**2 + p1n * qn + p0n
print(f"(R5) q'^2 - P(q): residual max {np.max(np.abs(qp_n**2 - Pn)):.1e}  (max q'^2 = {np.max(qp_n**2):.3f})")
roots = np.sort(np.roots([-1, p2n, p1n, p0n]))
print(f"     roots of P: q3 = {roots[0]:.6f} (<0), q1 = {roots[1]:.6f} = kappa_min^2 ({qn.min():.6f}), q2 = {roots[2]:.6f} = kappa_max^2 ({qn.max():.6f})")
print(f"     p2 = q1+q2+q3 = {p2n:.6f}, p1 = {p1n:.6f}, p0 = {p0n:.6f}")
print(f"     rod moduli in this parametrisation: lambda2 = {lam2v:.6f}, lambda3 = {lam3:.6f}, alpha = {al:.6f}, c = {cfit:.6f}; tau0 = l2/(2 l3) = {lam2v/(2*lam3):.6f}, C = {Cn:.6f}")
print(f"     kappa range [{kap_n.min():.4f}, {kap_n.max():.4f}], tau range [{tau_n.min():.4f}, {tau_n.max():.4f}], z_s at kappa max {zs[0]:.4f}, at kappa min {zs[np.argmin(kap_n)]:.4f}")

# ------------------------------------------------------------------ which Killing fields xi have N.xi = 0 along the rod?
# xi = a + b x x ; N.xi = N.a + b.(x x N): 6 functions of s; rank of their Gram matrix
F = np.concatenate([N_, np.cross(X_, N_)], 1)          # rows: s samples, cols: coefficients of (a, b)
sv = np.linalg.svd(F, compute_uv=False)
print("\nKilling fields with N.xi = 0 along the rod: singular values of the 6-column sample matrix:", np.array2string(sv, precision=3))
print("   => null space dimension 1 (only the screw I = alpha e_z + e_z x x):", sv[-2] / sv[0] > 1e-3 and sv[-1] / sv[0] < 1e-6)
print("z_s at the curvature extrema (must vanish for a separable V_h(x,y)+V_z(z) to have the rod as orbit):", zs[0], zs[np.argmin(kap_n)])
np.save('rod12.npy', {'alpha': al, 'beta': bet, 'L': L, 'lambda2': lam2v, 'lambda3': lam3, 'c': cfit, 'C': Cn, 'roots': roots,
                      's': ss, 'kappa': kap_n, 'tau': tau_n, 'zs': zs, 'x': X_, 'T': T_}, allow_pickle=True)
