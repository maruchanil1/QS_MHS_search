"""Resonant case iota = 0, N = 1 (closed field lines, D = d_phi; D_j v^j = 0): the J = 3, D = 1 hierarchy (task3_J3.py 0)
leaves the branch  X_1 = g v e_z, X_-1 = gb/v e_z  (z gains a mode e^{i theta} = v E invisible to W = x_phi).
Here: (i) re-derive the branch, (ii) impose the remaining E^0 condition W.x_chi = kappa W.W and the reality conditions,
(iii) describe the surface and test embeddedness (self-intersection) numerically.
Run: python3 task3_resonant_branch.py
"""
import sympy as sp, sys, time
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from vE_surface import v, E, eps, epsb, ez, surface_equations, frame_vector, laurent_poly, laurent
t0 = time.time()
io, Nn = sp.Integer(0), 1
kap = sp.Symbol('kappa')
lam, lam_s = laurent_poly('lam', 1); lamb, lamb_s = laurent_poly('lamb', 1)
ga0, ga0_s = laurent_poly('ga0', 1)
g, gb, c0, cb0 = sp.symbols('g gb c0 cb0')
# the branch found by task3_J3.py 0:  X_3 = lam eps, X_-3 = lamb epsbar, X_2 = X_-2 = 0, X_1 = g v e_z, X_-1 = gb/v e_z,
# X_0 = c0 eps + cb0 epsbar + ga0(v) e_z
X = {3: lam*eps, -3: lamb*epsb, 1: g*v*ez, -1: gb/v*ez, 0: c0*eps + cb0*epsb + ga0*ez}
c2, c1, WW, Wx = surface_equations(X, io, Nn, kappa=kap)
nz = lambda eqs: [(k, m, sp.factor(cc)) for (k, m, cc) in eqs if sp.expand(cc) != 0]
print("E^k (k != 0) equations of C2:", nz([e for e in c2 if e[0] != 0]))
print("E^k (k != 0) equations of C1:", nz([e for e in c1 if e[0] != 0]))
e0 = nz([e for e in c1 if e[0] == 0])
print("E^0 condition W.x_chi = kappa W.W, v-modes:")
for k, m, cc in e0:
    print(f"   v^{m}: {cc}")
sol = sp.solve([cc for _, _, cc in e0], ga0_s + [kap], dict=True)
print("solve for (gamma_0 coefficients, kappa):", sol)
# reality: lamb_m = conj(lam_{-m}), gb = conj(g), cb0 = conj(c0), ga0_{-m} = conj(ga0_m): impose with real/imag parts
print("\nInterpretation: with iota = 0 the field lines are the phi-curves; W = x_phi does not see any z = Z(theta):")
x = sum((X[j]*E**j for j in X), sp.zeros(3, 1))
W = sp.I*E*x.diff(E) + (io - Nn)*sp.I*v*x.diff(v)          # iota - N = -1: W = x_u - x_chi = x_phi at fixed theta
print("   W_z =", sp.factor(sp.expand(W[2])), "   (the g-modes drop out; vanishes iff gamma_0' = 0)")
# E^0 condition with gamma_0' = 0 (forced by E^+-1 when g != 0):
e0c = [sp.factor(cc.subs({ga0_s[0]: 0, ga0_s[2]: 0})) for _, _, cc in e0]
print("   E^0 with gamma_0' = 0:", e0c)
print("   => kappa = I/G = 1/8 from v^{+-2}; with reality lamb_m = conj(lam_{-m}): 2 lam_0 conj(lam_{-1}) = lam_1 conj(lam_0)/2 and"
      " -(9/8)|lam_0|^2 + (3/2)|lam_1|^2 - 6|lam_{-1}|^2 = 0  (e.g. lam_0 = 0, |lam_1| = 2|lam_{-1}|): the branch SURVIVES C1 + C2.")
# The surface for gamma_0 = const:  x+iy = c + E^3 lam(v), z = z0 + 2 Re(g e^{i theta}):
# field line at label theta: planar closed curve at height z(theta), rotated by 3 theta.  Two field lines at the same
# height are rotated copies of one closed curve about the same centre => they intersect => immersed, not embedded.
import numpy as np
lam_n = {lam_s[0]: sp.Rational(1, 5), lam_s[1]: sp.Integer(1), lam_s[2]: sp.Rational(1, 4)}     # lam_{-1}, lam_0, lam_1
th = np.linspace(0, 2*np.pi, 400, endpoint=False)
def curve(theta, ph):
    vv = np.exp(1j*(theta - ph)); EE = np.exp(1j*ph)
    lamv = float(lam_n[lam_s[0]])/vv + float(lam_n[lam_s[1]]) + float(lam_n[lam_s[2]])*vv
    return EE**3*lamv
def z_of(theta): return np.cos(theta)
# heights equal for theta and -theta (Z = cos theta); curves at theta = 1 and theta = -1
c_a, c_b = curve(1.0, th), curve(-1.0, th)
d = np.abs(c_a[:, None] - c_b[None, :])
i, j = np.unravel_index(np.argmin(d), d.shape)
print(f"   numeric: field lines at theta = +-1 lie in the same plane z = cos(1); min distance between them = {d.min():.2e}"
      f" (an intersection => the torus is immersed, not embedded).  Turning number of the field line: ",
      int(round((np.angle(np.diff(c_a))[1:] - np.angle(np.diff(c_a))[:-1] + np.pi) % (2*np.pi) - np.pi).sum()/(2*np.pi) + 1) if False else
      int(round(np.sum(np.diff(np.unwrap(np.angle(np.diff(np.append(c_a, c_a[0])))))) / (2*np.pi))))
print(f"done in {time.time()-t0:.0f} s")
