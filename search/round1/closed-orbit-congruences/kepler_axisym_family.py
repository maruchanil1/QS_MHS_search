"""The unique surviving Kepler branch (a' = 0): an exact AXISYMMETRIC force-free (p = const) field
whose field lines are congruent Kepler ellipses, all with the same semi-major axis a, focus at the
origin, line of apsides in the equatorial plane, inclination i(psi) with sin i = eps/(C sqrt(1-eps^2)).

x(psi,s,eta) = Rz(s) Rx(i(psi)) e(eta),  e = a(cos eta - eps, sqrt(1-eps^2) sin eta, 0),
t = sqrt(a^3/mu)(eta - eps sin eta),  B = d_t x,  V = -mu/rho,  E = -mu/(2a) for all tori.

Checks: (1) symbolic: d_eta J = 0 for J = det(d_psi x, d_s x, d_t x)  (=> div B = 0);
        (2) numeric, independent of the framework: invert x(psi,s,eta) by Newton, finite-difference
            div B and curl B at a few points, check div B = 0 and (curl B) x B = 0 (force-free).
Run: python3 kepler_axisym_family.py
"""
import sympy as sp
import numpy as np
from scipy.optimize import fsolve

psi, s, eta = sp.symbols('psi s eta', real=True)
a, mu, C = sp.symbols('a mu C', positive=True)
eps = sp.Function('epsilon')(psi)
beta = sp.sqrt(1 - eps**2)
sin_i = eps/(C*beta); cos_i = sp.sqrt(1 - sin_i**2)
def Rz(t): return sp.Matrix([[sp.cos(t), -sp.sin(t), 0], [sp.sin(t), sp.cos(t), 0], [0, 0, 1]])
Rx_i = sp.Matrix([[1, 0, 0], [0, cos_i, -sin_i], [0, sin_i, cos_i]])
e = sp.Matrix([a*(sp.cos(eta) - eps), a*beta*sp.sin(eta), 0])
x = Rz(s)*Rx_i*e
dt_deta = sp.sqrt(a**3/mu)*(1 - eps*sp.cos(eta))
xt = x.diff(eta)/dt_deta                       # B = d_t x at fixed (psi, s)
J = sp.Matrix.hstack(x.diff(psi), x.diff(s), xt).det()
J = sp.simplify(J)
print("J =", sp.factor(J))
print("d_eta J = 0:", sp.simplify(sp.diff(J, eta)) == 0)

# ---------------- numeric independent check ----------------
aN, muN, CN = 1.0, 1.0, 2.0
def epsN(p): return 0.1 + 0.3*p          # eps(psi) monotone => eps' != 0
xf = sp.lambdify((psi, s, eta), x.subs({a: aN, mu: muN, C: CN, eps: 0.1 + 0.3*psi}), 'numpy')
Bf = sp.lambdify((psi, s, eta), xt.subs({a: aN, mu: muN, C: CN, eps: 0.1 + 0.3*psi}), 'numpy')
def B_at(X):
    """B at spatial point X by inverting the parametrisation (Newton from a nearby guess)."""
    def res(q): return (xf(*q).ravel() - X)
    q0 = B_at.guess
    q = fsolve(res, q0, xtol=1e-13)
    B_at.guess = q
    return Bf(*q).ravel()
rng = np.random.default_rng(1)
for trial in range(4):
    q0 = np.array([rng.uniform(0.3, 0.8), rng.uniform(0, 6.28), rng.uniform(0, 6.28)])
    X0 = xf(*q0).ravel()
    h = 1e-4
    B_at.guess = q0
    grad = np.zeros((3, 3))     # grad[i,j] = d B_i / d x_j
    for j in range(3):
        dX = np.zeros(3); dX[j] = h
        B_at.guess = q0; Bp = B_at(X0 + dX)
        B_at.guess = q0; Bm = B_at(X0 - dX)
        grad[:, j] = (Bp - Bm)/(2*h)
    B_at.guess = q0; B0 = B_at(X0)
    divB = np.trace(grad)
    curlB = np.array([grad[2, 1] - grad[1, 2], grad[0, 2] - grad[2, 0], grad[1, 0] - grad[0, 1]])
    JxB = np.cross(curlB, B0)
    rho = np.linalg.norm(X0)
    print(f"point {trial}: |B|={np.linalg.norm(B0):.4f}  divB={divB:.2e}  |curlB|={np.linalg.norm(curlB):.3f}  "
          f"|JxB|/(|J||B|)={np.linalg.norm(JxB)/(np.linalg.norm(curlB)*np.linalg.norm(B0)):.2e}  "
          f"B^2/2 + V - E = {0.5*B0@B0 - muN/rho + muN/(2*aN):.2e}")
print("Expected: divB ~ 1e-7 (FD error), |JxB| ~ 1e-7, energy residual ~ 0  => exact axisymmetric force-free Kepler field")
