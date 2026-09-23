"""Parity obstruction for isotropic-oscillator congruences (orchestrator seed check).

If every field line is a centred ellipse x(t) = A cos t + B sin t (all bounded orbits of V = |x|^2/2),
with A, B depending on the torus label psi and the orbit label s, then the Jacobian
J = det(d_psi x, d_s x, d_t x) is a cubic trigonometric polynomial in t containing only odd harmonics.
Hence its mean over a period vanishes, so J changes sign: the coordinates (psi, s, t) are singular and
div B = 0 (which needs d_t J = 0) cannot hold with J != 0.  Run: python3 parity_check.py
"""
import sympy as sp

t = sp.symbols('t')
A = sp.Matrix(sp.symbols('a1:4'));   B = sp.Matrix(sp.symbols('b1:4'))
Ap = sp.Matrix(sp.symbols('ap1:4')); Bp = sp.Matrix(sp.symbols('bp1:4'))   # d/dpsi of A, B
As = sp.Matrix(sp.symbols('as1:4')); Bs = sp.Matrix(sp.symbols('bs1:4'))   # d/ds of A, B
x = A*sp.cos(t) + B*sp.sin(t)
xpsi = Ap*sp.cos(t) + Bp*sp.sin(t)
xs = As*sp.cos(t) + Bs*sp.sin(t)
xt = sp.diff(x, t)
J = sp.expand(sp.Matrix.hstack(xpsi, xs, xt).det())
mean = sp.simplify(sp.integrate(J, (t, 0, 2*sp.pi)))
print("mean of J over one period:", mean)
# Check the half-period antisymmetry directly
print("J(t+pi) + J(t) == 0:", sp.simplify(J.subs(t, t+sp.pi) + J) == 0)
