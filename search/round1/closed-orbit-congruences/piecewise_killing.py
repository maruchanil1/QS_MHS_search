"""Side lemma (closes a loophole suggested by the Kepler analysis): flux surfaces that are surfaces of
revolution about a psi-DEPENDENT axis N(psi), with B invariant under the rotation about N(psi) on each
surface, cannot be MHS+QS unless N' = 0.

Argument: weak QS + MHS + nested surfaces => strong QS (Rodriguez-Helander-Bhattacharjee), i.e.
(L_u g)(B, .) = 0 with u = N(psi) x x  (rotation about N(psi), tangent to the surfaces).
grad u = N^x + (N' x x) (grad psi)^T  =>  L_u g = w (grad psi)^T + (grad psi) w^T,  w := N' x x,
so (L_u g)(B,.) = (w.B) grad psi  =>  strong QS  <=>  B . w = 0, i.e. B is perpendicular to N' x x.
On the standard torus about z with N' = alpha' xhat, a rotation-invariant tangent field
B = b_phi(theta) e_phi/|e_phi| + b_theta(theta) e_theta/|e_theta| satisfies B.w = 0 for all phi only if
b_phi = b_theta = 0.  Verified symbolically below.
Run: python3 piecewise_killing.py
"""
import sympy as sp

R, r, th, ph, alp = sp.symbols('R r theta phi alpha_p', real=True)
bphi, bth = sp.Function('b_phi')(th), sp.Function('b_theta')(th)
X = sp.Matrix([(R + r*sp.cos(th))*sp.cos(ph), (R + r*sp.cos(th))*sp.sin(ph), r*sp.sin(th)])
e_ph = X.diff(ph); e_th = X.diff(th)
Bvec = bphi*e_ph/sp.sqrt(e_ph.dot(e_ph)) + bth*e_th/sp.sqrt(e_th.dot(e_th))
w = sp.Matrix([1, 0, 0]).cross(X)*alp          # N' x x with N = zhat, N' = alpha' xhat
cond = sp.simplify(Bvec.dot(w))
print("B . w =", sp.factor(cond))
# Fourier in phi
c_cos = sp.simplify(sp.integrate(cond*sp.cos(ph), (ph, 0, 2*sp.pi)))
c_sin = sp.simplify(sp.integrate(cond*sp.sin(ph), (ph, 0, 2*sp.pi)))
print("cos(phi) coefficient:", sp.factor(c_cos), "  => b_phi sin(theta) = 0")
print("sin(phi) coefficient:", sp.factor(c_sin), "  => b_theta (r + R cos(theta)) = 0")
print("=> B = 0 unless alpha' = 0: nested tori of revolution about varying axes with rotation-invariant B are not strongly QS.")
# check the gradient formula for u = N(psi) x x with N = (sin alpha(psi), 0, cos alpha(psi))
x1, x2, x3 = sp.symbols('x1 x2 x3', real=True)
psi = sp.Function('psi')(x1, x2, x3); alpha = sp.Function('alpha')
N = sp.Matrix([sp.sin(alpha(psi)), 0, sp.cos(alpha(psi))])
xx = sp.Matrix([x1, x2, x3])
u = N.cross(xx)
gradu = u.jacobian(xx)
Lg = gradu + gradu.T
wv = N.diff(psi).cross(xx)  # N' x x  (N' = dN/dpsi)
gpsi = sp.Matrix([psi.diff(v) for v in (x1, x2, x3)])
print("L_u g == w gradpsi^T + gradpsi w^T:", sp.simplify(Lg - (wv*gpsi.T + gpsi*wv.T)) == sp.zeros(3, 3))
