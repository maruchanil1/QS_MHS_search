"""Algebraic-B formulation of QS-MHS, derived and verified on a Solov'ev equilibrium.

Mechanical formulation: field lines are orbits of  xdd = -grad V,  xd = B,  V = -Pi = -(p + |B|^2/2),
E(psi) = -p(psi) is the orbit energy, and QS <=> the clock  Vd^2 = F(psi, V)  with Vd := B.grad V.

CLAIM (algebraic-B):  a smooth B on a solid torus with nested surfaces psi is QS-MHS  <=>  there are
scalar fields V, psi and functions E(psi), F(psi,V) such that, POINTWISE,

   (A1)  |B|^2       = 2 (E(psi) - V)                      (energy; p = -E)
   (A2)  B . grad psi = 0                                  (surfaces)
   (A3)  B . grad V   = s sqrt(F(psi,V)),  s = +-1         (QS clock, built in)

and, for the B so determined (two-fold, see (R) below), the residual system

   (a)  div B = 0
   (b)  B . Hess(psi) . B = grad V . grad psi              (normal force balance = Maupertuis normal-curvature law)
   (c)  B . Hess(V) . B   = F_V(psi,V)/2 + |grad V|^2      (time derivative of the clock)

holds.  Proof of the residual system: let e := B.grad B + grad V (force-balance error).  Identities
   e.B      = B.grad(|B|^2/2 + V) = B.grad E = 0                 (from A1, A2)
   e.grad psi = B.grad(B.grad psi) - [B Hess(psi) B - grad V.grad psi] = -[B Hess psi B - grad V.grad psi]
   e.grad V   = B.grad(B.grad V)   - [B Hess(V) B - |grad V|^2]      and  B.grad(B.grad V) = d/dt(Vd) = F_V/2 by (A3)
so e = 0  <=>  (b) and (c)  whenever grad V is not in span(B, grad psi) (true off the |B|-contour direction).
Hence QS-MHS  <=>  (A1)-(A3) + (a),(b),(c):  THREE scalar equations for the TWO scalar fields (V, psi)
(+ free functions E, F): the Garren-Boozer +1 overdetermination, with only (a) genuinely differential in B.

Reconstruction (R): with n = grad psi, m = grad V, c = s sqrt F, the line {b.n = 0, b.m = c} meets the sphere
|b|^2 = 2(E-V) in
   B = c |n|^2 (m - (m.n) n/|n|^2)/|n x m|^2  +- sqrt( 2(E-V) - F |n|^2/|n x m|^2 ) (n x m)/|n x m|.

Solov'ev test: psi = A(R^2-R0^2)^2/8 + C R^2 z^2/2 + D z^2/2,  Delta* psi = (A+C) R^2 + D,
p = p0 - (A+C) psi,  F_pol^2 = F0^2 - 2 D psi,  B = grad psi x grad phi + F_pol grad phi.
Run:  python3 algebraicB_solovev.py      (~30 s)
"""
import sympy as sp
import time

t0 = time.time()
x, y, z = X = sp.symbols('x y z', real=True)
s = sp.symbols('s', positive=True)        # stands for F_pol = sqrt(F0^2 - 2 D psi)

grad = lambda f: sp.Matrix([sp.diff(f, v) for v in X])
hess = lambda f: sp.hessian(f, X)
def div(F): return sum(sp.diff(F[i], X[i]) for i in range(3))
def curl(F):
    return sp.Matrix([sp.diff(F[2], y) - sp.diff(F[1], z),
                      sp.diff(F[0], z) - sp.diff(F[2], x),
                      sp.diff(F[1], x) - sp.diff(F[0], y)])

# ---------------- Solov'ev equilibrium (rational parameters) ----------------
A, C, D, R0, F0, p0 = sp.Rational(1), sp.Rational(1, 2), sp.Rational(1, 3), sp.Integer(1), sp.Integer(3), sp.Integer(2)
R2 = x**2 + y**2
psi = A*(R2 - R0**2)**2/8 + C*R2*z**2/2 + D*z**2/2
Q = F0**2 - 2*D*psi                     # F_pol^2 as a polynomial
p = p0 - (A + C)*psi
gradphi = sp.Matrix([-y, x, 0])/R2
Fpol = sp.sqrt(Q)
B = grad(psi).cross(gradphi) + Fpol*gradphi

def red(expr):
    """Replace sqrt(Q) by s and reduce modulo s^2 - Q; return (a, b) with expr = a + b s (a, b rational)."""
    e = sp.together(sp.expand(expr.subs(sp.sqrt(Q), s)))
    num, den = sp.fraction(e)
    num = sp.expand(num)
    rem = sp.rem(sp.Poly(num, s), sp.Poly(s**2 - Q, s)).as_expr()
    rem = sp.expand(rem)
    a = sp.cancel(rem.coeff(s, 0)/den)
    b = sp.cancel(rem.coeff(s, 1)/den)
    return a, b
def is_zero(expr):
    a, b = red(expr)
    return sp.simplify(a) == 0 and sp.simplify(b) == 0

# force balance and div
J = curl(B)
fb = J.cross(B) - grad(p)
print("[Solov'ev] J x B - grad p == 0 :", all(is_zero(fb[i]) for i in range(3)))
print("[Solov'ev] div B == 0          :", is_zero(div(B)))

# mechanical quantities (rational: the sqrt only sits in the phi-component, which is orthogonal to all gradients)
B2 = sp.cancel(sp.expand((B.T*B)[0]).subs(sp.sqrt(Q)**2, Q))
V = -(p + B2/2)
E = -p                                 # E(psi) = -p(psi)
print("(A1) |B|^2 - 2(E - V) == 0     :", sp.simplify(B2 - 2*(E - V)) == 0)
print("(A2) B.grad psi == 0           :", is_zero((B.T*grad(psi))[0]))
Vd = sp.cancel((B.T*grad(V))[0])       # rational (phi-component drops)
print("(A3) Vd = B.grad V is rational :", Vd.has(sp.sqrt(Q)) is False)
# QS clock via Helander's criterion in potential form: grad psi x grad V . grad(Vd) = 0
trip = sp.cancel(grad(psi).cross(grad(V)).dot(grad(Vd)))
print("QS clock: grad psi x grad V . grad(B.grad V) == 0 :", trip == 0)

# ---------------- residual system (a),(b),(c) ----------------
BgradB = sp.Matrix([(B.T*grad(B[i]))[0] for i in range(3)])
e = BgradB + grad(V)
print("(fb) e = B.grad B + grad V == 0 :", all(is_zero(e[i]) for i in range(3)))
cond_b = (B.T*hess(psi)*B)[0] - grad(V).dot(grad(psi))
print("(b)  B Hess(psi) B - grad V.grad psi == 0 :", is_zero(cond_b))
# F_V at fixed psi from grad(Vd^2) = F_psi grad psi + F_V grad V, evaluated in the plane y = 0 (R = x)
Fc = sp.cancel(Vd**2)
gF, gpsi, gV = [g.subs(y, 0) for g in (grad(Fc), grad(psi), grad(V))]
den = sp.cancel(gV[0]*gpsi[2] - gV[2]*gpsi[0])
F_V = sp.cancel((gF[0]*gpsi[2] - gF[2]*gpsi[0])/den)
cond_c = ((B.T*hess(V)*B)[0] - grad(V).dot(grad(V))).subs(y, 0) - F_V/2
print("(c)  B Hess(V) B - |grad V|^2 - F_V/2 == 0 (y=0 plane) :", is_zero(cond_c))

# ---------------- generic identities behind (b), (c) (arbitrary B, psi, V) ----------------
Bg = sp.Matrix([sp.Function(f'b{i}')(*X) for i in range(3)])
psig = sp.Function('Psi')(*X); Vg = sp.Function('W')(*X)
BgB = sp.Matrix([(Bg.T*grad(Bg[i]))[0] for i in range(3)])
id1 = sp.simplify(BgB.dot(grad(psig)) + (Bg.T*hess(psig)*Bg)[0] - (Bg.T*grad((Bg.T*grad(psig))[0]))[0])
id2 = sp.simplify(BgB.dot(grad(Vg)) + (Bg.T*hess(Vg)*Bg)[0] - (Bg.T*grad((Bg.T*grad(Vg))[0]))[0])
print("identity (B.grad B).grad f + B Hess f B = B.grad(B.grad f) for f = psi, V :", id1 == 0, id2 == 0)

# ---------------- reconstruction (R) at random rational points ----------------
def reconstruct(pt, sgn1, sgn2):
    sub = dict(zip(X, pt))
    n = grad(psi).subs(sub); m = grad(V).subs(sub)
    Fv = Fc.subs(sub); Ev = E.subs(sub); Vv = V.subs(sub)
    c = sgn1*sp.sqrt(Fv)
    nxm = n.cross(m); nxm2 = nxm.dot(nxm)
    tang = c*n.dot(n)*(m - m.dot(n)*n/n.dot(n))/nxm2
    rad2 = 2*(Ev - Vv) - Fv*n.dot(n)/nxm2
    return tang + sgn2*sp.sqrt(rad2)*nxm/sp.sqrt(nxm2)
pts = [(sp.Rational(11, 10), sp.Rational(1, 5), sp.Rational(1, 7)),
       (sp.Rational(9, 10), -sp.Rational(3, 10), sp.Rational(1, 4)),
       (sp.Rational(6, 5), sp.Rational(1, 3), -sp.Rational(1, 5))]
for pt in pts:
    Btrue = B.subs(dict(zip(X, pt))).evalf(30)
    best = min(((sp.N((reconstruct(pt, s1, s2) - Btrue).norm(), 20), s1, s2) for s1 in (1, -1) for s2 in (1, -1)),
               key=lambda r: r[0])
    print(f"(R) point {tuple(map(float, pt))}: min_signs |B_rec - B| = {best[0]:.3e}  (signs {best[1]:+d},{best[2]:+d})")
print(f"done in {time.time() - t0:.1f} s")
