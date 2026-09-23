"""Strong QS as a strain-kernel condition, and its use on conjugated circle actions u = Psi_* d_phi.

(A) Identity (any u, B in R^3, g = Euclidean):   L_u(B^flat) = (L_u g)(B, .) + ([u,B])^flat,
    (L_u g)_ij = d_i u_j + d_j u_i = 2 S_ij,  S = sym(grad u).
    Hence weak QS ([u,B] = 0) + strong QS (L_u B^flat = 0)  <=>  S(u) B = 0.
    Since tr S = div u = 0, S has eigenvalues (mu, -mu, 0): wherever u is not infinitesimally Killing (S != 0)
    the direction of B is FIXED by u alone:  B || ker S(u).   Necessary condition on u:  det S(u) = 0.
    (RHB 2020: weak => strong for MHS with nested surfaces and isotropic p, so this is a condition on every
    QS-MHS symmetry vector.)

(B) Conjugated rotations u = Psi_* d_phi for polynomial volume-preserving shears:
    (i)   vertical shear   Psi_z: (x,y,z) -> (x, y, z + f(x,y)):    u = d_phi + (d_phi f) d_z
          det S = 0 automatically, ker S = e_z x grad(h), h := d_phi f  =>  B horizontal, tangent to level
          curves of h  =>  psi = psi(h, z); u.grad psi = 0 forces d_phi h = h Q(h) along every circle, which is
          incompatible with periodicity unless d_phi h = 0  =>  h = h(R)  =>  cylinders or u Killing. DEAD.
    (ii)  horizontal shear Psi_x: (x,y,z) -> (x + g(y), y, z):  u is a planar field => ker S = e_z
          => B vertical, straight field lines => no toroidal surfaces.  DEAD.  (For g = g(y,z), det S = 0 is a
          genuine constraint, printed.)
    (iii) (a) linear Psi in SL(3): S constant => B || fixed vector => straight lines. DEAD.
          (b) Psi_x o Psi_z with quadratic f, g: det S = 0 is NOT automatic (degree-11 polynomial identity,
          197 coefficient equations); classification left open (see notes).

Run:  python3 strain_kernel_conjugations.py     (~1 min)
"""
import sympy as sp
import time
t0 = time.time()
x, y, z = X = sp.symbols('x y z', real=True)
grad = lambda f: sp.Matrix([sp.diff(f, v) for v in X])
def jac(u):  # J_ij = d_j u_i
    return sp.Matrix(3, 3, lambda i, j: sp.diff(u[i], X[j]))
def lie_bracket(u, v):  # [u,v] = u.grad v - v.grad u
    return jac(v)*u - jac(u)*v

# ---------------- (A) the identity ----------------
u = sp.Matrix([sp.Function(f'u{i}')(*X) for i in range(3)])
B = sp.Matrix([sp.Function(f'B{i}')(*X) for i in range(3)])
Ju = jac(u)
LuBflat = jac(B)*u + Ju.T*B            # (L_u B^flat)_i = u.grad B_i + B_j d_i u_j
S = (Ju + Ju.T)/2
lhs = LuBflat - (2*S*B + lie_bracket(u, B))
print("(A) L_u B^flat - [2 S B + [u,B]^flat] == 0 :", sp.simplify(lhs) == sp.zeros(3, 1))

# ---------------- (B) pushed-forward rotations ----------------
dphi = sp.Matrix([-y, x, 0])
def pushforward_dphi(Psi, Psi_inv):
    """u(X) = DPsi(Psi^{-1} X) . d_phi(Psi^{-1} X)."""
    DPsi = sp.Matrix(3, 3, lambda i, j: sp.diff(Psi[i], X[j]))
    pre = dict(zip(X, Psi_inv))
    return sp.simplify((DPsi*dphi).subs(pre, simultaneous=True))

# (i) vertical shear with generic f
f = sp.Function('f')(x, y)
Psi_z = sp.Matrix([x, y, z + f]); Psi_z_inv = sp.Matrix([x, y, z - f])
u_z = pushforward_dphi(Psi_z, Psi_z_inv)
S_z = (jac(u_z) + jac(u_z).T)/2
h = dphi.dot(grad(f))                       # d_phi f
kern = sp.Matrix([0, 0, 1]).cross(grad(h))  # e_z x grad h
print("(B-i)  u = d_phi + (d_phi f) d_z :", list(u_z))
print("       div u == 0 :", sp.simplify(sum(sp.diff(u_z[i], X[i]) for i in range(3))) == 0)
print("       det S == 0 identically :", sp.simplify(S_z.det()) == 0)
print("       S (e_z x grad d_phi f) == 0 :", sp.simplify(S_z*kern) == sp.zeros(3, 1))
print("       S != 0 unless d_phi f const (S_13, S_23) =", sp.simplify(S_z[0, 2]), ",", sp.simplify(S_z[1, 2]))

# (ii) horizontal shear with generic g(y,z): u is horizontal (u_z = 0) but depends on z, so S e_z != 0 in general
#      and det S = -(S11 S23^2 - 2 S12 S13 S23 + S22 S13^2) is a genuine constraint.  If g = g(y) (z-independent
#      shear) u is a planar field, ker S = e_z, B vertical: DEAD.
g = sp.Function('g')(y, z)
Psi_x = sp.Matrix([x + g, y, z]); Psi_x_inv = sp.Matrix([x - g, y, z])
u_x = pushforward_dphi(Psi_x, Psi_x_inv)
S_x = (jac(u_x) + jac(u_x).T)/2
print("(B-ii) u = Psi_x* d_phi :", list(u_x))
print("       u_z == 0 :", sp.simplify(u_x[2]) == 0, "| det S =", sp.factor(sp.simplify(S_x.det())), " (not identically 0 for z-dependent g)")
g1 = sp.Function('g1')(y)
u_x1 = pushforward_dphi(sp.Matrix([x + g1, y, z]), sp.Matrix([x - g1, y, z]))
S_x1 = (jac(u_x1) + jac(u_x1).T)/2
print("       g = g(y): S e_z == 0 :", sp.simplify(S_x1*sp.Matrix([0, 0, 1])) == sp.zeros(3, 1),
      "| S == 0 only if g1' = 0: S_11 =", sp.simplify(S_x1[0, 0]))

# (iii) compositions.  (a) LINEAR volume-preserving Psi (M in SL(3), not orthogonal): u = M (e_z x M^-1 X) is linear,
# S is a CONSTANT matrix, so B || fixed vector k: straight parallel field lines, no toroidal surfaces.  Verified:
M = sp.Matrix([[1, 0, sp.Rational(1, 2)], [0, 1, 0], [sp.Rational(1, 3), sp.Rational(-1, 4), 1]])  # det = 1 - 1/6 = 5/6 -> normalise
M = M/M.det()**sp.Rational(1, 3)
u_lin = M*dphi.subs(dict(zip(X, list(M.inv()*sp.Matrix(X)))), simultaneous=True)
S_lin = sp.simplify((jac(u_lin) + jac(u_lin).T)/2)
print("(B-iii-a) linear Psi: S constant :", all(sp.diff(S_lin[i, j], v) == 0 for i in range(3) for j in range(3) for v in X),
      "| det S =", sp.nsimplify(sp.simplify(S_lin.det())), "(generically != 0 => no strong-QS B at all)")
# (b) Psi = Psi_x o Psi_z with quadratic f, g: det S is a polynomial identity of degree 11 in (x,y,z) with 197
# coefficient equations in the 10 shear coefficients (computed; classification NOT completed within the time box).
# Generic instance: det S != 0, so det S = 0 is a genuine restriction on the conjugation.
fq = x*y + sp.Rational(1, 2)*y**2 + x; gq = lambda Y, Z: Y*Z - Z**2 + 2*Y
Psi = sp.Matrix([x + gq(y, z + fq), y, z + fq])
Psi_inv = sp.Matrix([x - gq(y, z), y, z - fq.subs(x, x - gq(y, z))])
assert sp.expand(Psi.subs(dict(zip(X, Psi_inv)), simultaneous=True) - sp.Matrix(X)) == sp.zeros(3, 1)
u_c = sp.expand((sp.Matrix(3, 3, lambda i, j: sp.diff(Psi[i], X[j]))*dphi).subs(dict(zip(X, Psi_inv)), simultaneous=True))
S_c = (jac(u_c) + jac(u_c).T)/2
detS = sp.expand(S_c.det())
print("(B-iii-b) generic quadratic composition: div u == 0 :", sp.expand(sum(sp.diff(u_c[i], X[i]) for i in range(3))) == 0,
      "| det S == 0 identically :", detS == 0, f"| degree {sp.Poly(detS, x, y, z).total_degree()}")
print(f"done in {time.time()-t0:.1f} s")
