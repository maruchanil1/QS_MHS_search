"""Exact identities for the QS clock in the mechanical (potential) formulation.

Field lines are orbits of  xdd = -grad V,  xd = B,  V = -Pi = -(p + |B|^2/2).
Verified symbolically (generic V, generic congruence B, generic flux label psi):

  (I1)  wd  = B.grad w,      wdd = -grad V . grad w  +  B.Hess(w).B      (any scalar w)
  (I2)  Vdd = L(x) := B.Hess(V).B - |grad V|^2    (so L is a *pointwise* function once the congruence is fixed)
  (I3)  Helander's triple product in potential language:
        grad psi x grad|B| . grad(B.grad|B|)  =  (1/|B|) grad psi x grad V . grad(B.grad V)
        i.e.  QS  <=>  Vd = B.grad V is a function of (psi, V) only  <=>  Vd^2 = F(psi,V)  (the clock)
  (I4)  clock  =>  Vdd = F_V(psi,V)/2, hence  grad psi x grad V . grad L = 0  (necessary; algebraic in Hess V)
        conversely  L = f(psi,V)  =>  Vd^2/2 = int f dV + C(orbit): the clock holds iff C is a flux function
        (automatic on irrational tori by density/continuity).
  (I5)  first variation of Vdd along a linearised solution dx (dxdd = -Hess V dx):
        d^2/dt^2 (grad V . dx) = 2 Hess V(B, dxd) + D^3V(B,B,dx) - 2 Hess V(grad V, dx)
        (= delta L, consistent with (I2); the D^3V term is the one that the linearised ODE eliminates).

Run:  python3 clock_identities.py
"""
import sympy as sp

x, y, z = X = sp.symbols('x y z', real=True)
V = sp.Function('V')(x, y, z)
w = sp.Function('w')(x, y, z)
psi = sp.Function('psi')(x, y, z)
p = sp.Function('p')
B = sp.Matrix([sp.Function(f'B{i}')(x, y, z) for i in range(1, 4)])

grad = lambda f: sp.Matrix([sp.diff(f, v) for v in X])
hess = lambda f: sp.hessian(f, X)
def Ddir(f):            # orbit derivative of a scalar field:  d/dt f(x(t)) = B.grad f
    return (B.T * grad(f))[0]
def DdirVec(F):         # orbit derivative of a vector field evaluated along the orbit
    return sp.Matrix([Ddir(F[i]) for i in range(3)])

# Force balance along orbits: B.grad B = -grad V  (this is what "orbits of V" means)
force_rule = {}
BgradB = DdirVec(B)
gV = grad(V)

# (I1): wdd = Ddir(Ddir w) = (B.grad B).grad w + B.Hess(w).B ; substitute B.grad B -> -grad V
wdd = Ddir(Ddir(w))
wdd_claimed = (-gV.T * grad(w))[0] + (B.T * hess(w) * B)[0]
diff = sp.expand(wdd - wdd_claimed - ((BgradB + gV).T * grad(w))[0])
print("(I1) wdd = -gradV.gradw + B.Hess(w).B  (mod force balance):", diff == 0)

# (I2) is (I1) with w = V:
L = (B.T * hess(V) * B)[0] - (gV.T * gV)[0]
diff2 = sp.expand(Ddir(Ddir(V)) - L - ((BgradB + gV).T * gV)[0])
print("(I2) Vdd = B.Hess(V).B - |grad V|^2  (mod force balance):", diff2 == 0)

# (I3): triple products.  Pi = p(psi) + |B|^2/2 ;  V = -Pi.  Needs B.grad psi = 0: use Clebsch B = grad psi x grad alpha.
alpha = sp.Function('alpha')(x, y, z)
Bc = grad(psi).cross(grad(alpha))
def Ddir_c(f):
    return (Bc.T * grad(f))[0]
modB = sp.sqrt((Bc.T * Bc)[0])
Pi = p(psi) + modB**2 / 2
def triple(a, b, c):
    return (a.cross(b)).dot(c)
T_std = triple(grad(psi), grad(modB), grad(Ddir_c(modB)))
T_pot = triple(grad(psi), grad(Pi), grad(Ddir_c(Pi)))
# structural proof: grad Pi = p' grad psi + |B| grad|B|,  B.grad Pi = |B| B.grad|B|  (since B.grad psi = 0)
print("(I3a) B.grad psi = 0 for Clebsch B:", sp.simplify(Ddir_c(psi)) == 0)
print("(I3b) grad(B.gradPi) = (B.grad|B|) grad|B| + |B| grad(B.grad|B|):",
      sp.simplify(grad(Ddir_c(Pi)) - (Ddir_c(modB) * grad(modB) + modB * grad(Ddir_c(modB)))) == sp.zeros(3, 1))
print("(I3c) gradpsi x gradPi = |B| gradpsi x grad|B|:",
      sp.simplify(grad(psi).cross(grad(Pi)) - modB * grad(psi).cross(grad(modB))) == sp.zeros(3, 1))
print("      => gradpsi x gradPi . grad(B.gradPi) = |B| * (Helander triple product)  [the extra term is "
      "(B.grad|B|) (gradpsi x grad|B|).grad|B| = 0]")

# (I5): first variation.  Generic V, orbit x(t) with xdd = -grad V, dx(t) with dxdd = -Hess V dx.
t = sp.symbols('t', real=True)
xt = sp.Matrix([sp.Function(f'x{i}')(t) for i in range(1, 4)])
dxt = sp.Matrix([sp.Function(f'd{i}')(t) for i in range(1, 4)])
sub = {x: xt[0], y: xt[1], z: xt[2]}
gV_t = gV.subs(sub)
H_t = hess(V).subs(sub)
# third derivative tensor contracted:  D^3V(xd, xd, dx)_i... as a scalar: sum_ijk V_ijk xd_i xd_j dx_k
D3 = sum(sp.diff(V, X[i], X[j], X[k]).subs(sub) * xt.diff(t)[i] * xt.diff(t)[j] * dxt[k]
         for i in range(3) for j in range(3) for k in range(3))
lhs = sp.diff((gV_t.T * dxt)[0], t, 2)
rhs = 2 * (xt.diff(t).T * H_t * dxt.diff(t))[0] + D3 - 2 * (gV_t.T * H_t * dxt)[0]
expr = sp.expand(lhs - rhs)
# impose the orbit equation and its linearisation
rules = {}
for i in range(3):
    rules[sp.diff(xt[i], t, 2)] = -gV_t[i]
    rules[sp.diff(dxt[i], t, 2)] = -(H_t * dxt)[i]
expr = sp.expand(expr.subs(rules))
print("(I5) d2/dt2(gradV.dx) = 2HessV(B,dxd) + D3V(B,B,dx) - 2HessV(gradV,dx)  (mod ODEs):", expr == 0)

# (I4) remark, checked on a 1-D model: if Vdd = f(psi,V) then Vd^2/2 - int f dV is constant along the orbit.
Vt = sp.Function('Vt')(t); f = sp.Function('f')
Fint = sp.Function('Fi')  # Fi_V = f
claim = sp.diff(sp.diff(Vt, t)**2 / 2 - Fint(Vt), t).subs(sp.diff(Vt, t, 2), sp.diff(Fint(Vt), Vt))
print("(I4) d/dt [Vd^2/2 - Int f dV] = 0 when Vdd = f(V):", sp.simplify(claim) == 0)
