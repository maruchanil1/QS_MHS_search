"""vE_surface.py -- trigonometric-polynomial Boozer surfaces in the symmetry-adapted variables
        v = e^{i chi} (chi = theta - N phi),   E = e^{i phi}  (phi along the QS symmetry direction u = d_phi + N d_theta).
A surface x(chi, phi) = sum_{j=-J}^{J} X_j(v) E^j,  X_j in C^3 Laurent polynomials in v (X_{-j} = conj X_j on |v| = 1).
Tangents: x_chi = i v d_v x,  x_u = i E d_E x;  field-line tangent W = D x = x_u + (iota - N) x_chi,  a-direction
d_a x = I x_u - (G + N I) x_chi.  The two psi-frozen surface conditions of a QS-MHS flux surface in Boozer form are
        C2:  |W|^2 = W.W  has no E^k component for k != 0            (|B| is u-invariant)
        C1:  W.x_chi = kappa |W|^2,  kappa := I/(G + iota I)         (equivalently (1),(2) with sqrt g = |W|^2/(G+iota I))
(W.W is the complex-bilinear square of the formal Laurent expansion; for a real surface it equals |W|^2.)
This module builds the coefficient equations {E^k v^m} of C1 and C2 for a general ansatz.

Null frame: eps = (1, -i, 0)/2, epsbar = (1, i, 0)/2, e_z: eps.eps = 0, eps.epsbar = 1/2.
Components: X = alpha eps + beta epsbar + gamma e_z;  X.Y = (alpha_X beta_Y + beta_X alpha_Y)/2 + gamma_X gamma_Y.
"""
import sympy as sp

v, E = sp.symbols('v E')
eps = sp.Matrix([1, -sp.I, 0])/2
epsb = sp.Matrix([1, sp.I, 0])/2
ez = sp.Matrix([0, 0, 1])


def laurent(expr, var, lo, hi):
    """coefficients {m: c} of a Laurent polynomial in var."""
    expr = sp.expand(expr)
    out = {}
    for m in range(lo, hi + 1):
        c = expr.coeff(var, m) if m != 0 else expr.subs(var, 0) if False else None
    # robust: multiply by var^(-lo) and use Poly
    P = sp.Poly(sp.expand(expr*var**(-lo)), var)
    for (k,), c in P.terms():
        out[k + lo] = c
    return out


def surface_equations(X, iota, N, kappa=None, keep_zero=False):
    """X: dict {j: 3-vector Laurent polynomial in v}.  Returns (eqsC2, eqsC1): lists of coefficient equations
    of the E^k v^m modes (k != 0) of W.W and of W.x_chi - kappa W.W (all k if keep_zero)."""
    x = sp.zeros(3, 1)
    for j, Xj in X.items():
        x += Xj*E**j
    x_chi = sp.I*v*x.diff(v)
    x_u = sp.I*E*x.diff(E)
    W = x_u + (iota - N)*x_chi
    WW = sp.expand(W.dot(W))
    Wx = sp.expand(W.dot(x_chi))
    J = max(abs(j) for j in X)
    Dv = max(sp.degree(sp.expand(comp*v**(2*J*10)), v) - 2*J*10 for Xj in X.values() for comp in Xj if comp != 0)
    # collect E-coefficients
    def Ecoeffs(expr):
        P = sp.Poly(sp.expand(expr*E**(2*J)), E)
        return {k - 2*J: c for (k,), c in P.terms()}
    cWW, cWx = Ecoeffs(WW), Ecoeffs(Wx)
    eqsC2, eqsC1 = [], []
    for k, c in cWW.items():
        if k != 0 or keep_zero:
            eqsC2 += [(k, m, cc) for m, cc in laurent(c, v, -4*Dv - 4, 4*Dv + 4).items() if cc != 0]
    for k, c in cWx.items():
        if k != 0:
            eqsC1 += [(k, m, cc) for m, cc in laurent(c, v, -4*Dv - 4, 4*Dv + 4).items() if cc != 0]
    if kappa is not None:
        c0 = cWx.get(0, 0) - kappa*cWW.get(0, 0)
        eqsC1 += [(0, m, cc) for m, cc in laurent(c0, v, -4*Dv - 4, 4*Dv + 4).items() if cc != 0]
    return eqsC2, eqsC1, WW, Wx


def frame_vector(alpha, beta, gamma):
    return alpha*eps + beta*epsb + gamma*ez


def laurent_poly(name, D, lo=None):
    """generic Laurent polynomial sum_{m=-D}^{D} name_m v^m with complex symbols; returns (expr, symbols)."""
    lo = -D if lo is None else lo
    syms = [sp.Symbol(f'{name}_{m}'.replace('-', 'm')) for m in range(lo, D + 1)]
    return sum(s*v**m for s, m in zip(syms, range(lo, D + 1))), syms
