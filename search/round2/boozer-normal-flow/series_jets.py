"""series_jets.py -- exact jets of a Boozer embedding given in orbit coordinates, by truncated Taylor series.

Problem: a congruence x(lam, s, tau) with explicit Boozer angles
        psi = psi(lam),  theta_B = iota F(lam, tau) + const,  phi_B = s + F(lam, tau),
gives the Boozer embedding x(psi, theta, phi) only implicitly.  Symbolic differentiation of the composite map
(round-2 first attempt) is hopeless (minutes per derivative).  Here everything is done on truncated multivariate
Taylor series at a point with EXACT coefficients (Taylor-mode automatic differentiation on the expression tree):
   1. series of the forward map Phi = (psi, theta, phi)(lam, s, tau) and of any scalar/vector g(lam, s, tau),
   2. formal inversion of Phi (fixed point on truncated series),
   3. composition g o Phi^{-1}  ->  the jet of g in (psi, theta, phi).
Series are dicts {exponent tuple: exact sympy number} in the displacements (dP, dT, dF) = (d psi, d theta, d phi)
or (dl, ds, dt).  No numerical approximation anywhere; no nsimplify (it mangles exact irrational constants).
"""
import sympy as sp
from itertools import product
from math import factorial

dP, dT, dF = sp.symbols('dP dT dF')          # displacements in (psi, theta, phi)
dl, ds, dt = sp.symbols('dl ds dt')          # displacements in (lam, s, tau)


def monomials(nvars, order):
    return [e for e in product(range(order + 1), repeat=nvars) if sum(e) <= order]


class TSeries:
    """truncated multivariate power series in given generators, exact coefficients."""

    def __init__(self, gens, order, data=None):
        self.gens, self.order = tuple(gens), order
        self.d = {}
        if isinstance(data, dict):
            self.d = {k: v for k, v in data.items() if sum(k) <= order and v != 0}
        elif data is not None:              # constant
            data = sp.sympify(data)
            if data != 0:
                self.d = {tuple([0]*len(gens)): data}

    def copy(self):
        return TSeries(self.gens, self.order, dict(self.d))

    def __add__(self, o):
        if not isinstance(o, TSeries):
            o = TSeries(self.gens, self.order, o)
        r = dict(self.d)
        for k, v in o.d.items():
            r[k] = r.get(k, 0) + v
        return TSeries(self.gens, min(self.order, o.order), {k: v for k, v in r.items() if v != 0})

    __radd__ = __add__

    def __neg__(self):
        return TSeries(self.gens, self.order, {k: -v for k, v in self.d.items()})

    def __sub__(self, o):
        return self + (-o if isinstance(o, TSeries) else -sp.sympify(o))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, TSeries):
            o = sp.sympify(o)
            return TSeries(self.gens, self.order, {k: v*o for k, v in self.d.items()})
        order = min(self.order, o.order)
        r = {}
        for k1, v1 in self.d.items():
            s1 = sum(k1)
            for k2, v2 in o.d.items():
                if s1 + sum(k2) > order:
                    continue
                k = tuple(a + b for a, b in zip(k1, k2))
                r[k] = r.get(k, 0) + v1*v2
        return TSeries(self.gens, order, {k: v for k, v in r.items() if v != 0})

    __rmul__ = __mul__

    def const(self):
        return self.d.get(tuple([0]*len(self.gens)), sp.Integer(0))

    def inv(self):
        """1/self (constant term must be nonzero)."""
        c = self.const()
        assert c != 0
        rest = (self - c)*(1/c)                 # self = c (1 + rest)
        out = TSeries(self.gens, self.order, 1)
        term = TSeries(self.gens, self.order, 1)
        for _ in range(self.order):
            term = term*(-rest)
            out = out + term
        return out*(1/c)

    def sqrt(self):
        c = self.const()
        assert c != 0
        rc = sp.sqrt(c)
        rest = (self - c)*(1/c)
        out = TSeries(self.gens, self.order, 1)
        term = TSeries(self.gens, self.order, 1)
        coef = sp.Integer(1)
        for n in range(1, self.order + 1):
            coef = coef*(sp.Rational(1, 2) - (n - 1))/n
            term = term*rest
            out = out + term*coef
        return out*rc

    def truncate(self, order):
        return TSeries(self.gens, order, {k: v for k, v in self.d.items() if sum(k) <= order})

    def deriv(self, i):
        r = {}
        for k, v in self.d.items():
            if k[i] > 0:
                kk = list(k); kk[i] -= 1
                r[tuple(kk)] = v*k[i]
        return TSeries(self.gens, self.order - 1, r)

    def compose(self, subs_list):
        """substitute gens[i] -> subs_list[i] (TSeries in new gens, zero constant term)."""
        new = subs_list[0]
        order = min(self.order, new.order)
        out = TSeries(new.gens, order, 0)
        powers = [[TSeries(new.gens, order, 1)] for _ in subs_list]
        for i, s_ in enumerate(subs_list):
            for n in range(1, self.order + 1):
                powers[i].append(powers[i][-1]*s_.truncate(order))
        for k, v in self.d.items():
            term = TSeries(new.gens, order, v)
            for i, ki in enumerate(k):
                if ki:
                    term = term*powers[i][ki]
            out = out + term
        return out

    def coeff(self, k):
        return self.d.get(tuple(k), sp.Integer(0))

    def jet(self, k):
        """partial derivative of the represented function: coefficient times factorials."""
        fac = 1
        for ki in k:
            fac *= factorial(ki)
        return self.coeff(k)*fac

    def simplify(self, fn=sp.simplify):
        return TSeries(self.gens, self.order, {k: fn(v) for k, v in self.d.items()})

    def __repr__(self):
        return f"TSeries({self.gens}, order {self.order}, {len(self.d)} terms)"


def invert_map(Phi):
    """Phi: list of TSeries in (dl, ds, dt) with zero constant term (the forward map).  Returns list of TSeries
    in (dP, dT, dF) giving (dl, ds, dt) as functions of (dP, dT, dF): fixed point on truncated series."""
    order = Phi[0].order
    gens_new = (dP, dT, dF)
    n = len(Phi)
    L = sp.Matrix(n, n, lambda i, j: Phi[i].coeff(tuple(1 if jj == j else 0 for jj in range(n))))
    Linv = L.inv()
    Y = [TSeries(gens_new, order, {tuple(1 if jj == j else 0 for jj in range(n)): 1}) for j in range(n)]
    X = [sum((Linv[i, j]*Y[j] for j in range(n)), TSeries(gens_new, order, 0)) for i in range(n)]
    for _ in range(order):
        PhiX = [Phi[i].compose(X) for i in range(n)]
        NL = [PhiX[i] - sum((L[i, j]*X[j] for j in range(n)), TSeries(gens_new, order, 0)) for i in range(n)]
        X = [sum((Linv[i, j]*(Y[j] - NL[j]) for j in range(n)), TSeries(gens_new, order, 0)) for i in range(n)]
    return X


# ---------------- Taylor-mode automatic differentiation on expression trees ----------------
def const_eval(v):
    """exact evaluation of a constant (no free symbols): trig of atan-rationals -> rationals; never approximates."""
    v = sp.sympify(v)
    if v.is_Rational:
        return v
    v = sp.expand_trig(v)
    v = sp.radsimp(sp.expand(v))
    if v.is_Rational:
        return v
    v2 = sp.simplify(v)
    return v2


def _series_cos_sin(u):
    """(cos u, sin u) for a TSeries u."""
    u0 = u.const()
    v = u - u0
    c0, s0 = const_eval(sp.cos(u0)), const_eval(sp.sin(u0))
    one = TSeries(u.gens, u.order, 1)
    cv, sv = one.copy(), TSeries(u.gens, u.order, 0)
    term = one
    for n in range(1, u.order + 1):
        term = term*v*sp.Rational(1, n)
        if n % 2 == 0:
            cv = cv + term*((-1)**(n//2))
        else:
            sv = sv + term*((-1)**((n - 1)//2))
    return cv*c0 - sv*s0, sv*c0 + cv*s0


def series_eval(expr, env, gens=(dl, ds, dt), order=4):
    """Evaluate a sympy expression in truncated-series arithmetic.  env: {symbol: TSeries}.  Handles Add, Mul,
    Pow (integer or half-integer exponents), cos, sin, exact constants."""
    expr = sp.sympify(expr)
    if not expr.free_symbols:
        return TSeries(gens, order, const_eval(expr))
    if expr.is_Symbol:
        return env[expr]
    if expr.is_Add:
        out = TSeries(gens, order, 0)
        for a in expr.args:
            out = out + series_eval(a, env, gens, order)
        return out
    if expr.is_Mul:
        out = TSeries(gens, order, 1)
        for a in expr.args:
            out = out*series_eval(a, env, gens, order)
        return out
    if expr.is_Pow:
        base, e = expr.args
        b = series_eval(base, env, gens, order)
        if e.is_Integer:
            n = int(e)
            if n < 0:
                b = b.inv(); n = -n
            out = TSeries(gens, order, 1)
            for _ in range(n):
                out = out*b
            return out
        if e.is_Rational and e.q == 2:
            r = b.sqrt()
            n = int(e.p)
            if n < 0:
                r = r.inv(); n = -n
            out = TSeries(gens, order, 1)
            for _ in range(n):
                out = out*r
            return out
        raise NotImplementedError(f"Pow exponent {e}")
    if isinstance(expr, sp.Abs):
        u = series_eval(expr.args[0], env, gens, order)
        c = u.const()
        assert c != 0 and c.is_real, "Abs at a zero or non-real constant"
        return u*(1 if c > 0 else -1)
    if isinstance(expr, sp.cos):
        return _series_cos_sin(series_eval(expr.args[0], env, gens, order))[0]
    if isinstance(expr, sp.sin):
        return _series_cos_sin(series_eval(expr.args[0], env, gens, order))[1]
    raise NotImplementedError(f"node {type(expr)}: {expr}")


def series_of_ad(expr, vars_, point, order, gens=(dl, ds, dt)):
    """Taylor series of expr(vars_) at point via Taylor-mode AD (fast, exact)."""
    env = {v: TSeries(gens, order, {tuple([0]*len(gens)): point[v], tuple(1 if j == i else 0 for j in range(len(gens))): 1})
           for i, v in enumerate(vars_)}
    return series_eval(expr, env, gens, order)


def taylor_dict(expr, vars_, point, order):
    """(reference implementation, slow) {exponent tuple: exact coefficient} by repeated differentiation."""
    derivs = {tuple([0]*len(vars_)): expr}
    out = {}
    for e in sorted(monomials(len(vars_), order), key=sum):
        if e not in derivs:
            for i, ei in enumerate(e):
                if ei > 0:
                    pred = list(e); pred[i] -= 1; pred = tuple(pred)
                    derivs[e] = sp.diff(derivs[pred], vars_[i])
                    break
        val = const_eval(derivs[e].subs(point))
        if val != 0:
            fac = 1
            for ei in e:
                fac *= factorial(ei)
            out[e] = sp.Rational(1, fac)*val
    return out


def series_of(expr, vars_, point, order, gens=(dl, ds, dt), post=None):
    return TSeries(gens, order, taylor_dict(expr, vars_, point, order))
