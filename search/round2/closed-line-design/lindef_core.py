"""Core of the first-order closed-line deformation problem (TASK 2), reusable for any exact congruence.

Given an exact closed-line congruence x0(lab, s, t) (B = x0_t, u = d_s, J = J(lab)) and a basis of vector fields
b_i(x) (polynomial), the first-order deformation x = x0 + eps xi(x0), xi = sum c_i b_i, satisfies weak QS + MHS +
div B = 0 to first order iff (see notes.md, Sec. 2; all conditions linear in xi)
    (dJ)  d_s[div xi o x0] = d_t[div xi o x0] = 0
    (W)   d_s dg_tt = 0,            dg_ab := 2 x0_a . S(xi)(x0) . x0_b
    (M1)  d_t dg_ts - d_s dg_tt = 0
    (M2)  d_s Q = d_t Q = 0,        Q := d_t dg_tlab - d_lab dg_tt
Strong QS adds (SQ) d_s dg_ts = d_s dg_tlab = 0 (reported, not imposed).
Everything is turned into Laurent polynomials in E = e^{is}, T = e^{it} (and, if the congruence is trigonometric in
the label, L = e^{i lab}); an optional multiplier clears denominators and an optional substitution evaluates
label-dependent constants at a rational point AFTER the label-differentiation.
"""
import sympy as sp

X, Y, Z = sp.symbols('x y z')
L, E, T = sp.symbols('L E T')

def basis_fields(deg):
    monos = [X**i*Y**j*Z**k for i in range(deg+1) for j in range(deg+1) for k in range(deg+1) if i+j+k <= deg]
    fields = []
    for comp in range(3):
        for m in monos:
            v = sp.zeros(3, 1); v[comp] = m
            fields.append(v)
    return fields

def monomial_dict(expr, gens):
    """Laurent polynomial expr in gens -> {monomial: coefficient} (coefficients free of gens)."""
    expr = sp.expand(expr)
    out = {}
    for term in sp.Add.make_args(expr):
        c, m = term.as_independent(*gens, as_Add=False)
        out[m] = out.get(m, 0) + c
    return out

def condition_columns(x0, dlab, ds, dt, fields, clear=1, post=None, gens=(L, E, T), with_strong=True):
    """Return (rows_dict_list, strong_rows) where each entry is {monomial: [coeff per field]} per condition."""
    x0_l, x0_s, x0_t = [x0.applyfunc(D) for D in (dlab, ds, dt)]
    sub = {X: x0[0], Y: x0[1], Z: x0[2]}
    cond_names = ['dJ_s', 'dJ_t', 'W', 'M1', 'M2_s', 'M2_t']
    strong_names = ['SQ_ts', 'SQ_tl']
    cols = {n: [] for n in cond_names + strong_names}
    for b in fields:
        G = sp.Matrix(3, 3, lambda i, j: sp.diff(b[i], (X, Y, Z)[j]))
        Sy = ((G + G.T)/2).subs(sub, simultaneous=True)
        dv = sum(G[i, i] for i in range(3)).subs(sub, simultaneous=True)
        def dg(a, c): return 2*(a.T*Sy*c)[0, 0]
        g_tt, g_ts, g_tl = dg(x0_t, x0_t), dg(x0_t, x0_s), dg(x0_t, x0_l)
        Q = dt(g_tl) - dlab(g_tt)
        exprs = {'dJ_s': ds(dv), 'dJ_t': dt(dv), 'W': ds(g_tt), 'M1': dt(g_ts) - ds(g_tt), 'M2_s': ds(Q), 'M2_t': dt(Q)}
        if with_strong:
            exprs.update({'SQ_ts': ds(g_ts), 'SQ_tl': ds(g_tl)})
        for n, ex in exprs.items():
            ex = ex*clear
            if post is not None:
                ex = post(ex)
            cols[n].append(monomial_dict(ex, gens))
    return cols

def assemble(cols, names, nfields):
    rows = []
    for n in names:
        monos = set()
        for d in cols[n]:
            monos |= set(d.keys())
        for m in monos:
            row = [cols[n][i].get(m, 0) for i in range(nfields)]
            if any(v != 0 for v in row):
                rows.append(row)
    return sp.Matrix(rows) if rows else sp.zeros(0, nfields)

def field_from(vec, fields):
    f = sp.zeros(3, 1)
    for c, b in zip(vec, fields):
        f += c*b
    return f.applyfunc(sp.expand)

def is_killing(f):
    G = sp.Matrix(3, 3, lambda i, j: sp.diff(f[i], (X, Y, Z)[j]))
    return (G + G.T).applyfunc(sp.expand) == sp.zeros(3, 3)

def divergence(f):
    return sp.expand(sum(sp.diff(f[i], (X, Y, Z)[i]) for i in range(3)))

def s_modes(f, x0):
    """set of |n| of the e^{ins} modes of xi(x0) (x0 given in E)."""
    fx = f.subs({X: x0[0], Y: x0[1], Z: x0[2]}, simultaneous=True).applyfunc(sp.expand)
    modes = set()
    for comp in fx:
        for term in sp.Add.make_args(comp):
            if not term.has(E):
                modes.add(0); continue
            p = sp.Poly(term.as_independent(E, as_Add=False)[1]*E**50, E)
            modes.add(abs(p.degree() - 50))
    return modes

def clean_vec(v):
    den = sp.lcm([sp.fraction(sp.Rational(a))[1] for a in v if a != 0]) if any(a != 0 for a in v) else 1
    return v*den


def fast_nullspace(M):
    """nullspace via DomainMatrix (exact, much faster than Matrix.nullspace for large systems); falls back if needed."""
    try:
        from sympy.polys.matrices import DomainMatrix
        dM = DomainMatrix.from_Matrix(M).to_field()
        ns = dM.nullspace().to_Matrix()          # rows = basis vectors
        return [ns.row(i).T for i in range(ns.rows)]
    except Exception as e:
        print("  (DomainMatrix nullspace failed:", e, "-> falling back to Matrix.nullspace)")
        return M.nullspace()
