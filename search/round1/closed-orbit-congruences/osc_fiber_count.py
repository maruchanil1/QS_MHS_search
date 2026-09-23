"""Profile-fibre count for rational oscillators with three distinct frequencies (w1,w2,w3), e.g. (1,2,3).
x_i = Re(a_i e^{i w_i t});  |B|^2 = sum w_i^2 |a_i|^2/2 - sum (w_i^2/2) Re(a_i^2 e^{2 i w_i t}).
Same profile up to shift tau  <=>  |a_i| equal and a_i'^2 = a_i^2 e^{2 i w_i tau}.  Gauge a_1 real > 0 on both
orbits => tau in {0, pi/w1, ...} finite => finitely many (a_2', a_3') (sign choices) : 0-dim fibre, all related by
the reflections x_i -> -x_i (isometries of V).  So three distinct frequencies never give a torus of orbits with
one |B|-profile; two equal frequencies give exactly the SO(2) rotation (see osc112_qs.py).
Run: python3 osc_fiber_count.py
"""
import sympy as sp

def fibre(w):
    # generic orbit
    a = [sp.Rational(3, 2), sp.Rational(1, 2) + sp.I, 2 - sp.I/3]
    # unknown orbit with a1' real positive (time gauge), a2', a3' complex
    r1 = sp.symbols('r1', positive=True)
    u2, v2, u3, v3 = sp.symbols('u2 v2 u3 v3', real=True)
    ap = [r1, u2 + sp.I*v2, u3 + sp.I*v3]
    tau = sp.symbols('tau', real=True)
    eqs = []
    eqs.append(sp.Eq(r1**2, sp.Abs(a[0])**2))
    # a1'^2 = a1^2 e^{2 i w1 tau}: fixes tau mod pi/w1 (a1 real => e^{2 i w1 tau} real positive => tau = k pi / w1)
    sols = []
    for k in range(2*w[0]):
        tk = sp.pi*k/w[0]
        e2 = [sp.expand(a[i]**2*sp.exp(2*sp.I*w[i]*tk), complex=True) for i in range(3)]
        E = [sp.Eq(sp.expand(ap[i]**2, complex=True), e2[i]) for i in (1, 2)]
        E = [sp.re(e.lhs - e.rhs) for e in E] + [sp.im(e.lhs - e.rhs) for e in E]
        S = sp.solve(E, [u2, v2, u3, v3], dict=True)
        sols += [(tk, s) for s in S]
    return sols

for w in [(1, 2, 3), (1, 3, 4), (2, 3, 5)]:
    S = fibre(w)
    print(f"w={w}: number of orbits with the same |B|-profile as the reference (mod time shift): {len(S)} (finite => 0-dim fibre)")
    for tk, s in S[:4]:
        print("    tau =", tk, " a2' =", sp.nsimplify(s[sp.Symbol('u2', real=True)] + sp.I*s[sp.Symbol('v2', real=True)]),
              " a3' =", sp.nsimplify(s[sp.Symbol('u3', real=True)] + sp.I*s[sp.Symbol('v3', real=True)]))
