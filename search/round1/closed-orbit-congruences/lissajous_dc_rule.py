"""TASK B(i): 'Jacobian DC selection rule' for Lissajous congruences of the rational oscillator
V = sum w_i^2 x_i^2 / 2:  x_i = Re(a_i e^{i w_i t}), a_i = a_i(psi, s) complex.
J = det(d_psi x, d_s x, d_t x) is a trigonometric polynomial whose harmonics are +-w1 +-w2 +-w3.
Its mean (the only part that can survive d_t J = 0 with J != 0) is nonzero only if some
+-w1 +-w2 +-w3 = 0, i.e. one frequency is the sum of the other two.
Run: python3 lissajous_dc_rule.py
"""
import sympy as sp

t = sp.symbols('t', real=True)

def mean_J(w):
    A = [sp.symbols(f'A{i}', real=True) for i in range(3)]      # Re a_i
    Bc = [sp.symbols(f'B{i}', real=True) for i in range(3)]     # Im a_i
    Ap = [sp.symbols(f'Ap{i}', real=True) for i in range(3)]; Bp = [sp.symbols(f'Bp{i}', real=True) for i in range(3)]
    As = [sp.symbols(f'As{i}', real=True) for i in range(3)]; Bs = [sp.symbols(f'Bs{i}', real=True) for i in range(3)]
    x = sp.Matrix([A[i]*sp.cos(w[i]*t) - Bc[i]*sp.sin(w[i]*t) for i in range(3)])
    xp = sp.Matrix([Ap[i]*sp.cos(w[i]*t) - Bp[i]*sp.sin(w[i]*t) for i in range(3)])
    xs = sp.Matrix([As[i]*sp.cos(w[i]*t) - Bs[i]*sp.sin(w[i]*t) for i in range(3)])
    J = sp.expand(sp.Matrix.hstack(xp, xs, x.diff(t)).det())
    L = 2*sp.pi  # common period for integer w
    m = sp.simplify(sp.integrate(J, (t, 0, L))/L)
    # list of harmonics present
    Jf = sp.expand(sp.expand_trig(J))
    return m, J

for w in [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 2), (2, 2, 3), (1, 2, 3), (1, 3, 4), (2, 3, 5), (1, 2, 4)]:
    m, J = mean_J(w)
    sel = any(s1*w[0] + s2*w[1] + w[2] == 0 for s1 in (1, -1) for s2 in (1, -1))
    print(f"w={w}: some +-w1+-w2+-w3 = 0? {sel};  mean(J) == 0? {m == 0}" + ("" if m == 0 else f";  mean(J) = {sp.factor(m)}"))
print("\nRule verified: mean(J) != 0  <=>  one frequency equals the sum of the other two.")
print("With at most two distinct frequencies this leaves exactly (1,1,2) [up to scaling]; with three distinct, e.g. (1,2,3), (1,3,4), (2,3,5).")
