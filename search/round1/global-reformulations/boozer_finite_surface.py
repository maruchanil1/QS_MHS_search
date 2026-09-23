"""Finite trigonometric-polynomial Boozer surfaces: the two psi-frozen SURFACE CONDITIONS of boozer_flow_reduction.py
   C1 :  I (W.x_p) - G (W.x_t) = 0                (W = x_p + iota x_t)
   C2 :  |W|^2 is a function of chi = theta - N phi
are necessary for a surface x(theta,phi) to be a flux surface of a QS-MHS field in Boozer coordinates.  Vacuum here
(I = 0, G = 1):  C1 = W.x_t = 0,  C2 = u(|W|^2) = 0.

Ansatz:  x + i y = e^{i phi} rho(theta,phi),  z = zeta(theta,phi),  rho complex and zeta real trig polynomials.
The e^{i phi} factor drops out of all metric coefficients, so C1, C2 are finite Fourier systems in the coefficients.

Part 1 (axisymmetric, degree 1 in theta; rho = r0 + r1 e^{i th} + rm e^{-i th}, zeta = Re(z1 e^{i th})):
        THEOREM (this script): C1 forces  iota * |z1|^2 = 0, i.e. iota = 0 (or a degenerate surface).
        The iota = 0 solutions are exactly the shifted-ellipse vacuum family used as a test in boozer_flow_reduction.py.
Part 2 (smallest non-axisymmetric class: modes m in {-1,0,1}, n in {-N,0,N}):  sets up C1, C2 as polynomial
        equations, solves the extreme-mode equations and reports what they force.
Run:  python3 boozer_finite_surface.py     (~1 min)
"""
import sympy as sp
import time, itertools
t0 = time.time()
th, ph = sp.symbols('theta phi', real=True)
E1, E2 = sp.symbols('E1 E2')            # e^{i theta}, e^{i phi}
iota = sp.symbols('iota', real=True)

def laurent_coeffs(expr):
    """expr: polynomial in E1, E2, 1/E1, 1/E2 -> dict {(m,n): coeff}."""
    expr = sp.expand(expr)
    out = {}
    for term in sp.Add.make_args(expr):
        c, m, n = term, 0, 0
        for fac in sp.Mul.make_args(term):
            b, e = fac.as_base_exp()
            if b == E1: m += int(e); c = c/fac
            elif b == E2: n += int(e); c = c/fac
        out[(m, n)] = out.get((m, n), 0) + c
    return {k: sp.expand(v) for k, v in out.items() if sp.expand(v) != 0}

def metric_conditions(rho, zeta, N):
    """rho complex, zeta real expressions in E1, E2 (Laurent). Returns Fourier dicts of C1 and of |W|^2."""
    dth = lambda f: sp.I*(E1*sp.diff(f, E1))       # d/dtheta of f(E1 = e^{i theta})
    dph = lambda f: sp.I*(E2*sp.diff(f, E2))
    # complex conjugation: E -> 1/E (unit modulus) and i -> -i in the (real-symbol) coefficients
    conj = lambda f: sp.expand(f.subs({E1: 1/E1, E2: 1/E2}, simultaneous=True)).subs(sp.I, -sp.I)
    P_t = dth(rho); P_p = sp.I*rho + dph(rho)       # P = e^{i phi} rho: P_theta, P_phi (prefactor dropped)
    z_t, z_p = dth(zeta), dph(zeta)
    g_tt = sp.expand(P_t*conj(P_t) + z_t**2)
    g_tp = sp.expand((P_t*conj(P_p) + conj(P_t)*P_p)/2 + z_t*z_p)
    g_pp = sp.expand(P_p*conj(P_p) + z_p**2)
    C1 = sp.expand(g_tp + iota*g_tt)
    W2 = sp.expand(g_pp + 2*iota*g_tp + iota**2*g_tt)
    return laurent_coeffs(C1), laurent_coeffs(W2)

# ---------------- Part 1: axisymmetric degree-1 ----------------
r0a, r0b, r1a, r1b, rma, rmb, z1a, z1b = sp.symbols('r0a r0b r1a r1b rma rmb z1a z1b', real=True)
r0 = r0a + sp.I*r0b; r1 = r1a + sp.I*r1b; rm = rma + sp.I*rmb; z1 = z1a + sp.I*z1b
rho = r0 + r1*E1 + rm/E1
zeta = (z1*E1 + sp.conjugate(z1)/E1)/2
C1c, W2c = metric_conditions(rho, zeta, 0)
print("Part 1: C1 Fourier modes (m,n):", sorted(C1c))
eqs = [sp.re(sp.expand(v)) for v in C1c.values()] + [sp.im(sp.expand(v)) for v in C1c.values()]
eqs = [sp.expand(e) for e in eqs if sp.expand(e) != 0]
for k, v in sorted(C1c.items()):
    print("   mode", k, ":", sp.factor(v))
# consequence: combine the modes
print("   C2 automatic for axisymmetric surfaces (|W|^2 has n = 0 modes only):", all(n == 0 for (m, n) in W2c))
sols = sp.solve(eqs, [r1a, r1b, rma, rmb, iota], dict=True)
print("   solutions of C1 for (r1, rm, iota) given r0, z1:")
for s in sols:
    print("     ", s)
# direct derivation printed for the record
print("   => every branch has iota = 0 or z1 = 0 (degenerate: z-independent 'surface');  Part 1 THEOREM holds:",
      all((s.get(iota, iota) == 0) or (sp.simplify(s.get(z1a, z1a)) == 0 and sp.simplify(s.get(z1b, z1b)) == 0) for s in sols))

# ---------------- Part 2: smallest non-axisymmetric class ----------------
Nv = 1
modes = [(m, n) for m in (-1, 0, 1) for n in (-Nv, 0, Nv)]
R = {}; Zc = {}
syms = []
for (m, n) in modes:
    a_, b_ = sp.symbols(f'a{m+1}{n+1} b{m+1}{n+1}', real=True); syms += [a_, b_]
    R[(m, n)] = a_ + sp.I*b_
zsyms = []
for (m, n) in modes:
    if (m, n) == (0, 0): continue
    if (-m, -n) in Zc: continue
    c_, d_ = sp.symbols(f'c{m+1}{n+1} d{m+1}{n+1}', real=True); zsyms += [c_, d_]
    Zc[(m, n)] = c_ + sp.I*d_; Zc[(-m, -n)] = c_ - sp.I*d_
rho2 = sum(R[(m, n)]*E1**m*E2**n for (m, n) in modes)
zeta2 = sum(Zc[(m, n)]*E1**m*E2**n for (m, n) in Zc)/2
C1c2, W2c2 = metric_conditions(rho2, zeta2, Nv)
bad = {k: v for k, v in W2c2.items() if k[1] != -Nv*k[0]}
print(f"\nPart 2 (N = {Nv}): unknowns {len(syms) + len(zsyms)} real + iota;  C1 modes: {len(C1c2)};  C2 (|W|^2 modes with n != -N m): {len(bad)}")
def realeqs(d):
    out = []
    for v in d.values():
        for e in (sp.re(sp.expand(v)), sp.im(sp.expand(v))):
            e = sp.expand(e)
            if e != 0: out.append(e)
    return out
eqs1, eqs2 = realeqs(C1c2), realeqs(bad)
print(f"   real polynomial equations: C1: {len(eqs1)}, C2: {len(eqs2)}")
# extreme modes: |m| = 2 or |n| = 2N in C1, and the same in C2
ext1 = {k: v for k, v in C1c2.items() if abs(k[0]) == 2 or abs(k[1]) == 2*Nv}
ext2 = {k: v for k, v in bad.items() if abs(k[0]) == 2 or abs(k[1]) == 2*Nv}
print("   extreme C1 modes:", sorted(ext1), "\n   extreme C2 modes:", sorted(ext2))
for k in sorted(ext1):
    print("     C1", k, ":", sp.factor(ext1[k]))
allunk = syms + zsyms + [iota]
# Structure of the extreme modes (read off above): for iota not in {0, +-1, +-1/2} they are ISOTROPY conditions
#   rho_{-1,-N} conj(rho_{1,N}) + zeta_{-1,-N}^2/4 = 0,   rho_{-1,N} conj(rho_{1,-N}) + zeta_{-1,N}^2/4 = 0,  etc.
# The full 26-unknown system is too large for sympy's solve in the time box; impose STELLARATOR SYMMETRY
# (R(-th,-ph) = R(th,ph), Z(-th,-ph) = -Z(th,ph)  <=>  rho coefficients real, zeta coefficients imaginary):
stell = {s_: 0 for s_ in syms if s_.name.startswith('b')}
stell.update({s_: 0 for s_ in zsyms if s_.name.startswith('c')})
unk_st = [s_ for s_ in allunk if s_ not in stell]
ext_eqs = [sp.expand(e.subs(stell)) for e in realeqs(ext1) + realeqs(ext2)]
ext_eqs = list(dict.fromkeys(e for e in ext_eqs if e != 0))
print(f"   stellarator-symmetric class: {len(unk_st)} unknowns {unk_st};  {len(ext_eqs)} extreme-mode equations")
t1 = time.time()
sol_ext = sp.solve(ext_eqs, unk_st, dict=True)
print(f"   {len(sol_ext)} solution branches in {time.time()-t1:.1f} s")
nonax = [s_ for s_ in unk_st if s_ != iota and s_.name[-1] != '1']       # index n+1 = 1 <=> n = 0 (axisymmetric mode)
print("   non-axisymmetric amplitudes:", nonax)
def forced_axisym(s):
    return all(v in s and sp.simplify(s[v]) == 0 for v in nonax)
nfa = sum(forced_axisym(s) for s in sol_ext)
print("   branches with ALL non-axisymmetric amplitudes forced to zero:", nfa, "of", len(sol_ext))
full_eqs = list(dict.fromkeys(sp.expand(e.subs(stell)) for e in eqs1 + eqs2 if sp.expand(e.subs(stell)) != 0))
zsyms_st = [s_ for s_ in zsyms if s_ not in stell]     # d's: the (imaginary) z-amplitudes
terminal = []          # (combined solution dict, classification)
# rho_{0,-1} (= a10) is a rigid translation x -> x + a10 (e^{i phi} a10 e^{-i phi}); exclude it from the non-axisymmetric test
a10 = [s_ for s_ in unk_st if s_.name == 'a10'][0]
nonax_eff = [v for v in nonax if v != a10]
def classify(comb):
    vals = [sp.simplify(comb.get(v, v)) for v in unk_st]
    isreal = all(v.is_real is not False and not v.has(sp.I) for v in vals)
    nz = all(sp.simplify(comb.get(v, v)) == 0 for v in nonax_eff)
    zz = all(sp.simplify(comb.get(v, v)) == 0 for v in zsyms_st)                 # z == 0: planar, degenerate
    io = comb.get(iota, iota)
    # degenerate parametrisation: rank of (x_theta, x_phi) < 2 at a generic point
    rho_c = sum(comb.get(sp.re(R[(m, n)]), sp.re(R[(m, n)]))*E1**m*E2**n for (m, n) in modes)
    zeta_c = sum((sp.I*comb.get(sp.im(Zc[(m, n)]), sp.im(Zc[(m, n)])) if (m, n) in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
                  else -sp.I*comb.get(sp.im(Zc[(-m, -n)]), sp.im(Zc[(-m, -n)])))*E1**m*E2**n for (m, n) in Zc)/2
    Pc = sp.exp(sp.I*ph)*rho_c.subs({E1: sp.exp(sp.I*th), E2: sp.exp(sp.I*ph)})
    zc = zeta_c.subs({E1: sp.exp(sp.I*th), E2: sp.exp(sp.I*ph)})
    xv = sp.Matrix([sp.re(Pc), sp.im(Pc), sp.re(zc)])
    Jm = sp.Matrix.hstack(xv.diff(th), xv.diff(ph))
    pt_ = {th: sp.Rational(7, 10), ph: sp.Rational(13, 10)}
    free_ = [v for v in unk_st if v not in comb and v != iota]
    pt_.update({v: sp.Rational(3 + 2*i_, 7) for i_, v in enumerate(free_)})
    if iota not in comb: pt_[iota] = sp.Rational(1, 3)
    try:
        Jn = sp.Matrix(Jm.subs(pt_).evalf(30)).applyfunc(lambda q: sp.re(q) if q.is_real is None else q)
        rank = Jn.evalf().rank(iszerofunc=lambda q: abs(complex(q)) < 1e-9) if isreal else None
    except Exception as ex:
        rank = None
    return dict(isreal=isreal, nonax_zero=nz, z_zero=zz, iota_pm_N=(io in (1, -1)), iota=io, rank=rank)
for s in sol_ext:
    if forced_axisym(s):
        terminal.append((s, classify(s))); continue
    rem = [sp.expand(e.subs(s)) for e in full_eqs]
    rem = list(dict.fromkeys(e for e in rem if e != 0))
    print(f"     surviving branch {s}")
    print(f"        {len(rem)} remaining C1/C2 equations", ("; e.g. " + str(sp.factor(min(rem, key=lambda e: len(sp.Add.make_args(e)))))) if rem else "")
    if not rem:
        terminal.append((s, classify(s))); continue
    sub_sol = sp.solve(rem, [v for v in unk_st if v not in s], dict=True)
    print(f"        full C1+C2 solve on this branch: {len(sub_sol)} solutions:", sub_sol[:6])
    for ss in sub_sol:
        comb = {k: sp.simplify(v.subs(ss)) for k, v in s.items()}; comb.update(ss)
        terminal.append((comb, classify(comb)))
print(f"\n   {len(terminal)} terminal solutions of C1 + C2 in the stellarator-symmetric class (N = {Nv}):")
ok = True
for comb, cl in terminal:
    trivial = (not cl['isreal']) or cl['nonax_zero'] or cl['z_zero'] or cl['iota_pm_N'] or (cl['rank'] is not None and cl['rank'] < 2)
    ok &= trivial
    if not trivial:
        print("     NON-TRIVIAL candidate surface:", comb, cl)
print("   every real terminal solution is (translated axisymmetric) or degenerate (z == 0 / rank < 2) or has iota = +-N:", ok)
print("   tallies: complex", sum(not cl['isreal'] for _, cl in terminal), "| axisymmetric(+translation)", sum(cl['isreal'] and cl['nonax_zero'] for _, cl in terminal),
      "| degenerate", sum(cl['isreal'] and not cl['nonax_zero'] and (cl['z_zero'] or (cl['rank'] is not None and cl['rank'] < 2)) for _, cl in terminal),
      "| iota = +-N only", sum(cl['isreal'] and cl['iota_pm_N'] and not cl['nonax_zero'] and not cl['z_zero'] and not (cl['rank'] is not None and cl['rank'] < 2) for _, cl in terminal))
print(f"done in {time.time()-t0:.1f} s")
