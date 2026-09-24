"""TASK 3 / Part D (and task 4): the E-degree-3 class of trigonometric Boozer surfaces with v-degree 1,
   x(chi, phi) = sum_{j=-3}^{3} X_j(v) E^j,  X_j in C^3 Laurent polynomials of degree <= 1 in v = e^{i chi}, E = e^{i phi},
in the null frame (eps, epsbar, e_z) with the top mode X_3 = lambda(v) eps (top-mode lemma, WLOG after a rotation),
X_{-3} = lambdab(v) epsbar.  This class CONTAINS the round-1 smallest non-axisymmetric class (m, n in {-1,0,1}, N = 1:
x+iy = E sum rho_{mn} v^m E^{m+n}, z = sum zeta_{mn} v^m E^{m+n}) WITHOUT stellarator symmetry (26 real unknowns, open in
round 1), because there J = L + N M + 1 = 3 and the v-degree is 1.
Conditions: C2 = the E^k (k != 0) coefficients of W.W vanish; C1 = the E^k (k != 0) coefficients of W.x_chi vanish and
W.x_chi = kappa W.W at E^0.  Complexified: the conjugate modes carry independent symbols (every real solution is a
solution of this larger system).  Sequential solve of the hierarchy E^5, E^4, ..., E^1 for generic iota = 2/11, N = 1.
Run: python3 task3_J3.py    (log: task3_J3.log)
"""
import sympy as sp, time, sys, random
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from vE_surface import v, E, eps, epsb, ez, surface_equations, frame_vector, laurent_poly
t0 = time.time()
# usage: python3 task3_J3.py [iota] [numeric_lambda_seed]     (default iota = 2/11, symbolic lambda)
io = sp.Rational(sys.argv[1]) if len(sys.argv) > 1 else sp.Rational(2, 11)
seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
Nn = 1
kap = sp.Symbol('kappa')

P = {}
S = {}
for nm in ('lam', 'lamb', 'al2', 'be2', 'ga2', 'alm2', 'bem2', 'gam2', 'al1', 'be1', 'ga1', 'alm1', 'bem1', 'gam1', 'al0', 'be0', 'ga0'):
    P[nm], S[nm] = laurent_poly(nm, 1)
X = {3: P['lam']*eps, -3: P['lamb']*epsb,
     2: frame_vector(P['al2'], P['be2'], P['ga2']), -2: frame_vector(P['alm2'], P['bem2'], P['gam2']),
     1: frame_vector(P['al1'], P['be1'], P['ga1']), -1: frame_vector(P['alm1'], P['bem1'], P['gam1']),
     0: frame_vector(P['al0'], P['be0'], P['ga0'])}
numlam = {}
if seed is not None:                      # random nonzero numeric top mode lambda, lambdab (removes all parameters)
    random.seed(seed)
    numlam = {s_: sp.Rational(random.randint(1, 9)*random.choice((-1, 1)), random.randint(1, 4)) for s_ in S['lam'] + S['lamb']}
    X = {j: Xj.subs(numlam) for j, Xj in X.items()}
    print("numeric lambda:", numlam)
c2, c1, WW, Wx = surface_equations(X, io, Nn, kappa=kap)
allunk = [s_ for nm in S for s_ in S[nm] if s_ not in numlam] + [kap]
S['lam'] = [s_ for s_ in S['lam'] if s_ not in numlam]; S['lamb'] = [s_ for s_ in S['lamb'] if s_ not in numlam]
print(f"J = 3, D = 1, iota = {io}, N = {Nn}, lambda {'numeric' if seed is not None else 'symbolic'}: {len(allunk)} complex unknowns; C2 equations {len(c2)}, C1 equations {len(c1)}  ({time.time()-t0:.0f} s)")
# D_j kernels on v^m (resonances): j + (iota - N) m = 0
res = [(j, m) for j in range(-6, 7) for m in range(-3, 4) if j != 0 and j + (io - Nn)*m == 0]
print("resonant (j, m) with D_j v^m = 0 in range:", res)
byk = lambda eqs, k, sub: [e for e in list(dict.fromkeys(sp.expand(cc.subs(sub)) for (kk, m_, cc) in eqs if abs(kk) == k)) if e != 0]

sub = {}
# ---- level E^{+-5}: 2 W_3.W_2 = -D_3 lam D_2 beta_2 = 0 (and conj) => beta_2 = 0, alpha_{-2} = 0 (lam, lamb != 0) ----
print("\nE^5 equations:", [sp.factor(e) for e in byk(c2, 5, sub)][:3], "...")
sub.update({s_: 0 for s_ in S['be2'] + S['alm2']})
print("impose beta_2 = 0, alpha_-2 = 0 => E^+-5 equations all vanish:", byk(c2, 5, sub) + byk(c1, 5, sub) == [])

# ---- level E^{+-4}: -(D_2 gamma_2)^2 - D_3 lam D_1 beta_1 = 0 and the C1 partner (and conj) ----
e4 = byk(c2, 4, sub) + byk(c1, 4, sub)
print(f"\nE^+-4: {len(e4)} equations; solve for beta_1, alpha_-1 (linear, coefficient D_3 lam) and the constraints on (lam, gamma_2):")
t1 = time.time()
sol4 = sp.solve(e4, S['be1'] + S['alm1'] + S['ga2'] + S['gam2'], dict=True)
print(f"   {len(sol4)} branches ({time.time()-t1:.0f} s)")
for s4 in sol4:
    print("   ", {k: sp.factor(vv) for k, vv in s4.items()})
branches = []
for s4 in sol4:
    sub4 = dict(sub); sub4.update(s4)
    # E^{+-3}
    e3 = byk(c2, 3, sub4) + byk(c1, 3, sub4)
    unk3 = [s_ for s_ in S['al2'] + S['bem2'] + S['be0'] + S['al0'] + S['ga1'] + S['gam1'] if s_ not in sub4]
    t1 = time.time()
    sol3 = sp.solve(e3, unk3, dict=True)
    print(f"\n   branch {s4}: E^+-3: {len(e3)} equations -> {len(sol3)} solutions ({time.time()-t1:.0f} s)")
    for s3 in sol3:
        print("      ", {k: sp.factor(vv) for k, vv in s3.items()})
        sub3 = dict(sub4); sub3.update(s3)
        e2 = byk(c2, 2, sub3) + byk(c1, 2, sub3)
        unk2 = [s_ for s_ in allunk if s_ not in sub3 and s_ not in S['lam'] + S['lamb'] + [kap]]
        t1 = time.time()
        sol2 = sp.solve(e2, unk2, dict=True)
        print(f"         E^+-2: {len(e2)} equations -> {len(sol2)} solutions ({time.time()-t1:.0f} s)")
        for s2 in sol2:
            print("            ", {k: sp.factor(vv) for k, vv in s2.items()})
            sub2 = dict(sub3); sub2.update(s2)
            e1 = byk(c2, 1, sub2) + byk(c1, 1, sub2)
            unk1 = [s_ for s_ in allunk if s_ not in sub2 and s_ not in S['lam'] + S['lamb'] + [kap]]
            t1 = time.time()
            sol1 = sp.solve(e1, unk1, dict=True)
            print(f"            E^+-1: {len(e1)} equations -> {len(sol1)} solutions ({time.time()-t1:.0f} s)")
            for s1 in sol1:
                comb = dict(sub2); comb.update(s1)
                Xs_ = {j: X[j].subs(comb).applyfunc(sp.simplify) for j in (2, 1, -1, -2)}
                x0xy = sp.simplify(sp.expand(X[0].subs(comb).dot(epsb)*2))
                print(f"               X_2 = {list(Xs_[2])}, X_1 = {list(Xs_[1])}, X_-1 = {list(Xs_[-1])}, X_-2 = {list(Xs_[-2])}; (x+iy) of X_0 = {sp.factor(x0xy)}")
                branches.append((comb, Xs_, x0xy))
trivial = all(all(Xs_[j] == sp.zeros(3, 1) for j in Xs_) and sp.diff(x0xy, v) == 0 for comb, Xs_, x0xy in branches)
print(f"\nRESULT J = 3, D = 1 (iota = {io}, N = {Nn}): {len(branches)} terminal solutions of the E^5..E^1 hierarchy with lambda != 0;"
      f" all have X_2 = X_1 = X_-1 = X_-2 = 0 and X_0 = const + gamma_0(v) e_z (x+iy = c + lambda(v) E^3, triply covered surface of revolution): {trivial}")
print(f"done in {time.time()-t0:.0f} s")
