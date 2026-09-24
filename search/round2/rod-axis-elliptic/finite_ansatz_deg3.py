"""TASK 5 (part 1): finite global ansatz around an exactly solvable (elliptic) first-order QS point on a Kirchhoff rod --
the degree-3 coefficient identities, decided EXACTLY in the finite-gap ring.

Setting.  Frenet tube coordinates x = gamma(s) + X N + Y Bn, scaled transverse variables Xh = kappa X, Yh = kappa Y
(so that all coefficient functions are even in kappa and live in the ring
        R = Q[q, 1/q, q'] / (q'^2 - P(q)),   q = kappa^2,   P = -q^3 + p2 q^2 + p1 q + p0,   tau = tau0 + C/q,
i.e. R = the elliptic function field of the rod's curvature).  Ansatz (all polynomial in (Xh, Yh)):
     Pi  = Pi0 + Xh + Pi2 + Pi3,        Pi2 = exact on-axis Hessian of the first-order QS point (round 1 closed forms),
     psi = psi2 + psi3,                 psi2 = the first-order flux ellipse,
     B   = T + B1 + B2,                 B1 = exact first-order field (transverse block M of grad B),
     p   = -E(psi), E = E0 + E1 psi.
Unknowns: B2 (9 coefficient functions), Pi3 (4), psi3 (4), E1 (constant).  Exact equations imposed as polynomial identities:
     (FB)  h [B.grad B - grad Pi] = 0                    at degree 2          (h = 1 - Xh)
     (DIV) h div B = 0                                   at degree 1
     (SURF) h B.grad psi = 0                             at degree 3
     (A1)  Pi - |B|^2/2 + E(psi) = 0                     at degree 2
     (QS)  h^3 grad psi x grad Pi . grad(B.grad Pi) = 0  at degree 2   (Helander triple product, (I3) of round 1)
and, as consistency checks, all lower-degree parts must vanish identically in R (they are the first-order QS conditions).
The 21 identities are linear in the unknown functions and their first s-derivatives.  Each unknown is sought in R with
bounded pole order at q = 0:  u = sum_{n=-N}^{N} (c_n q^n + d_n q^n q').  The result is a linear system over Q whose
solution space is computed exactly (DomainMatrix over QQ).  A nonempty solution set = "the finite-gap structure closes at
degree 3"; an empty one = the obstruction (reported equation by equation).

Validation: the same code, with constant coefficients, is run on the exact Solov'ev equilibrium of osc112_qs.py (circular
axis, iota = 2, p' != 0): the Taylor coefficients of its exact (Pi, psi, B) must satisfy all identities.
Run: python3 finite_ansatz_deg3.py   (~1-2 min)
"""
import sympy as sp, time, itertools, sys
from sympy.polys.matrices import DomainMatrix
from sympy import QQ

Xh, Yh = sp.symbols('Xh Yh')
t_start = time.time()

# ------------------------------------------------------------------ polynomial helpers in (Xh, Yh)
def poly_terms(e):
    e = sp.expand(e)
    if e == 0:
        return {}
    Pl = sp.Poly(e, Xh, Yh)
    return {m: c for m, c in Pl.terms()}

def trunc(e, deg):
    return sum((c * Xh**i * Yh**j for (i, j), c in poly_terms(e).items() if i + j <= deg), sp.Integer(0))

def deg_part(e, d):
    return {m: c for m, c in poly_terms(e).items() if sum(m) == d}

# ------------------------------------------------------------------ generic builder of the identities
def identities(kap, tau_c, ds, kap1_over_kap, BT, bN, bB, Pi, psi, E0, E1, need=None):
    """returns dict name -> polynomial in (Xh,Yh), each truncated at its own degree need[name]
    (FB 2, DIV 1, SURF 3, A1 2, QS 2 by default; all intermediate products truncated accordingly)."""
    if need is None:
        need = {'FB': 2, 'DIV': 1, 'SURF': 3, 'A1': 2, 'QS': 2}
    h = 1 - Xh
    def invh(n):
        return sum(Xh**i for i in range(n + 1))
    def Ds(f, n):
        return trunc(ds(f) + kap1_over_kap * (Xh * sp.diff(f, Xh) + Yh * sp.diff(f, Yh)) + tau_c * (Yh * sp.diff(f, Xh) - Xh * sp.diff(f, Yh)), n)
    def T(e, n):
        return trunc(e, n)
    BN = kap * bN; BB = kap * bB
    Bvec = [BT, BN, BB]
    Gam = sp.Matrix([[0, -kap, 0], [kap, 0, -tau_c], [0, tau_c, 0]])
    out = {}
    # (FB)
    n = need['FB']
    gradPi_h = [Ds(Pi, n), T(h * kap * sp.diff(Pi, Xh), n), T(h * kap * sp.diff(Pi, Yh), n)]
    for i, nm in enumerate('TNB'):
        conv = T(T(BT, n) * T(Ds(Bvec[i], n) + sum(Gam[i, j] * Bvec[j] for j in range(3)), n), n) \
             + T(h * kap**2 * T(T(bN, n) * T(sp.diff(Bvec[i], Xh), n) + T(bB, n) * T(sp.diff(Bvec[i], Yh), n), n), n)
        out['FB_' + nm] = T(conv - gradPi_h[i], n)
    # (DIV)
    n = need['DIV']
    out['DIV'] = T(Ds(BT, n) - kap**2 * bN + h * kap**2 * (sp.diff(bN, Xh) + sp.diff(bB, Yh)), n)
    # (SURF)
    n = need['SURF']
    out['SURF'] = T(T(BT, n) * Ds(psi, n) + h * kap**2 * T(T(bN, n) * T(sp.diff(psi, Xh), n) + T(bB, n) * T(sp.diff(psi, Yh), n), n), n)
    # (A1)
    n = need['A1']
    out['A1'] = T(Pi - (T(BT, n)**2 + kap**2 * (T(bN, n)**2 + T(bB, n)**2)) / 2 + E0 + E1 * psi, n)
    # (QS): W = B.grad Pi = BT Ds(Pi)/h + kap^2 (bN Pi_X + bB Pi_Y); h^3 triple = det[h grad psi; h grad Pi; h grad W]
    n = need['QS']
    W = T(T(T(BT, n) * Ds(Pi, n), n) * invh(n) + kap**2 * T(T(bN, n) * T(sp.diff(Pi, Xh), n) + T(bB, n) * T(sp.diff(Pi, Yh), n), n), n)
    gpsi = [Ds(psi, n), T(h * kap * sp.diff(psi, Xh), n), T(h * kap * sp.diff(psi, Yh), n)]          # degree <= n
    gPi = [Ds(Pi, n - 1), T(h * kap * sp.diff(Pi, Xh), n - 1), T(h * kap * sp.diff(Pi, Yh), n - 1)]   # degree <= n-1 suffices (psi row has degree >= 1)
    gW = [Ds(W, n - 1), T(h * kap * sp.diff(W, Xh), n - 1), T(h * kap * sp.diff(W, Yh), n - 1)]
    det = 0
    for perm, sgn in (((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1), ((0, 2, 1), -1), ((2, 1, 0), -1), ((1, 0, 2), -1)):
        det += sgn * T(T(gpsi[perm[0]] * gPi[perm[1]], n) * gW[perm[2]], n)
    out['QS'] = T(det, n)
    return out

# ================================================================== (A) VALIDATION on the exact Solov'ev equilibrium
print("=" * 30, "(A) validation: Solov'ev equilibrium from the 1:1:2 congruence (circular axis, tau = 0)", "=" * 30)
rho, z = sp.symbols('rho z', positive=True)
S, kp, dl = sp.Rational(2), sp.Rational(3, 2), sp.pi / 3          # rational-ish parameters (delta = pi/3: sin, cos algebraic)
m2 = (rho**2 - S)**2 / 4 + (sp.cos(dl) * (rho**2 - S) / 2 - z / kp)**2 / sp.sin(dl)**2
Psi = -kp * sp.sin(dl) * m2
F = sp.sqrt(S**2 - 4 * m2)
Brho = -sp.diff(Psi, z) / rho; Bz = sp.diff(Psi, rho) / rho; Bphi = F / rho
V = (rho**2 + 4 * z**2) / 2
kap_S = 1 / sp.sqrt(S); B0 = sp.sqrt(S)
# Frenet: T = e_phi, N = -e_rho, Bn = e_z; rho = sqrt(S) - X, z = Y; scaled Xh = kap X, Yh = kap Y; units B0 = 1, length: kappa in units of R0? we keep dimensional but divide B by B0 and Pi by B0^2
eps = sp.symbols('epsilon')
X = Xh / kap_S; Y = Yh / kap_S
sub_XY = {rho: sp.sqrt(S) - eps * X, z: eps * Y}
def taylor(expr, deg):
    e = expr.subs(sub_XY)
    ser = sp.series(e, eps, 0, deg + 1).removeO()
    return sp.expand(ser.subs(eps, 1))
BT_S = taylor(Bphi / B0, 3)
BN_S = taylor(-Brho / B0, 3); BB_S = taylor(Bz / B0, 3)
Pi_S = taylor(-V / B0**2, 3)
psi_S = taylor(m2, 3)
E_S = taylor((Brho**2 + Bz**2 + Bphi**2) / (2 * B0**2) + V / B0**2, 2)     # E = B^2/2 + V  (units B0^2), must be E0 + E1 psi2
psi2_S = trunc(psi_S, 2)
E0_S = E_S.subs({Xh: 0, Yh: 0}); E1_S = sp.simplify((E_S - E0_S) / psi2_S)
print("   E = B^2/2 + V to degree 2 is E0 + E1 psi2 with E1 =", E1_S, "; ratio simplifies to a constant:", E1_S.is_constant())
print("   axis check: B_T(0) =", BT_S.subs({Xh: 0, Yh: 0}), " Pi_1 =", deg_part(Pi_S, 1), " (should be {(1,0): 1} in scaled units)")
# unit conventions: in the scaled variables Pi = Pi0 + Xh + ... requires Pi in units B0^2 and lengths so that kappa X = Xh: ok by construction
ids_S = identities(kap_S, sp.Integer(0), lambda f: sp.Integer(0), sp.Integer(0), BT_S, BN_S / kap_S, BB_S / kap_S, Pi_S, psi_S, -(Pi_S.subs({Xh: 0, Yh: 0})) + sp.Rational(1, 2), E1_S, need={'FB': 3, 'DIV': 3, 'SURF': 3, 'A1': 3, 'QS': 3})
for nm, e in ids_S.items():
    parts = {d: {m: sp.nsimplify(sp.simplify(c)) for m, c in deg_part(e, d).items()} for d in range(4)}
    nz = {d: {m: c for m, c in p.items() if c != 0} for d, p in parts.items()}
    print(f"   {nm:5s}: nonzero coefficients by degree: {nz}")
print(f"   [Solov'ev validation done, {time.time()-t_start:.1f}s]  (expected: all zero up to the degrees FB<=2, DIV<=1, SURF<=3, A1<=2, QS<=2; degree-3 parts of FB/DIV/A1 need B3 and are not meaningful)")

# ================================================================== (B) the elliptic point
print("\n" + "=" * 30, "(B) exactly solvable S=1 branch on a rod: degree-3 closure in the elliptic ring", "=" * 30)
q, qp, kap = sp.symbols('q qp kappa')

def run_point(k, eta2, p1, tau0, Nmax=4, label=""):
    t0 = time.time()
    C = k * eta2                       # branch A: eta^2 = C/k
    p2 = -4 * k**2; p0 = -4 * C**2
    P = -q**3 + p2 * q**2 + p1 * q + p0; Pp = sp.diff(P, q)
    rts = sp.Poly(P, q).nroots()
    print(f"\n--- point {label}: k = {k}, eta^2 = {eta2}, C = {C}, p2 = {p2}, p1 = {p1}, p0 = {p0}, tau0 = {tau0};  roots of P: {[sp.N(r, 5) for r in rts]}")
    tau = tau0 + C / q
    sig = -qp / (2 * k * q)
    kap1 = qp / (2 * kap)                       # kappa'
    # jets of the unknown coefficient functions
    mon2 = [(2, 0), (1, 1), (0, 2)]; mon3 = [(3, 0), (2, 1), (1, 2), (0, 3)]
    U = {}; U1 = {}; U2 = {}
    for nm, mons in (('bT', mon2), ('bN', mon2), ('bB', mon2), ('Pi3', mon3), ('psi3', mon3)):
        for m in mons:
            U[(nm, m)] = sp.Symbol(f'{nm}_{m[0]}{m[1]}'); U1[(nm, m)] = sp.Symbol(f'{nm}_{m[0]}{m[1]}_s'); U2[(nm, m)] = sp.Symbol(f'{nm}_{m[0]}{m[1]}_ss')
    E1 = sp.Symbol('E1')
    def ds(f):
        r = sp.diff(f, q) * qp + sp.diff(f, qp) * Pp / 2 + sp.diff(f, kap) * kap1
        for key, u in U.items():
            r += sp.diff(f, u) * U1[key]
        for key, u1 in U1.items():
            r += sp.diff(f, u1) * U2[key]
        for key, u2 in U2.items():
            if f.has(u2):
                raise RuntimeError("third derivative of an unknown appeared: " + str(key))
        return r
    def red(e):
        """reduce qp^2 -> P and kappa^n -> q^(n//2) kappa^(n%2); return expression"""
        e = sp.expand(e)
        d = sp.collect(e, qp, evaluate=False)
        r = 0
        for key, c in d.items():
            n = 0 if key == 1 else (1 if key == qp else int(key.exp))
            r += c * qp**(n % 2) * P**(n // 2)
        r = sp.expand(r)
        d = sp.collect(r, kap, evaluate=False)
        out = 0
        for key, c in d.items():
            n = 0 if key == 1 else (1 if key == kap else int(key.exp))
            a, b = n // 2, n % 2
            out += c * q**a * kap**b
        return sp.expand(out)
    # first-order data
    kap2 = ds(kap1)
    tau1 = ds(tau)
    sig1 = ds(sig)
    jval = p1 / (2 * k * eta2) + 2 * tau0
    ric = sp.simplify(red(sig1 - (-k * (1 + sig**2) - k * eta2**2 / q**2 + eta2 * (jval - 2 * tau) / q)))
    print("   Riccati residual (must be 0):", ric)
    # closed forms (round 1) with symbols kappa kappa1 kappa2 tau tau1 sigma eta k j
    import pickle
    d = pickle.load(open('/home/user/QS_MHS_search/search/round1/rod-axis-potential/hessian_closed_forms.pkl', 'rb'))
    sy = {n: sp.Symbol(n, real=True) for n in d['symbols'].split()}
    sy_pos = {n: sp.Symbol(n, positive=True) for n in ('eta', 'k')}
    def load(name):
        e = eval(d[name], vars(sp))
        e = e.subs({sp.Symbol('kappa', real=True): kap, sp.Symbol('kappa1', real=True): kap1, sp.Symbol('kappa2', real=True): kap2,
                    sp.Symbol('tau', real=True): tau, sp.Symbol('tau1', real=True): tau1, sp.Symbol('sigma', real=True): sig,
                    sp.Symbol('eta', positive=True): sp.sqrt(eta2), sp.Symbol('k', positive=True): k, sp.Symbol('j', real=True): jval})
        return red(e)
    PNN, PNB, PBB = load('P_NN'), load('P_NB'), load('P_BB')
    # transverse block M of grad B on the axis (strain_kernel_axis.py)
    MNN = red(k * sig - kap1 / kap); MNB = red(-eta2 * k / q - tau)
    MBN = red((eta2 * tau + k * q * sig**2 + k * q + q * sig1) / eta2); MBB = -MNN
    jchk = sp.simplify(red(MBN - MNB) - jval)
    print("   j = M_BN - M_NB - j_closed (must be 0):", jchk)
    # fields (hatted): B1_N = M_NN X + M_NB Y = kap * (M_NN Xh + M_NB Yh)/q
    bN1 = (MNN * Xh + MNB * Yh) / q; bB1 = (MBN * Xh + MBB * Yh) / q
    Pi2 = (PNN * Xh**2 + 2 * PNB * Xh * Yh + PBB * Yh**2) / (2 * q)
    psi2 = ((q / eta2) * (1 + sig**2) * Xh**2 - 2 * sig * Xh * Yh + (eta2 / q) * Yh**2) / (2 * q)
    bT2 = sum(U[('bT', m)] * Xh**m[0] * Yh**m[1] for m in mon2)
    bN2 = sum(U[('bN', m)] * Xh**m[0] * Yh**m[1] for m in mon2)
    bB2 = sum(U[('bB', m)] * Xh**m[0] * Yh**m[1] for m in mon2)
    Pi3 = sum(U[('Pi3', m)] * Xh**m[0] * Yh**m[1] for m in mon3)
    psi3 = sum(U[('psi3', m)] * Xh**m[0] * Yh**m[1] for m in mon3)
    Pi0 = sp.Rational(1, 2) * 0            # Pi0 arbitrary constant; E0 = 1/2 - Pi0
    BT = 1 + Xh + bT2
    Pi = Pi0 + Xh + Pi2 + Pi3
    psi = psi2 + psi3
    ids = identities(kap, tau, ds, kap1 / kap, BT, bN1 + bN2, bB1 + bB2, Pi, psi, sp.Rational(1, 2) - Pi0, E1)
    print(f"   identities built [{time.time()-t0:.1f}s]")
    # reduce and split by degree and kappa-parity
    need = {'FB_T': 2, 'FB_N': 2, 'FB_B': 2, 'DIV': 1, 'SURF': 3, 'A1': 2, 'QS': 2}
    eqs = []          # (name, monomial, expression in ring with unknown jets)
    for nm, e in ids.items():
        er = red(e)
        # kappa parity: after red, expression is c0 + kappa*c1 ; both must vanish separately
        dpar = sp.collect(er, kap, evaluate=False)
        for d_ in range(need[nm] + 1):
            for key, c in dpar.items():
                part = deg_part(c, d_)
                for m, cc in part.items():
                    cc = sp.together(cc)
                    num = sp.expand(sp.numer(cc)); den = sp.denom(cc)
                    if d_ < need[nm]:
                        chk = sp.simplify(red(num))
                        if chk != 0:
                            print(f"   !! lower-degree consistency FAILED: {nm} degree {d_} monomial {m} kappa^{0 if key==1 else 1}: {chk}")
                    else:
                        if key != 1:
                            pass
                        eqs.append((nm, m, key, num))
    print(f"   lower-degree (first-order) identities checked; {len(eqs)} closure equations at the required degrees [{time.time()-t0:.1f}s]")
    # count how many involve unknowns
    unk_syms = set(U.values()) | set(U1.values()) | set(U2.values()) | {E1}
    bad = [(nm, m) for nm, m, key, ex in eqs if ex.free_symbols & set(U2.values())]
    print(f"   equations containing second derivatives of unknowns (must be none): {bad}")
    eqs_u = [(nm, m, key, ex) for nm, m, key, ex in eqs if ex.free_symbols & unk_syms]
    eqs_0 = [(nm, m, key, ex) for nm, m, key, ex in eqs if not (ex.free_symbols & unk_syms)]
    for nm, m, key, ex in eqs_0:
        chk = sp.simplify(red(ex))
        print(f"   equation without unknowns: {nm} {m} kappa^{0 if key==1 else 1} -> {chk}")
    print(f"   {len(eqs_u)} equations involve unknowns; kappa-parity classes present: {set(str(k_) for _,_,k_,_ in eqs_u)}")
    # ---------------- ring ansatz for the unknowns
    coeffs = []
    ans = {}
    for key, u in U.items():
        cs = [sp.Symbol(f'c_{u.name}_{n+Nmax}') for n in range(-Nmax, Nmax + 1)]
        dsym = [sp.Symbol(f'd_{u.name}_{n+Nmax}') for n in range(-Nmax, Nmax + 1)]
        expr = sum(cs[i] * q**n + dsym[i] * q**n * qp for i, n in enumerate(range(-Nmax, Nmax + 1)))
        ans[u] = expr
        ans[U1[key]] = sp.expand(sp.diff(expr, q) * qp + sp.diff(expr, qp) * Pp / 2)
        coeffs += cs + dsym
    coeffs.append(E1)
    rows = []; row_tags = []
    for nm, m, key, ex in eqs_u:
        e = red(ex.subs(ans))
        e = sp.expand(e * q**60)          # clear poles (generous)
        # collect coefficients of q^n and q^n qp
        Pl = sp.Poly(e, q, qp)
        for mono, c in Pl.terms():
            rows.append(c); row_tags.append((nm, m, key, mono))
    print(f"   linear system: {len(rows)} equations, {len(coeffs)} unknown rational constants [{time.time()-t0:.1f}s]")
    # build matrix over QQ
    A = []; bvec = []
    for r_ in rows:
        r_ = sp.expand(r_)
        lin = sp.Poly(r_, *coeffs)
        rowd = dict(zip(coeffs, [0] * len(coeffs)))
        const = 0
        for mono, c in lin.terms():
            if sum(mono) == 0:
                const = c
            elif sum(mono) == 1:
                idx = mono.index(1); rowd[coeffs[idx]] = c
            else:
                raise RuntimeError("nonlinear term in unknown constants")
        A.append([sp.Rational(rowd[c_]) for c_ in coeffs]); bvec.append(sp.Rational(-const))
    Am = DomainMatrix([[QQ.convert(x) for x in row] for row in A], (len(A), len(coeffs)), QQ)
    Aug = DomainMatrix([[QQ.convert(x) for x in row] + [QQ.convert(bb)] for row, bb in zip(A, bvec)], (len(A), len(coeffs) + 1), QQ)
    rA = Am.rank(); rAug = Aug.rank()
    print(f"   rank A = {rA}, rank [A|b] = {rAug}, unknowns = {len(coeffs)}  ->  {'CONSISTENT' if rA == rAug else 'INCONSISTENT'}; solution-space dimension = {len(coeffs) - rA if rA == rAug else 'n/a'}  [{time.time()-t0:.1f}s]")
    if rA != rAug:
        # find a minimal inconsistent subsystem by equation name: drop one identity family at a time
        fams = sorted(set(nm for nm, _, _, _ in row_tags))
        for fam in fams:
            idx = [i for i, t in enumerate(row_tags) if t[0] != fam]
            Am2 = DomainMatrix([[QQ.convert(x) for x in A[i]] for i in idx], (len(idx), len(coeffs)), QQ)
            Aug2 = DomainMatrix([[QQ.convert(x) for x in A[i]] + [QQ.convert(bvec[i])] for i in idx], (len(idx), len(coeffs) + 1), QQ)
            print(f"      without {fam:5s}: rank A = {Am2.rank()}, rank [A|b] = {Aug2.rank()} -> {'consistent' if Am2.rank()==Aug2.rank() else 'inconsistent'}")
        return None
    # particular solution and nullspace
    rref, pivs = Aug.rref()
    M_ = rref.to_Matrix()
    piv = [j for j in pivs if j < len(coeffs)]
    part = {c_: sp.Integer(0) for c_ in coeffs}
    for i, j0 in enumerate(piv):
        part[coeffs[j0]] = M_[i, len(coeffs)]           # free (non-pivot) variables set to 0
    free = [coeffs[j] for j in range(len(coeffs)) if j not in piv]
    # verify the particular solution exactly
    resid = max(abs(sum(A[i][j] * part[coeffs[j]] for j in range(len(coeffs))) - bvec[i]) for i in range(len(A)))
    print(f"   particular solution residual (exact): {resid}")
    print(f"   E1 is {'FREE' if E1 in free else 'DETERMINED'}: E1 = {part[E1]} (free constants = 0)")
    print(f"   free constants ({len(free)}): {free[:20]}{' ...' if len(free) > 20 else ''}")
    # print the particular solution for the unknown functions (free constants = 0)
    print("   particular degree-3 closure (free constants set to 0):")
    for key, u in U.items():
        val = sp.factor(sp.expand(ans[u].subs(part)))
        if val != 0:
            print(f"      {u.name} = {val}")
    print(f"      E1 = {part[E1]}   (E1 = -dp/dpsi on axis; toroidal flux/2pi as psi, B0 = 1)")
    # which unknown functions are free (appear in the nullspace)?
    Nsp = Am.nullspace().to_Matrix()
    print(f"   nullspace dimension {Nsp.rows}: unknown functions carrying freedom:")
    for i in range(Nsp.rows):
        v = Nsp.row(i)
        sub = {c_: v[j] for j, c_ in enumerate(coeffs)}
        desc = []
        for key, u in U.items():
            val = sp.factor(sp.expand(ans[u].subs(sub)))
            if val != 0:
                desc.append(f"{u.name}={val}")
        if sub[E1] != 0:
            desc.append(f"E1={sub[E1]}")
        print("      ", "; ".join(desc))
    return part

if len(sys.argv) > 1 and sys.argv[1] == "A":
    sys.exit(0)
which = sys.argv[2] if len(sys.argv) > 2 else "1"
Nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
if which == "1":
    run_point(sp.Rational(-1, 2), sp.Integer(2), sp.Integer(7), sp.Rational(9, 10), Nmax=Nmax, label="near lamODE=-0.5 rod (branch A)")
elif which == "2":
    run_point(sp.Rational(-1, 3), sp.Rational(5, 2), sp.Integer(6), sp.Rational(-3, 5), Nmax=Nmax, label="generic second point (branch A, tau0 < 0)")
elif which == "3":   # branch B-like sign structure: C/k < 0 is branch B; here we keep branch A but flip both signs
    run_point(sp.Rational(1, 2), sp.Integer(2), sp.Integer(7), sp.Rational(9, 10), Nmax=Nmax, label="mirror point (k > 0, C > 0)")
print(f"\n[total {time.time()-t_start:.1f}s]")
