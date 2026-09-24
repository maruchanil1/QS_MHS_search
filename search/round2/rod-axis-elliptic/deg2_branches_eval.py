"""TASK 4a (evaluation): are the deg S = 2 Kovacic case-1 branches of kovacic_deg2.py genuine, and what are their
(k, eta^2, j) on closed rods?  For each exponent pattern the K-polynomial factors common to all resultants are loaded
from deg2_branches.pkl; for a numerical rod (q1,q2,q3,C,tau0,L) every positive root K of each factor is tested:
   s1 = s1(K) (linear equation), s0 = common root of the remaining equations (checked to 1e-8 relative),
   S = q^2 + s1 q + s0 must have no zero on [q1, q2] (sigma finite),  eta^2 from K eta^4 + 2 C k eta^2 - kb = 0 (eta^2 > 0),
   j = ka/(k eta^2) + 2 tau0 = a/eta^2 + 2 tau0,  k = +-sqrt(K),  iota0 - N = k L/(2 pi).
Then sign(j) sign(k) and min|j| over the closed (1,2) and (1,3) families.
Run: python3 deg2_branches_eval.py     (~1.5 min; needs deg2_branches.pkl from kovacic_deg2.py)
"""
import pickle, sys, io, contextlib, time
import numpy as np, sympy as sp
from sympy import *          # names for eval(srepr)
sys.path.insert(0, '/home/user/QS_MHS_search/search/round2/rod-axis-elliptic')
from rods import family12, family13, find_rod

K, s0, s1, q1, q2, q3 = sp.symbols('K s0 s1 q1 q2 q3')
D = pickle.load(open('deg2_branches.pkl', 'rb'))
branches = []
for tag, d in D.items():
    s1e = eval(d['s1']); piv = [eval(p) for p in d['pivots']]; Kf = [eval(f) for f in d['Kfactors']]
    A = eval(d['A']); B = eval(d['B'])
    fs1 = sp.lambdify((K, q1, q2, q3), s1e, 'numpy')
    fpiv = [sp.lambdify((K, s0, q1, q2, q3), p, 'numpy') for p in piv]
    piv_coeffs = [sp.lambdify((K, q1, q2, q3), sp.Poly(p, s0).all_coeffs(), 'numpy') for p in piv]
    fA = sp.lambdify((K, s0, s1, q1, q2, q3), A, 'numpy'); fB = sp.lambdify((K, s0, s1, q1, q2, q3), B, 'numpy')
    for fi, f in enumerate(Kf):
        degK = sp.degree(f, K)
        fKc = sp.lambdify((q1, q2, q3), sp.Poly(f, K).all_coeffs(), 'numpy')
        branches.append(dict(tag=tag, e0=d['e0'], e3=d['e3'], factor_deg=degK, fKc=fKc, fs1=fs1, fpiv=fpiv, piv_coeffs=piv_coeffs, fA=fA, fB=fB, fi=fi))
print(f"loaded {len(branches)} (pattern, K-factor) branches:", [(b['tag'], b['factor_deg']) for b in branches])

def evaluate(rod, verbose=False):
    qq3, qq1, qq2 = rod['roots']; C = rod['C']; tau0 = rod['tau0']; L = rod['L']
    out = []
    for br in branches:
        co = np.atleast_1d(np.array(br['fKc'](qq1, qq2, qq3), float))
        Kroots = [r.real for r in np.roots(co) if abs(r.imag) < 1e-9 * max(1, abs(r)) and r.real > 1e-10] if len(co) > 1 else []
        for Kr in Kroots:
            s1v = float(br['fs1'](Kr, qq1, qq2, qq3))
            # s0 candidates: roots of the lowest-degree pivot; keep those where all pivots vanish
            cands = []
            for pc in br['piv_coeffs']:
                cc = np.atleast_1d(np.array(pc(Kr, qq1, qq2, qq3), float))
                if len(cc) > 1:
                    cands += [r.real for r in np.roots(cc) if abs(r.imag) < 1e-7 * max(1, abs(r))]
            good = []
            for s0v in cands:
                res = max(abs(float(fp(Kr, s0v, qq1, qq2, qq3))) / (1 + abs(float(fp(Kr, 0.0, qq1, qq2, qq3))) + abs(s0v)**3) for fp in br['fpiv'])
                if res < 1e-7 and not any(abs(s0v - g) < 1e-7 for g in good):
                    good.append(s0v)
            for s0v in good:
                # S has no zero in [q1, q2]?
                Sroots = np.roots([1, s1v, s0v])
                pole = any(abs(r.imag) < 1e-9 and qq1 - 1e-9 <= r.real <= qq2 + 1e-9 for r in Sroots)
                Av = float(br['fA'](Kr, s0v, s1v, qq1, qq2, qq3)); Bv = float(br['fB'](Kr, s0v, s1v, qq1, qq2, qq3))
                for ksign in (+1, -1):
                    kv = ksign * np.sqrt(Kr)
                    e2 = np.roots([Kr, 2 * C * kv, -Bv])
                    for idx, ev in enumerate(sorted(e2.real)):
                        if abs(e2[0].imag) > 1e-9 or ev <= 1e-12:
                            continue
                        jv = Av / (kv * ev) + 2 * tau0
                        out.append(dict(tag=br['tag'] + f"#f{br['fi']}(deg{br['factor_deg']})", K=Kr, k=kv, s0=s0v, s1=s1v, eta2=ev, j=jv,
                                        iota=kv * L / (2 * np.pi), pole=pole, key=(br['tag'] + f"#f{br['fi']}", ksign, idx)))
    return out

if __name__ == '__main__':
    t0 = time.time()
    r0 = find_rod(-0.4273, [0.4632, -0.3408], 2)
    print(f"\n#### deg S = 2 branches on the QA-matched (1,2) rod (lamODE = -0.4273): q = {np.round(r0['roots'], 4)}, C = {r0['C']:.4f}, tau0 = {r0['tau0']:.4f} ####")
    pts = evaluate(r0)
    print(f"   {len(pts)} admissible (eta^2 > 0) points, of which {sum(p['pole'] for p in pts)} have a zero of S in [q1,q2] (sigma pole, excluded below)")
    for p in pts:
        if not p['pole']:
            print(f"   {p['tag']:34s} k={p['k']:+.4f} iota0-N={p['iota']:+.4f} s0={p['s0']:+.4f} s1={p['s1']:+.4f} eta^2={p['eta2']:.4f} j={p['j']:+.4f}")

    for label, fam in (("closed (1,2) rods", family12()), ("closed (1,3) rods", family13())):
        table = {}
        for r in fam:
            for p in evaluate(r):
                if not p['pole']:
                    table.setdefault(p['key'], []).append((r['lam'], p['j'], p['k'], p['iota'], p['eta2']))
        print(f"\n#### {label}: {len(fam)} rods, lamODE in [{fam[0]['lam']:.3f}, {fam[-1]['lam']:.3f}] ####")
        viol = 0; minj = np.inf
        for key, rows in sorted(table.items()):
            j = np.array([x[1] for x in rows]); kk = np.array([x[2] for x in rows]); io_ = np.array([x[3] for x in rows])
            same = np.all(np.sign(j) == np.sign(kk)); viol += (not same); minj = min(minj, np.min(np.abs(j)))
            print(f"   branch {key[0]:28s} sign k={key[1]:+d} eta-root#{key[2]}: {len(rows):3d} rods, sign(j)=sign(k) on all: {same}, min|j| = {np.min(np.abs(j)):.3f}, j in [{j.min():+.2f},{j.max():+.2f}], iota0-N in [{io_.min():+.3f},{io_.max():+.3f}]")
        print(f"   => deg-2 branches violating sign j = sign k: {viol}; overall min |j| = {minj:.3f}   [{time.time()-t0:.0f}s]")
