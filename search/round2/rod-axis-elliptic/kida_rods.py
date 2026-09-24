"""TASK 4c: Kida rods (alpha = 0: the Killing field of the rod is a pure rotation, N meridional) and their exactly solvable
first-order points.  With alpha = 0 the closure conditions (curvature extremum at phi = pi/nfp with Z = 0) are two equations
for (beta, lamODE) at fixed R_max = 1: isolated rods.  All deg S <= 1 branches (round 1, branches_scan.evaluate) and the
deg S = 2 branches (deg2_branches_eval.evaluate) are evaluated; we record sign(j) sign(k) and |j|.
Run: python3 kida_rods.py     (~3-5 min: 20 fsolve starts per nfp)
"""
import sys, io, contextlib, time
import numpy as np
from scipy.optimize import fsolve
sys.path.insert(0, '/home/user/QS_MHS_search/search/round2/rod-axis-elliptic')
sys.path.insert(0, '/home/user/QS_MHS_search/search/round1/rod-axis-potential')
from rods import shoot, rod_moduli
with contextlib.redirect_stdout(io.StringIO()):
    from branches_scan import evaluate as eval_d01
    import deg2_branches_eval as d2
t0 = time.time()
for nfp in (2, 3):
    sols = []
    for bg in (0.15, 0.4, 0.7, -0.3):
        for lg in (-0.4, -0.8, -1.5, 0.6, 1.2):
            try:
                p, info, ier, msg = fsolve(lambda p: shoot(0.0, p[0], p[1], nfp), [bg, lg], xtol=1e-10, maxfev=80, full_output=True)
            except Exception:
                continue
            if ier == 1 and np.sum(np.abs(shoot(0.0, p[0], p[1], nfp))) < 1e-7 and abs(p[0]) > 2e-2 and abs(p[1]) > 1e-3:
                r = rod_moduli(0.0, p[0], p[1], nfp)
                if r is not None and r['closure'] < 1e-6 and not any(abs(x['beta'] - r['beta']) < 1e-5 and abs(x['lam'] - r['lam']) < 1e-5 for x in sols):
                    sols.append(r)
    print(f"\n#### alpha = 0 (Kida) closed (1,{nfp}) rods with R_max = 1: {len(sols)} found  [{time.time()-t0:.0f}s] ####")
    for r in sols:
        print(f"  lamODE={r['lam']:+.5f} beta={r['beta']:+.5f} L={r['L']:.4f} l2={r['l2']:.4f} l3={r['l3']:.4f} c={r['c']:.4f} C={r['C']:.4f} tau0={r['tau0']:.4f} p2={r['p2']:+.4f} p1={r['p1']:.4f} p0={r['p0']:+.4f} roots={np.round(r['roots'],4)} closure={r['closure']:.0e}  (q1-kmin^2={r['roots'][1]-r['kmin']**2:+.1e}, q2-kmax^2={r['roots'][2]-r['kmax']**2:+.1e})")
        pts = eval_d01(r) + [p for p in d2.evaluate(r) if not p['pole']]
        for p in pts:
            print(f"      {p['tag']:34s} k={p['k']:+.4f} iota0-N={p['iota']:+.4f} eta^2={p['eta2']:.4f} j={p['j']:+.4f}  sign(j)=sign(k): {np.sign(p['j'])==np.sign(p['k'])}")
        if pts:
            print(f"      => {len(pts)} points, all with sign j = sign k: {all(np.sign(p['j'])==np.sign(p['k']) for p in pts)}, min|j| = {min(abs(p['j']) for p in pts):.3f}")
