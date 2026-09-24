# rod-axis-elliptic — round 2 notes

Direction: exactly solvable (elliptic) first-order QS points on Kirchhoff-rod axes as seeds for a global closed-form
equilibrium; vacuum-vs-current decision; finite global ansatz at an elliptic point decided in the finite-gap ring.
All scripts run with `python3 <script>` from this directory (sympy 1.14, numpy, scipy). Units: B0 = 1, arclength s;
q = κ², q′ = dq/ds, q′² = P(q) = −q³ + p2 q² + p1 q + p0 = −(q−q1)(q−q2)(q−q3), q3 < 0 < q1 < q2, τ = τ0 + C/q,
p0 = −4C², p1 = −(q1q2+q1q3+q2q3), p2 = q1+q2+q3 (rod_exact.py conventions; note p1 > 0 on all rods met so far).
"S=1 branch" = the simplest Kovacic case-1 point: k² = −p2/4, σ = −κ′/(kκ), η² ∈ {C/k (A), −3C/k (B)}, j = p1/(2kη²) + 2τ0.
The exact first-order machinery (closed forms P_NN, P_NB, P_BB, Riccati) is used only as a filter/check.

## 0. Summary of statuses

| # | statement | status | script |
|---|---|---|---|
| 1 | VMEC precise-QA axis: p2 = q1+q2+q3 ≠ 0. Full 9-harmonic axis: E1 (moduli formula) p2 = −0.102 ± 0.018, E4 (exact extrema + moduli C) −0.050 ± 0.012, E2/E3 (Frenet-only fits) −0.038 ± 0.19 / −0.061 ± 0.11; all negative; 4cλ3 − λ2² = −0.040 ± 0.007; p2/(q1+q2) ≈ −0.02…−0.04. The round-1 conjecture p2 = 0 is not supported (4–5σ formal in the two robust estimators; the estimator spread 0.06 is the systematic error from the 0.1–0.3 % non-rod-ness). The axis is *near* the depressed-cubic locus (|p2| ≲ 4 % of the curvature scale) but not on it. | refuted (as an exact identity, at the stated precision) | `vmec_axis_p2.py` |
| 2 | Strain kernel with the Boozer-normalised u = e_φ + N e_θ = [(G+NI)B + (N−ι)B×∇ψ]/B²: S(u)B = 0 holds on the axis and, at O(r), its N,B components determine the transverse gradient of u₂_T (3 unknowns, 4 equations); the single compatibility condition equals (1/2) dj/ds. **No identity beyond the Riccati / j = const.** On the axis S(u)/G0 has transverse block [[−κ′/κ, κ²σ′/(2η²)],[κ²σ′/(2η²), κ′/κ]]: u is Killing on the axis iff κ′ = σ′ = 0. | verified_symbolically | `strain_kernel_axis.py` |
| 3 | Closed forms P_NN, P_BB and the Riccati hold exactly on the Solov'ev-from-1:1:2 congruence (circular axis, ι = 2, p′ ≠ 0, j = 2(Sκ_²+1)/(Sκ_ sin δ) ≠ 0, σ = −cot δ·sign) and on the Kepler force-free family (ι = 1, j = (C+1/C)/a); the j-terms are essential (without them P_BB is wrong). j from curl B agrees with the Grad–Shafranov current. | verified_symbolically | `circ_axis_checks.py` |
| 4a | Kovacic case 1, deg S = 2: for every rod and each exponent pattern (e3, m) the K-polynomial has a NEW cubic and a NEW sextic factor (plus the deg-0 value as a degenerate factor). The branches are genuine (s0 common root of all remaining equations, residual < 1e−7) and admissible (S ≠ 0 on [q1,q2], η² > 0). On 62 closed (1,2) rods and 19 closed (1,3) rods all of them have sign j = sign k; min |j| = 1.16 ((1,2)) and 1.93 ((1,3)). | verified_symbolically (branches) + verified_numerically (sign law) | `kovacic_deg2.py`, `deg2_branches_eval.py` |
| 4b | Kovacic case 2 (log-derivative quadratic over C(q)): E-sets computed from the local exponents; d = 0 patterns solved with symbolic (q1,q2,q3): rod-independent branches exist only for n ∈ {2,3,4,6,8} (n = exponent-difference parameter at q = 0, e0 = 1/2 − n/4); the even-n ones reproduce the case-1 deg-0 branches (or their q_i-permutations with K < 0 or a σ-pole at a turning point); n = 2 is the trivial a = b = 0 solution (σ = ±i, unphysical); **n = 3 gives new quarter-exponent branches**, e.g. K = −p2/16 = K_{S=1}/4, k a = 3p1/16, k b = −5p0/16. Physical admissibility (real Floquet multiplier of Hill's equation, η² > 0) is tested on two exact rational rods (see §4). | verified_symbolically (branch list) ; admissibility: see §4 | `kovacic_case2.py` |
| 4c | Sign law on the S=1 branch: j k = (k/C) f_A/λ3² (A), (−k/C) f_B/(3λ3²) (B) with f_A = λ3²(p1/2 + 2τ0C), f_B = λ3²(p1/2 − 6τ0C); i.e. sign j = sign k ⇔ p1 + 4τ0C > 0 (A), p1 − 12τ0C > 0 (B); j = 0 ⇔ τ0 = −p1/(4C) (A). The moduli (q1,q2,q3,τ0) are independent (Jacobian rank 4), so for every cubic there are (non-closed) rods with j k < 0 and with j = 0: **the sign law is a property of closure**, not of the rod identities; it cannot be proved from the root ordering. On the closed families τ0C < 0 and 4|τ0C|/p1 ∈ [0.025, 0.60] ((1,2)), [0.007, 0.20] ((1,3)). Same structure (j affine in τ0 with slope 2) on all deg-0 branches. | proved (reduction) + verified_numerically (margins) | `sign_j_reduction.py` |
| 4d | Kida rods (α = 0, isolated closed rods): three (1,2)/(1,3) Kida rods found; all exactly solvable points have sign j = sign k; the λ_ODE = −0.5242 Kida rod carries a NEAR-VACUUM point j = −0.00650 (K = −q3/4 branch, k = −0.774, η² = 11.9; refined to 9 digits, exact condition F = +0.060 ≠ 0). Smallest \|j\| found in the whole classification (previous minimum 1.16). | verified_numerically | `kida_rods.py`, `kida_refine.py` |
| 4e | Case-2 physical points: n = 3 quarter-exponent branches are complex (unphysical); n = 5 (e = (1,1,2,3), d = 1) and the n = 4, d = 1 one-parameter family c0 ≥ c0_crit are real, periodic, NON-stellarator-symmetric elliptic σ ∈ C(q,q′) (even part c q/(2k(q+c0))); \|j\| ≥ 1.7 on all of them, sign j = sign k. | verified_symbolically (ω solves the Riccati, 1e−14) + verified_numerically (Floquet) | `case2_admissibility.py`, `case2_verify.py`, `case2_family_scan.py` |
| 5 | Finite global ansatz at the S=1 point, degree-3 coefficient identities decided exactly in the ring R = Q[q,1/q,q′]/(q′²−P): the 18 identities (FB at degree 2, div B at degree 1, B·∇ψ at degree 3, p = −E(ψ) at degree 2, Helander triple product at degree 2) for the 17 unknown coefficient functions (B2: 9, Π3: 4, ψ3: 4) and E1 are **consistent in R**, with a 16-dimensional solution space (Laurent range q^{−4..4}); all lower-degree identities vanish identically (first-order consistency). The elliptic structure propagates exactly to the next degree — no transcendental function is needed. Details, E1 status and robustness (Nmax = 6, second rational point) in §5. | verified_symbolically (exact rational arithmetic at rational rod moduli) | `finite_ansatz_deg3.py` |

No candidate solution. Nothing here is an equilibrium.

## 1. Task 1 — VMEC axis test of p2 = 0 (`vmec_axis_p2.py`, ~40 s)

Data: (A) the 7 axis harmonics printed in handoff §6; (B) all 9 harmonics (n ≤ 16) from `wout_LandremanPaul2021_QA_lowres.nc`
(downloaded from the simsopt test files, handoff §10.1; the .nc files are not committed — see `.gitignore`); (C) truncation
n ≤ 12; (D) the reactor-scale QA (independent VMEC run, ns = 50) rescaled to R_00 = 1.00396. Finite Fourier series ⇒ exact
derivatives (sympy → lambdify), 4096 points, arclength-weighted means. Screw fit α from T·𝓘 = const (least squares),
λ2 = ⟨T·𝓘⟩, λ3 = ⟨𝐁·𝓘/κ⟩, c = ⟨z_s + λ3κ²/2⟩.
Estimators of p2: E1 = 4c/λ3 − λ2²/λ3² (moduli), E2 = q1 + q2 − 4C²/(q1q2) with C from the fit τ = τ0 + C/q,
E3 = coefficient of q² in the least-squares fit of q′² + q³, E4 = q1 + q2 + p0/(q1q2) with exact curvature extrema and the
moduli C. Uncertainties: pointwise stds propagated (E1), split-sample differences (E2, E3), moduli-vs-fit C (E4).

Results (full axis; A is noisier because of the truncation, its E2/E3 are unreliable):
```
        E1              E2              E3              E4          q1+q2   N.I rms  std(l3)/l3
B n<=16  -0.1024+-0.018  -0.0379+-0.193  -0.0610+-0.113  -0.0497+-0.012  2.392   1.6e-3   5.5e-3
C n<=12  -0.1024+-0.018  -0.0379+-0.188  -0.0614+-0.151  -0.0497+-0.012  2.392
D reactor-0.1010+-0.017  -0.0380+-0.179  -0.0617+-0.090  -0.0496+-0.012  2.392
```
λ2 = 0.9646, λ3 = −0.6285 (VMEC orientation), c = −0.3540, α = −0.5626, C = 0.5599; 4cλ3 − λ2² = −0.0404 ± 0.0073.
Verdict: exact zero is disfavoured (all estimators negative; the two robust ones at 4–5σ formal); the axis is close to the
locus (|p2|/(q1+q2) ≈ 2–4 %; the QA-matched exact rod of round 1 has −0.5 %). The S=1 branch on this axis would have
ι0 − N = ±0.170 (k = 0.160): not the QA value 0.4232, consistent with round 1 (the QA point is not elliptic).

## 2. Task 2 — strain kernel on the axis with the Boozer-normalised u (`strain_kernel_axis.py`, ~5 s)

u/G0 = [B − k B×∇ψ]/B² + O(r²) (G = G0 + O(r²), I = O(r²), N − ι0 = −kG0), ψ = (1/2)[(κ²/η²)(1+σ²)X² − 2σXY + (η²/κ²)Y²]
(toroidal flux/2π, det of the quadratic form = 1). General unknown quadratic u₂. S = sym∇u in the Frenet tube frame.
Results: div u = 0 on axis; S(u)B = 0 at O(1); at O(r) the T-component equations vanish identically; the N,B components
contain only the transverse gradient (a0,b0,c0) of u₂_T (4 equations, 3 unknowns); the constraint equals exactly
(1/2)·dj/ds with j = M_BN − M_NB = kη²/κ² + 2τ + (κ²/η²)(k(1+σ²) + σ′). **Decided: the strong-QS strain-kernel condition
adds nothing to the first-order Riccati.** (Coordinator note 1 normalisation used; the sign k → −k would break the
identity, so this also confirms the normalisation.)

## 3. Task 3 — circular-axis j ≠ 0 checks (`circ_axis_checks.py`, ~10 s)

(A) Solov'ev from 1:1:2 (osc112_qs.py §4): Ψ = −κ_ sin δ m², F = ±√(S² − 4m²), V = (ρ² + 4z²)/2. Verified: force balance
(B·∇)B = ∇Π, E = B²/2 + V a function of m² only. Axis ρ = √S, z = 0, κ = 1/√S, B0 = √S. Independent first-order data:
σ = ∓cos δ/|sin δ|, η²/κ² = 1/(√S κ_ |sin δ|) (from Hess m²), k = ±2/√S (ι = 2, sign from M_BN), j = (curl B)·T/B0 =
±2(Sκ_² + 1)/(Sκ_ sin δ) (= GS current ρp′ + FF′/ρ). True Hess Π: P_NN = −1/S, P_NB = 0, P_BB = −4/S. Closed forms:
P_NN = 3κ² − k² = −1/S ✓, P_BB = k²(1 + 2η⁴/κ⁴ + 2σ²) − 2kη²j/κ² = −4/S ✓, Riccati residual 0 ✓. Without the j-term
P_BB would be −4/S + 8/(S sin²δ) + 8/(S²κ_² sin²δ).
(B) Kepler force-free (kepler_axisym_family.py, ε → 0): axis circle radius a, ι = 1, X = a cos η, Y = (a/C) sin η,
σ = 0, η²/κ² = C, k = 1/a, j = (C + 1/C)/a = −λB0. True P_NN = 2/a², P_BB = −1/a²; closed forms agree; Riccati 0.
**The j-dependence of the round-1 closed forms is validated on two exact non-vacuum equilibria.**

## 4. Task 4 — completing the classification of exactly solvable first-order points

### 4.1 Case 1, deg S = 2 (`kovacic_deg2.py` ~2.5 min → `deg2_branches.pkl`; `deg2_branches_eval.py` ~50 s)
φ = q^{e0}(q−q3)^{e3}(q² + s1 q + s0), e0 = m − e3 − 2. After eliminating (a, b) linearly, s1 is fixed by the top coefficient
(s1 = 2K, (4K+p2)/6, (4K+q3)/2, …), s0 by resultants; the gcd of the resultants in K factors as (deg-0 value)·(cubic)·(sextic)
for each of the four patterns, e.g. (e3 = 0, m = 0): 4K³ + 4K²p2 − 3K p1 + 2p0 and a sextic (printed in
`kovacic_deg2.out`). All factors are rod-independent (no constraint on (q1,q2,q3)). Genuineness and admissibility are
checked numerically rod by rod (§0, row 4a). Together with round 1 (deg ≤ 1), every rod carries 3 + 4·(1…3) + 8 case-1
branch values of K; **none is current-free on the closed (1,2) and (1,3) families** (|j| ≥ 1.16).

### 4.2 Case 2 (`kovacic_case2.py [NMAX] [DMAX]`, default 8 1; d = 0 symbolic part ~5 min, rod part heavy)
Normal form y = φP^{1/4}, r = p1²/4 + p1′/2 − kQ/P. Double poles: q1,q2,q3 (b = −3/16 ⇒ E = {1,2,3}), ∞ (E = {1,2,3}),
0 (b0 = −kb/p0 ⇒ E_0 = {2, 2 ± n}, n = 2√(1 + 4b0) ∈ Z, i.e. kb = p0(4 − n²)/16); d = (e_∞ − Σe_c)/2 ≥ 0 forces e_0 = 2 − n, n ≥ 2.
Exponent at q = 0: e0 = 1/2 − n/4 (case 1 has even n). For d = 0 the condition θ″ + 3θθ′ + θ³ − 4rθ − 2r′ = 0 is linear in
(K, ka); with symbolic (q1,q2,q3) the augmented rank decides consistency: rod-independent branches (§0 row 4b) at
n = 2 (trivial), n = 3 (NEW), n = 4, 6, 8 (case-1 values and their q_i-permutations); all other patterns are inconsistent for
generic rods (constraints such as q1q2 − q1q3 − q2q3 = 0, or a nonzero constant). Physical admissibility of the new
n = 3 branches (σ real ⇔ Hill's equation has a real Floquet multiplier, |tr M| ≥ 2; η² > 0; σ finite) — see the rod part of
`kovacic_case2.out` (§4.3 below, filled in when the run finished).

### 4.3 Case-2 admissibility on the two rational rods (`kovacic_case2.py 8 1` rod part ~4 min; `case2_admissibility.py`,
`case2_verify.py`, `case2_family_scan.py` ~1 min each)
Exact rational rods: QA-matched (1,2) (q = 0.222842, 2.212881, −2.448875) and λ_ODE = −0.5 (q = 0.684788, 1.769765, −3.388971).
Every case-2 pattern with n ≤ 8, d ≤ 1 was solved exactly (d = 0: linear in (K, ka); d = 1: bilinear, eliminated by hand);
each solution with K > 0 was tested: (i) Floquet multipliers of Hill's equation φ″ + (K + kb/q² − ka/q)φ = 0 over one period
of κ² (σ real ⇔ |tr M| ≥ 2), (ii) sign of R = 4r − φ_K² − 2φ_K′ on (q1,q2), (iii) η² > 0 and j.
Findings:
- All case-1 points reappear (n even) with tr M = ±2 and R ≡ 0; the numerical Floquet σ agrees with σ = q′(φ_K/2 − P′/(4P))/k
  to 1e−5 and is odd about the curvature extrema (stellarator symmetric).
- n = 3 (quarter exponents), d = 0: K = K_{S=1}/4 on both rods has tr M = 1.33/1.40 (< 2) and R < 0: σ complex, unphysical.
  n = 5, d = 1: two patterns with K > 0; one has tr M = 0 (unphysical), one (e = (1,1,2,3)) has tr M = 2.07/2.04 > 2 and
  R > 0: a REAL, periodic, NON-stellarator-symmetric σ (hyperbolic Floquet solution, nonzero mean), k = ±1.03/±1.24,
  η² ∈ {1.87, 0.80}/{2.85, 1.22}, j = +4.36/−5.10 and +3.93/−3.06 (sign j = sign k).
- **A one-parameter family** (n = 4, e = (1,1,1,3), d = 1, P_K = q + c0): for every c0 the step-3 equation holds with
  K(c0), A(c0) rational (printed in `case2_family_scan.out`); `case2_verify.py` confirms ω± = (φ_K ± √R)/2 solve
  ω′ + ω² = r to 1e−14. The radical is elementary: R·P·(q + c0)²/q² = c² = const (c² = 1.029 on the QA rod), i.e.
  √R = c q/((q + c0) q′) and
      σ(s) = q′[θ + 1/(q+c0) − P′/(2P)]/(2k) + c q/(2k(q + c0)),   θ = ½[1/(q−q1) + 1/(q−q2) + 1/(q−q3) − 2/q],
  an ELLIPTIC σ ∈ C(q, q′) with an even (in q′) part: non-stellarator-symmetric, hyperbolic Floquet multipliers
  ρ = e^{±∫√R dq}. Exactly: c² = (q1 + c0)(q2 + c0)(q3 + c0)·(rational)/c0² ∝ Π(q_i + c0) (printed in `case2_family_scan.out`),
  so the family is physical (c² ≥ 0 ⇔ real σ ⇔ |tr M| ≥ 2) iff c0 ≥ −q3, i.e. iff the zero of P_K = q + c0 lies below q3; it
  ends at the S=1 point (c0 → ∞, K → −p2/4). Along the physical part |j| ∈ [3.18, 3.90] (QA rod) resp. [1.69, 3.19]
  (λ_ODE = −0.5 rod), no sign change, sign j = sign k throughout. The closed-form σ agrees with the numerical Floquet σ to
  1e−10 away from the turning points (the 4–24 % "deviations" printed by `case2_admissibility.py` are the 1/P cancellation of
  the closed form at the grid points nearest the turning points).
- Conclusion: the complete Kovacic classification (case 1 deg S ≤ 2; case 2 with n ≤ 8, d ≤ 1) adds non-stellarator-symmetric
  exactly solvable points and a continuous family, **none current-free**; round 1's "σ ∈ q′C(q)" is the stellarator-symmetric
  subclass. Case 3 (finite primitive Galois groups, all solutions algebraic) was not examined.

### 4.4 Kida rods (α = 0) (`kida_rods.py` ~12 min with the reduced grid; `kida_refine.py` ~40 s)
Closed α = 0 rods (R_max = 1) are isolated: for nfp = 2 the scan found λ_ODE = ∓1.2393 (one rod and its mirror, L = 17.42,
p2 = +2.54) and λ_ODE = −0.5242 (L = 14.31, p2 = +7.71, κ² ∈ [1.93, 8.18]); for nfp = 3: λ_ODE = −3.266 (L = 23.4, p2 = +0.86)
plus a duplicate detection of the −0.5242 rod (its shooting event also fires at φ = π/3; same L and moduli). All have p2 > 0
(no S=1 branch); the branches K = −q3/4 (deg 0), deg 1 and deg 2 exist. Sign law holds on all 24 points of each rod
(min |j| = 0.40 on the λ_ODE = ∓1.239 rod, 0.62 on the (1,3) rod). **Near-vacuum point:** on the λ_ODE = −0.5242 rod the branch K = −q3/4 (e3 = ½, e0 = −½),
k = −0.7744 (ι0 − N = −1.764), η² = 11.925 has
    j = −0.00650233  (refined with rtol 1e−10…1e−13: identical to 9 digits; kb = 28.43, ka = 12.13, τ0 = 0.6534),
a 0.5 % cancellation between ka/(kη²) = −1.3132 and 2τ0 = +1.3068; the exact j = 0 condition
F = −q3(q1+q2)/2 − 2τ0(C + √(C² + kb)) = +0.0600 ≠ 0. So it is NOT a vacuum point, but by far the smallest |j| found
(previous minimum 1.16), and it sits on a rod whose Killing field is a pure rotation (α = 0 ⇒ compatible with an axisymmetric
V). Since Kida rods are isolated, one cannot continue to j = 0 within nfp = 2; other (m, n) Kida rods are the obvious place to
look (suggestion for the next round).

### 4.5 The sign law (`sign_j_reduction.py`, ~30 s)
See §0 row 4c. Reduced statement for the other agents: **a vacuum (j = 0) elliptic first-order point on a closed rod requires
the closed-rod family to meet an explicit algebraic locus in (q1,q2,q3,τ0)** — for the S=1 branch A: τ0 = −p1/(4C), i.e.
4τ0C = −p1 (with τ0 C < 0 on all closed rods found); the (1,2) family reaches at most 4|τ0C|/p1 = 0.60 (near its
large-|λ_ODE| end), the (1,3) family 0.20. Closure is a transcendental (elliptic-integral) condition, so a *proof* of
sign j = sign k needs the closure integrals, not the root ordering. Conversely, if some closed family (other (m,n), or the
second sheet of the (1,2) family) crosses 4|τ0C| = p1, a vacuum elliptic point exists there.

## 5. Task 5 — finite ansatz at the S=1 point: degree-3 closure in the finite-gap ring (`finite_ansatz_deg3.py`, ~1–2 min)

Coordinates: Frenet tube (s, X, Y), scaled X̂ = κX, Ŷ = κY so that every coefficient function is even in κ and lies in
R = Q[q, 1/q, q′]/(q′² − P) (the elliptic function field of κ²); parity bookkeeping B_N = κ b_N, B_B = κ b_B (checked: every
identity is κ-parity homogeneous). Operators: d_T = D_s/(1−X̂), D_s = ∂_s + (κ′/κ)(X̂∂_X̂ + Ŷ∂_Ŷ) + τ(Ŷ∂_X̂ − X̂∂_Ŷ),
d_N = κ∂_X̂, d_B = κ∂_Ŷ, Frenet connection Γ = [[0,−κ,0],[κ,0,−τ],[0,τ,0]]. Identities (h = 1 − X̂):
h(B·∇B − ∇Π) = 0, h div B = 0, h B·∇ψ = 0, Π − |B|²/2 + E0 + E1ψ = 0, h³ ∇ψ×∇Π·∇(B·∇Π) = 0 (Helander, (I3)).
Validation: the exact Solov'ev equilibrium (Taylor data to degree 3, S = 2, κ_ = 3/2, δ = π/3, E1 = 9/4 = 2κ_²/S) satisfies
all identities to the meaningful degrees (only the degree-3 part of div B, which needs B3, is nonzero) — this tests the
connection terms, the h-factors and the triple product on a j ≠ 0, p′ ≠ 0 case.
Elliptic point: k = −1/2, η² = 2 (C = kη² = −1, branch A), p2 = −1, p1 = 7, p0 = −4, τ0 = 9/10 (rational proxy of the
λ_ODE = −0.5 rod: k = −0.483, η² = 2.10, p1 = 7.11, τ0 = 0.916); roots −3.40, 0.684, 1.72. Checks printed: Riccati residual 0,
j = M_BN − M_NB = p1/(2kη²) + 2τ0 exactly, all first-order identities vanish identically in R.
Unknowns: b_T2, b_N2, b_B2 (3 monomials each), Π3, ψ3 (4 each), E1; each function = Σ_{n=−N}^{N}(c_n q^n + d_n q^n q′).
Result (N = 4): 380 linear equations over Q for 307 constants, rank 291 = rank of the augmented matrix ⇒ **consistent**,
solution space of dimension 16; the particular solution is verified exactly (residual 0); **E1 = −dp/dψ|axis is a free
parameter** (not fixed by the degree-3 closure, as in GB theory). Robustness: N = 6 (Laurent range q^{−6..6}) gives rank 413,
consistent, dimension 30 (= 16 + 14): the freedom grows with the range, i.e. it consists of *free functions in the ring*
(≈ 3.5 ring functions, cf. the three GB second-order free functions B20, B2c, B2s), not of a finite parameter set.
Second rational point (k = −1/3, η² = 5/2, p1 = 6, τ0 = −3/5, roots −2.87, 0.503, 1.92): identical structure (rank 291,
consistent, dimension 16). Particular solution at point 1 (free constants 0): b_T2 = [−(250q² − 127q − 20), q′, 25q² + 81q − 10]/q²
for (X̂², X̂Ŷ, Ŷ²), b_N,20 = q′(9q² − 20q − 63)/q³, … (full list in `finite_ansatz_deg3.out`).
Interpretation: the second-order data of the exactly solvable point can be taken in the same finite-gap ring; the
Garren–Boozer second-order freedom survives inside the ring. This is the first nontrivial step of the finite ansatz and it
does NOT fail. The decisive next step (not done, see suggestions) is the degree-4 closure — the GB "+1" order — with the
degree-3 free ring functions as unknown parameters (a bilinear system: linear in the new unknowns B3, Π4, ψ4, E2, quadratic in
the degree-3 parameters).
Caveats: (i) B is algebraic, not polynomial, so "finite Π, ψ" does not make the B-identities finite: the exact decision for a
polynomial (Π, ψ) needs the algebraic-B identities (I)–(III) of the round-1 global-reformulations agent; (ii) the
degree-3 equations coincide with what any expansion would give at second order — here they are used only as the first
coefficient identities of the finite ansatz and decided in the ring (no order-by-order construction is attempted).

## 6. Reproduction
```
python3 vmec_axis_p2.py            # needs wout_QA.nc, wout_QA_reactor.nc (handoff 10.1 URLs) for parts B-D; part A runs without
python3 strain_kernel_axis.py
python3 circ_axis_checks.py
python3 kovacic_deg2.py            # ~2.5 min, writes deg2_branches.pkl
python3 deg2_branches_eval.py      # ~50 s
python3 kida_rods.py               # slow (fsolve grid), see .out
python3 kovacic_case2.py 8 1       # two exact rational rods, n <= 8, d <= 1 (~4 min); add 'sym' as 3rd arg for the symbolic d = 0
                                   #   classification (~12 min; saved in kovacic_case2_sym.out)
python3 case2_admissibility.py     # Floquet / R-sign / eta^2 / j tests of the case-2 candidates (~1 min)
python3 case2_verify.py            # Riccati residual of omega = (phi_K +- sqrt R)/2 for the candidates (exact, 1e-14)
python3 case2_family_scan.py       # the n = 4, d = 1 one-parameter family: radical structure, physical range c0 > -q3, j (~1 min)
python3 kida_refine.py             # high-precision j on the near-vacuum Kida point (j = -0.00650233)
python3 sign_j_reduction.py
python3 finite_ansatz_deg3.py A    # Solov'ev validation only (2 s)
python3 finite_ansatz_deg3.py 4 1  # elliptic point, Laurent range 4 (~1-2 min);  '6 1' range 6;  '4 2' second point
```
