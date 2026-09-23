# rod-axis-potential — round 1 notes

Direction: global closed-form potential V = −Π with an exact Kirchhoff-rod axis and the QS clock.
All scripts in this directory run with `python3 <script>` from this directory (sympy 1.14, numpy, scipy).
Units: B0 = 1, arclength s = time along the axis; P_ij = Frenet components of Hess Π on the axis (units B0²).

## 0. Summary of what was established (see the structured report for status labels)

1. Exact clock identities in the potential language (proved symbolically, `clock_identities.py`).
2. Exact linearisation of ẍ = −∇V about a constant-speed closed orbit in its Frenet frame, and the exact
   first-order QS clock conditions: closed-form on-axis Hessian (P_NN, P_NB, P_BB) + one Riccati ODE whose
   first integral is the on-axis parallel current (`axis_linearization.py`; numerically verified to 4e-12).
3. Rod identities in Frenet form (all Frenet data of a rod are rational in q = κ², q′, with q′² = cubic) and a
   numerically exact closed (1,2) rod reproducing handoff §7.1 (`rod_exact.py`).
4. The first-order QS problem around a rod axis is Hill's equation with an elliptic potential; its exactly
   solvable (Kovacic case-1 = "σ elliptic") points are classified for deg S ≤ 1: for EVERY rod there are
   explicit branches k² = −(q1+q2+q3)/4, −q3/4, −(q1+q2+4q3)/4 and roots of four explicit cubics, with
   σ, η̄², j in closed form (`elliptic_sigma.py`, `elliptic_sigma_d1.py`, `branches_scan.py`); the simplest
   one is σ = −κ′/(kκ), verified directly (`check_S1_solution.py`).
5. None of these exactly solvable points is current-free (j = 0) on the closed (1,2) rod family (scan,
   `branches_scan.py`, `vacuum_S1_search.py`), and the QA point (j = 0, ι0 = 0.4232) of the QA-matched rod has
   a non-elliptic σ (`numeric_sigma_QArod.py`): the "algebraic first-order conjecture" (handoff §9.1) is
   refuted in its strong form; the §6.4 fits are good approximations only.
6. Which potentials can have a rod orbit: the screw pitch is unique (rank test), no separable V_h(x,y)+V_z(z)
   and no axisymmetric/central V has a non-planar α ≠ 0 rod as orbit.

## 1. Clock identities (`clock_identities.py`, ~15 s)

Field lines: ẍ = −∇V, ẋ = B, V = −Π. For any scalar w: ẇ = B·∇w, ẅ = −∇V·∇w + B·Hess(w)·B (I1); with w = V:
V̈ = L(x) := B·Hess V·B − |∇V|² (I2) — a pointwise function once the congruence is fixed.
(I3) With B·∇ψ = 0 (Clebsch form): ∇ψ×∇Π·∇(B·∇Π) = |B|·[∇ψ×∇|B|·∇(B·∇|B|)], so Helander's triple product
criterion ⇔ V̇ = B·∇V is a function of (ψ, V) ⇔ V̇² = F(ψ, V) (the clock).
(I4) Clock ⇒ V̈ = F_V/2 ⇒ ∇ψ×∇V·∇L = 0 (necessary, algebraic in Hess V on the axis). Conversely L = f(ψ,V) gives
V̇²/2 = ∫f dV + C(orbit): clock iff C is a flux function (automatic on irrational tori).
(I5) First variation along a linearised solution δx: d²/dt²(∇V·δx) = 2HessV(B,δẋ) + D³V(B,B,δx) − 2HessV(∇V,δx);
the D³V term (transverse derivative of V_TT, not axis data) is exactly what the linearised ODE eliminates — so
the first-order clock is decided by Hess V on the axis alone (next section).

## 2. Linearisation about the axis and the first-order clock (`axis_linearization.py`, ~45 s)

Frenet components f = (f_T, f_N, f_B), covariant derivative D f = (f_T′ − κf_N, f_N′ + κf_T − τf_B, f_B′ + τf_N).
Axis: ∇Π = κN ⇒ Hess Π·T = (−κ², κ′, κτ). Free axis data: P_NN, P_NB, P_BB. δx = ξT + XN + Y𝐁, δx″ = Hess Π δx.
(L1) T-component ⇔ (ξ′ − 2κX)′ = 0; ξ′ − 2κX = δE. For the field-line pair δE = O(ψ) ⇒ 0 at first order.
(L2) ξ drops out:
    X″ + (3κ² − τ²)X − 2τY′ − τ′Y = P_NN X + P_NB Y
    Y″ − τ²Y + 2τX′ + τ′X          = P_NB X + P_BB Y
Verified against direct integration of δẍ = −Hess V δx for two explicit potentials with exact unit-speed orbits
(helix in V(r,…) with τ ≠ 0; catenary-type planar curve with κ′ ≠ 0 and P_NB ≠ 0): agreement 4e-12.
(L3) First-order clock ⇔ κX is a pure harmonic on the field-line pair: X = (η̄/κ)e^{iks}, k = 2π(ι0 − N)/L;
flux conservation (∇·B = 0 at leading order, ellipse area = const) ⇔ Im(X̄Y) = const ⇒ Y = X(κ²/η̄²)(σ − i).
Solving (L2) for the P's (4 real equations, 3 unknowns) gives
    P_NN = 3κ² + 3τ² − 2τj − k² + 2kη̄²τ/κ² − 2kσκ′/κ − κ″/κ + 2κ′²/κ²
    P_NB = 2kη̄²κ′/κ³ + 2kστ − τ′ − 2τκ′/κ
    P_BB = k²(1 + 2η̄⁴/κ⁴ + 2σ²) + 2kη̄²(τ − j)/κ² − 2kσκ′/κ − τ² + κ″/κ
    ∇²Π  = 2κ² + 2(τ + kη̄²/κ²)² + 2(kσ − κ′/κ)² − 2j(τ + kη̄²/κ²)      (vacuum j = 0: a sum of squares = |∇B|²)
and the compatibility condition, which is EXACTLY d j/ds = 0 with
    j := (curl B)·T on the axis = kη̄²/κ² + 2τ + (κ²/η̄²)(k(1+σ²) + σ′),
i.e. the Riccati   σ′ = −k(1 + σ²) − kη̄⁴/κ⁴ + η̄²(j − 2τ)/κ²,   j = const.
(Structure identical to the known first-order equation with j ↔ 2I₂/B0, but derived here from the clock identity
and Floquet theory of the axis orbit, not from a near-axis expansion.)  The helical (screw-invariant) values
P_NB^hel = −(λ2/λ3)κ′/κ, P_BB^hel = (z_s − λ2τ)/λ3 are not of this form: the O(1) "breaking" of handoff §6.4 is
P_BB − P_BB^hel = k²(1+2η̄⁴/κ⁴+2σ²) + 2kη̄²(τ−j)/κ² − 2kσκ′/κ − τ² + κ″/κ + λ2τ/λ3 − z_s/λ3 (printed by the script).

## 3. Rod identities and the exact (1,2) rod (`rod_exact.py`, ~3 s)

With 𝓘 = αe_z + e_z×x = λ2T + λ3κ𝐁 and e_z = (z_s, N_z, 𝐁_z) in Frenet components, D e_z = 0 and D𝓘 = e_z×T give
(R1) N_z = −λ3κ′, 𝐁_z = κ(λ2 − λ3τ); (R2) z_s = c − λ3κ²/2; (R3) τ = λ2/(2λ3) + C/κ², C = (λ2c − α)/λ3²;
(R4) r² = λ2² − α² + λ3²κ²; (R5) q = κ² obeys q′² = P(q) = −q³ + p2q² + p1q + p0,
p2 = 4c/λ3 − λ2²/λ3², p1 = 4(1−c²)/λ3² + 4λ2C/λ3, p0 = −4C² < 0 ⇒ roots q3 < 0 < q1 = κ²_min < q2 = κ²_max.
Every Frenet quantity of a rod is rational in (q, q′): κ²/τ/z_s/κ′/κ etc. Sign convention: T×𝓘 = −λ3κN, so the ODE
γ″ = T×𝓘/λ_ODE has λ_ODE = −λ3 (the handoff's "λ3 = −0.6283" is λ_ODE; Frenet λ3 = +0.6283).
Numerical (1,2) rod (shooting as handoff §7.1): α = 0.561636, β = −0.340808, L = 6.65696, closure 2e-12;
T·𝓘 = 0.955131 (std 1e-13), 𝐁·𝓘/κ = 0.6283 (std 1e-16); (R2)–(R5) hold to 1e-9…1e-12.
Moduli: c = 0.360928, C = −0.549454, τ0 = 0.760091, q3 = −2.448875, q1 = 0.222842, q2 = 2.212881,
p2 = q1+q2+q3 = −0.013151 (≈ 0!), p1 = 5.471658, p0 = −1.207599.  This rod is the mirror image of the VMEC axis
(τ ∈ [−1.71, 0.51], z_s(κmax) = −0.334 vs VMEC τ ∈ [−0.51, 1.57], z_s = +0.325).
Killing fields ξ with N·ξ = 0 along the rod: singular values of the 6-column sample matrix
[32.9, 25.9, 21.6, 8.0, 6.7, 6e-15] ⇒ unique (the screw 𝓘). z_s ≠ 0 at both curvature extrema.

## 4. Elliptic first-order solutions (`elliptic_sigma.py`, `elliptic_sigma_d1.py`, `branches_scan.py`, `check_S1_solution.py`)

σ = φ′/(kφ) linearises the Riccati to Hill's equation with an elliptic potential
    φ″ + [k² + kb/κ⁴ − ka/κ²]φ = 0,   a = η̄²(j − 2τ0),  b = kη̄⁴ + 2η̄²C
(in q: Pφ_qq + (P′/2)φ_q + kQφ = 0, Q = k + b/q² − a/q; Fuchsian, singular points q1,q2,q3 (exponents 0,1/2),
0 (e(1−e) = kb/(q1q2q3)), ∞ (0, −1/2)).  σ elliptic (rational in q, q′) ⇔ φ = q^{e0}(q−q3)^{e3}S(q), S polynomial
(Kovacic case 1; e1 = e2 = 0 forced by finiteness of σ at the curvature extrema; e3 ∈ {0,½}; e0 = m − e3 − deg S,
m ∈ {0, −½}).  Exact solutions (a, b eliminated linearly; s0 from the equation linear in it; the remaining
coefficient equations have a common factor — no constraint on the rod):
  deg S = 0:  4K + q1+q2+q3 = 0 (e3=0) ;  4K + q3 = 0 (e3=½, e0=−½) ;  4K + q1+q2+4q3 = 0 (e3=½, e0=−1),  K = k²
  deg S = 1:  four explicit cubics in K (printed by branches_scan.py), s0 = 2K, 2K/3+(q1+q2+q3)/6, 2K+q3/2, 2K/3+(q1+q2)/6+2q3/3
For each: a = ã/k, b = b̃/k explicit; η̄² from kη̄⁴ + 2Cη̄² − b = 0; j = a/η̄² + 2τ0.  Simplest branch:
    k = ±√(−(q1+q2+q3))/2,  σ = −κ′/(kκ),  X = (η̄/κ)e^{iks},  Y = −(κe^{iks})′/(kη̄),  η̄² ∈ {C/k, −3C/k},  j = p1/(2kη̄²) + 2τ0.
Direct check on the QA-matched rod (p2 < 0 ⇒ k = ±0.05734, ι0−N = ±0.0608): Riccati residual 4e-8 (relative),
pointwise (L2) residual 1e-8 (N) / 1e-5 (B, FFT third derivative), monodromy of (L2) has rotation numbers exactly
±0.06075 (the other transverse pair is strongly hyperbolic, |ev| ~ 3e4 — irrelevant for the field, only for the
Lagrangian-graph/vacuum degeneracy), P_NN, P_NB/q′, P_BB Laurent polynomials in q to 1e-8…1e-12.
Exactly solvable points on the QA-matched rod (branches_scan.out): ι0−N ∈ {±0.061, ±0.829, ±1.437, ±1.536, ±2.305,
±2.324, ±3.140}, all with |j| ≥ 2.6 (large on-axis current) — none near the QA point (ι0 = 0.4232, j = 0).
Scan of the closed (1,2) family (continuation from the QA rod; circles excluded): see `branches_scan.out`, §7 below.

## 5. The QA point is not elliptic (`numeric_sigma_QArod.py`, check only)

On the QA-matched rod with j = 0 and |k| = 2π·0.4232/L: for k > 0 (mirror rod) two admissible stellarator-symmetric
periodic σ exist, η̄ = 0.574747 and 0.853616 (a third, η̄ = 1.86, has a pole); for k < 0 none with η̄ ∈ [0.3, 2.5].
Fits of the exact P's in the §6.4 functional forms (mirror the signs of τ, z_s, P_NB to compare with VMEC):
  η̄ = 0.5747: P_NN = 3.13κ² + 0.41τ + 1.24 (rms 1e-3), P_BB = 1.28z_s − 0.98τ − 0.15, P_NB = −1.18κ′/κ + 0.02κ′
  VMEC (§6.4):  P_NN = 3.19κ² − 0.86τ + 0.93,  P_BB = −1.56z_s + 0.86τ − 0.11,  P_NB = 0.78κ′/κ + 0.29κ′
Same structure and signs (after mirroring), coefficients 10–50 % off (the rod is a 3 % approximation of the axis, and
Hess Π is sensitive).  Rationality test: max residual of σq′ vs a Laurent polynomial in q of degree ±2, ±3, ±4 is
1.3e-3, 6.5e-5, 3.9e-6 (geometric decay of an analytic non-rational function; an exact branch gives 1e-8…1e-12).
⇒ the strong "algebraic first-order" conjecture (handoff §9.1) is false at the QA point; §6.4's fits are approximate.

## 6. Which potentials can have a rod orbit (elementary, verified numerically where stated)

- ∇V = −κN along the rod constrains only ∇V on a curve; any V with this property has the rod as an exact orbit.
- V with a Killing symmetry ξ: ξ·∇V = 0 on the rod ⇒ N·ξ = 0 along the rod ⇒ ξ ∝ 𝓘 (rank test §3): the only
  symmetric potentials with the rod as orbit are screw-invariant with the rod's own pitch. In particular no
  axisymmetric (α = 0 Kida rods only), no central (planar orbits), no V(r), no V(z).
- Separable V = V_h(x,y) + V_z(z): z″ = −V_z′(z) ⇒ z_s² must be a function of z alone ⇒ z_s = 0 at the curvature
  extrema (κ² even, z odd there otherwise); the QA rod has z_s = −0.334, +0.291 there ⇒ excluded.
- Consequence: a torus swept by symmetric copies of the rod (Kepler-type trick of CONTEXT §3.2) is impossible; the
  congruence must be built from a non-symmetric V.

## 7. Scan of the closed (1,2) rod family (`branches_scan.py`, `vacuum_S1_search.py`; ~40 s each)

Closed (1,2) rods with R_max = 1 exist for λ_ODE ∈ [−0.575, −0.311] (beyond, the shooting degenerates to circles;
the first scan of branches_scan.py in an earlier version silently sat on the circle family — that is why the current
version continues from the QA rod in both directions and rejects |β| < 1e-3).  Result on 68 rods:
- all 18 (branch, sign k, η̄-root) combinations with deg S ≤ 1 have sign(j) = sign(k) and |j| ≥ 1.16 everywhere
  (typically |j| ≥ 2.7 for k > 0); **no current-free (j = 0) elliptic first-order point exists on the closed (1,2)
  family for deg S ≤ 1** (vacuum_S1_search.py: the S=1 vacuum conditions f_A = 2(1−c²)+3λ2λ3C, f_B = 2(1−c²)−λ2λ3C
  stay in [0.22, 1.26] and [1.2, 2.33]).
- the S=1 branch (needs p2 = q1+q2+q3 < 0) exists only for λ_ODE < −0.4273: p2 changes sign essentially AT the
  QA-matched rod (p2 = −0.0131 there; the fitted λ3 has 0.6 % error ⇒ δp2 ≈ 0.07).  Observation/conjecture: the
  precise-QA axis has q1+q2+q3 = 0, i.e. κ² = −4℘(s + ω₃) exactly (depressed Weierstrass cubic, 4cλ3 = λ2²).
  Testable on the VMEC axis with the handoff's rod fit: evaluate 4cλ3 − λ2² with c = z_s + λ3κ²/2.
- Caveat on completeness: the classification covers σ ∈ C(q)·q′, i.e. σ rational in (κ², (κ²)′) (period L/2 of κ²
  for the (1,2) rod; e3 = ½ terms are included since only φ′/φ enters).  Elliptic σ with only the full period L
  (Kovacic case 2, φ′/φ quadratic over C(q)) and deg S ≥ 2 are not yet enumerated (resultant method extends).

## 8. Reproduction commands

    python3 clock_identities.py          # (I1)-(I5)
    python3 axis_linearization.py        # (L1)-(L5) + closed forms -> hessian_closed_forms.pkl, axis_linearization.out
    python3 rod_exact.py                 # (R1)-(R5), exact rod -> rod12.npy
    python3 elliptic_sigma.py 0          # deg S = 0 branches (deg 1-2 with blind sp.solve is too slow; use next two)
    python3 elliptic_sigma_d1.py         # deg S = 1 by resultants
    python3 branches_scan.py             # all deg<=1 branches, QA-rod points, scan of the closed (1,2) family for j = 0
    python3 check_S1_solution.py         # direct verification of sigma = -kappa'/(k kappa)
    python3 numeric_sigma_QArod.py [±1]  # QA point check (sign = handedness), fits, rationality test
    python3 vacuum_S1_search.py          # S=1 branch along the closed (1,2) family (f_A, f_B never vanish)
