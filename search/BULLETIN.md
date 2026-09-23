# QS–MHS exact-solution search — coordinator bulletin

Shared state of the three-agent search for an exact, analytic, toroidal, quasisymmetric MHS equilibrium
that is not isometry-invariant. Nothing below is a solution; no such solution is known.
Statuses: `proved`, `verified_symbolically`, `verified_numerically`, `conjecture`, `refuted`.

## Round 1

### Verification verdicts
No candidate solutions were produced in round 1; nothing to verify.

### Direction `closed-orbit-congruences` (Kepler tori, rational oscillators) — both tasks refuted
- Kepler tori (congruent ellipses, orientations g(s;ψ)): ∂_tJ = 0 with J ≠ 0 forces a′ = 0 (equal energy on all
  tori ⇒ p′ = 0), then zero-curvature forces a fixed rotation axis ⇒ axisymmetric. `proved`
  (`kepler_jacobian.py`, `kepler_cases.py`; Gröbner bases with saturation for every branch).
- Exact axisymmetric force-free field with Kepler-ellipse field lines, Π = μ/ρ, ι = 1, all lines same a,
  sin i = ε/(C√(1−ε²)). `verified_symbolically` (+ numerical div B ~ 5e-9). Not the target (axisymmetric, p′ = 0).
- Lissajous DC rule: mean J ≠ 0 only if one frequency is the sum of the other two. `verified_symbolically` (9 triples).
- 1:1:2 oscillator: identical |B|-profiles ⇒ orbits on a torus are z-rotations of one orbit ⇒ axisymmetric.
  `proved`. Jacobian condition on that family ⇒ exactly a Solov'ev equilibrium (GS residual 0). `verified_symbolically`.
- Three distinct frequencies, and Evans barriers a/x²: profile fibre finite / 0-dim ⇒ no torus. `verified_symbolically`.
- Side lemma: tori of revolution about ψ-dependent axes N(ψ): strong QS ⇒ B·(N′×x) = 0 ⇒ N′ = 0. `verified_symbolically`
  (conditional on weak ⇒ strong QS, RHB 2020).
- Helander local criterion ⇔ same-profile-up-to-shift for closed lines with non-degenerate profiles. `conjecture`.
- General principle: when the profile invariants are complete invariants of the isometry group K of V on orbit
  space, closed-line QS forces B to be K-symmetric. `conjecture` (instances above proved).
- Dead ends: Kepler tori (any a, ε, g); 1:1:2; 1:2:3-type oscillators; Evans barriers; local-criterion relaxation;
  "piecewise Killing" tori. **Reduced open problem:** a potential V with a 1-parameter family of equal-energy closed
  orbits with identical V(t)-profiles that is not an isometry orbit.

### Direction `rod-axis-potential` (user's preferred line) — exact first-order structure, no vacuum elliptic point
- Clock identities: ẅ = −∇V·∇w + B·Hess w·B; V̈ = B·Hess V·B − |∇V|²; Helander triple product =
  (1/|B|)∇ψ×∇V·∇(B·∇V); clock ⇒ ∇ψ×∇V·∇(V̈) = 0. `proved` (`clock_identities.py`).
- Exact linearisation about a constant-speed closed orbit in its Frenet frame; first-order QS clock ⇔ κX harmonic
  ⇒ closed-form on-axis Hessian (P_NN, P_NB, P_BB) and ONE compatibility ODE = dj/ds = 0 with j = (curl B)·T on
  the axis, i.e. the Riccati σ′ = −k(1+σ²) − kη̄⁴/κ⁴ + η̄²(j−2τ)/κ². Derived from the clock identity + Floquet
  theory (not NAE). `verified_symbolically` (+ numerical 4e-12) (`axis_linearization.py`, `hessian_closed_forms.pkl`).
- Rod identities in Frenet form: all Frenet data rational in (κ², (κ²)′), (κ²)′² = cubic with explicit coefficients,
  q3 < 0. `proved` (`rod_exact.py`). Sign fix: handoff's λ3 is −λ_Frenet.
- Kovacic case-1 classification of first-order QS points on ANY rod (σ rational in κ², (κ²)′, deg S ≤ 1): explicit
  k² values, σ, η̄², j. Simplest: k = ±√(−(q1+q2+q3))/2, σ = −κ′/(kκ). `verified_symbolically` + direct check
  (Riccati residual 4e-8, monodromy exactly kL/2π).
- No current-free (j = 0) exactly solvable point on the closed (1,2) rod family (68 rods): sign j = sign k,
  |j| ≥ 1.16. `verified_numerically`. QA point (j = 0, ι0 = 0.4232): σ exists but is NOT rational in κ²
  (handoff §9.1 algebraic conjecture) — `refuted` at the QA point; §6.4 fits reproduced in structure only.
- Potentials with a rod orbit: any Killing symmetry must be the rod's own screw; no axisymmetric/central/separable
  V_h + V_z. `proved` (numerical rank test for uniqueness of the screw).
- Observation: QA-matched rod has p2 = q1+q2+q3 ≈ 0 (κ² a pure Weierstrass ℘; 4cλ3 = λ2²). `conjecture`.
- Dead ends: symmetric-V tricks with a rod axis; separable V; vacuum elliptic first-order data with deg S ≤ 1;
  blind sympy solve for deg S ≥ 1 (use resultants).

### Direction `global-reformulations` — algebraic-B system, strain kernel, Boozer normal flow, finite surfaces
- QS-MHS ⇔ (V, ψ, E(ψ), F(ψ,V)) with B fixed pointwise by |B|² = 2(E−V), B·∇ψ = 0, B·∇V = ±√F plus
  (a) div B = 0, (b) B·Hess ψ·B = ∇V·∇ψ, (c) B·Hess V·B = F_V/2 + |∇V|². `proved`, verified on Solov'ev;
  reconstruction formula reproduces B to 1e-31.
- Strain-kernel lemma: L_uB♭ = 2S(u)(B,·) + [u,B]♭ ⇒ strong QS ⇔ S(u)B = 0 ⇒ det S(u) = 0, B ∥ ker S(u). `proved`.
  Kills u = Ψ_*∂_φ for vertical shears, z-independent horizontal shears, linear SL(3) maps. `proved`.
  Quadratic shear compositions: det S = 0 is a genuine degree-11 identity, not classified.
- Boozer identities (1),(2),(4′) ⇔ one vector equation W = K x_θ×x_φ + I x_φ×x_ψ + G x_ψ×x_θ, algebraic in x_ψ;
  solvable iff C1 := I(W·x_φ) − G(W·x_θ) = 0; then √g = |W|²/(G+ιI) is metric-only and QS is
  C2 := (∂_φ+N∂_θ)[|W|²/(G+ιI)] = 0. μ (tangential part of x_ψ) is algebraic where f′(χ) ≠ 0; ∂_ψC1 = 0 gives one
  scalar COMPAT per surface (the GB +1). `verified_symbolically`; tested only on a vacuum ι = 0 family and the
  degenerate screw pinch.
- Finite trigonometric Boozer surfaces: degree-1 axisymmetric ⇒ ι = 0 (`proved`); smallest stellarator-symmetric
  non-axisymmetric class (14 unknowns): all 64 terminal solutions of C1 = C2 = 0 are axisymmetric, degenerate or
  ι = ±N. `verified_symbolically`. Non-stellarator-symmetric class (26 unknowns) open.
- Dead ends: shear/linear conjugations; Gröbner for the larger classes in the time box; transnormal V = Φ(w).

### Coordinator's spot-check notes (errors, overclaims, missed connections)
1. **Normalisation of u for strong QS.** L_uB♭ = 0 is not invariant under u → c(ψ)u (it gains (G+NI)c′dψ).
   In Boozer covariant form B×∇ψ = (G e_θ − I e_φ)/√g, so the correctly normalised vector is
   u = e_φ + N e_θ = [(G+NI)B + (N−ι)B×∇ψ]/B². CONTEXT §2.3's u = (G̃B + B×∇ψ)/B² is off by the flux
   function 1/(N−ι) and gives a spurious ι′ = 0 if used in L_uB♭ = 0. All round-1 applications used 2π-periodic
   circle actions (Ψ_*∂_φ, N(ψ)×x) and are unaffected; any on-axis use of S(u)T = 0 must use the Boozer-normalised u.
2. rod-axis §6 (separable V): the stated reason "z_s = 0 at curvature extrema" does not follow from stellarator
   symmetry; the valid argument is z_s² = 2E_z − 2V_z(z) must agree at the two z = 0 points, but 0.334² ≠ 0.291².
   Conclusion stands.
3. `vacuum_S1_search.out` lists circle rods (β = 0, L = 2π) for λ_ODE ≤ −0.587 with "ι0(S1)" values; ignore those
   rows (`branches_scan.py` rejects them; 68 genuine rods).
4. **Missed connection.** closed-orbit-congruences produced two exact axisymmetric MHS congruences with explicit
   x(ψ,s,t), B = ∂_tx: Solov'ev via 1:1:2 (ι = 2, p′ ≠ 0, I ≠ 0) and the Kepler force-free family (ι = 1, j ≠ 0).
   Their Boozer angles are elementary: B·∇φ_B = B²/(G+ιI) ⇒ φ_B = s + ∫B²dt/(G+ιI) with G+ιI = ⟨B²⟩_t,
   θ_B = ιφ_B + const (|B|² a trig polynomial in t; for Kepler ∫B²dt = √(aμ)(η + ε sin η)). These are the
   non-degenerate ι, I, K ≠ 0 tests global-reformulations lacked, and circular-axis j ≠ 0 tests of the rod-axis
   P_ij/Riccati closed forms.
5. Convergent negative result from two sides: a continuous isometry of V is incompatible with a non-symmetric QS
   congruence (rod: the Killing field must be the rod's own screw; congruences: profile fibres = isometry orbits).
   Any candidate V must have trivial isometry group.
6. Closed-line (ψ,s,t) reformulation (coordinator): with s relabelled so J = J(ψ) and t shifted so profiles
   coincide, u = ∂_s and strong QS ⇔ ∂_s g_tt = ∂_s g_ts = ∂_s g_tψ = 0 (the three "t-row" metric components are
   s-independent); force balance ⇔ the 1-form x_tt·dx is closed. Together with J = J(ψ) this is a finite set of
   identities for any design-first ansatz x = Φ_s(x0(ψ,t)). Used in assignment C below.

### Cross-agent suggestions carried forward
- closed-orbit → all: profile-fibre rank test (slice sums A_n on resonant Liouville tori) as a cheap necessary
  condition; algebraic-B (b),(c) as filters; strain-kernel test B·w = 0 for screw actions with ψ-dependent pitch.
- rod-axis → all: on-axis Hessian closed forms + constancy of j as a fast filter for any candidate axis;
  closed-line clock as a finite Fourier identity; global construction at a finite-current elliptic point.
- global → rod-axis: strain-kernel S(u)T = 0 on the axis (with the normalisation caveat above);
  global → closed-orbit: (b) B·Hess ψ·B = ∇V·∇ψ as a Jacobian-free kill test;
  global → self: non-degenerate ι ≠ 0 test, ∂_ψCOMPAT chain, one-surface problem with richer ansätze.

### Round 2 assignments
- **A. `rod-axis-elliptic`** (continues rod-axis-potential; user's line): VMEC-axis test of p2 = 0 from the handoff
  Fourier data; Boozer-normalised strain-kernel check on the axis; complete the Kovacic classification (deg S = 2,
  case 2, (1,3) and Kida rods) and try to prove sign j = sign k; circular-axis j ≠ 0 tests of the P_ij/Riccati
  formulas on Solov'ev/Kepler; a finite global ansatz in (℘, ℘′, tube coordinates) at a finite-current elliptic
  point decided by algebraic-B (b),(c) + div B = 0 as polynomial identities.
- **B. `boozer-normal-flow`** (continues global-reformulations, absorbs the exact axisymmetric congruences):
  non-degenerate validation on Solov'ev-from-1:1:2 and Kepler force-free with explicit Boozer angles; derive
  ∂_ψCOMPAT and decide the length of the compatibility chain; attempt the theorem "no trigonometric-polynomial
  Boozer surface satisfies C1 and C2 unless axisymmetric" via top-degree isotropy conditions.
- **C. `closed-line-design`** (new angle; absorbs closed-orbit-congruences' reduced problem and rod-axis's
  closed-line clock idea): the (ψ,s,t) formulation of note 6 proved and checked on the two exact congruences;
  design-first ansatz x = Φ_s(x0(ψ,t)) with non-isometric volume-preserving Φ_s decided by finite identities;
  quick profile-fibre rank tests on Stäckel systems (elliptic-cylindrical Family-2 core, prolate/parabolic
  axisymmetric Stäckel potentials).
- stop = false: no candidate, and each direction has decidable next steps.
