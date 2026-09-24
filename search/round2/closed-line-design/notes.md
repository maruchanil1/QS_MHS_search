# Round 2 — closed-line-design

Direction: closed-line (rational ι) QS-MHS by design in orbit coordinates (ψ, s, t). Builds on round-1
`closed-orbit-congruences` (reduced open problem) and BULLETIN notes 1, 5, 6. All scripts: `python3 <name>.py`
from this directory (sympy 1.14, scipy 1.17); logs `<name>*.log`.

## 0. Summary of outcomes

| item | statement | status | script |
|---|---|---|---|
| orbit-coordinate formulation | with B = x_t, u = ∂_s, g_ab = x_a·x_b: div B = ∂_tJ/J, div u = ∂_sJ/J; (L_uB♭)_a = ∂_s g_ta; x_tt·x_a = ∂_t g_ta − ∂_a g_tt/2. Hence weak QS ⇔ ∂_s g_tt = 0; MHS (x_tt = ∇Π, Π = p(ψ)+g_tt/2) ⇔ **(M1) ∂_t g_ts = ∂_s g_tt** [J^ψ = 0] and **(M2) ∂_t g_tψ − ∂_ψ g_tt = p′(ψ)** [(J×B)_ψ = p′]; strong QS ⇔ ∂_s g_tt = ∂_s g_ts = ∂_s g_tψ = 0 | proved (identities [A]) | `orbit_coords_formulation.py` |
| closed-line weak ⇏ strong | weak QS + MHS give g_ts = I(ψ) + k(ψ,s), g_tψ = K_p(ψ,t) + κ(ψ,s) with line-constant k, κ; the Boozer covariant form B_θ = I + k(α), B_φ = G − ιk(α) satisfies J^ψ = 0, √g = (G+ιI)/B², (J×B)_ψ identically (k drops out) ⇒ on a rational surface the local equations do **not** force k = κ = 0; the gap is a line-constant parallel current δJ = (∂_ψk)B. RHB's proof needs B_θ = I(ψ) (irrational surfaces / continuity in a sheared ι) | proved ([B]) | same |
| exact congruences in this language | Solov'ev-from-1:1:2 (ι = 2, p′ ≠ 0): J = −(S²κ/4)[cos(δ+4λ) − cos(δ−4λ)] λ-only, g_ts = I = −S sin 2λ, (M2) gives p′(λ) = S²κ² sin 4λ = d/dλ[−S − 2κ²m²] (Grad–Shafranov value), x_tt·dx closed, α_s = 0, all ∂_s g_ta = 0; Kepler force-free (ι = 1) with ∂_ψ at fixed t: J = −a^{3/2}√μ εε′/C, g_ts = I(ψ), (M2) = 0 = p′, closed, x_tt = −μx/ρ³ | verified_symbolically (exact rational points for Kepler) | same ([C], [D]) |
| first-order deformation = linear system | every first-order deformation of an exact congruence is x = x0 + εξ(x0); δg_ab = 2x0_a·S(ξ)·x0_b, δJ = J0 div ξ ⇒ first-order closed-line weak-QS-MHS ⇔ div ξ = 0 (mode n ≠ 0), B·S(ξ)·B = 0, B·∇[B·S(ξ)·x_s] = 0, B·∇[B·S(ξ)·x_ψ] = 0, i.e. **S(ξ)B = c_ψ(ψ,s)∇ψ + c_s(ψ,s)∇s** with line-constant c's (strong QS: S(ξ)B = 0 — the strain-kernel lemma with ξ in place of u). Physically: ξ is an incompressible neutral ideal-MHD displacement (neighbouring equilibrium) with vanishing Lagrangian perturbation of \|B\| | proved | `lindef_core.py` (docstring), §2 |
| polynomial ξ, Solov'ev | deg ≤ 3 (60 coeffs, two parameter sets), deg ≤ 4 (105 coeffs, see log): nullspace = 6 Killing + (x,y,0) [S-scaling] + (0,0,z) [κ-scaling] + (0,0,ρ²) [δ-shear]; **no non-trivial first-order deformation, in particular none with s-mode n ≠ 0** | verified_symbolically (exact linear algebra) | `lindef_solovev.py`, logs `lindef_solovev_d*_c*.log` |
| polynomial ξ, Kepler | deg ≤ 1: nullspace = 6 Killing (dilation is NOT a solution at fixed period: Kepler is not homogeneous of degree 2); deg ≤ 2 at the point (C,ε) = (5/4,3/5): rank 24 of 30 ⇒ nullspace = 6 Killing | verified_symbolically | `lindef_kepler.py`, logs |
| trigonometric displacements, Solov'ev | Ξ(λ,t) e^{ins} with (λ,t)-degrees (2,3), (3,4): mode 1 → only the two mode-1 Killing combinations (translation, rotation about a horizontal axis); modes 2, 3 → nullspace 0 (both parameter sets) | verified_symbolically | `lindef_trig_solovev.py`, logs `lindef_trig_n*_*.log` |
| Stäckel profile-fibre tests | (b1) two-centre (μ₁≠μ₂, ℓ≠0) at the 1:2 and 2:3 meridional resonances, (b2) Kepler–Stark at 17:16: ⟨V²⟩,⟨V³⟩,⟨V⁴⟩ vary with the transverse phase (spreads 1e-1…1e-4 ≫ 1e-10 noise) ⇒ profile constant only along the rotation ⇒ **only axisymmetric QS tori**; (a) Family-2 horizontal Stäckel system + pendulum at three doubly resonant tori (10:21, 3:7, 5:11 horizontal; 1:9, 1:4, 1:6 vertical): Jacobian of (⟨V^k⟩) w.r.t. the two orbit-space phases has rank 2 at all tested points ⇒ 0-dim fibres ⇒ **refuted** | verified_numerically (1e-11 tolerances, closure errors 1e-12) | `profile_fibre_rank.py`, `profile_fibre_rank.log` |
| task 4 (algebraic filter on survivors) | no surviving family from task 3 ⇒ not needed | — | — |

Nothing here is a candidate solution. The structural products are (i) the closed-line system (M1),(M2),(W),(J) with the
precise statement of what strong QS adds on rational surfaces, and (ii) the first-order deformation problem as an exact
finite linear system, which is empty (beyond isometries and family parameters) in every finite class tested.

## 1. Orbit-coordinate formulation (task 1) — `orbit_coords_formulation.py` (≈20 s)

Coordinates (ψ, s, t), B = x_t, u = ∂_s, J = det(x_ψ, x_s, x_t), g_ab = x_a·x_b.

* div B = ∂_tJ/J, div u = ∂_sJ/J: J = J(ψ) ⇔ div B = div u = 0 (the s-relabelling s → σ(ψ,s) with ∂_sσ ∝ J(ψ,s)
  makes J s-independent; after that s is fixed up to s → s + s₀(ψ); the t-shift is fixed up to t₀(ψ) by aligning
  the |B|-profiles, unless |B| is constant along lines — the isodynamic case excluded by handoff Thm 5.1).
* (L_uB♭)_a = ∂_s g_ta, so strong QS ⇔ ∂_s g_tt = ∂_s g_ts = ∂_s g_tψ = 0; weak QS (u·∇|B| = 0) ⇔ ∂_s g_tt = 0.
  Equivalently L_uB♭ = 2S(u)B (since [u,B] = 0), so weak QS ⇔ B·S(u)·B = 0 (B on the null cone of the strain of u),
  strong QS ⇔ S(u)B = 0 (B in its kernel).
* x_tt·x_a = ∂_t g_ta − ∂_a g_tt/2 (calculus). Force balance x_tt = ∇Π with Π = p(ψ) + g_tt/2 (single valued) is
  therefore: a = t identity; a = s: ∂_t g_ts = ∂_s g_tt (M1); a = ψ: ∂_t g_tψ − ∂_ψ g_tt = p′(ψ) (M2). In terms of the
  current J^a = ε^{abc}∂_b g_tc/J: (M1) ⇔ J^ψ = 0, (M2) ⇔ (J×B)_ψ = p′. With weak QS, (M1) ⇔ ∂_t g_ts = 0
  ⇔ x_tt·x_s = 0 ⇔ u·∇Π = 0 ⇔ the 1-form x_tt·dx has no ds-component; closedness of x_tt·dx is then (M2) plus
  ∂_s(x_tt·x_ψ) = 0, which follows from (M2) and ∂_s g_tt = 0. Periodicity of g_tψ in t gives the closed-line
  version of the Boozer relation: p′T(ψ) + d/dψ∮g_tt dt = 0, T = ∮dt.
* **Closed-line system:** (J) J = J(ψ); (W) ∂_s g_tt = 0; (M1) ∂_t g_ts = 0; (M2) ∂_t g_tψ − ∂_ψ g_tt = p′(ψ).
  Consequences: g_ts = I(ψ) + k(ψ,s), g_tψ = K_p(ψ,t) + κ(ψ,s) with ∂_tK_p = p′ + ∂_ψ g_tt. Strong QS ⇔ k_s = κ_s = 0.
* **k is not fixed by the local equations on a rational surface** ([B]): with B♭ = K dψ + (I + k(α))dθ + (G − ιk(α))dφ,
  k constant along lines (α = θ − ιφ) and zero-mean, one has √g J^ψ = 0, B_φ + ιB_θ = G + ιI (so √g = (G+ιI)/B² is
  unchanged), and √g(J×B)_ψ = (∂_φ+ι∂_θ)K − (G′+ιI′) — k drops out of every equation of §2.2 of CONTEXT; it only adds
  the parallel current δJ = (∂_ψk)B, constant along lines. RHB's weak ⇒ strong argument uses B_θ = I(ψ), which on a
  rational surface is not implied. Consequence for this direction: **only (J),(W),(M1),(M2) are to be imposed on a
  closed-line design; ∂_s g_ts = ∂_s g_tψ = 0 is an extra condition to report, not to impose.** (The strain-kernel kills
  of round 1, S(u)B = 0, therefore do not apply as stated to closed-line configurations; the weak version B·S(u)·B = 0
  is what remains.) Whether some global argument forces k = 0 is open; for the two exact congruences u is Killing and
  k = κ = 0 trivially ([C],[D]).
* Checks on the exact congruences: Solov'ev x = R_z(s)y0(λ,t), y0 = √(2S)(cos λ cos t, −sin λ sin t, 0) +
  (κS/2)cos 2λ cos(2t+δ) e_z (m = (S/2)cos 2λ, from round 1): J λ-only, (W),(M1) hold, (M2) gives
  p′(λ) = S²κ² sin 4λ, equal to d/dλ[p₀ − S − 2κ²m²] of the Grad–Shafranov identification, x_tt·dx closed, and the
  Cartesian J×B = ∇p check of the GS field passes. Kepler force-free (label ψ = ε, ∂_ψ at fixed t including the
  −(τ_ψ/τ_η)∂_η term): at the rational points (C,ε) = (5/4,3/5), (25/36,5/13) all identities hold exactly, (M2) = 0 = p′.

## 2. Design-first search at first order (task 2)

### 2.1 Reduction to a linear system
For an exact congruence x0(ψ,s,t) any first-order deformation is x = x0 + εξ(x0) (x0 is a diffeomorphism onto the
tube), and δg_ab = x0_a·(∇ξ)x0_b + x0_b·(∇ξ)x0_a = 2x0_a·S(ξ)·x0_b, δJ = J0 div ξ. The closed-line system is linear in
g, so the first-order problem is the **linear** system
    div ξ|_{mode n≠0} = 0,  ∂_s δg_tt = 0,  ∂_t δg_ts = ∂_s δg_tt,  ∂_s,∂_t[∂_t δg_tψ − ∂_ψ δg_tt] = 0,
i.e. S(ξ)B = c_ψ(ψ,s)∇ψ + c_s(ψ,s)∇s with line-constant c's — the weak (null-cone) relaxation of the strain-kernel
lemma, with the deformation ξ in the role of u. Eulerian reading: δB = ∇×(ξ×B), δp = −ξ·∇p, so (M1),(M2) say ξ is a
neutral incompressible ideal-MHD displacement (neighbouring equilibrium) and (W) says the Lagrangian perturbation of
|B| vanishes (same |B|-profile on the displaced lines). Mode n ≠ 0 non-Killing solutions would be first-order
non-axisymmetric closed-line QS-MHS fields; the class x = Φ_s(x0) with Φ_s = exp(sW), W = Ψ_*∂_φ (CONTEXT §3.4) is
the special case ξ = generator of Ψ.

### 2.2 Results (exact rational linear algebra, Laurent polynomials in e^{iλ}, e^{is}, e^{it})
* Solov'ev, polynomial ξ of degree ≤ 2, 3 (S = 2, κ = 3/2, cos δ = 3/5 and S = 8, κ = 1, δ = π/2): nullspace 9 =
  6 Killing + (x,y,0) + (0,0,z) + (0,0,x²+y²). The three non-Killing ones are the family parameters: horizontal
  dilation (S), vertical scaling (κ), and the vertical shear z → z + ερ², which is a δ-shift combined with a
  translation (κ′cos δ′ = κ cos δ + 2ε, κ′sin δ′ = κ sin δ, exact in the family). All have s-modes {0, 1} only and
  satisfy strong QS trivially. Degree ≤ 4: see `lindef_solovev_d4_c0.log`.
* Kepler force-free, polynomial ξ: degree ≤ 1 gives exactly the 6 Killing fields (the dilation fails because a
  fixed-period dilation is not a Kepler symmetry); degree ≤ 2 at (C,ε) = (5/4,3/5): rank 24 ⇒ nullspace = 6 Killing.
* Solov'ev, trigonometric displacements in orbit coordinates, one s-mode at a time (Ξ(λ,t)e^{ins}, (λ,t)-degrees
  (2,3) and (3,4)): n = 1 → exactly the two mode-1 Killing combinations (translation ∥ (1,i,0), rotation
  (1,i,0)×y0); n = 2, 3 → nullspace 0 (both parameter sets).
* Conclusion (status: verified_symbolically in the finite classes): **no non-axisymmetric first-order closed-line
  weak-QS-MHS deformation of the Solov'ev (ι = 2) or Kepler (ι = 1) congruences exists among polynomial vector fields
  of degree ≤ 3 (4) or trigonometric displacements of degree ≤ (3,4).** The weak/strong distinction of §1 did not
  matter: all solutions found are strongly QS. The class "congruent closed lines x = Φ_s(x0) with a non-isometric
  one-parameter family Φ_s" is therefore dead to first order around both exact congruences in these classes; the
  general (non-polynomial) statement is open (see §4).

## 3. Profile-fibre rank tests on Stäckel systems (task 3) — `profile_fibre_rank.py` (≈1 min)
Method (round-1 §5): at fixed energy the closed orbits of a non-degenerate integrable system lie on isolated doubly
resonant Liouville tori L; a closed-line QS torus is a curve in O = L/flow along which the profile is constant; the
shift-invariants ⟨V^k⟩ (physical-time averages, k = 1..4) must be constant on that curve. Liouville form
h = h₁(q₁)+h₂(q₂), V = (F+G)/h, Stäckel time σ (dτ = h dσ), periods by quadrature, resonance P₁/P₂ = m/n by brentq
in the separation constant, closed orbits integrated in σ (DOP853, 1e-12) with prescribed transverse phases.
* Axisymmetric V: the rotation direction of O is trivially a profile symmetry, so a non-axisymmetric QS torus needs the
  profile constant on all of O (analyticity). ⟨V⟩ is always constant on a resonant torus (Lagrangian action), the
  higher moments are not: two-centre μ₁ = 1, μ₂ = 1/2, ℓ = 0.3, E = −0.6 at 1:2 (spread of ⟨V²⟩ 0.19) and 2:3
  (0.039); Kepler–Stark μ = 1, F = 0.02, ℓ = 0.3, E = −0.5 at 17:16 (spread 1e-4, noise 1e-12). **Refuted:** in
  axisymmetric Stäckel potentials separable in prolate or parabolic coordinates every closed-line QS torus is a
  rotation orbit (axisymmetric field).
* Family 2 (V_h = −(sin²A + sinh²B)/(8ε(sinh²u + cos²v)), pendulum V_z = sin²λz/(2λ²), ε = S = λ = 1) at E_h = 0
  (Landreman's value), −0.02, −0.01: horizontal resonances 10:21, 3:7, 5:11 and pendulum resonances T_z = T_h/9,
  T_h/4, T_h/6; Jacobian of (⟨V^k⟩)_{k≤4} with respect to the two orbit-space phases has rank 2 at four generic
  points each (singular values well separated from 0). **Refuted** (0-dim fibres) at these tori; rank drops only on
  measure-zero sets, which cannot carry a 1-dim fibre with identical profiles (a fibre is a level set of an analytic
  map of rank 2 on an open dense set).
* Task 4 filter not needed (no survivor).

## 4. Open statements for the other agents
1. Closed-line QS in orbit coordinates is the system (J),(W),(M1),(M2) of §1. Strong QS is an extra pair of
   conditions on rational surfaces, not implied locally; whether it is implied globally (or whether a weak-not-strong
   closed-line QS-MHS field can exist) is a clean open question; the gap is a line-constant parallel current.
2. First-order problem (§2.1) in general: Ξ(λ,t)e^{ins} with arbitrary t-Fourier content. The four conditions are
   (W) b·Ξ_t = 0, (M1) ∂_t[b·(e_z×Ξ + inΞ) + w·Ξ_t] = 0, (M2) ∂_t[b·Ξ_λ + y_λ·Ξ_t] − 2∂_λ[b·Ξ_t] = 0,
   (dJ) det(Ξ_λ,w,b) + det(y_λ, e_z×Ξ + inΞ, b) + det(y_λ,w,Ξ_t) = 0, with b = y0_t, w = e_z×y0 (finite trig
   polynomials in (λ,t)). A top-mode argument in t (as in boozer-normal-flow §3.2) could turn the finite results of
   §2.2 into a theorem: "no first-order non-axisymmetric closed-line QS-MHS deformation of any axisymmetric
   closed-line equilibrium".
3. Reduced open problem of round 1 (a profile-isospectral non-congruent family of closed orbits) is now refuted for
   all Stäckel systems tested; a positive example would need a resonant torus on which all slice sums |A_n| are
   constant along a non-isometric curve — for axisymmetric V this means constant on the whole torus.

## 5. Reproduce
```
python3 orbit_coords_formulation.py                 # 20 s   -> orbit_coords_formulation.log
python3 lindef_solovev.py 3 0 ; python3 lindef_solovev.py 3 1     # 3-5 min each -> lindef_solovev_d3_c*.log
python3 lindef_solovev.py 4 0                       # long (see lindef_solovev_d4_c0.log)
python3 lindef_kepler.py 1 ; python3 lindef_kepler.py 2 1         # 3.5 min / ~15 min -> lindef_kepler_d*.log
python3 lindef_trig_solovev.py n Dl Dt [case]      # e.g. 1 2 3 ; 2 2 3 ; 3 2 3 ; 1 3 4 ; 2 3 4 (1-10 min) -> lindef_trig_n*_*.log
python3 profile_fibre_rank.py                       # 1-2 min -> profile_fibre_rank.log
```
