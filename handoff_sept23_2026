# CONTEXT: exact quasisymmetric MHS equilibria via elastic-rod magnetic axes

Handoff file for continuing this research in Claude Code. Written 2026-09-23 at the end of a long
multi-session investigation. Treat everything below as the authoritative state of the project.
Companion code lives in `code/` (see §10); the container that produced it does not persist, so
regenerate data with the scripts and the download commands in §10.

---

## 0. How to use this file (instructions to the assistant)

- Read §1–§2 first (problem + binding rules), then §5–§8 (results), then §9 (open items).
- The user is a stellarator-theory expert. Tone: concise, technical, no hedging filler, no overclaiming.
  **No exact non-axisymmetric QS+MHS solution exists yet; do not imply one does.**
- Every numerical claim in this file was computed in-session; quote it as "measured on the low-res
  precise-QA VMEC file" etc. Do not invent new numbers; recompute with the scripts.
- Prefer global constructions. Near-axis expansion (NAE) only as a check or boundary condition (§2).

---

## 1. Research question

Find explicit, smooth magnetic fields that (a) satisfy magnetohydrostatics (MHS) `J×B = ∇p`,
(b) have nested toroidal flux surfaces, (c) are quasisymmetric (QS). Constant rotational transform
ι is acceptable (even zero); non-constant ι would be a major win. The user believes the
Garren–Boozer (GB) non-existence argument is not a proof (it is an overdetermination count at third
order in the near-axis expansion), and wants to exploit an empirical link between QS and Kirchhoff
elastic-rod centerlines:

- (User-supplied, unverifiable) unpublished computations: magnetic axes of *precise* QS fields, when
  continued along a parameter, satisfy the elastic-rod centerline equations to small tolerance and
  hug the 1-D families of closed rods identified by Ivey–Singer in their 2-D moduli space.
- Schief (2003): every field line of an *isodynamic* MHS field obeys the vortex-filament (LIA)
  equation; soliton/travelling-wave solutions are elastic rods (genus-1 finite-gap potentials).
- Landreman (project file `analytic_3d_equilibrium.txt`): first analytic smooth MHS solutions
  with nested surfaces and nontrivial ι (two families, neither QS).

Line of attack adopted: (1) fields whose magnetic axis is an exact elastic-rod centerline;
(2) MHS with pressure a function of flux; constant ι acceptable.

## 2. Binding instructions from the user (do not violate)

1. **"Stop doing anything with near-axis, pyQSC or using Rodriguez et al."** No NAE derivations,
   no pyQSC. **"Start from global constructions and use NAE only as checks/boundary condition near
   the axis."** Extracting first-order data (η̄, σ) from a *global* equilibrium at definition level
   is allowed as a check; deriving order-by-order NAE equations is not.
2. **Current chosen direction: option (ii)** — "nonlinear potential V with a non-planar
   constant-speed closed orbit as axis; this may give enough expressiveness to impose QS on top of
   MHS." (Options (i) bifurcation from axisymmetric equilibria and (iii) special evolutions in the
   Boozer normal-flow system were offered and not chosen.)
3. Reasonably concise replies; honest counting of what is and is not established.

## 3. Sources (project files, read-only, at `/mnt/project/` in the original environment)

- `analytic_3d_equilibrium.txt` — Landreman's two analytic families (details §5.5).
- `elastrod.txt` — Ivey–Singer, closed elastic rods: first integrals, κ² elliptic, Killing field
  𝓘 = λ₂T + λ₃κ𝐁 = aJ + γ×J, cylindrical relations (κ² affine in r², z_s affine in κ², screw
  momentum r²θ_s − a z_s = const), closure Δθ = 2πm/n, 1-D (m,n) families connecting m- and
  n-covered circles, Kida curve a = 0.
- Schief, J. Plasma Phys. 65 (2003) 465–484 — isodynamic MHS ⇔ Heisenberg spin/LIA
  `r_b = r_s × r_ss` with a Jacobian constraint; Thm 3.1 abnormality Ω_ss = 0; travelling waves ⇒
  helicoidal Π-surfaces = flux surfaces, field lines are rods (a ≠ 0 helicoids; a = 0 Palumbo
  surfaces of revolution).
- Levien (elastica history, §7: rod curvature linear in displacement), Calini–Ivey (finite-gap ↔
  rods), Calini (knotted filaments) — background, not read in detail.

## 4. Mechanical formulation (the framework everything else uses)

Field lines of an MHS field are trajectories of `ẍ = −∇V`, `V = −Π`, `Π = p + |B|²/2`, with
velocity `ẋ = B` (so `ẍ = B·∇B = ∇Π`). Energy `E = |B|²/2 + V = −p(ψ)` is a flux function.

- An MHS field with nested surfaces = a one-parameter family of invariant **isotropic 2-tori** of
  the 3-DOF Hamiltonian `H = |v|²/2 + V` projecting onto the flux surfaces (ω|_torus = J-flux = 0
  since J·∇ψ = 0). The graph Γ_B = {(x,B(x))} is Lagrangian iff curl B = 0 (vacuum).
- `∇·B = 0` ⇔ the Jacobian of the projection from torus coordinates (ψ,θ₂,θ₃) to x is a flux
  function (hard constraint; Landreman's families satisfy it by special structure).
- **QS ⇔ V is invariant under the symmetry flow** `u = (G̃B + B×∇ψ)/B²` ⇔ `B·∇Π = D(ψ,Π)` ⇔ along
  each field line `V̇² = F(ψ,V)` (autonomous 1-D clock; in Boozer coordinates
  `χ̇ = ι̃B²(ψ,χ)/(G̃+ι̃I)`). Standard equivalent form: `G̃ B·∇|B| = |B|²κ 𝐁·∇ψ`.
- Count (Clebsch form B = ∇ψ×∇α): MHS is 3 equations for 3 unknowns (ψ, α, V); QS adds one.
  **The GB overdetermination (+1) is global, not an artifact of the expansion.** Choosing V cleverly
  cannot change the count; it can only supply structure that makes the extra equation consistent.
- Axis: QS ⇒ the axis is a constant-speed closed trajectory, V constant on it, `∇Π = B₀²κN`, and
  differentiating along the axis: `Hess Π·T = B₀²(κ′N − κ²T + κτ𝐁)`, i.e.
  `Π_TT = −B₀²κ²`, `Π_TN = B₀²κ′`, `Π_TB = B₀²κτ` (Frenet frame T,N,𝐁).

## 5. Established theory (with proof sketches)

### 5.1 Theorem (ι ≡ N ⇒ axisymmetric)
QS + MHS + nested tori with rotational transform identically equal to the helicity integer N (so
|B| is constant along every closed field line) ⇒ isodynamic ⇒ (Schief) Ω constant ⇒ travelling
waves ⇒ helicoidal surfaces ⇒ compactness forces surfaces of revolution ⇒ axisymmetric. Hence
ι − N ≠ 0 is necessary; the Palumbo mechanism ("all field lines are rods") is excluded.

### 5.2 Lemma (rod ⇔ two conservation laws along the axis)
For the Killing screw field `𝓘 = α e_z + e_z×x` (∇𝓘 antisymmetric, ∇_e𝓘 = e_z×e), a unit-speed
curve is a Kirchhoff rod iff (i) `T·𝓘 = const` and (ii) `|𝓘|² = α² + r²` is affine in κ².
Proof: `d/ds(T·𝓘) = κ N·𝓘`, so (i) ⇔ N·𝓘 = 0 ⇔ 𝓘 = λ₂T + h𝐁; (ii) then gives h = ±λ₃κ.
For a QS axis: (i) ⇔ **𝓘·∇Π = 0 on the axis**, (ii) ⇔ **|∇Π|² = B₀⁴(A + B(α²+r²)) on the axis**.
Kida rods (α = 0) ⇔ N meridional ⇔ compatible with an axisymmetric V. α ≠ 0 (QA axis has
α ≈ −0.56 R₀) forces a non-axisymmetric V whose total pressure is screw-invariant along the axis.
Companion identity used as a convention check: `κ(λ₂ − λ₃τ) = e_z·𝐁` (holds to 1e−11 on the exact
rod, 0.5% on the VMEC QA axis).

### 5.3 Rods as charged-particle orbits
`𝓘 = λ₂T + λ₃κ𝐁` ⇒ `κN = T×𝓘/λ₃`, so every Kirchhoff rod is a trajectory of
`γ″ = T×𝓘(γ)/λ₃`, i.e. a charged particle in the magnetic field of a uniform axial field plus a
uniform current density along z (screw pinch, B_θ ∝ r). All such trajectories have constant speed.
A congruence of them is isodynamic (⇒ axisymmetric by 5.1), so only the axis can be a rod in QS.
Also: rod ⇔ the guiding-centre velocity `v_∥b̂ + v_drift` on the axis is a rigid screw motion;
`u|_axis ∝ 𝓘 − λ₃κ𝐁` (Killing minus LIA/grad-B drift direction).

### 5.4 First-order geometric theorem (any QS field)
The |B| = B₀ symmetry lines χ = ±π/2 on the flux surface at radius r are the LIA displacements
`γ ± (r/η̄) κ𝐁` of the axis to first order (uses only X ∝ (η̄/κ)cos χ, Y ∝ (κ/η̄)sin χ). If the
axis is a rod, these are rigid screw copies `e^{±t𝓘}γ` to first order. Verified numerically (§6.2).

### 5.5 Landreman's families in this language
- Family 1: V = (x²+y²+4z²)/2 (1:1:2 oscillator, superintegrable), ι = 2, axis is a 1:1:2 Lissajous
  curve, |B| varies on the axis.
- Family 2 (sheared ι): ω = x+iy, K = √(ω̄²+ε), Ξ = ωK + π/2 − S, B_x+iB_y = e^{−iλz}W,
  W = i sinΞ/(2K), B_z = (1/λ)Re(e^{−iλz}cosΞ), ψ = ½[sin²λz + (Re(e^{−iλz}cosΞ))²], p = p_a − ψ/λ²,
  Q = −|W|²/2 + sin²(λz)/(2λ²); identities W_ωω = −K²W, W_ωω̄ = −|ω|²W, WW_ωω − W_ω² = 1/4; nfp = 2;
  planar axis {Re(ωK) = S, z = 0} with |B| = cosh(Im ωK)/(2|K|) varying; ε = 0 seed is the linear
  Grad–Shafranov solution ψ = cos(λz)cos(R²−S)/(2λ).
  Mechanism: V separable = V_h(x,y) + V_z(z); all field lines have zero horizontal energy; flux
  label = vertical pendulum energy ⇒ planar axis at the bottom of the well.
  **Completeness within the ansatz** B_h = e^{−iλz}W, B_z = (2/λ)Im(e^{−iλz}W_ω): MHS ⇔
  WW_ωω − W_ω² = C, Im(W̄W_ωω̄) = 0 ⇒ W = (√−C/k)sin(kω + b), k² = c₀ω̄² + 2c₁ω̄ + c₂,
  kb′ = c̄₁ω̄ + d; single-valuedness ⇒ d = 0; complex ε ~ rotation, complex S ~ tilt. Cannot be tuned
  to QS. A helical-phase variant (pattern rotating with z, simple B_z ansatz) forces axisymmetric W.
  **Stäckel structure (verified analytically):** with ω = √ε sinh ζ, ζ = u+iv: Ξ = A(u) + iB(v),
  A = (ε/2)sinh 2u + π/2 − S, B = (ε/2)sin 2v; |sinΞ|² = sin²A + sinh²B; |K|² = ε(sinh²u + cos²v)
  = |dω/dζ|². Hence (E_h − V_h)|dω|² is of Liouville form for every E_h: Family 2's horizontal
  system is Stäckel-separable in elliptic-cylindrical coordinates, V_h = [F(u)+G(v)]/(ε(sinh²u+cos²v))
  with the specific F = −sin²A/8, G = −sinh²B/8. Both Landreman families are "separable
  congruences" (resonant sub-tori of Liouville tori). ∇·B = 0 (Jacobian condition) is what selects
  the specific F, G.

### 5.6 Integrable/separable V and the resonance structure
In an integrable V the flux tori are resonant sub-tori `k₁θ₁ + k₂θ₂ + k₃θ₃ = c(ψ)` of Liouville
3-tori. If V is axisymmetric and the resonance does not involve the toroidal angle (k₃ = 0), the
rotation preserves each sub-torus ⇒ B axisymmetric. Non-axisymmetric fields in axisymmetric V need
k₃ ≠ 0 (nfp set by the resonance). For V = V_h(r) + V_z(z) separable: a constant-speed closed orbit
with ż = 0 at the apsides is mirror-symmetric (τ = 0 there) ⇒ not a rod unless planar; a
stellarator-symmetric non-planar orbit needs V_h with an interior minimum r* on the orbit's radial
range (ż = 0 where r = r*, z = z_mid at the apsides). Kida-rod condition for such an axis reduces to
one first-order ODE: with U = ż² as a function of r, `(√U)′ = ±B₀² √((A + Br²) r² /(B₀²r² − ℓ²))`
(elliptic integral). **Derived, not numerically verified.**

### 5.7 Global reformulations (from earlier phases; available if needed)
- Vacuum QS ⇔ coordinates (ψ,α,Φ) with ∇Φ ⊥ ∇ψ, ∇α, |∇Φ| = |∇ψ×∇α| = B(ψ, α + ι̃Φ/G).
- ADM normal flow in ψ with lapse G/B and prescribed mean curvature tr K = −(ι̃/G)∂_χB; the radial
  (ψ) evolution of the Boozer embedding is overdetermined by one function per step (GB globally):
  2 algebraic + 2 first-order PDE constraints on the 3 components of ∂_ψx, with one free function of χ.
- Strain rate of u annihilates B (pure shear ⊥ B): QS = inextensible unsheared transport; u is a
  steady Euler flow only if axisymmetric; L_u B♭ = 0 ⇒ σ = G̃′.
- Maupertuis: on each flux surface the field lines are simultaneously geodesics and asymptotic
  lines of the surface in the Jacobi metric g_E = (E(ψ) − V)δ (normal curvature = ∂_νΠ/|B|²).
- Hess Π = Ġ + G² on the axis (G = ∇B on the axis, Ġ = B·∇G): **the axis Hessian is first-order data.**
- Vacuum: Γ_B Lagrangian ⇒ monodromy spectrum of the axis orbit in the field-line Hamiltonian
  system is {1, 1, e^{±2πiι₀}, e^{±2πiι₀}} (doubly degenerate elliptic pair). Verified (§6.4).

## 6. Numerical findings (all on `wout_LandremanPaul2021_QA_lowres.nc` unless stated)

Data: precise QA (Landreman–Paul 2021), vacuum, nfp = 2, ns = 75, mpol = ntor = 8, ι = 0.416–0.423
(ι₀ = 0.4232), R₀ ≈ 1, a_minor = 0.168, B₀ = 1.00648. Axis Fourier (R: n = 0,2,4,6; Z: n = 2,4,6):
R = [1.00396, 0.18397, 0.02169, 0.00259], Z = [−0.15804, −0.02057, −0.00255]. Axis length
L = 6.6776, κ ∈ [0.488, 1.468], τ ∈ [−0.513, 1.565]. Rod fit (Frenet convention §10.4, VMEC axis
orientation): α = −0.5628, λ₂ = T·𝓘 = 0.9646 (std 3.7e−4), λ₃ = 𝐁·𝓘/κ = −0.6283 (std 3.6e−3),
N·𝓘 rms 1.65e−3. Rod tests T1–T5 residuals 0.1–0.3% (relative to range); sensitivity: random
perturbations of 3e−3 R₀ → ~20%, 1e−3 R₀ → ~1%.

### 6.1 Only the axis is a rod
|B|-contours (u-lines) on surfaces s = 0.05–0.9: rod residuals 2–6%, insensitive to 1e−2 rad
perturbations, only 2–3× better than θ = const curves ⇒ not rods. Field lines have inflection
points ⇒ not rods. Screw momentum aB_z + RB_φ: best a ≈ −0.57 near axis but surface variation
≈ 0.19–0.20 (r/R₀) (linear in r) ⇒ not a flux function. QH lowres (nfp = 4, ι ≈ −1.24,
R₀ = 13.7): axis 2–3.5%, contours 6–15% (unresolved file).

### 6.2 Screw copies and LIA displacements of the axis (this session)
Relative std of s along the displaced curve (s = normalized flux):
- LIA displacement `γ + cκ𝐁`: 0.54% (c=0.005), 0.93% (0.01), 1.79% (0.02), 3.59% (0.04) — linear in c
  ⇒ first-order theorem 5.4 confirmed. Constant-magnitude binormal displacement `γ + d𝐁`: 74% (O(1)).
- Rigid screw copy `e^{t𝓘}γ`, α = −0.563: 1.34% (t=0.005), 1.37% (0.01), 1.46% (0.02), 1.71% (0.04),
  2.35% (0.08; displacement ≈ 0.09 R₀). Wrong pitch at t = 0.02: 41% (+0.563), 38% (α=0), 41% (0.281),
  41% (0.844). Pitch scan minimum at α ≈ −0.575. The ~1.3% floor is the data/rod-fit noise floor.
- |B| along screw copies: std 6.8e−5, 7.1e−5, 8.3e−5, 1.4e−4, 4.1e−4 for t = 0.005…0.08 — O(t²) with
  coefficient ≈ 0.04, ~25× below the generic second-order scale. Mean offset −3e−4 is B₀ estimation.
- Normal displacement d N: |B| − B₀ = 5.7e−3 (d=0.005), 9.6e−3 (0.01): linear, as expected.

### 6.3 Screw invariance of Π (= B²/2, vacuum)
RMS over flux surfaces of cos∠(𝓘, ∇Π) with 𝓘 = α e_z + e_z×x, α = −0.563: 3.99e−3, 5.86e−3,
7.43e−3, 1.06e−2, 1.40e−2, 2.10e−2 at mean distance r = 0.021, 0.031, 0.041, 0.061, 0.081, 0.122
⇒ **cos ≈ 0.19 r/R₀** (0.191, 0.190, 0.182, 0.173, 0.172, 0.172). Controls: 0.24 (α = 0), 0.41 (−α),
flat in r. So Π is screw-invariant to first order (forced by the rod) and broken at O(r).
(The ρ = 0.05 row of this scan is spline-extrapolation garbage; ignore ρ < 0.1.)

### 6.4 Hessian of Π on the axis (Hess Π = Ġ + G², G from FD of the B vector with h = 0.02 R₀)
- Consistency: antisymmetric part 0.5%; vacuum identity |B·∇B − ∇(B²/2)|/|∇(B²/2)| = 0.2–0.3%,
  |curl B|/|∇B| = 0.1–0.3%, div B/|∇B| = 0.2%; Π_N = B₀²κ to 0.1%.
- Trajectory row: Π_TT vs −B₀²κ² 0.3%; Π_TN vs B₀²κ′ 0.9%; Π_TB vs B₀²κτ 1.2%.
- 6D monodromy of the axis orbit (true field): |ev| = [1.141, 1, 1, 1, 1, 0.877] (the 1.141/0.877 pair
  is the trivial pair perturbed by noise), rotation numbers ±0.4176 and ±0.431 — field-line pair
  reproduces ι₀ = 0.4232 (direct field-line linearization δẋ = Gδx gives 0.422); the second pair is
  degenerate with it (Lagrangian graph, §5.7). Direct second differences of Π (h = 0.03) agree in rms
  with Ġ+G² but are too noisy for stability (gave rotation 0.088) — always use Ġ + G².
- **Helical (screw-invariant) predictions fail at second order:** Π_NB vs −(λ₂/λ₃)B₀²κ′/κ:
  rel. mismatch 0.344; Π_BB vs B₀²(z_s − λ₂τ)/λ₃: 0.373. Substituting helical Π_NB, Π_BB (true Π_NN):
  monodromy |ev| = [238.9, 1.05, 1.05, 0.95, 0.95, 0.0042] — strongly hyperbolic. Helical Π_NB only:
  123.7; helical Π_BB only: 16.4. Scan Π_NN → Π_NN + cB₀² with helical off-diagonals: only c = −2 is
  elliptic (rotations 0.127, 0.316 — not ι₀, not degenerate). **The screw-symmetry breaking is O(1)
  in the torsion couplings and is what makes the axis elliptic with the right ι₀.**
- **Fitted structure of the true Hessian (units B₀², lengths in R₀):**
  `Π_NN = 3.19κ² − 0.86τ + 0.93` (rel. resid 0.1%; adding τ² changes nothing);
  `Π_BB = −1.56 z_s + 0.86τ − 0.11` (0.8%) vs helical −1.59 z_s + 1.54τ  [1/λ₃ = −1.59];
  `Π_NB = 0.78 κ′/κ + 0.29 κ′` (1.1%) vs helical 1.54 κ′/κ;
  trace = ∇²Π = |∇B|² ≈ 1.71κ² + 1.36 (0.5%).
  Deviations from helical: ΔΠ_BB ≈ −0.68τ − 0.11 (1.4%), ΔΠ_NB ≈ −0.76κ′/κ + 0.29κ′ (2%). The z_s
  coupling is the helical one; the breaking is entirely in the τ and κ′ couplings (as if λ₂ → ≈0.5).
- Reactor-scale QA (`..._QA_reactorScale_lowres_reference.nc`, R₀ = 10.17, B₀ = 5.91): identical
  dimensionless coefficients (3.19, −0.89, 0.94 / −0.157·R₀-scaled z_s, 0.85 / 0.78, 0.30), same
  monodromy (rotations 0.4164, 0.4311).
- QH lowres (R₀ = 13.72, α/R₀ = 0.219, λ₂ = 11.06, λ₃ = 63.08, N·𝓘 rms 3.3% ⇒ unresolved rod):
  trajectory row fine (1.1–1.5%); helical predictions off by 100%; Π_NN = 3.19κ² − 0.75τ + 2.86 (1.4%)
  — same κ² coefficient as QA; Π_BB, Π_NB fits poor (24–33%); monodromy rotations 0.02, 0.15, 0.49 do
  not reproduce ι₀ (0.7625 mod 1) ⇒ near-axis data unreliable. Needs a resolved QH equilibrium.

### 6.5 Values of the fitted Hessian along one field period (QA, units B₀²)
```
 s/L    kappa   tau     z_s    Pi_NN   Pi_NB   Pi_NB_hel  Pi_BB   Pi_BB_hel
 0.000  1.468  -0.513   0.325   8.260  -0.000   -0.000   -1.034   -1.305
 0.074  1.294  -0.433   0.174   6.638  -0.587   -0.776   -0.755   -0.941
 0.141  0.940  -0.124  -0.076   3.860  -0.974   -1.429   -0.089   -0.069
 0.198  0.625   0.667  -0.234   1.596  -1.051   -1.693    0.832    1.395
 0.251  0.488   1.565  -0.282   0.301   0.000    0.000    1.689    2.852
```

## 7. Explicit constructions (this session)

### 7.1 Exact closed (1,2) rod matched to the QA axis (`code/rodexact.py`)
Integrate γ″ = T×𝓘/λ₃ from the stellarator-symmetric point (R_max, 0, 0), T(0) = (0, cos β, sin β),
with λ₃ = −0.6283, R_max = 1.2126 (QA values); shoot on (α, β) so that the next κ-extremum lies at
φ = π/2, Z = 0. Result: α = 0.561636, β = −0.340808, L = 6.65696, closure 5e−13, κ ∈ [0.472, 1.488];
in its own Frenet frame T·𝓘 = 0.955131, 𝐁·𝓘/κ = 0.6283 (signs differ from the VMEC-oriented fit
because of curve orientation). Harmonics R = [0.99684, 0.18974, 0.0229, 0.00276],
Z = [−0.16208, −0.02164, −0.00272] (≈3% from QA, since λ₃ and R_max were fixed rather than fitted).
Saved as `rod_exact.npy` (regenerate by running the script).

### 7.2 Explicit helically symmetric potential with the rod as exact orbit (`code/v0exact.py`)
Orbit-space coordinate ζ(x) = e^{−iz/α}(x + iy) (𝓘·∇ζ = 0). With C the rod's projection in polar
form ρ_C(ϑ): `V₀(x) = (|ζ| − ρ_C(arg ζ))·H(arg ζ) + (|ζ| − ρ_C)²·K(arg ζ)`, H fixed by
|∇V| = B₀²κ on the rod (sign so that ∇V = −B₀²κN). Rod is an exact orbit: return error 9.1e−10,
speed drift 3.2e−10 (with the exact rod; with the VMEC axis instead: 3.7e−2 — the construction needs
an *exact* rod). All screw copies are orbits; the helicoid S = {e^{t𝓘}γ} is Schief's isodynamic
travelling-wave surface. Monodromy (K = 0): [20.22, 1.0008, 1.0002 ± 0.0241i, 0.999, 0.0495] —
hyperbolic. Only Π_NN is free in a helically symmetric V (Π_NB, Π_BB are fixed by symmetry, §5.2);
the K-scan for other values was not completed (timed out), but §6.4 shows constant shifts of Π_NN
cannot give ι₀ with the degenerate structure.

## 8. Current interpretation

Near the axis a QS field coincides, on the helicoid S through the axis (≈ the Π = Π₀ surface),
with Schief's α ≠ 0 travelling-wave isodynamic equilibrium: S is swept by congruent rods with
|B| = B₀. QS is a **toroidalization** of that non-compact, helically symmetric object: exact on the
axis (rod ⇔ two conservation laws), first-order on S (screw copies are the |B| = B₀ symmetry lines),
and broken at second order in V by an O(1) change of the torsion couplings (Π_BB: τ-coefficient
0.86 instead of 1.54; Π_NB: 0.78κ′/κ + 0.29κ′ instead of 1.54κ′/κ) that is precisely what makes the
axis an elliptic orbit with the correct ι₀. For α = 0 the underlying object is Palumbo's
equilibrium (circular axis), which is why nontrivial QS axes need α ≠ 0. Because Hess Π on the
axis is first-order data, the constant-coefficient fits of §6.4 are statements about the first-order
QS solution around a rod axis: they suggest (η̄, σ(s)) are algebraic in the rod's (κ, τ, z_s).
This is the sharpest constraint on an exact solution obtained so far; it is not yet a solution.

## 9. Open problems and prioritized next steps

1. **Test the algebraic-first-order conjecture.** Extract η̄ and σ(s) from the VMEC near-axis surfaces
   at *definition level* (first-order ellipse in the Frenet frame; no NAE derivation) and test whether
   σ is an algebraic function of (κ, τ, z_s) of the rod (e.g. linear fits as in §6.4). If yes, the
   first-order QS problem around a rod axis closes in finite-gap terms — first analytic piece beyond
   the axis. Separate which of the §6.4 fits are generic first-order QS identities (true for any QS
   field) from rod-specific facts (compare with a non-rod QS-optimized axis if one is available).
2. **Global V with the correct axis Hessian.** Build V = V₀ (helical core) + quadratic-in-distance
   correction so that Hess V on the axis matches §6.4, integrate the trajectory congruence around the
   axis, and check whether nested tori with ι ≈ 0.42 form beyond the linear regime; measure ι(ψ).
3. **Resolved QH equilibrium** (higher ns, mpol, ntor; e.g. from the simsopt/DESC repositories or by
   rerunning VMEC): repeat §6.2–6.4. Test universality: is the 3.19κ² coefficient universal? Are the
   Π_BB/Π_NB coefficients functions of the rod moduli (λ₂, λ₃, α)?
4. **Structured class for exact solutions:** Stäckel-separable V in elliptic-cylindrical (or other
   Eisenhart) coordinates with resonant sub-tori whose resonance involves the toroidal frequency;
   impose ∇·B = 0 as the Jacobian condition (this selects Landreman's F, G in Family 2) and then QS as
   u-invariance of V. Also: deformation/toroidalization of Schief's α ≠ 0 helicoidal equilibrium.
5. Numerically verify the Kida-rod separable construction of §5.6 (ODE for U(r)); check whether a
   stellarator-symmetric non-planar Kida axis in an axisymmetric separable V exists.
6. Repeat the screw-copy / cos∠(𝓘,∇Π) / Hessian tests on the QH and on other precise QA/QH
   configurations to confirm the 0.19 r/R₀ breaking coefficient and its dependence on the rod moduli.

## 10. Code, data, conventions

### 10.1 Data (downloadable; `scipy.io.netcdf_file` reads them; GitHub API is rate-limited, raw URLs work)
```
BASE=https://raw.githubusercontent.com/hiddenSymmetries/simsopt/master/tests/test_files
curl -sL -o wout_LandremanPaul2021_QA_lowres.nc $BASE/wout_LandremanPaul2021_QA_lowres.nc
curl -sL -o wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc $BASE/wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc
curl -sL -o wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc $BASE/wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc
```
Python deps: numpy, scipy (netcdf_file, CubicSpline, solve_ivp DOP853, fsolve).

### 10.2 Scripts (in `code/`; run from that directory with the .nc files present)
- `geom.py` — class `Vmec`: cubic splines in ρ = √s of rmnc/zmns (full mesh) and bmnc (half mesh,
  parity-correct even/odd extension (−1)^m to ρ < 0); `axis(ph)` (correct Z sign); `RZ(rho,th,ph,
  deriv)`; `Bmag`; `invert(pts)` = damped Newton in quasi-Cartesian (u,v) = (ρcosθ, ρsinθ) at fixed
  cylindrical angle (robust near the axis; residual ~1e−16).
- `rodlines.py` — `frenet(x)` (FFT Frenet data of a closed curve sampled uniformly in a periodic
  parameter), `rod_tests(x)` (T1 (κ²)_s² cubic in κ²; T2 τ = A + C/κ²; T3 z_s affine in κ²;
  T4 κ² affine in r²; T5 screw momentum, returns fitted a), |B|-contour and field-line tracing.
  Its `__main__` uses the raw zaxis_cs sign (mirror image of the true axis; rod tests unaffected).
- `rodexact.py` — exact closed (1,2) rod by shooting (§7.1); writes `rod_exact.npy`.
- `v0exact.py` — helically symmetric V₀ from the exact rod, orbit verification, monodromy K-scan (§7.2).
  Slow (finite-difference gradients inside DOP853); reduce the scan or use analytic gradients.
- `screwcopy.py <wout> <alpha>` — screw copies / LIA / control displacements of the axis vs flux
  surfaces and |B| (§6.2). Use alpha = −0.563 for the QA file (sign depends on orientation!).
- `screwinv.py <wout> <alpha>` — RMS cos∠(𝓘, ∇|B|) on surfaces (§6.3).
- `bvec.py` — B vector from bsupumnc/bsupvmnc (parity-extended splines), ∇B on the axis by central
  differences, vacuum-identity checks, field-line linearization monodromy (ι₀); writes `gradB_axis.npy`.
- `hess2.py` — Hess Π = Ġ + G² on the axis, comparisons, 6D monodromy, helical substitutions and
  Π_NN scans (§6.4); needs `gradB_axis.npy` and `hess_h0.03.npy` (from `hess.py`, direct FD Hessian).
- `axisfit.py <wout>` — self-contained version of the whole axis analysis (screw fit, Ġ + G² Hessian,
  trajectory-row and helical checks, constant-coefficient fits, 6D monodromy). Start here.
- `monod.py`, `hess.py` — earlier/auxiliary versions (FD Hessian; superseded by Ġ + G²).

### 10.3 Gotchas learned the hard way
- VMEC axis: `Z_axis = −Σ zaxis_cs sin(nφ)` (matches rmnc/zmns at s = 0). Using +Σ gives the mirror
  image: rod tests unchanged, but everything involving the surfaces breaks (inversion fails, 41%
  garbage). `geom.py` has the correct sign.
- Half-mesh quantities (bmnc, bsupumnc, bsupvmnc) must be extended to ρ < 0 with parity (−1)^m before
  spline interpolation; plain even extension gives 2e−3 errors in |B| near the axis.
- Newton in (ρ,θ) is singular at the axis; use (u,v) = (ρcosθ, ρsinθ) with step clipping.
- Never take second differences of |B| for stability analysis; use Hess Π = Ġ + G².
- Spline extrapolation below ρ ≈ 0.1 is unreliable for surface-averaged quantities (the ρ = 0.05 row).
- The rod property is sharp: 1e−3 R₀ axis errors → 1% residuals; 3e−3 → 20%. The low-res QH file
  is not adequate for any axis-Hessian statement.
- Sign of the screw pitch α flips with curve orientation; always refit `T·𝓘 = const` for the
  orientation in use and verify with `κ(λ₂ − λ₃τ) = e_z·𝐁`.

### 10.4 Conventions
- Frenet: N = T′/κ, 𝐁 = T×N, τ = −𝐁′·N (standard; `frenet()` returns this τ).
- Killing screw: 𝓘 = α e_z + e_z×x; on a unit-speed rod 𝓘 = λ₂T + λ₃κ𝐁 with λ₂ = T·𝓘, λ₃ = 𝐁·𝓘/κ.
  Rod ODE: γ″ = T×𝓘/λ₃. Ivey–Singer's (a, μ) normalization differs by scale.
- V = −Π; field lines parametrized so that ẋ = B. Hess Π(𝓘, e) = −(∇_e𝓘)·∇Π = −B₀²κ e_z·(e×N) on
  the axis for a screw-invariant Π (gives Π_NB = −λ₂B₀²κ′/(λ₃κ), Π_BB = B₀²(z_s − λ₂τ)/λ₃).
- Boozer: χ = θ_B − Nφ_B; |B| = B(ψ,χ); u = (G̃B + B×∇ψ)/B²; first-order ellipse X = r(η̄/κ)cos χ,
  Y = r(κ/η̄)(sin χ + σcos χ) (up to the usual √(2/B₀) scaling) — used only as definitions.

## 11. Glossary
Π total pressure p + B²/2; V = −Π mechanical potential; E = −p(ψ) energy; B₀ axis field strength;
κ, τ axis curvature/torsion; z_s = e_z·T; T, N, 𝐁 Frenet frame; 𝓘 Killing screw field of the rod;
α screw pitch (translation per radian); λ₂, λ₃ rod constants; S helicoid {e^{t𝓘}γ};
u QS symmetry vector; χ Boozer helical angle; η̄, σ first-order NAE data; G = ∇B on axis;
ι₀ on-axis rotational transform; nfp field periods; LIA localized induction approximation
(vortex filament equation) r_b = κ𝐁; GB Garren–Boozer.
