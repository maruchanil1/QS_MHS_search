# Round 1 — global-reformulations

Direction: reformulate QS-MHS so that the Garren–Boozer +1 overdetermination becomes finitely many algebraic
identities on structured ansätze (CONTEXT §2.2, §3.4, §3.6; handoff §5.6–5.7).
All scripts: `python3 <name>.py` from this directory (sympy 1.14). Runtimes are given per script.

## 0. Summary of outcomes

| item | statement | status | script |
|---|---|---|---|
| algebraic-B formulation | QS-MHS ⇔ (V, ψ, E(ψ), F(ψ,V)) with B fixed pointwise by \|B\|² = 2(E−V), B·∇ψ = 0, B·∇V = ±√F, plus (a) div B = 0, (b) B·Hess ψ·B = ∇V·∇ψ, (c) B·Hess V·B = F_V/2 + \|∇V\|² | proved (2-line identities) + verified on Solov'ev | `algebraicB_solovev.py` (3 s) |
| reconstruction formula | B = ±√F \|∇ψ\|²(∇V)_∥/\|∇ψ×∇V\|² ± √(2(E−V) − F\|∇ψ\|²/\|∇ψ×∇V\|²) ∇ψ×∇V/\|∇ψ×∇V\| reproduces the Solov'ev B to 1e−31 | verified | same |
| strain-kernel lemma | strong QS ⇔ S(u)B = 0, S = sym∇u, tr S = 0 ⇒ det S(u) = 0 and **B ∥ ker S(u)** wherever u is not Killing | proved (identity checked symbolically) | `strain_kernel_conjugations.py` (1 s) |
| conjugated rotations, shears | u = Ψ_*∂_φ dead for: vertical shear z→z+f(x,y) (B horizontal & tangent to level curves of ∂_φf ⇒ cylinders), z-independent horizontal shear (B vertical), any linear Ψ∈SL(3) (S const ⇒ straight lines); for Ψ_x∘Ψ_z quadratic, det S = 0 is a genuine degree-11 polynomial identity (not classified) | proved / partially open | same |
| Boozer identities as ONE vector equation | (1),(2),(4') ⇔ W = K x_θ×x_φ + I x_φ×x_ψ + G x_ψ×x_θ, algebraic in x_ψ; solvable iff surface condition C1 := I(W·x_φ) − G(W·x_θ) = 0; then x_ψ = Y×a/\|a\|² + μ a with a = I x_φ − G x_θ, Y = W − K x_θ×x_φ | verified symbolically (residuals vanish on C1 = 0) | `boozer_flow_reduction.py` (~3.5 min) |
| √g is metric-only on a QS surface | √g = (W·x_θ)/I = (W·x_φ)/G = \|W\|²/(G+ιI) on C1 = 0; QS (3) ⇔ C2 := (∂_φ + N∂_θ)[\|W\|²/(G+ιI)] = 0, a second **surface** condition | verified symbolically | same |
| μ is algebraic | ∂_ψ f = f_ψ(χ) (f := \|W\|²/(G+ιI)) has μ_θ, μ_φ coefficients 2C1/(G+ιI)·(ι,1) ⇒ zero on the surface; coefficient of μ is 2W·Da/(G+ιI) with W·Da = −(G+ιI)(G+NI) f′(χ)/2 ⇒ **μ = μ_alg(x-jets; f_ψ(χ), ι′, G′, I′)** wherever f′(χ) ≠ 0 | verified symbolically + on the vacuum family (μ_alg = μ_true at rational points) | same |
| COMPAT | ∂_ψC1 = 0 is a first-order linear PDE for μ; with μ = μ_alg it is a single scalar condition COMPAT(x-jets ≤ 3, f_ψ, f_ψ′, ι′, G′, I′, K, K′) = 0 on each surface: the Garren–Boozer +1 made explicit | derived; verified = 0 on the exact vacuum family | same |
| explicit finite Boozer embedding | vacuum ι = 0: x+iy = (R₀(ψ) + a(ψ)cos θ)e^{iφ}, z = −k a sin θ, R₀ = ψ/(kG)+c, a² = ψ²/(kG)² + 2cψ/(kG) + d: exact solution of (1)–(4) with B = G∇φ; used as sanity test | verified symbolically | same |
| finite trig surfaces, axisymmetric degree 1 | C1 forces ι·\|z₁\|² = 0: **no degree-1 (in θ) axisymmetric Boozer surface with ι ≠ 0** | proved (Fourier modes solved) | `boozer_finite_surface.py` |
| finite trig surfaces, smallest non-axisymmetric class | see §4 | see §4 | same |

Nothing here is a candidate solution. The main product is the reduced system of §3, which is the right
finite-dimensional playground for round 2.

## 1. Algebraic-B formulation (Step 1)

Mechanical formulation: field lines are orbits of ẍ = −∇V, ẋ = B, V = −Π, E(ψ) = −p. QS ⇔ clock V̇² = F(ψ,V).
The three *pointwise algebraic* conditions
(A1) |B|² = 2(E(ψ) − V), (A2) B·∇ψ = 0, (A3) B·∇V = s√F(ψ,V)
fix B up to a two-fold choice (line ∩ sphere). Writing e := B·∇B + ∇V for the force-balance error,
e·B = B·∇E = 0 automatically, e·∇ψ = −[B Hess ψ B − ∇V·∇ψ], e·∇V = F_V/2 − [B Hess V B − |∇V|²]
(using B·∇(B·∇ψ) = 0 and B·∇(B·∇V) = V̈ = F_V/2). Hence QS-MHS ⇔ (A1)–(A3) and
(a) div B = 0, (b) B·Hess ψ·B = ∇V·∇ψ, (c) B·Hess V·B = F_V/2 + |∇V|²,
three equations for the two scalar fields (V, ψ): the +1 overdetermination, with only (a) differential in B.
(b) is the Maupertuis normal-curvature law κ_n|B|² = ∂_νΠ of handoff §5.7. Verified on the Solov'ev equilibrium
ψ = A(R²−R₀²)²/8 + C R²z²/2 + D z²/2 (A = 1, C = 1/2, D = 1/3, F₀ = 3): J×B = ∇p, div B = 0, clock (triple product) = 0,
(b) = 0, (c) = 0 with F_V computed at fixed ψ, and the reconstruction formula reproduces B at three rational
points (|B_rec − B| ~ 1e−31 with the appropriate signs).
Use for the other agents: given any candidate congruence (potential V + orbits), (b) and (c) are *algebraic* QS-MHS
tests that need only Hessians of V and ψ and the clock F — no field-line integration.

The ansatz V = Φ(w), ψ = Ψ(w,q) was not pursued beyond this: for transnormal w (spheres/cylinders/planes) the
clock reduces to constancy of angular momentum / axial momentum on tori, which the closed-orbit-congruences agent
already showed forces isometry (their §3.2–3.3); for other w the reduced system (a)–(c) is the honest statement.

## 2. Strain-kernel lemma and conjugated circle actions (Step 3)

Identity (checked symbolically for arbitrary u, B): L_u(B♭) = 2 S(u)(B,·) + ([u,B])♭, S = sym∇u.
So weak QS + strong QS (RHB 2020: automatic for MHS with nested surfaces) ⇔ **S(u) B = 0**. With tr S = div u = 0,
S has eigenvalues (μ, −μ, 0), so **det S(u) = 0 everywhere and B ∥ ker S(u)** wherever u is not infinitesimally
Killing. The symmetry vector alone fixes the direction of B. Consequences for u = Ψ_*∂_φ:
- Ψ_z: z → z + f(x,y): u = ∂_φ + (∂_φf)∂_z, det S ≡ 0, ker S = e_z×∇(∂_φf). B is horizontal and tangent to the
  level curves of h := ∂_φf, so ψ = ψ(h,z) and u·∇ψ = 0 gives ∂_φh = hQ(h) on every circle, impossible for a
  periodic h unless ∂_φh = 0 ⇒ h = h(R) ⇒ ψ_z = 0 (cylinders) or u Killing. Dead.
- Ψ_x: x → x + g(y): planar u, ker S = e_z, B vertical. Dead. (For g(y,z), det S = 0 is a genuine constraint.)
- Linear Ψ ∈ SL(3): S constant ⇒ B ∥ fixed vector ⇒ straight lines. Dead.
- Ψ_x∘Ψ_z with quadratic f, g: det S ≡ 0 is a degree-11 polynomial identity in (x,y,z) (197 coefficient
  equations, 10 unknowns); generic instance has det S ≠ 0. Not classified (Gröbner too slow in the time box).
Note also: in Boozer coordinates u = ∂_φ + N∂_θ, so the conjugation Ψ *is* the Boozer embedding; Step 3 and
Step 2 are the same problem, and the Boozer form below is the efficient way to write it.

## 3. Boozer identities as a constrained normal flow (Step 2, main result)

Notation: x_θ, x_φ, x_ψ tangents, W = x_φ + ιx_θ (B = W/√g), D = ∂_φ + ι∂_θ, u = ∂_φ + N∂_θ, χ = θ − Nφ.
Strong QS makes K = B_ψ a function of (ψ,χ), so (4) becomes (ι−N)∂_χK = G′ + ιI′ + p′√g, and (1),(2),(4′) are
the single vector equation

    (E)  W = K x_θ×x_φ + I x_φ×x_ψ + G x_ψ×x_θ      [B♭ = K dψ + I dθ + G dφ]

which is algebraic in x_ψ: with a := I x_φ − G x_θ, Y := W − K x_θ×x_φ, (E) ⇔ a × x_ψ = Y, solvable iff
C1 := a·Y = I(W·x_φ) − G(W·x_θ) = 0, and then x_ψ = Y×a/|a|² + μ a (μ = free tangential component).
On C1 = 0: √g = (W·x_θ)/I = (W·x_φ)/G = |W|²/(G+ιI) — metric only — so QS (3) is the second surface condition
C2 := u(f) = 0, f := |W|²/(G+ιI). Both C1 and C2 involve only the induced metric of the surface in its Boozer
parametrization (no ψ-derivatives).

Propagation (formal ψ-derivation on jets, `boozer_flow_reduction.py` part [B]):
- ∂_ψf = f_ψ(χ): coefficients of μ_φ, μ_θ are 2C1/(G+ιI) and 2ιC1/(G+ιI) (zero on the surface); coefficient of μ
  is 2W·Da/(G+ιI) with W·Da = −(G+ιI)(G+NI)f′(χ)/2. Hence μ = μ_alg is **algebraic** wherever f′(χ) ≠ 0
  (i.e. away from the |B| extrema on the surface and excluding surfaces with |B| constant).
- ∂_ψC1 = 0 is linear first-order in μ (coefficients printed by the script) and with μ = μ_alg becomes
  COMPAT(x-jets up to order 3; f_ψ(χ), f_ψ′(χ), ι′, G′, I′; K(χ), K′(χ)) = 0.

**Theorem (reduction).** A QS-MHS field with nested surfaces in Boozer form is the same as a one-parameter family
of surfaces x(ψ;θ,φ) with C1 = C2 = COMPAT = 0 on each surface, evolving by x_ψ = Y×a/|a|² + μ_alg a, with the
free data f_ψ(χ) (one function of χ per surface), ι′, G′, I′, p′ (constants per surface) and K from (4).
The overdetermination is exactly: COMPAT must be preserved by the flow (its ψ-derivative is a further condition,
etc.). For an axisymmetric surface everything is φ-independent and COMPAT reduces to one ODE relating f_ψ, f_ψ′,
ι′ at each χ, which is why axisymmetric solutions exist. For a non-axisymmetric surface COMPAT must hold along
each u-orbit (φ at fixed χ) with χ-constant data (f_ψ, f_ψ′) and global constants (ι′, …): a linear-dependence
condition on functions of φ.

Tests (exact rational arithmetic at rational points): (i) the shifted-ellipse vacuum family (exact B = G∇φ,
finite Fourier content, ι = 0, N = 0, |B| = G/R varies on the surface): (1)–(4) hold symbolically, C1 = C2 = 0,
x_ψ = Y×a/|a|² + μ_true a, μ_alg = μ_true, COMPAT = 0 at two points; (ii) screw pinch (periodic cylinder,
ι(ψ) = ψ, I ≠ 0, p′ ≠ 0, K = 0): (1)–(4) and C1 = C2 = 0 hold; here f′(χ) = 0 (|B| constant on the surface) so
W·Da = 0 and μ is not algebraic — the expected degenerate case. A non-degenerate test with ι ≠ 0, I ≠ 0, K ≠ 0
was not available in closed form (no exact axisymmetric equilibrium with explicit Boozer angles); the ι, I, K
dependence is covered only by the symbolic identity checks [A] and by the formal derivation.

## 4. Finite trigonometric Boozer surfaces (`boozer_finite_surface.py`)

Ansatz x+iy = e^{iφ}ρ(θ,φ), z = ζ(θ,φ), ρ complex and ζ real trig polynomials; the e^{iφ} drops out of the
metric, so C1, C2 are finite Fourier systems.
- Axisymmetric, degree 1 in θ (ρ = ρ₀ + ρ₁e^{iθ} + ρ₋₁e^{−iθ}, ζ = Re(ζ₁e^{iθ})), vacuum: C1 has modes 0, ±1,
  ±2; modes ±2 ⇒ ρ₁ρ̄₋₁ = ζ₁²/4 (if ι ≠ 0), mode ±1 ⇒ |ρ₁| = |ρ₋₁|, mode 0 ⇒ ιζ₁² = 0. **Theorem: ι = 0 in this
  class** (checked by sympy's solve on the real/imaginary parts); the ι = 0 solutions are the shifted-ellipse family.
- Smallest non-axisymmetric class (m ∈ {−1,0,1}, n ∈ {−N,0,N}, N = 1): 26 real unknowns + ι; C1 gives 49 and C2
  40 real polynomial equations. The extreme modes (|m| = 2 or |n| = 2N) are isotropy (null-vector) conditions, e.g.
  (ι+1)[ρ₋₁,₋ₙ ρ̄₁,ₙ + ζ₋₁,₋ₙ²/4] = 0, (ι−1)[ρ₋₁,ₙ ρ̄₁,₋ₙ + ζ₋₁,ₙ²/4] = 0, (2ι±1)[…] = 0, ι[…] = 0. The full
  26-unknown system is beyond sympy's solve in the time box; with **stellarator symmetry** (ρ real, ζ imaginary:
  14 unknowns incl. ι) the extreme modes give 21 branches and the full C1+C2 system 64 terminal solutions:
  20 complex (discarded), 19 axisymmetric about the z-axis or about a translated axis (the mode ρ₀,₋₁ is a rigid
  x-translation), 12 degenerate (z ≡ 0, a cylinder, or a curve: rank of the parametrization < 2), 11 with ι = ±N
  (the isodynamic case excluded by handoff Thm 5.1; relabelled axisymmetric ι = 0 tori or |B|-constant surfaces),
  and 2 surfaces of revolution with swapped Boozer labels (θ_B winds around the z-axis:
  x+iy = e^{iθ}(a + b e^{2i(φ−θ)}), z = c sin(θ−φ), a = √((2−ι)/ι)|c|/2, b = c²/(4a), 0 < ι ≤ 2; checked by
  invariance under (θ,φ) → (θ+c, φ+c)). **No non-axisymmetric embedded torus in this class satisfies even the two
  ψ-frozen surface conditions.** (All classification steps are automatic in the script; see
  `boozer_finite_surface.log`.) Status: verified_symbolically for the stellarator-symmetric subclass; the
  non-stellarator-symmetric subclass (26 unknowns) is open.

Remark on sign conventions: for the screw pinch, (4) with K = 0 gives p′ = −ιI′ = −4ψ² for ι = ψ, r = √(2ψ),
which agrees with the direct pinch force balance d/dr(p + B²/2) + B_θ²/r = 0 (dp/dψ = −r⁴ = −4ψ²): (4) as written
in CONTEXT §2.2 has the right signs.

## 5. Reproduce
```
python3 algebraicB_solovev.py
python3 strain_kernel_conjugations.py
python3 boozer_flow_reduction.py      # ~3.5 min; log in boozer_flow_reduction.log
python3 boozer_finite_surface.py      # log in boozer_finite_surface.log
```
