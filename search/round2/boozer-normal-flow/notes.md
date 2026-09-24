# Round 2 — boozer-normal-flow

Continues round-1 `global-reformulations` (Boozer identities as a constrained normal flow in ψ). All scripts:
`python3 <name>.py` from this directory (sympy 1.14); logs `<name>.log`. Notation as in round 1 / `flow_core.py`:
x(ψ,θ,φ) Boozer embedding, x_θ, x_φ tangents, W = x_φ + ιx_θ = Dx (D = ∂_φ + ι∂_θ, B = W/√g), a = Ix_φ − Gx_θ = ∂_a x
(∂_a = I∂_φ − G∂_θ), u = ∂_φ + N∂_θ, χ = θ − Nφ, f = |W|²/(G+ιI) (= √g on a QS surface), n = x_θ×x_φ, P = |n|²/f,
K = B_ψ = K(ψ,χ) with (ι−N)K′ = G′ + ιI′ + p′f [eq. (4)], f′ := ∂_χ f.

## 0. Summary of outcomes

| item | statement | status | script |
|---|---|---|---|
| Task 1: non-degenerate validation | Boozer angles of the exact Solov'ev (1:1:2) congruence (ι=2, I≠0, K≠0, p′≠0, K′≠0 at pt 2) and of the Kepler force-free congruence (ι=1, I≠0, p′=0) built explicitly; at 2+2 exact points all of (1),(2),(4′),(4), √g = (G+ιI)/B², G, I, p flux functions, K = K(ψ,θ), C1 = C2 = 0, W·Da closed form, x_ψ = Y×a/|a|² + μ_true a, μ_alg = μ_true, COMPAT = 0 hold exactly | verified_symbolically | `task1_validate.py` (2 min) |
| reusable module | exact Boozer jets of any congruence x(λ,s,τ) = R_z(s)X0(λ,τ) by truncated-series inversion of the explicit map (ψ,θ_B,φ_B)(λ,s,τ) — seconds per point, no numerics | tool | `boozer_axisym.py`, `series_jets.py` |
| Task 2, structure of ∂_ψC1 | exact polynomial identities: coeff(μ_φ) = \|a\|² + I·C1, coeff(μ_θ) = ι\|a\|² − G·C1, coeff(μ) = ∂_aC1 ⇒ on a C1-surface **∂_ψC1 = \|a\|² D(μ) + R** (a magnetic differential equation for the Boozer gauge μ), R explicit (§2.2) | proved + verified | `task2_compat_structure.py` [A],[B] |
| Task 2, μ | μ = −h(χ) + m₀^χ(χ) + m₀^h with h = f_ψ/((G+NI)f′), m₀^h = −2(DW·n)/(P(G+ιI)(G+NI)f′) (Maupertuis normal-curvature law) | proved + verified | same [C] |
| Task 2, COMPAT | **COMPAT = \|a\|²[−(ι−N)ĥ′(χ) + Q̂]** with ĥ := h − m₀^χ and Q̂ a function of the 3-jet of the surface, K′ and (ι′,G′,I′) only. Hence COMPAT = 0 ⇔ **(C3) u(Q̂) = 0 and (C4) ∮Q̂dχ = 0**, and then **f_ψ is determined** by the surface (up to the Boozer gauge constant): the QS profile's ψ-derivative is not free data | proved + verified (4 constrained random points, N = 0,1,2; both exact congruences) | same [D],[E],[F] |
| Task 2, chain | ∂_ψ^k COMPAT has the same form with the new datum ∂_ψ^k ĥ: the hierarchy is **C3_k: u(∂_ψ^k Q̂) = 0**, k ≥ 0, with 2 free constants per order (∂_ψ^{k+1} of the two free profiles) and ι^{(k+1)} fixed by ∮. No free functions anywhere. Axisymmetric surfaces: every C3_k is an identity (closes at k = 0). Non-axisymmetric: one function of two variables per order against 2 constants — never closes by count | proved (structure) / conjecture (non-closure) | §2.4 |
| Task 3, reformulation | in v = e^{iχ}, E = e^{iφ}: C1+C2 ⇔ W·W and W·x_χ are E-independent and proportional (κ = I/(G+ιI)) | proved | `vE_surface.py` |
| Task 3, top-mode lemma | top E-mode X_J(v) of any trig-polynomial Boozer surface with J ≥ 1 and non-resonant ι: X_J·X_J = 0 = X_J′·X_J′ ⇒ **X_J = λ(v)ν₀, ν₀ a fixed null vector** (WLOG ε = (1,−i,0)/2 after a rotation) | proved (identities checked) | `task3_finite_fourier.py` B |
| Task 3, J = 2 | any v-degree, non-resonant ι: X_1 = 0, X_0 = c + γ₀(v)e_z ⇒ x+iy = c + λ(v)E²: a doubly covered surface of revolution | proved (§3.3) + verified at D = 1 | same, C |
| Task 3/4, J = 3, D = 1 | contains round-1's open 26-unknown class (m,n ∈ {−1,0,1}, N = 1, no stellarator symmetry). Hierarchy E⁵…E¹: **X_2 = X_1 = 0, X_0 = c + γ₀e_z** (triply covered surface of revolution) for ι = 2/11 (symbolic λ and random numeric λ) and ι = −3/13 | verified_symbolically | `task3_J3.py` |
| resonant ι − N ∈ Z (ι = 0, N = 1) | D_j has a kernel (functions of θ are invisible to W = x_φ): the hierarchy admits x+iy = c + E³λ(v), z = z₀ + Z(θ), which passes C1 and C2 (κ = I/G = 1/8, two conditions on λ) but has planar self-intersecting field lines stacked at periodic heights: immersed, **not an embedded torus** | verified_symbolically (existence) / argued (non-embeddedness) | `task3_J3.py 0`, `task3_resonant_branch.py` |
| Task 3, general J | conjecture: same collapse for all J, D and non-resonant ι; proved for the "holomorphic" subclass (all X_{j>0} ∥ ε) | conjecture / partial proof (§3.4) | — |

Nothing here is a candidate solution. The main products: (i) the **one-surface problem is now three explicit conditions
C1, C2, C3 on a single surface with no free functions**; (ii) finite trigonometric Boozer surfaces are dead for
non-resonant ι in every class tested, with a proof for J = 2 and a proof mechanism for all J.

## 1. Task 1 — exact Boozer angles and non-degenerate validation

Congruence x(λ,s,τ) = R_z(s)X0(λ,τ), B = x_τ/(dt/dτ), one toroidal transit per τ-period, poloidal winding ι ∈ Z.
Λ(λ) := (1/2π)∮B²dt = G + ιI; F(λ,τ) := ∫₀^τ B²dt/Λ; φ_B = s + F, θ_B = ιF (rotation covariance fixes the s-dependence);
√g_B = (G+ιI)/B² requires dψ/dλ = −J/ι with J = det(x_λ, x_s, x_t) (λ-only by div B = 0). Then G = B·x_φ (= L_z),
I = (Λ−G)/ι, K = B·x_ψ. Caveat (1:1:2): x(s,t+π) = x(s+π,t), so s is defined mod π on the quotient torus; F(π) = π
(B² is π-periodic), so (θ_B,φ_B) ↦ (θ_B+2π, φ_B+2π) — the Boozer lattice is fine, local statements unaffected.

Jets: the first attempt (symbolic composite derivatives, `d_theta = −∂_s/ι + ∂_τ/(ιF_τ)` applied three times and
simplified) did not finish in 10 min. `series_jets.py` instead expands everything in exact truncated Taylor series
at the point (Taylor-mode AD on the expression tree; the only irrational constant, cτ₀ in F, is removed with the
constant term), inverts the map (dψ,dθ,dφ)(dλ,ds,dτ) as a formal series and composes: < 1 s per point (Solov'ev),
~20–100 s when the point is irrational (Kepler pt 2 has cos i = 2√2/3). Everything exact; `nsimplify` is never used
(it mangles large exact rationals into fake radicals — this bit an earlier run).

Results (`task1_validate.log`, 64 s): Solov'ev λ = 2 (G = −15/4, I = 5, K = −6/5, G′ = 4/3, I′ = 11/6, p′ = −5, √g = 1)
and λ = 3 (K = −596412978/423416825, K′ = −2713223205/20085551968 ≠ 0); Kepler C = 5/4, ε = 3/5 (G = 16/25, I = 9/25,
G′ = −I′ = −205/64, p′ = 0) and C = 25/36, ε = 5/13 (G = 48/65, I = 17/65): all residuals exactly 0. Remark: for the
force-free Kepler family (4) gives ιK′ = G′ + ιI′ = 0, so K ≡ 0 there (j ≠ 0 but B_ψ = 0); the fully non-degenerate test
(ι, I, K, K′, p′ all nonzero) is Solov'ev. The round-1 propagation formulas are validated in that regime.

## 2. Task 2 — the compatibility condition and its chain

### 2.1 ∂_ψC1 is a magnetic differential equation for μ
With x_ψ = v + μa (v := Y×a/|a|²), the μ-dependent part of ∂_ψC1 is, exactly (polynomial identities checked in [A]),
(Dμ)|a|² + (∂_aμ)C1 + μ ∂_aC1, because a·Da + W·∂_aa = ∂_a(W·a) = ∂_aC1 (D and ∂_a commute, Da = ∂_aW). On a surface
with C1 ≡ 0: **∂_ψC1 = |a|² Dμ + R**. Also, on such a surface a ⊥ W (C1 = a·W), so (W, a) is an orthogonal tangent
frame with |W|² = (G+ιI)f, |a|² = (G+ιI)P, and the flow is
    x_ψ = [K/(G+ιI)] W + μ a + √(f/P) n̂ .
Geometric reading: C1 ⇔ the field-line direction D and the direction ∂_a = I∂_φ − G∂_θ are orthogonal on the surface;
C2 ⇔ |Dx| is constant along every u-orbit (symmetry line); in coordinates (χ, φ) the surface metric is E = |x_χ|² (free),
x_χ·x_u = If − (ι−N)E, |x_u|² = (G−ιI+2NI)f + (ι−N)²E — C1+C2 leave one free metric function.

### 2.2 Explicit R, μ, COMPAT
Using ∂_ψ|W|² = 2W·Dx_ψ + 2ι′If and ∂_ψ(W·a) = a·Dx_ψ + W·∂_ax_ψ + ι′a·x_θ + (I′G − G′I)f with the frame above
(h(X,Y) = ∂_XY·n̂ second fundamental form, √(f/P) h(W,W) = (DW·n)/P, √(f/P) h(W,a) = (∂_aW·n)/P):

    R   = −(G+NI) f K′ − 2(∂_aW·n)/P − ι′|a|²/(G+ιI) + (I′G − G′I) f                                   [B]
    μ   = −h + m₀^χ + m₀^h,  h = f_ψ/((G+NI)f′),
          m₀^χ = [(ι−N)(2K′f + Kf′) + (ι′I − G′ − ιI′)f]/((G+ιI)(G+NI)f′),  m₀^h = −2(DW·n)/(P(G+ιI)(G+NI)f′)   [C]
    COMPAT = |a|² [ −(ι−N) ĥ′(χ) + Q̂ ],   ĥ := h − m₀^χ,
    Q̂  = D(m₀^h) − 2(∂_aW·n)/(P|a|²) + f[(I′G − G′I) − (G+NI)K′]/|a|² − ι′/(G+ιI)                          [D]

The μ-equation is the Maupertuis law κ_n|B|² = ∂_νΠ of the handoff (κ_n = h(W,W)/|W|² normal curvature along B).
Verified: [B],[C],[D] as exact identities at random rational jets constrained to C1 = ∂C1 = ∂²C1 = 0, u(f) = ∂u(f) = 0
(N = 0, 1, 2; ι = 2/5, −3/7, 3/4, 5/3), and on the exact Solov'ev/Kepler surfaces where (ι−N)ĥ′_true = Q̂ holds with
the true f_ψ. COMPAT is affine in (f_ψ, f_ψ′) with the coefficients of [D] and independent of f_ψ^{(n≥2)} [E]; Q̂ is
affine in (ι′,G′,I′) and ι′ enters only through the u-invariant constant −ι′/(G+ιI) [F].

### 2.3 Consequences for one surface
Since ĥ is a function of χ alone, COMPAT = 0 ⇔
    (C3) u(Q̂) = 0   and   (C4) ∮ Q̂ dχ = 0,
after which f_ψ = (G+NI)f′[m₀^χ + (ι−N)⁻¹∫Q̂dχ + c]: **the free datum f_ψ(χ) of round 1 is not free**; c is the Boozer
gauge θ → θ − Gc(ψ), φ → φ + Ic(ψ) (which shifts μ by c′), and the constant of K is the other gauge (θ,φ) → (θ+c₁, φ+c₂)
with Ic₁′ + Gc₂′ = K₀ (K₀ drops out of Q̂, which contains only K′). Flux constants: (ι′, G′, I′, p′) with the K-periodicity
relation 2π(G′+ιI′) + p′∮f = 0; C3 involves only the two combinations c₁ = (I′G − G′I) − (G+NI)(G′+ιI′)/(ι−N),
c₂ = −(G+NI)p′/(ι−N) (the two free profiles), and C4 fixes ι′. Cauchy-problem reading: surface + |B| on it + flux constants
determine the neighbourhood, exactly as for Grad–Shafranov; QS costs one scalar condition C3 per surface.
Regular form: multiply by f′² (u-invariant): C3 ⇔ u(f′²Q̂) = 0. At the |B|-extremal orbits (f′ = 0) it reduces to
u((DW·n)/P) = 0: **κ_n/|∇ψ| is constant along the two extremal symmetry lines** (extremal-orbit lemma; cheap test).
For trigonometric surfaces f′²Q̂·P³ is a trigonometric polynomial, so C3 is a finite Fourier identity.

### 2.4 The chain
COMPAT ≡ |a|²[−(ι−N)ĥ′ + Q̂] holds identically on the constraint manifold, which the flow preserves. Differentiating along
the flow and using COMPAT = 0: ∂_ψ^kCOMPAT|₀ = |a|²[−(ι−N)∂_χ(∂_ψ^kĥ) + ∂_ψ^kQ̂ + (u-invariant terms from ι^{(j)}∂_χ∂_ψ^{k−j}ĥ)],
because ∂_ψ and ∂_χ commute and every ∂_ψ^jĥ is a function of χ. Hence the k-th condition is
    C3_k: u(∂_ψ^kQ̂) = 0  (given C3_0..C3_{k−1}),  plus  ∮: fixes ι^{(k+1)},
with the new free constants G^{(k+1)}, I^{(k+1)}, p^{(k+1)} (one relation) — 2 free constants per order and no new free
functions. Counting: C1, C2 use two of the three functions of (θ,φ) describing a surface; C3_0 uses the third; each
C3_k (k ≥ 1) is a further function of two variables against two constants. **Axisymmetric surfaces** (N = 0, φ-independent
family): u = ∂_φ annihilates every ∂_ψ^kQ̂, so all C3_k hold identically and the chain closes at k = 0; C4_k fix ι^{(k+1)}
given the two free profiles — consistent with the Cauchy problem for Grad–Shafranov (two free profiles p, F). **Non-axisymmetric
surfaces**: the chain never closes by count; closure at finite order would need an identity expressing u(∂_ψ^{k+1}Q̂)
through the lower C3_j — no mechanism is visible, and none is needed to state the reduced problem:

**QS-MHS (Boozer form, f′ ≢ 0) ⇔ a surface with C1 = C2 = 0 satisfying C3_k for all k ≥ 0, flowed by the determined
normal flow of §2.1.** The one-surface problem (C1, C2, C3_0) is the first non-trivial filter beyond round 1.

## 3. Task 3 — finite trigonometric Boozer surfaces

### 3.1 Symmetry-adapted variables
v = e^{iχ}, E = e^{iφ} (φ along u): x = Σ_{j=−J}^{J} X_j(v)E^j, X_j ∈ C³ Laurent polynomials, X_{−j} = conj X_j on |v| = 1.
x_χ = iv∂_vx, x_u = iE∂_Ex, W = x_u + (ι−N)x_χ, so W = Σ iD_jX_j E^j with **D_j := j + (ι−N)v∂_v** (diagonal on v^m
with eigenvalue j + (ι−N)m). C2 ⇔ W·W has no E^k, k ≠ 0; C1 ⇔ W·x_χ = κ W·W (κ = I/(G+ιI)); both use the complex-bilinear
product of the formal expansion. Non-resonance hypothesis (NR): j + (ι−N)m ≠ 0 for the (j,m) that occur (always true for
ι−N irrational; violated e.g. for ι−N ∈ Z, in particular **ι = 0, N = 1**). Sanity: N = 0 axisymmetric x+iy = Eρ(v),
z = ζ(v) has no E^{k≠0} modes and its E⁰ C1 reproduces round-1's degree-1 equations (v^{±2}: ι(r₁r_{−1}+ζ₁²) = 0, …).

Degrees: the round-1 ansatz (θ-degree M, φ-degree L, helicity N) has E-degree J = L + NM + 1 and v-degree M; so J = 1 ⇔
N = 0 axisymmetric-type, and J ≥ 2 for every N ≥ 1. J = 1 in general means all u-orbits are round circles (X_1 null).

### 3.2 Top-mode lemma (NR)
E^{2J}: W_J·W_J = 0 and W_J·vX_J′ = 0. Identities (checked): W_J·X_J = (i/2)D_{2J}(X_J·X_J) and
J W_J·X_J = −iW_J·W_J − (ι−N)W_J·vX_J′, so s := X_J·X_J satisfies D_{2J}s = 0 ⇒ s = 0; then
W_J·W_J = −[J²s + J(ι−N)vs′ + (ι−N)²v²X_J′·X_J′] ⇒ X_J′·X_J′ = 0. Two orthogonal null vectors in C³ are parallel
(totally isotropic subspaces have dimension ≤ 1), so X_J′ ∥ X_J and X_J = λ(v)ν₀ with ν₀ fixed null; on the null cone
ν(w) = (1−w², i(1+w²), 2w)/2 one has X′·X′ = λ²w′² (checked), confirming w′ = 0. Null vectors are rotations of ε up to
scale, so WLOG X_J = λ(v)ε: **the top mode of a trigonometric Boozer surface is holomorphic in a fixed plane.**

### 3.3 J = 2, any v-degree (NR) — proof
Null frame X_j = α_jε + β_jε̄ + γ_je_z (X·Y = (α_Xβ_Y + β_Xα_Y)/2 + γ_Xγ_Y), Λ := D_2λ ≠ 0.
E³: 2W_2·W_1 = −ΛD_1β_1 = 0 ⇒ β_1 = 0. E²: W_1·W_1 + 2W_2·W_0 = −(D_1γ_1)² − ΛD_0β_0 = 0 and (C1)
Λvβ_0′ + 2(D_1γ_1)vγ_1′ + vλ′D_0β_0 = 0, i.e. with g := D_1γ_1, w := vβ_0′: g² = −2(ι−N)wΛ, gγ_1′ = −2wD_1λ.
If g = 0 then w = 0. If g ≠ 0, dividing gives g D_1λ = (ι−N)vγ_1′D_2λ, which reduces to γ_1λ + (ι−N)v(γ_1λ′ − γ_1′λ) = 0,
i.e. r := γ_1/λ satisfies r = (ι−N)vr′ ⇒ r = r₀v^{1/(ι−N)} ⇒ (NR) r = 0, contradiction. So γ_1 = 0 and β_0 = const.
E^{±1}: 2W_2·W_{−1} = −ΛD_{−1}(conj α_1) = 0 ⇒ α_1 = 0. Hence X_1 = 0, X_0 = c + γ₀(v)e_z: **x+iy = c + λ(v)E²**, a doubly
covered surface of revolution; the remaining E⁰ condition is the axisymmetric C1 (which at v-degree 1 has no ι ≠ 0
solution, round 1). Verified at D = 1 in `task3_finite_fourier.py` (Part C).

### 3.4 General J (NR): structure and conjecture
With X_J = λε the level E^{2J−k} equations contain the new unknown only through β_{J−k} = 2ε·X_{J−k}, linearly with
coefficient Λ = D_Jλ (|W|²: −ΛD_{J−k}β_{J−k}; C1: (i/2)[Λvβ_{J−k}′ + vλ′D_{J−k}β_{J−k}]), plus lower data. Two equations
per level for one new function: 2(2J−1) equations for the 3(J−1)+1+2 unknown Laurent polynomials — overdetermined for J ≥ 3,
and the divisibility Λ | (…)² in the Laurent ring is the rigidity mechanism (units are monomials). Holomorphic subclass
(all X_{j>0} ∥ ε): E^k (k ≥ 1) gives Σ_{j−j″=k} D_jα_j conj(D_{j″}α_{j″}) − 2(ι−N)D_kα_k v(ε·X_0)′ = 0; k = J forces
ε·X_0 = const, then k = J−1, …, 1 force α_1 = … = α_{J−1} = 0: only x+iy = c + λ(v)E^J survives. **Conjecture:** the same
for the full class, all J, D, under NR. Descending induction as in §3.3 is the route; not completed.

### 3.5 J = 3, v-degree 1 (contains round-1's open class) — computation
`task3_J3.py`: 52 complex unknowns (complexified: conjugate modes independent, so every real solution is a solution),
105 coefficient equations; sequential solve E⁵ → E¹. ι = 2/11, N = 1 (symbolic λ): E⁵ ⇒ β_2 = 0 = conj-partner;
E⁴ ⇒ γ_2 = 0, β_1 = 0; E³ ⇒ ε-components of X_0 constant; E² ⇒ X_1 = 0; E¹ ⇒ X_2 = 0. Unique terminal solution:
x+iy = c + λ(v)E³, z = γ₀(v) — a triply covered surface of revolution (12 s). Cross-checks with the same outcome:
random numeric λ, λ̆ (no free parameters left for `solve`; `task3_J3_num1.log`) and ι = −3/13 (`task3_J3_iota2.log`).
**Round-1 task 4 (26-unknown non-stellarator-symmetric class) is thereby closed for non-resonant ι**: no
non-axisymmetric torus satisfies C1, C2. Caveat on completeness: sympy's `solve` on the parametric system could in
principle drop branches; the numeric-λ run (a parameter-free polynomial system) agrees, and every level is linear in the
new unknown with coefficient D_Jλ, so the only possible loss is at the divisibility steps, which the numeric run covers.

### 3.6 Resonant case ι − N ∈ Z (here ι = 0, N = 1) — the kernel of D_j
For ι = 0 the field-line derivative is D = ∂_φ|_θ and D_j(v^j) = 0: any vector function Y(θ) added to x leaves W = x_φ
unchanged. `task3_J3.py 0` (`task3_J3_iota0.log`) finds exactly this: the hierarchy leaves x+iy = c + E³λ(v),
z = γ₀(v) + 2Re(g e^{iθ}) (e^{iθ} = vE). `task3_resonant_branch.py`: the E^{±1} C1 equations give g·γ₀′ = 0, so γ₀ = const
(W_z = 0: **planar field lines**); the E⁰ condition then gives κ = I/G = 1/8 and two real conditions on λ
(2λ₀λ̄₋₁ = λ₁λ̄₀/2, −(9/8)|λ₀|² + (3/2)|λ₁|² − 6|λ₋₁|² = 0), which are solvable (λ₀ = 0, |λ₁| = 2|λ₋₁|). So **a
non-axisymmetric trigonometric surface satisfying C1 and C2 exists in the resonant case**, but it is not an embedded
torus: each field line is a planar closed curve of turning number 3 (self-intersecting), and field lines at labels θ, θ′
with Z(θ) = Z(θ′) lie in the same horizontal plane as rotated copies of one curve (they cross; numerically the lines at
θ = ±1 approach to 4·10⁻³ on a 400-point grid). The J = 1 analogue (limaçon field lines of turning number 1 at heights
Z(θ)) fails embeddedness for the same reason (periodic Z ⇒ two coplanar rotated copies). General statement: in the
resonant case the finite classes only add "stacked planar field line" immersions. Whether non-planar closed-line
(ι = 0, N ≠ 0) trigonometric surfaces exist at higher v-degree is open; the (v,E) hierarchy handles it mechanically.

## 4. Reproduce
```
python3 task1_validate.py          # ~2 min   (exact Boozer angles + all residuals; log task1_validate.log)
python3 task2_compat_structure.py  # ~1.5 min (structural identities [A]-[F], constrained random points, exact congruences)
python3 task3_finite_fourier.py    # ~2 min   (v,E reformulation, top-mode lemma, J = 2 class)
python3 task3_J3.py [iota] [seed]  # ~15 s    (J = 3, D = 1 hierarchy; e.g. `2/11`, `2/11 1`, `-3/13`, `0`)
python3 task3_resonant_branch.py   # 1 s      (the iota = 0, N = 1 branch: E^0 condition, planar field lines, immersion)
```
Modules: `flow_core.py` (round-1 jet machinery, nsimplify removed), `boozer_axisym.py` + `series_jets.py` (exact
Boozer jets of congruences, reusable), `vE_surface.py` (trigonometric surfaces in (v,E)).
