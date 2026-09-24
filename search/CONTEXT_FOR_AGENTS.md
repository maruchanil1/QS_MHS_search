# Shared context for the QS–MHS exact-solution search agents

Read this file first, then `../handoff_sept23_2026.md` (§1–§2, §4–§5, §8–§9 at minimum).
Everything below is the orchestrator's distilled state; the handoff is authoritative where they differ
on previously established facts.

## 0. Mission

Find an **exact, analytic (closed-form) magnetic field in a toroidal region of R³** that is

1. **MHS**: `J×B = ∇p`, `J = ∇×B`, `∇·B = 0`, smooth, `B ≠ 0`;
2. has **nested toroidal flux surfaces** `ψ = const` (an embedded solid torus, a magnetic axis);
3. is **quasisymmetric (QS)** and **not** invariant under a continuous Euclidean isometry
   (i.e. not axisymmetric, not helically symmetric).

Constant rotational transform ι is acceptable, **even ι = 0 or rational** (closed field lines), as long
as the field is genuinely QS and non-symmetric. Isotropic pressure `p = p(ψ)` is required
(anisotropic-pressure QS equilibria already exist in the literature and are *not* the target;
they may be used as stepping stones only if a path back to isotropic p is visible).

## 1. Binding rules (from the user; do not violate)

- **No near-axis-expansion (NAE) derivations, no pyQSC, no Rodriguez et al. order-by-order theory.**
  NAE may be used only as a *check* of a global construction, never as the construction.
- **Global constructions only.** Preferred line (user's choice): a nonlinear mechanical potential
  `V = −Π` whose orbits are the field lines, with a non-planar constant-speed closed orbit as axis
  (see §2). Other global routes are welcome if they are creative and rigorous.
- **Symbolic first.** Use sympy (installed) for algebra and exact verification. Small numerical
  sanity checks (seconds, not minutes) are fine; **no parameter scans, optimizations, VMEC/DESC
  runs, or heavy numerics** unless there is a compelling reason, and then say why.
- **Honesty.** No exact non-axisymmetric QS+MHS solution exists yet. Do not imply one does.
  Label every claim as `proved`, `verified_symbolically`, `verified_numerically`, `conjecture`,
  or `refuted`. A "solution" that is secretly axisymmetric/helical in disguised coordinates,
  or that fails `∇·B = 0`, or whose surfaces are not embedded tori, is not a solution — check.
- Concise technical writing. The user is a stellarator-theory expert.

## 2. Formulations you can use (all equivalent for MHS with nested surfaces)

### 2.1 Mechanical (Hamiltonian) formulation — the user's preferred language
Field lines are trajectories of `ẍ = −∇V`, `V := −Π`, `Π := p + |B|²/2`, parametrized so that `ẋ = B`
(indeed `B·∇B = ∇Π` ⇔ `J×B = ∇p` with `p = Π − |B|²/2`). Energy `E = |B|²/2 + V = −p` is constant
along orbits; MHS with `p = p(ψ)` ⇔ every orbit on a flux surface has the same energy `E(ψ)`.

**Key fact:** given *any* potential V, *any* congruence of its orbits (a 2-parameter family of orbits per
flux torus, all with energy E(ψ)) automatically satisfies force balance. The **only** extra condition
is `∇·B = 0`, which in torus coordinates `(ψ, s, t)` (s = orbit label, t = time along orbit,
`B = ∂x/∂t`) reads

    ∂_t J = 0,   J := det(∂_ψ x, ∂_s x, ∂_t x)            (Jacobian condition)

(J may still depend on (ψ, s); relabelling s then makes J a flux function.)

**QS in this language (Helander's triple-product criterion, local):**
`∇ψ × ∇|B| · ∇(B·∇|B|) = 0` ⇔ `B·∇Π = D(ψ, Π)` ⇔ along every orbit `V̇² = F(ψ, V)`:
the total pressure obeys an *autonomous one-dimensional clock* on each flux surface.

### 2.2 Boozer-coordinate (embedding) formulation — the global overdetermination made explicit
Unknown: embedding `x(ψ, θ, φ)` of the solid torus, `e_ψ, e_θ, e_φ` the coordinate tangents,
`√g = e_ψ·(e_θ×e_φ)`, `W := e_φ + ι e_θ` (so `B = W/√g`). Free flux functions `ι, G, I, p`; QS profile
`B(ψ, χ)`, `χ = θ − Nφ`. Then MHS + QS ⇔

    (1) W·e_φ = G √g
    (2) W·e_θ = I √g
    (3) √g = (G + ιI) / B(ψ, χ)²                       [QS]
    (4) (∂_φ + ι∂_θ)[ W·e_ψ / √g ] = G' + ιI' + p' √g   [force balance; K := W·e_ψ/√g = B_ψ]

Three unknown functions of three variables, four equations: **the Garren–Boozer overdetermination is
global (+1)**. Any construction must supply structure that makes (1)–(4) compatible. Drop (3) and the
system is determined (generic 3-D MHS with nested surfaces). Vacuum: `I = 0`, `G = const`, `p = 0`,
and `φ` is the scalar potential (`B = G∇φ`).

### 2.3 Weak vs strong QS
Weak QS: ∃ `u ≠ 0` with `∇·u = 0`, `[u, B] = 0`, `u·∇|B| = 0`. Strong QS adds `L_u(B♭) = 0`
(`B♭ = g(B,·)`). For MHS with nested surfaces and isotropic p, weak ⇒ strong
(Rodriguez–Helander–Bhattacharjee 2020). **Normalisation matters for the strong condition**: `L_u B♭ = 0`
is not invariant under `u → c(ψ)u`. The correctly normalised symmetry vector is the Boozer coordinate
vector `u = e_φ + N e_θ = [(G + NI)B + (N − ι)B×∇ψ]/B²` (a 2π-periodic circle action); the often-quoted
`(G̃B + B×∇ψ)/B²` differs by the flux function `1/(N−ι)` and gives a spurious `ι′ = 0` if inserted into
`L_u B♭ = 0` (coordinator, round 1). Weak-QS statements are unaffected.
Sato (Sci. Rep. 2022; and Sato–Yamada) constructed **weakly QS fields with ι = 0 in asymmetric toroidal
domains, but only with anisotropic pressure**, not isotropic MHS. Rodriguez–Bhattacharjee: anisotropic
pressure removes the GB overdetermination. So isotropic p is exactly where the difficulty lives.

### 2.4 Closed-field-line (rational ι) configurations — a legitimate target
If every field line on a surface closes, QS in the weak sense ⇔ **all closed lines on a surface have
the same |B|-profile along the line, up to a shift of the time parameter t** (then `u = ∂_s` at fixed
shifted time is a weak QS symmetry after normalising J to a flux function; the Helander criterion holds
component-wise on the level sets of |B|). Theorem 5.1 of the handoff (ι ≡ N ⇒ axisymmetric) means the
helicity N of the |B| pattern must differ from ι: with ι = 0 the profile must vary along the line
(N ≠ 0); with ι = 1 (tilted-loop type) one needs N ≠ 1, e.g. N = 0 (quasi-axisymmetric pattern).
Existence of Boozer coordinates on a rational surface needs `∮ dl/B` equal on all lines of the surface;
identical profiles give this for free.

## 3. Seed analysis by the orchestrator (use, extend, or refute)

### 3.1 Parity obstruction (verified symbolically, `search/seed/parity_check.py`)
If every orbit is `x(t) = A cos t + B sin t` (isotropic oscillator `V = |x|²/2`, all A, B depending on
(ψ, s)), then `J(t)` is a cubic trigonometric polynomial with **only odd harmonics**, hence zero mean,
hence vanishes somewhere ⇒ no non-degenerate nested-torus congruence. General principle: **if the orbit
family has a symmetry `x(t+T/2) = −x(t)` the Jacobian condition fails.** Orbits need a non-zero "DC"
part (Kepler ellipses, offset Lissajous, etc.).

### 3.2 Central potentials ⇒ Kepler is the only candidate
If `V = V(ρ)`, `ρ = |x|`, orbits are planar; `|B|² = 2E − 2V(ρ)` and `ρ̇² = 2E − 2V − L²/ρ²`, so the
QS clock condition holds iff `|L|` (angular-momentum magnitude) is constant on each torus. Orbits must
close (else a field line fills a planar annulus, impossible on an embedded torus) ⇒ Bertrand ⇒ Kepler or
isotropic oscillator; the oscillator dies by §3.1. **Kepler candidate:** tori swept by *congruent*
Kepler ellipses (same a = a(ψ), eccentricity ε = ε(ψ), focus at the origin) in a one-parameter family of
orientations `g(s;ψ) ∈ SO(3)`; QS is automatic; the only condition is the Jacobian condition. In the
eccentric anomaly η (`t = η − ε sin η`), `x_body(η) = a(cos η − ε, √(1−ε²) sin η, 0)`, and one needs
`det(Ω_ψ×e + e_ψ, Ω_s×e, e′) ∝ (1 − ε cos η)` with `Ω_s, Ω_ψ` the body-frame angular velocities of
`g(s;ψ)` (∂_s g = g Ω_s^×, ∂_ψ g = g Ω_ψ^×, with the zero-curvature compatibility condition). This is a
finite trigonometric-polynomial identity in η ⇒ **a closed algebraic system; decide it symbolically.**
If g(s) is a one-parameter *subgroup* the field is axisymmetric (fine as a consistency check: an
axisymmetric MHS with `Π = 1/ρ` exactly).

### 3.3 Rational anisotropic oscillators with exactly two distinct frequencies
For `V = Σ ω_i² x_i²/2` with rational ω the orbits are Lissajous curves and `|B|² = 2E − 2V(t)` is a
trigonometric polynomial. The "same profile up to shift" condition (§2.4) fixes the shift-invariants of
the profile as flux functions. Two distinct frequencies (e.g. 1:1:2 — Landreman's Family 1 potential —
1:1:3, 1:2:2, 2:2:3) give a 4-dim profile space ⇒ 3 invariants ⇒ **the QS-admissible orbits on each
energy level form a 1-dim family: exactly a torus's worth.** Then test the Jacobian condition (a finite
Fourier identity in t). Three or more distinct frequencies over-constrain (0-dim family). Add the
superintegrable "barrier" terms (`a/x²`, `b/y²`) of Evans' classification to enlarge the class; and
consider offset/DC parts to evade §3.1.

### 3.4 Circle-action / conjugated-rotation formulation
QS ⇔ B is invariant under a circle action generated by u; write `u = Ψ_*∂_φ` for a volume-preserving
diffeomorphism Ψ (not an isometry). Then `B = Ψ_* B̃` with B̃ "axisymmetric" in the coordinate φ, and
MHS + |B| invariance become equations on B̃ with the φ-dependent pulled-back metric `Ψ*δ`. Polynomial or
trigonometric Ψ (twists, shears, `x ↦ x + ε∇×A`) turn the overdetermination into finitely many
polynomial identities — search that finite space.

### 3.5 Rod-axis / toroidalized Schief equilibria (the user's chosen direction (ii))
The QA axis is (empirically) an exact Kirchhoff elastic rod; §7 of the handoff built a helically
symmetric V₀ having the rod as exact orbit (Schief's α ≠ 0 isodynamic travelling wave), and §6.4 shows
the true QS field breaks the screw symmetry at second order by an O(1) change of the torsion couplings
that makes the axis elliptic with the right ι₀. Task: find a *global, closed-form* V (or embedding) that
(i) has an exact rod as a constant-speed closed orbit, (ii) is expressive enough to impose the QS clock
`V̇² = F(ψ,V)` on the neighbouring tori. Ideas: V built from rod-adapted coordinates (screw coordinate
`ζ = e^{−iz/α}(x+iy)` plus one transverse function), V = V₀ + (distance-to-helicoid)² × correction,
Stäckel-type V whose separated coordinate is the "clock".

### 3.6 Other angles worth a serious attempt
- Axisymmetric V (Π axisymmetric) with a *non-axisymmetric* congruence whose resonance involves the
  toroidal angle (handoff §5.6): QS then requires Π̇² = F(ψ,Π) on the resonant sub-torus.
- Landreman-type separable congruences (Family 2 is Stäckel-separable in elliptic-cylindrical
  coordinates) with the QS clock imposed; classify separable V for which a separated coordinate is
  clocked (`q̇² = f(q; integrals)` with unit metric factor: Cartesian x, cylindrical r or z, spherical ρ).
- Vacuum QS: harmonic Φ whose gradient lines lie on nested tori with `|∇Φ|` constant along the
  helical symmetry lines (handoff §5.7).
- Bogoyavlenskij-type symmetry transforms of MHD equilibria (need field-aligned flow; check if a static
  limit or a "flow ↔ pressure" trick exists).
- Closed-line equilibria with Hamada structure: is there an exact non-symmetric closed-line MHS field at
  all (even non-QS)? Tilted-loop tori from superintegrable potentials are the natural place.

## 4. Literature facts (from web search; arXiv itself is not reachable from this sandbox)
- Landreman (2026, arXiv:2609.26742): analytic toroidal 3D MHD equilibria — the two families in the
  handoff §5.5 (neither QS). Open problem stated there: exact QS non-axisymmetric equilibria unknown.
- Sato (Sci. Rep. 12, 11322, 2022) and Sato–Yamada (2022): weakly QS fields, ι = 0, asymmetric
  toroidal domains, nested surfaces, **anisotropic** MHD only.
- Rodriguez–Bhattacharjee: anisotropic pressure removes the GB overdetermination.
- "Periodic KdV soliton potentials generate quasisymmetric |B| in a finite-β equilibrium"
  (arXiv:2507.06480, 2025): finite-gap/KdV structure appears in QS |B| profiles — resonates with the
  elastic-rod (genus-1 finite-gap) empirical link. Details not accessible here; treat as a hint.
- "High-β large-aspect-ratio QA Palumbo-like configurations" (arXiv:2506.17528): Palumbo/isodynamic
  QA at large aspect ratio (asymptotic, not exact).

## 5. Working conventions

- Write all scripts and notes under `search/roundN/<your-direction-slug>/`
  (`notes.md` + `*.py`). Never write outside your own directory except your final report object.
  Read other agents' directories freely.
- Every symbolic claim must be backed by a runnable sympy script that prints the check
  (e.g. `simplify(divB) == 0`, `simplify(JxB - grad p) == 0`). Include the exact command in notes.
- When you obtain a candidate field, provide: explicit B(x), p(x), ψ(x) (or the parametrization
  x(ψ,s,t) with B = ∂_t x), the domain, the QS symmetry vector u, and a self-check script.
- Time-box: stop exploring a sub-idea once it is refuted or reduced to a clean open statement; report
  the statement precisely so the other agents can pick it up.
- Suggest new directions for the *other* agents explicitly; that is how collaboration works here.
