# Round 1 — closed-orbit-congruences: Kepler tori and rational oscillators

Direction: closed-field-line QS-MHS from superintegrable potentials (CONTEXT §2.1, §2.4, §3.1–3.3).
All scripts: `python3 <name>.py` from this directory (sympy 1.14; each < 2 min).

## 0. Summary of outcomes

| question | answer | status | script |
|---|---|---|---|
| Kepler tori (congruent ellipses, orientations g(s;ψ)) with J ≠ 0 | only if a' = 0 (same energy on all tori ⇒ p' = 0) | proved (symbolic case analysis) | `kepler_jacobian.py`, `kepler_cases.py` |
| the a' = 0 branch | zero-curvature ⇒ Ω_s ∥ fixed body axis n̂(ψ) and the spatial axis N = g n̂ is ψ-independent ⇒ **axisymmetric** | proved | `kepler_cases.py` |
| existence of axisymmetric Kepler tori | yes: exact axisymmetric **force-free** field, all lines Kepler ellipses with the same a, sin i = ε/(C√(1−ε²)), apsides in the equatorial plane | verified symbolically (∂_η J = 0) + numerically (div B, J×B ~ 1e-9) | `kepler_axisym_family.py` |
| Jacobian DC rule for Lissajous congruences | mean J ≠ 0 ⇔ some ±w₁±w₂±w₃ = 0 (one frequency = sum of the other two) | verified symbolically for 9 frequency triples | `lissajous_dc_rule.py` |
| 1:1:2 oscillator, QS-admissible congruence | identical profiles ⇒ the orbits on a torus are z-rotations of one orbit ⇒ **axisymmetric**; no Jacobian test needed | proved | `osc112_qs.py` |
| Jacobian condition for the axisymmetric 1:1:2 family | S' = 0, C ∝ m, δ' = 0 ⇒ exactly a **Solov'ev** equilibrium (Ψ quadratic in ρ², z; p', FF' const) | verified symbolically (GS residual 0) | `osc112_qs.py` |
| rational oscillators with three distinct frequencies (1:2:3, 1:3:4, 2:3:5, …) | profile fibre is finite (reflections) ⇒ no torus of equal-profile orbits | verified symbolically | `osc_fiber_count.py` |
| Evans barriers a/x², b/y² | make the profile rational with poles fixing the barrier coordinate's orbit ⇒ 0-dim fibre | verified symbolically (Pinney solution, pole structure) | `osc112_qs.py` |
| Helander local criterion vs "same profile up to shift" for closed lines | equivalent for smooth profiles with isolated turning points (§4) | argued | — |
| loophole "each torus a surface of revolution about its own axis N(ψ)" | strong QS ⇒ B ⊥ N'×x ⇒ B = 0 unless N' = 0 | verified symbolically (conditional on weak⇒strong QS, RHB 2020) | `piecewise_killing.py` |

Net: **both tasks are refuted**. Every closed-line QS congruence in the Kepler potential or in any rational
oscillator (with or without Evans barriers) is invariant under a continuous isometry of the potential.
The mechanism is general (§5): when the |B|-profile invariants are complete invariants of the isometry
group K of V acting on the orbit space, QS ⇔ the torus is a K-orbit ⇒ B is K-symmetric.

## 1. Kepler tori (Task A)

Setup. x(ψ,s,η) = g(s;ψ) e(η), e = a(cos η − ε, β sin η, 0), β = √(1−ε²), t = √(a³/μ)(η − ε sin η);
∂_s g = g Ω_s^×, ∂_ψ g = g Ω_ψ^× with Ω_s = (p1,p2,p3), Ω_ψ = (q1,q2,q3) in the body frame
(x̂ = line of apsides, ŷ = minor axis, ẑ = orbit normal). Two invariances used throughout:
J = det(∂_ψ x, ∂_s x, ∂_t x) is unchanged by t → t + t₀(ψ,s) and by s → s̃(ψ,s) (both add multiples of
∂_t x or ∂_s x to a column). Hence the secular term η_ψ|_t drops out and

    J = D(η) / (√(a³/μ)(1 − ε cos η)),   D := det(Ω_ψ×e + e_ψ, Ω_s×e, e′),

and ∂_t J = 0 ⇔ D(η) = c₀(1 − ε cos η). Structural formula (checked): with P = (p1,p2), Q = (q1,q2),
w = (e₂, −e₁), m = e·e′ = a²ε sin η (1 − ε cos η), n = (e_ψ × e′)₃:

    D = (P·w)(q3 m − n) − p3 m (Q·w).

Harmonics (script output): c3: ε(ε²−1)X = ε′p2 with X = p1q3 − p3q1; s3: εY = ε′p1 with Y = p2q3 − p3q2;
c2 − (2/ε)c3 ⇒ a′p2 = 0; s2, s1 ⇒ p1 · d/dψ[a(1−ε²)] = 0; c0 and c1 give two expressions for c₀ whose
difference is ∝ a′p2(ε²−2). Since c₀ ≠ 0 needs p2 ≠ 0 (c₀ = −a²βp2(2aε′+3a′ε)/2 from c0), **a′ = 0 is forced**.
Groebner bases (with saturation z·c₀ = 1) confirm: generic a′ ≠ 0 ⇒ c₀ = 0; ε′ = 0 ⇒ c₀ = 0;
constant semi-latus rectum ⇒ c₀ = 0. The unique surviving branch (a′ = 0, ε′ ≠ 0):

    Ω_s = (0, p2, p3),  Ω_ψ = ( ε′p2/(εβ²p3), q2, p3q2/p2 ),  c₀ = −a³βp2ε′.

So Ω_s = p2(0,1,k), k := p3/p2, and Ω_ψ = λΩ_s + (ε′/(εβ²k)) x̂. Zero curvature
∂_ψΩ_s − ∂_sΩ_ψ = Ω_s×Ω_ψ (sign convention verified on an explicit g = R_z(F)R_x(H)):
component 1 ⇒ ∂_s k = 0; components 2,3 ⇒ k′ = −(ε′/(εβ²))(k²+1)/k ⇒ (k²+1)ε²/(1−ε²) = C² const.
Therefore g(s;ψ) = g(0;ψ) exp(θ(s,ψ) n̂(ψ)^×), n̂ = (0,1,k)/√(1+k²): each torus is a surface of revolution
about N(ψ) = g(0;ψ)n̂(ψ), and ∂_ψN = g(Ω_ψ×n̂ + n̂′) = 0 on solutions (script). ⇒ globally axisymmetric.

Physical reading: a′ = 0 means E = −μ/(2a) is the same on all tori ⇒ p = −E constant ⇒ **J×B = 0**.
Explicit family (a = μ = 1 in the script): x = R_z(s) R_x(i(ψ)) e(η), sin i = ε/(C√(1−ε²)); J = −a^{3/2}√μ εε′/C
(η-independent). Numerical check by inverting the parametrisation: div B ~ 5e-9, |J×B|/(|J||B|) ~ 1e-8,
energy residual 1e-16. This is an exact axisymmetric force-free field with Π = μ/ρ and Kepler-ellipse field
lines (ι = 1, |B| = √(2μ/ρ − μ/a) spherically symmetric), presumably new but not the target.

**Refutation statement (Task A):** a congruence of congruent Kepler ellipses (focus at the origin) sweeping
nested tori with ∂_t J = 0 and J ≠ 0 exists only with a′ = 0, and then it is axisymmetric. There is no
axisymmetric MHS with Π = μ/ρ, nested tori and p′ ≠ 0 whose field lines are Kepler ellipses.

## 2. Lissajous congruences: Jacobian DC rule (Task B(i))

x_i = Re(a_i e^{iw_i t}): each column of J has frequency w_i in row i, so J only contains the harmonics
±w₁±w₂±w₃ and mean J ≠ 0 ⇔ one frequency is the sum of the other two. Verified for (1,1,1), (1,1,2),
(1,1,3), (1,2,2), (2,2,3), (1,2,3), (1,3,4), (2,3,5), (1,2,4). With ≤ 2 distinct frequencies only (1,1,2)
survives; with 3 distinct: (1,2,3), (1,3,4), (2,3,5), … survive the DC rule but die by §3.

## 3. Rational oscillators: QS ⇒ symmetric (Task B(ii)-(iii))

|B|² = Σ w_i²|a_i|²/2 − Σ (w_i²/2) Re(a_i² e^{2iw_i t}). "Same profile up to shift" ⇔ equal moduli
|a_i²| and phases 2 arg a_i equal mod (2w_iτ).

*Three distinct frequencies:* gauge a₁ real > 0 on both orbits ⇒ τ ∈ (π/w₁)Z ⇒ a₂′² , a₃′² determined ⇒
finite fibre (8 or 16 orbits, all related by reflections x_i → −x_i). No torus. (`osc_fiber_count.py`)

*1:1:2 (Landreman Family 1 potential):* invariants: S = (|a|²+|b|²)/2, |a²+b²| = 4m, |c| = C, arg c² − arg(a²+b²).
Gauge a²+b² = 4m real ⇒ c = Ce^{iδ} is fixed on the torus. Write a = 2√m cos w, b = 2√m sin w
(w ∈ C, always possible when m ≠ 0): a² + b² = 4m automatically and |a|²+|b|² = 4m cosh(2 Im w) ⇒
Im w = σ(ψ) fixed, Re w = φ free, and (x,y)(φ) = R_φ (x,y)(0) (checked). ⇒ **the QS-admissible orbit
family on a torus is the SO(2)-orbit of one orbit ⇒ B axisymmetric about z.** (m = 0 gives a doubly
covered cylinder, degenerate.)

*Jacobian condition on this axisymmetric family* (flux functions m, σ, C, δ): harmonics 2 ⇒
d/dψ[m cosh 2σ] = 0 (S′ = 0: horizontal energy is the same on all tori); harmonics 4 ⇒ (C/m)′ = 0,
δ′ = 0; mean J = −sin δ (Cm)′ ≠ 0. With ρ² − S = 2m cos 2t, z = κm cos(2t+δ), L_z = √(S²−4m²), the
poloidal flux is Ψ = −κ sin δ · m², m² = (ρ²−S)²/4 + [cos δ(ρ²−S)/2 − z/κ]²/sin²δ, p′ = 2κ/sin δ,
FF′ = 2/(κ sin δ): **Grad–Shafranov residual 0** — the Solov'ev equilibrium. (ι = 2, elliptic cross-section
in the (ρ², z) plane; the magnetic axis is the circle ρ² = S.) Good end-to-end check of the mechanical
framework.

*Evans barriers:* with a/x² the x-motion is x² = α + β cos(2t+φ), α² − β² = a (Pinney; checked), so
ẋ² = β² sin²(2t+φ)/(α + β cos(2t+φ)) has poles at complex t that pin (α, β, φ mod shift). Adding barriers
therefore removes the SO(2) degeneracy and makes the fibre 0-dimensional. No help.

*Helander criterion vs same-profile:* the triple-product criterion says (|B|, B·∇|B|) lies on a
ψ-dependent curve Γ_ψ in the phase plane. A smooth periodic profile with isolated turning points traverses
one loop of Γ_ψ; continuity in s forbids jumping between loops or through self-intersections; equal
periods (∮dl/B equal, needed for Boozer coordinates) then give identical profiles up to shift. So relaxing
to the local criterion changes nothing for closed lines with non-degenerate profiles (constant |B| along
lines is isodynamic ⇒ axisymmetric by handoff Thm 5.1).

## 4. Side lemma: tori of revolution about ψ-dependent axes (`piecewise_killing.py`)

If every flux surface is a surface of revolution about its own axis N(ψ) and B is invariant under that
rotation, u = N(ψ)×x is a weak QS vector (with the J-normalisation). MHS + nested surfaces ⇒ strong QS
(RHB 2020), i.e. (L_u g)(B,·) = 0. Since ∇u = N^× + (N′×x)∇ψᵀ, L_u g = w∇ψᵀ + ∇ψwᵀ with w = N′×x, so
strong QS ⇔ B·(N′×x) = 0. For a rotation-invariant tangent field on a torus about z with N′ = α′x̂ the
Fourier components in φ force b_φ sin θ = 0 and b_θ(r + R cos θ) = 0 ⇒ B = 0. Hence N′ = 0: the
"piecewise Killing" loophole is closed.

## 5. General principle and reduced open problem (for the other agents)

Closed-line QS torus in a potential V ⇔ a 1-parameter family of closed orbits, same energy E(ψ), with
the same V(t)-profile up to a time shift (plus ∂_t J = 0). Let K be the isometry group of V.
The K-orbit of any orbit is such a family, and gives a K-symmetric field. **Kepler and all rational
oscillators (± Evans barriers) have profile fibres equal to K-orbits (or finite), so QS forces symmetry.**
For a doubly-resonant Liouville torus L with frequency vector ω ∈ Z³ and V|_L = Σ V_κ e^{iκ·θ}, the
profile of the orbit through θ₀ has harmonics A_n(θ₀) = Σ_{κ·ω = n} V_κ e^{iκ·θ₀}; a 1-dim fibre needs all
|A_n| and n′arg A_n − n arg A_{n′} constant along a curve of the 2-torus L/flow. For quadratic V (oscillators)
each slice κ·ω = n contains one mode unless two frequencies coincide, and coincident frequencies give
exactly the rotation. Non-quadratic V has infinitely many slices ⇒ generically 0-dim fibres.
If the fibre is generated by the flow of an integral G = K^{ij}v_iv_j + U (Killing tensor K), then
{G − cH, V} = 0 on the torus family ⇔ (2K − c)∇V·B = 0 along every field line; if the velocities on the
Liouville torus at a point span R³ this forces ∇V to be an eigenvector of K with constant eigenvalue,
which in R³ (Stäckel systems) only happens for spherical (V radial ⇒ Kepler, dead), Cartesian/cylindrical
(unbounded motion) coordinates. **Reduced open problem:** find any V with a 1-parameter family of closed
orbits at fixed energy having identical V(t)-profiles that is not an isometry orbit (a "profile-isospectral"
non-congruent family). Without it, closed-line QS from a potential is impossible.
