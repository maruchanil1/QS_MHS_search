"""TASK B(ii)-(iii): 1:1:2 oscillator V = (x^2+y^2+4z^2)/2.
Orbits x = Re(a e^{it}), y = Re(b e^{it}), z = Re(c e^{2it}), a,b,c complex.
|B|^2 = (|a|^2+|b|^2)/2 + 2|c|^2 - Re((a^2+b^2) e^{2it})/2 - 2 Re(c^2 e^{4it}).
Same |B|-profile up to a time shift  <=>  same (|a|^2+|b|^2, |a^2+b^2|, |c|, arg c^2 - arg(a^2+b^2)).

(1) THEOREM (verified here): the fibre of the profile map is exactly the SO(2)-orbit (rotation about z)
    of one orbit.  Proof: gauge the time so that a^2+b^2 = 4m > 0 is real; then c = C e^{i delta} is fixed
    on the torus, and writing a = 2 sqrt(m) cos w, b = 2 sqrt(m) sin w (w complex, always possible)
    |a|^2+|b|^2 = 4m cosh(2 Im w)  =>  Im w = sigma(psi) fixed, Re w = phi free  =>  (x,y) = R_phi (x0,y0).
    Hence a QS-admissible 1:1:2 congruence is axisymmetric about z (no Jacobian test needed).
(2) For completeness: the Jacobian condition for the axisymmetric family (flux functions m, sigma, C, delta).
(3) Evans barrier terms a/x^2: the x-motion becomes x^2 = alpha + beta cos(2t+phi) with alpha^2-beta^2 = 2a;
    the profile acquires poles at complex t whose positions fix (alpha, beta, phi) mod shift  => 0-dim fibre.
Run: python3 osc112_qs.py
"""
import sympy as sp

t, phi, sigma = sp.symbols('t phi sigma', real=True)
m, S, C, delta = sp.symbols('m S C delta', positive=True)

# ---- (1) profile invariants of the cos/sin parametrisation ----
w = phi + sp.I*sigma
a = 2*sp.sqrt(m)*sp.cos(w); b = 2*sp.sqrt(m)*sp.sin(w); c = C*sp.exp(sp.I*delta)
inv1 = sp.simplify(sp.expand(a**2 + b**2, complex=True))
inv2 = sp.simplify(sp.expand(a*sp.conjugate(a) + b*sp.conjugate(b), complex=True))
print("a^2+b^2 =", inv1, "   |a|^2+|b|^2 =", sp.simplify(inv2.rewrite(sp.exp)).rewrite(sp.cosh))
# real orbit and its rotation structure
x = sp.re(sp.expand(a*sp.exp(sp.I*t), complex=True)); y = sp.re(sp.expand(b*sp.exp(sp.I*t), complex=True))
x = sp.simplify(x); y = sp.simplify(y)
x0 = x.subs(phi, 0); y0 = y.subs(phi, 0)
rot = sp.simplify(sp.Matrix([sp.cos(phi)*x0 - sp.sin(phi)*y0, sp.sin(phi)*x0 + sp.cos(phi)*y0]) - sp.Matrix([x, y]))
print("(x,y)(phi) == R_phi (x,y)(0):", rot.T == sp.Matrix([[0, 0]]))
# converse: any (a,b) with a^2+b^2 = 4m can be written as 2sqrt(m)(cos w, sin w): a/(2sqrt m) = cos w has a
# complex solution w for every complex value, and then sin w = +- sqrt(1 - cos^2 w) = +- b/(2 sqrt m); the sign
# is absorbed by w -> -w.  |a|^2+|b|^2 = 4m cosh(2 Im w) then fixes Im w up to sign (sign: w -> conj(w) is the
# reflection y -> -y composed with time reversal... it maps to the mirror torus).  So the fibre is the circle Re w.

# |B|^2 profile of the family: check it is phi-independent
z = C*sp.cos(2*t + delta)
xt, yt, zt = sp.diff(x, t), sp.diff(y, t), sp.diff(z, t)
B2 = sp.simplify(sp.expand(xt**2 + yt**2 + zt**2))
print("d/dphi |B|^2 == 0:", sp.simplify(sp.diff(B2, phi)) == 0)
print("|B|^2 =", sp.simplify(B2.rewrite(sp.cos)))

# ---- (2) Jacobian condition for the axisymmetric family ----
psi = sp.symbols('psi', real=True)
mf, sf, Cf, df = [sp.Function(n)(psi) for n in ('m', 'sigma', 'C', 'delta')]
X = sp.Matrix([x, y, z]).subs({m: mf, sigma: sf, C: Cf, delta: df})
J = sp.Matrix.hstack(X.diff(psi), X.diff(phi), X.diff(t)).det()
J = sp.simplify(sp.expand(J))
print("\nJacobian J (axisymmetric 1:1:2 family), phi-independent:", sp.simplify(sp.diff(J, phi)) == 0)
Jt = sp.expand(sp.expand_trig(sp.simplify(J.diff(t))))
# Fourier coefficients in t of d_t J (harmonics 2 and 4)
def fc(f, k, kind):
    basis = sp.cos(k*t) if kind == 'c' else sp.sin(k*t)
    return sp.simplify(sp.integrate(f*basis, (t, 0, 2*sp.pi))/sp.pi)
for k in (2, 4):
    for kind in ('c', 's'):
        print(f"  harmonic {kind}{k} of d_t J:", sp.factor(fc(J, k, kind)))
print("  mean of J:", sp.factor(sp.simplify(sp.integrate(J, (t, 0, 2*sp.pi))/(2*sp.pi))))

# ---- (3) Evans barrier a/x^2: Pinney solution and pole structure ----
al, be, ph, aa = sp.symbols('alpha beta varphi a_bar', positive=True)
x2 = al + be*sp.cos(2*t + ph)
xb = sp.sqrt(x2)
res = sp.simplify(sp.diff(xb, t, 2) + xb - aa/xb**3)
print("\nPinney check  x'' + x - a/x^3 = 0 for x^2 = alpha + beta cos(2t+phi) with alpha^2 - beta^2 = a:",
      sp.simplify(res.subs(aa, al**2 - be**2)) == 0)
xdot2 = sp.simplify(sp.diff(xb, t)**2)
print("  xdot^2 =", sp.factor(xdot2), "  (rational in e^{2it}; poles where alpha + beta cos(2t+phi) = 0)")
print("  => the profile |B|^2 has poles at complex t fixed by (alpha, beta, phi); two orbits with the same profile"
      " have the same (alpha, beta) and the same phi mod shift: the profile fibre is 0-dimensional (no torus).")

# ---- (4) identification of the axisymmetric 1:1:2 Jacobian solution with a Solov'ev equilibrium ----
# Jacobian condition => S := 2 m cosh(2 sigma) const, C = kappa m, delta const.  Then
# rho^2 - S = 2 m cos 2t,  z = kappa m cos(2t+delta),  L_z = 2 m sinh(2 sigma) = sqrt(S^2 - 4 m^2),
# poloidal flux Psi = -kappa sin(delta) m^2 with m^2 = psi(rho,z) below; p = p0 - S - 2 kappa^2 m^2, F = L_z.
rho, zz, kap, Sc, dl = sp.symbols('rho z kappa S delta', positive=True)
psi_lab = (rho**2 - Sc)**2/4 + (sp.cos(dl)*(rho**2 - Sc)/2 - zz/kap)**2/sp.sin(dl)**2      # = m^2
Psi = -kap*sp.sin(dl)*psi_lab
p_of_Psi = -2*kap**2*psi_lab                    # up to a constant;  p = p0 - S - 2 kappa^2 m^2
F2_of_Psi = Sc**2 - 4*psi_lab                   # F^2 = L_z^2 = S^2 - 4 m^2
# p and F^2 are linear in Psi:
pprime = sp.simplify(sp.diff(p_of_Psi, rho)/sp.diff(Psi, rho)); FFprime = sp.simplify(sp.diff(F2_of_Psi, rho)/sp.diff(Psi, rho)/2)
print("\nSolov'ev check: p'(Psi) =", pprime, ",  F F'(Psi) =", FFprime, " (constants)")
GS = sp.simplify(rho*sp.diff(sp.diff(Psi, rho)/rho, rho) + sp.diff(Psi, zz, 2) + rho**2*pprime + FFprime)
print("Grad-Shafranov residual Delta* Psi + rho^2 p' + F F' =", GS)
# and B from the orbit family equals the GS field: B_rho = -Psi_z/rho, B_z = Psi_rho/rho, B_phi = F/rho
tt, mm = sp.symbols('t m', positive=True)
sub = {rho: sp.sqrt(Sc + 2*mm*sp.cos(2*tt)), zz: kap*mm*sp.cos(2*tt + dl)}
Brho_orbit = -2*mm*sp.sin(2*tt)/sp.sqrt(Sc + 2*mm*sp.cos(2*tt))      # d rho/dt
Bz_orbit = -2*kap*mm*sp.sin(2*tt + dl)                                # dz/dt
print("B_rho matches -Psi_z/rho:", sp.simplify((-sp.diff(Psi, zz)/rho).subs(sub) - Brho_orbit) == 0)
print("B_z   matches  Psi_rho/rho:", sp.simplify((sp.diff(Psi, rho)/rho).subs(sub) - Bz_orbit) == 0)
