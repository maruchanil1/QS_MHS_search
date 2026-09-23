"""TASK A(ii)-(iii): complete case analysis of the Kepler Jacobian system and the zero-curvature
(compatibility) condition.  Run: python3 kepler_cases.py

Notation as in kepler_jacobian.py:  Om_s = (p1,p2,p3), Om_psi = (q1,q2,q3) body-frame angular
velocities of g(s;psi); a(psi), eps(psi) semi-axis and eccentricity; a_p, eps_p their psi-derivatives;
D(eta) = det(Om_psi x e + e_psi, Om_s x e, e') must equal c0 (1 - eps cos eta).
"""
import sympy as sp

eta = sp.symbols('eta', real=True)
a, ap, eps, epsp = sp.symbols('a a_p epsilon epsilon_p', real=True)
beta = sp.sqrt(1 - eps**2)
p1, p2, p3, q1, q2, q3, c0 = sp.symbols('p1 p2 p3 q1 q2 q3 c0', real=True)
Om_s = sp.Matrix([p1, p2, p3]); Om_psi = sp.Matrix([q1, q2, q3])
e = sp.Matrix([a*(sp.cos(eta) - eps), a*beta*sp.sin(eta), 0])
e_eta = e.diff(eta)
e_psi = e.diff(a)*ap + e.diff(eps)*epsp
D = sp.expand(sp.Matrix.hstack(Om_psi.cross(e) + e_psi, Om_s.cross(e), e_eta).det())
target = sp.expand(D - c0*(1 - eps*sp.cos(eta)))

def coeff(f, k, kind):
    if k == 0:
        return sp.integrate(f, (eta, 0, 2*sp.pi))/(2*sp.pi)
    basis = sp.cos(k*eta) if kind == 'c' else sp.sin(k*eta)
    return sp.integrate(f*basis, (eta, 0, 2*sp.pi))/sp.pi
harm = [(0,'c'),(1,'c'),(1,'s'),(2,'c'),(2,'s'),(3,'c'),(3,'s')]
E = {h: sp.simplify(coeff(target, *h)) for h in harm}
Elist = list(E.values())

X = p1*q3 - p3*q1; Y = p2*q3 - p3*q2
print("=== hand reductions (all should print 0) ===")
print(" c3  <->  eps(eps^2-1) X - eps_p p2 :", sp.simplify(E[(3,'c')]*(-4*beta)/(a**3*eps) - (eps*(eps**2-1)*X - epsp*p2)))
print(" s3  <->  eps Y - eps_p p1          :", sp.simplify(E[(3,'s')]*(4)/(a**3*eps) - (eps*Y - epsp*p1)))
r_c2 = sp.simplify(E[(2,'c')] - E[(3,'c')]*2/eps)          # uses c3 to remove X
print(" c2 - (2/eps) c3  ->", sp.factor(r_c2), "   [=> a_p p2 = 0]")
r_s2 = sp.simplify(E[(2,'s')] + E[(3,'s')]*2*(eps**2+1)/eps)  # remove Y
print(" s2 + ... s3      ->", sp.factor(r_s2), "   [=> p1 (2 a eps eps_p + a_p(eps^2-1)) = 0, i.e. p1 * d/dpsi[a(1-eps^2)] = 0]")
r_s1 = sp.simplify(E[(1,'s')] - 5*E[(3,'s')])
print(" s1 - 5 s3        ->", sp.factor(r_s1), "   [same condition p1 * d/dpsi[a(1-eps^2)] = 0]")
# c0, c1 after using c3
r_c0 = sp.simplify(E[(0,'c')] + E[(3,'c')]*2/eps**2 * (eps**2 - 1)/(eps**2-1) )
c0_from_c0 = sp.solve(sp.simplify(E[(0,'c')].subs(q1, sp.solve(E[(3,'c')], q1)[0])), c0)[0]
c0_from_c1 = sp.solve(sp.simplify(E[(1,'c')].subs(q1, sp.solve(E[(3,'c')], q1)[0])), c0)[0]
print(" c0 from harmonic 0 (after c3):", sp.factor(c0_from_c0))
print(" c0 from harmonic 1 (after c3):", sp.factor(c0_from_c1))
print(" difference:", sp.factor(c0_from_c0 - c0_from_c1), "   [=> a_p p2 = 0 again; combined with c0 != 0 => a_p = 0]")

print("\n=== Case a_p = 0 (all tori same semi-axis a, i.e. same energy: p' = 0) ===")
E0 = [sp.factor(v.subs(ap, 0)) for v in Elist]
sols0 = sp.solve(E0, [p1, q1, q3, c0], dict=True)
for S in sols0:
    print("  ", S)
print("  => Om_s = (0, p2, p3),  Om_psi = (eps_p p2/(eps(1-eps^2) p3), q2, p3 q2/p2),  c0 = -a^3 sqrt(1-eps^2) p2 eps_p")
# verify this family satisfies all harmonics identically
fam = {ap: 0, p1: 0, q1: epsp*p2/(eps*(1-eps**2)*p3), q3: p3*q2/p2, c0: -a**3*beta*p2*epsp}
print("  residuals of all 7 harmonic equations on the family:", [sp.simplify(v.subs(fam)) for v in Elist])
# saturation: impose c0 != 0 via z*c0 = 1 and compute a Groebner basis (no branch assumptions on p2, p3)
z = sp.symbols('z')
Gsat = sp.groebner(E0 + [z*c0 - 1], z, p1, q1, q3, c0, p2, p3, q2, order='lex')
print("  Groebner basis of {a_p=0 system, z*c0=1} (lex z>p1>q1>q3>c0>p2>p3>q2):")
for gg in Gsat.exprs:
    print("     ", sp.factor(gg))

print("\n=== Case eps_p = 0 (eccentricity constant across tori) ===")
E1 = [sp.factor(v.subs(epsp, 0)) for v in Elist]
G1 = sp.groebner(E1, p1, p2, p3, q1, q2, q3, c0, order='lex')
print("  Groebner basis (generic a_p, eps):", [sp.factor(g) for g in G1.exprs], "  => c0 = 0")

print("\n=== Case a_p != 0 generic: Groebner => c0 = 0 (already printed in kepler_jacobian.py) ===")
print("=== Case d/dpsi[a(1-eps^2)] = 0 (constant semi-latus rectum), a_p != 0 ===")
E2 = [sp.factor(v.subs(epsp, ap*(1-eps**2)/(2*a*eps))) for v in Elist]
G2 = sp.groebner(E2, p1, p2, p3, q1, q2, q3, c0, order='lex')
print("  Groebner basis:", [sp.factor(g) for g in G2.exprs], "  => c0 = 0")

# -------------------------------------------------------------------------------------------
print("\n=== Zero-curvature condition for the a_p = 0 family ===")
psi, s = sp.symbols('psi s', real=True)
k = sp.Function('k')(psi, s); P2 = sp.Function('p2')(psi, s); lam = sp.Function('lambda')(psi, s)
epsf = sp.Function('epsilon')(psi)
betaf = sp.sqrt(1 - epsf**2)
Oms = P2*sp.Matrix([0, 1, k])
q1f = sp.diff(epsf, psi)/(epsf*betaf**2*k)
Omp = lam*Oms + sp.Matrix([q1f, 0, 0])
# sign convention check with an explicit g: g = Rz(f(psi,s)) Rx(h(psi,s))
f, h = sp.symbols('f h', real=True)
def Rz(t): return sp.Matrix([[sp.cos(t), -sp.sin(t), 0], [sp.sin(t), sp.cos(t), 0], [0, 0, 1]])
def Rx(t): return sp.Matrix([[1, 0, 0], [0, sp.cos(t), -sp.sin(t)], [0, sp.sin(t), sp.cos(t)]])
F = sp.Function('F')(psi, s); H = sp.Function('H')(psi, s)
g = Rz(F)*Rx(H)
def vee(M): return sp.Matrix([M[2, 1], M[0, 2], M[1, 0]])
Os = vee(sp.simplify(g.T*g.diff(s))); Op = vee(sp.simplify(g.T*g.diff(psi)))
zc = sp.simplify(Os.diff(psi) - Op.diff(s) - Os.cross(Op))
print("  convention check d_psi Om_s - d_s Om_psi - Om_s x Om_psi = 0 for explicit g:", zc.T)
ZC = sp.simplify(Oms.diff(psi) - Omp.diff(s) - Oms.cross(Omp))
print("  component 1 (=> d_s of q1 = 0 => k = k(psi)):", sp.factor(ZC[0]))
kp = sp.Function('k')(psi)
ZCk = sp.simplify(ZC.subs(k, kp).doit())
print("  component 2:", sp.simplify(ZCk[1]))
print("  component 3:", sp.simplify(ZCk[2]))
comb = sp.simplify(ZCk[2] - kp*ZCk[1])
print("  comp3 - k*comp2 (an ODE for k(psi)):", sp.factor(comb))
ode = sp.Eq(sp.diff(kp, psi), sp.solve(comb, sp.diff(kp, psi))[0])
print("  ODE:", ode)
# solve: k' = -(eps'/(eps(1-eps^2))) (k^2+1)/k  =>  (k^2+1) eps^2/(1-eps^2) = const
inv = (kp**2 + 1)*epsf**2/(1 - epsf**2)
dinv = sp.simplify(sp.diff(inv, psi).subs(sp.diff(kp, psi), ode.rhs))
print("  d/dpsi[(k^2+1) eps^2/(1-eps^2)] on solutions:", dinv)
# spatial symmetry axis N = g n, n = (0,1,k)/sqrt(1+k^2); dN/dpsi = g (Om_psi x n + n')
n = sp.Matrix([0, 1, kp])/sp.sqrt(1 + kp**2)
Ompk = Omp.subs(k, kp)
dN_body = sp.simplify((Ompk.cross(n) + n.diff(psi)).subs(sp.diff(kp, psi), ode.rhs))
print("  body-frame d_psi N = g^{-1} dN/dpsi on solutions:", dN_body.T, "  => the rotation axis is the same for all tori")
