# Mathematical checks for the Section III.A outline

## Scope and sources

- Physical equations: src/rotary_pendulum/environment/jax_dynamics.py, state_derivative.
- Model constants: src/rotary_pendulum/environment/dynamics.py, derive_model.
- Nominal zero damping: src/rotary_pendulum/configs/physics.yaml.
- Physical energy partition: src/rotary_pendulum/heuristic/energy.py, encode.
- Existing derivation cross-check: docs/12-energy-transfer/formulation/analytical_contract.md. Its historical controller rules are not used as current implementation evidence.
- Manuscript scope: the independent Section III paragraph and III.A in docs/15-controller-paper/latex-project/main.tex.

The notation and definitions are given in ../outline.md. The calculations below additionally use v=(omega,nu)^T for the two angular velocities. This supporting vector is not needed in the manuscript plan.

## Euler-Lagrange equations

With A and C differentiated with respect to alpha, the two forced Euler-Lagrange equations reduce to

$$
A\dot\omega+C\dot\nu+A'\omega\nu+C'\nu^2=u,
$$

$$
C\dot\omega+J\dot\nu-\tfrac12A'\omega^2+V'=0.
$$

Substituting A'=2J sin(alpha) cos(alpha), C'=-b sin(alpha), and V'=G sin(alpha) gives the two components in the implementation. The cancellation of the C' omega nu terms in the second equation is required.

## Positive-definite mechanical inertia

The arm-pendulum inertia matrix has determinant

$$
\det M=I_aJ+m_pr^2J-b^2+(J^2+b^2)\sin^2\alpha.
$$

Using J=I_c+m_p c^2 and b=m_p r c,

$$
m_pr^2J-b^2=m_pr^2I_c>0.
$$

Thus det(M)>0 and A>0 for positive masses, lengths, and transverse inertia. These conditions establish positive definiteness and a well-defined acceleration for every configuration of the nominal model.

## Symbolic chain-rule verification

The state-to-energy map H=(I_a omega^2/2, K-I_a omega^2/2, V)^T was differentiated with respect to alpha, omega, and nu; its theta column is zero. Multiplication by (nu, dot(omega), dot(nu))^T, with accelerations obtained from M inverse times the implemented force vector, produced

$$
\dot K_a+\dot K_p+\dot V-u\omega=0.
$$

SymPy simplification returned exactly zero. The differentiated matrix agrees entry by entry with DH in the outline. The calculation assumes zero damping, matching the current configuration.

## Physical energy image

Completing the square gives

$$
K_p=\frac J2\left(\nu+\frac CJ\omega\right)^2
+\frac12\left(A-I_a-\frac{C^2}{J}\right)\omega^2.
$$

The symbolic difference between this expression and K-K_a simplified to zero. Its second coefficient is positive because

$$
(A-I_a)J-C^2=m_pr^2I_c+(J^2+b^2)\sin^2\alpha>0.
$$

Necessity follows by discarding the nonnegative square and substituting omega^2=2K_a/I_a. For sufficiency, choose an angle whose cosine is 1-V/G, choose omega with magnitude sqrt(2K_a/I_a), and solve the square for a real nu. Such a real solution exists exactly when the outline's kinetic-energy inequality holds. Theta is unrestricted by H and may be chosen inside any nonempty prescribed arm-position interval.

The image conditions are exact for the stated model with unrestricted velocities. They do not establish reachability from a particular initial state, existence of a constraint-satisfying trajectory, or invariance of the state-constraint set.

## Failure of autonomous energy-state closure

Compare physical states with pendulum angles alpha and minus alpha, the same theta, omega, and nu, and sin(alpha) nu nonzero. All three energy components are equal because K contains only sin squared and cos of alpha. Their potential-energy rates are opposite:

$$
\dot V(\alpha,\nu)=G\sin\alpha\,\nu,
\qquad
\dot V(-\alpha,\nu)=-G\sin\alpha\,\nu.
$$

Consequently, even for the same input, the energy vector does not uniquely determine its derivative. A globally exact relation depending only on e and u cannot represent these dynamics. This is stronger than simply observing that the energy map omits theta.

## Differential sensitivity and finite-time motion

At zero angular velocities, the kinetic-energy rows of DH vanish. At either vertical rest configuration, the potential-energy row also vanishes. Thus the instantaneous energy derivative is zero there even if an applied torque produces nonzero acceleration. Integration over a finite positive interval can still change the energies; first-order sensitivity does not establish finite-time uncontrollability.

For constant torque u and exact flow F_Delta, integrating the lossless power balance gives W_Delta=u times the unwrapped arm displacement. Therefore zero torque implies constant total energy, but zero signed work can also occur with nonzero torque and zero net arm displacement. Input admissibility alone does not establish the existence of a torque realizing an arbitrary work request.

## Validation record

- Exact symbolic total-power residual: zero.
- Exact symbolic completed-square residual: zero.
- Pendulum kinetic determinant lower bound: I_c m_p r^2, strictly positive under the stated assumptions.
- Jacobian: checked by symbolic differentiation of the implemented physical-energy expressions.
- Controller execution and performance: not evaluated by these mathematical checks.
- Manuscript and code: unchanged during this planning task.
