# Section III and III.A: argument and equation plan

This plan preserves the author's revised Sections I and II and does not replace manuscript prose. Use standard terms: state, input, trajectory, equilibrium, energy balance, observation interval, input admissibility, state constraints, and reachability. Retain act-coast as the author's established name. No Chinese companion is required.

Use one short introductory paragraph for Section III and four paragraphs for III.A. Combine the requested dynamics and Euler-Lagrange topics in the first paragraph, because deriving the equations from mechanical energy gives the reader one continuous argument. The remaining paragraphs cover the energy representation, its Jacobian, and feasibility. Keep total prose near the existing length; allow only the additional space needed for the explicit Jacobian and physical energy constraint.

## Section III introductory paragraph: the purpose of the construction

- Explain the division of the control problem: choose a desired change in total mechanical energy, then use the observed physical state to determine an admissible torque that approximately realizes that change while promoting the desired energy distribution.
- Identify III.A as the derivation of the mechanical and energy relations needed for this construction. Identify III.B as the work-request and torque-selection law.
- Explain why the physical state is retained: the energy representation is many-to-one and does not determine subsequent motion by itself.
- Use two or three short sentences and no equations. Describe an analytically motivated control law, without claiming that the complete numerical torque selection is a closed-form solution.

## III.A paragraph 1: mechanical model and its governing equations

- State the assumptions before applying the equations: rigid uniform slender rods, ideal joints, zero damping, and negligible pendulum inertia about its longitudinal axis. This is the nominal lossless model used by the present controller.
- Define positive physical parameters m_a and r as arm mass and length, m_p and l as pendulum mass and length, and g as gravitational acceleration. Define c=l/2 as the pendulum center-of-mass distance from the hinge.
- Define the derived quantities in prose: I_a=m_a r^2/3 is the arm inertia about the motor axis; I_c=m_p l^2/12 is the pendulum transverse inertia about its center of mass; J=I_c+m_p c^2 is its inertia about the hinge axis. Thus J follows from the parallel-axis theorem and is not a tuning parameter. Define b=m_p r c as the inertial coupling coefficient and G=m_p g c as the gravitational energy scale.
- Introduce generalized coordinates q=(theta,alpha)^T using the coordinates already defined in Section II. Define the following mechanical quantities together:

$$
\begin{aligned}
A(\alpha)&=I_a+m_pr^2+J\sin^2\alpha, & C(\alpha)&=b\cos\alpha,\\
K&=\tfrac12A\omega^2+C\omega\nu+\tfrac12J\nu^2, & V&=G(1-\cos\alpha),\\
\mathcal L&=K-V.
\end{aligned}
$$

- A and C are the configuration-dependent inertia coefficients; K and V are total kinetic and gravitational potential energy; L is the Lagrangian. Immediately identify C omega nu as the coupling term and explain that V is referenced to the downward configuration.
- Apply the forced Euler-Lagrange equation and show the resulting equations of motion in the second equation group:

$$
\begin{aligned}
\frac{\mathrm d}{\mathrm dt}\frac{\partial\mathcal L}{\partial\dot q}
-\frac{\partial\mathcal L}{\partial q}&=\begin{bmatrix}u\\0\end{bmatrix},\\
M(\alpha)&=\begin{bmatrix}A&C\\C&J\end{bmatrix},\\
M(\alpha)\begin{bmatrix}\dot\omega\\\dot\nu\end{bmatrix}
&=\begin{bmatrix}
u-2J\sin\alpha\cos\alpha\,\omega\nu+b\sin\alpha\,\nu^2\\
J\sin\alpha\cos\alpha\,\omega^2-G\sin\alpha
\end{bmatrix}.
\end{aligned}
$$

- The generalized force has a zero second component because the pendulum is unactuated. M is the positive-definite inertia matrix. The remaining terms describe velocity-dependent inertial effects and gravity. These relations, together with the two angular velocities, define the physical vector field f in dx/dt=f(x,u).
- Do not separately introduce a residual vector or expand the inverse inertia matrix. Positive definiteness follows from the stated positive masses and geometry; keep its algebraic verification in the supporting derivation.

## III.A paragraph 2: energy representation and power balance

- Motivate the representation by separating the energy available for motion from the energy associated with pendulum elevation.
- Define one map H from physical state to three physical energy components, with the target in the same equation group:

$$
\begin{aligned}
K_a&=\tfrac12I_a\omega^2, & K_p&=K-K_a,\\
e=H(x)&=(K_a,K_p,V)^{\mathsf T}, & E&=K_a+K_p+V,\\
e_\star&=(0,0,E_\star)^{\mathsf T}, & E_\star&=2G.
\end{aligned}
$$

- K_a is arm kinetic energy; K_p is the full kinetic energy of the pendulum body, including motion caused by the moving hinge and the coupling term. Using K-K_a avoids repeating the preceding expansion while preserving the exact partition. E is total mechanical energy. The upright-rest value E_star is computed from the potential difference between downward and upright; it is not a chosen gain.
- Derive the energy rates by differentiating these quantities along the mechanical equations:

$$
\begin{aligned}
\dot K_a&=I_a\omega\dot\omega,\\
\dot K_p&=u\omega-I_a\omega\dot\omega-G\sin\alpha\,\nu,\\
\dot V&=G\sin\alpha\,\nu,\qquad \dot E=u\omega.
\end{aligned}
$$

- Explain the terms immediately: motor power changes total energy; energy transferred to arm motion or gravitational potential is removed from the pendulum kinetic-energy balance. Internal exchanges cancel in the sum. No separate internal-power variable is needed.
- Explain why regulating only E is insufficient: total energy can equal E_star while kinetic energy remains nonzero. If both E and V approach E_star, their difference K_a+K_p approaches zero. This is an algebraic characterization, not a convergence proof for the controller.

## III.A paragraph 3: the Jacobian and the limits of the energy representation

- Interpret the requested Jacobian as DH(x), the derivative of the energy map with respect to the four physical-state coordinates. It is not the derivative of an assumed autonomous three-state energy model.
- Place the chain rule and the explicit Jacobian together:

$$
\dot e=DH(x)f(x,u),\qquad
DH(x)=\begin{bmatrix}
0&0&I_a\omega&0\\
0&\tfrac12A'\omega^2+C'\omega\nu&(A-I_a)\omega+C\nu&C\omega+J\nu\\
0&V'&0&0
\end{bmatrix}.
$$

- Columns correspond to theta, alpha, omega, and nu, in that order. A prime denotes differentiation with respect to alpha: A'=2J sin(alpha) cos(alpha), C'=-b sin(alpha), and V'=G sin(alpha). These are derivatives of already defined functions, not additional model parameters.
- Explain the chain rule: the mechanical model gives the rate of change of the physical state, and the Jacobian converts that rate into changes of the three energies. This is the differential form of the preceding power balance.
- Explain the zero first column through rotational symmetry: absolute arm angle does not change mechanical energy, although it remains needed to check the arm-position constraint.
- Explain the information loss precisely. Opposite pendulum-angle signs can give the same energies but different potential-energy rates. The map retains some information about angles and speed magnitudes, but does not uniquely identify physical phase. Do not claim that it contains no phase information.
- State that the map has no unique inverse and that the energy rates still depend on x. Vanishing first-order energy sensitivity at a stationary configuration does not imply absence of motion or energy change over a finite actuation interval.
- In the manuscript, use the transpose of the same matrix if needed to fit a single column. Do not introduce a new symbol merely to shorten one matrix entry.

## III.A paragraph 4: physically realizable energies and admissible motion

- Begin with the distinction between a physically realizable energy vector and an energy vector reachable from the current state under bounded torque.
- Give the exact conditions for the image of H under the nominal model with unrestricted velocities:

$$
K_a\ge0,\qquad 0\le V\le2G,\qquad
K_p\ge\frac{A-I_a-C^2/J}{I_a}\,K_a.
$$

- Evaluate A and C at any angle satisfying cos(alpha)=1-V/G. The coefficient is independent of the sign of that angle because it contains only sin squared and C squared. The inequality follows by completing the square in K_p. It is both necessary and sufficient for a real velocity pair under this model, not merely a nonnegativity condition.
- Explain that moving the arm also moves the pendulum body; therefore, the two kinetic-energy components cannot be assigned independently. These constraints concern physical realizability, not finite-time reachability or satisfaction of the arm-position constraint.
- Define F_Delta(x,u) as the state reached from x after a constant torque u acts for a positive duration Delta. Introduce signed motor work W_Delta through the exact lossless relation:

$$
\begin{aligned}
W_\Delta(x,u)&=u\bigl[\theta(F_\Delta(x,u))-\theta(x)\bigr],\\
E(F_\Delta(x,u))&=E(x)+W_\Delta(x,u).
\end{aligned}
$$

- Theta applied to a state extracts its unwrapped arm coordinate. The first relation integrates motor power over the actuation interval. Positive work adds mechanical energy and negative work removes it.
- Explain input admissibility using the existing bound |u|<=U. A physically realizable endpoint need not be reachable from the present state over Delta. A requested work value is attainable only if some admissible input satisfies the work relation; there need not be a unique such input.
- State that satisfaction of the arm-position constraint must be checked along the physical trajectory. It cannot be established from the energy vector or from the endpoint alone. The implemented penalty in III.B does not enforce a hard state constraint.
- End with the implication for III.B: the work request must be assessed through the physical dynamics and the available torques. Zero torque preserves total energy in this lossless model while permitting internal energy exchange; zero net work does not necessarily mean zero torque.

## Structure, objects, and notation budget

- One Section III introduction and four III.A paragraphs; at most seven displayed equation groups. The Euler-Lagrange identity may move inline if column layout requires it.
- Reuse physical quantities from Sections II and III.A. No additional independent physical variables or tunable parameters are needed.
- Derived constants: c, I_a, I_c, J, b, and G, each computed from the positive masses, lengths, and gravity. Retain their existing meanings.
- Mechanical objects: q, A, C, M, K, V, and L describe coordinates, inertia, and mechanical energy. The vector field f is defined by the displayed equations of motion, not by a new model.
- Energy objects: K_a, K_p, e, E, e_star, and E_star describe the physical partition and upright-rest energy. H is the already-used state-to-energy map, now explicitly defined. DH is its derivative, not a new state or control law.
- Finite-time objects: Delta is an actuation duration, F_Delta is the exact constant-input flow, and W_Delta is signed motor work. Numerical approximation and input selection belong to III.B.
- No new residual vector, internal-power variable, inverse energy map, autonomous energy-state model, reachability-set notation, or Jacobian acronym.
- No code changes: zero new classes, methods, functions, helpers, wrappers, data classes, runtime parameters, or configuration options. Preserve the project's minimal continuous logic and avoid fallback behavior or additional abstraction.

## Writing milestones and success conditions

| Milestone | Validation | Success condition |
| --- | --- | --- |
| Argument | Read the paragraph topics without equations. | Mechanical model leads to energy balance, then differential interpretation, then feasibility of the requested energy change. |
| Mechanics | Compare Euler-Lagrange equations with the current zero-damping implementation. | Inertial, gravitational, and input terms match; the hinge inertia and coupling term are fully explained. |
| Energy mathematics | Differentiate H symbolically, sum its rates, and complete the square in K_p. | The displayed Jacobian is correct, total power is u omega, and the physical-energy inequality follows exactly. |
| Claim calibration | Check the paragraph on information loss and the transition to III.B. | No inverse-map, exact reduced-state dynamics, closed-form torque solution, reachability, state-constraint, or stability guarantee is implied without support. |
| Prose and typesetting | Apply nature-polishing and academic-humanizer after the structural and control-theory passes. | Every symbol is defined; each equation is followed by its role; no unnecessary numerical settings, duplicated formulas, or task-specific substitutes for standard terminology remain. |

## Adjacent issues to retain for the later edit

- Section II currently describes rho_alpha as an arm-constraint violation count, although its formula measures the fraction of time satisfying the pendulum-angle tolerance. Correct that description separately; do not redefine the symbol here.
- Section II treats the arm-angle range as a task constraint, while III.B implements it through a finite penalty. Preserve the distinction between the specified constraint and its numerical treatment.
- The existing duplicate eq:encoder labels must become one energy-definition group or receive distinct labels within the edited scope.
- The existing main-text claim of a complete closed-form solution must be replaced by the actual analytical construction followed by numerical torque selection.
