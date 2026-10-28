# Paragraph plan: dynamics and demonstration teacher

This plan applies [the standing writing guidance](../writing_guidance.md) to the current draft's dynamics and heuristic-controller subsections. It refines [the subsection plan](revision_plan.md) and preserves `main.tex` for later revision. The argument is **coupled mechanics → physical prediction → energy distribution → physical phase → nominal torque → teacher trajectories for learning**.

Use four paragraphs for dynamics and two short paragraphs for the teacher. Aim for roughly 400–500 prose words in the eventual manuscript, with three to five sentences per paragraph and four compact display groups across both subsections. Sentence roles are specified below; these bullets are a drafting plan, rather than replacement manuscript prose. Configured values, unit lists, and experimental results are excluded.

## A. Rotary-pendulum dynamics and energy representation

### A1. Why the two motions must be modeled together

**Purpose:** Introduce the mechanical difficulty and the energy model from which the equations follow.

- **Opening sentence:** Identify the rotary pendulum as the chosen benchmark, then explain that arm torque raises the passive pendulum through their coupled motion.
- **Second sentence:** Define the arm angle $\theta$, downward-referenced pendulum angle $\alpha$, signed speeds $\omega=\dot\theta$ and $\nu=\dot\alpha$, coordinate vector $q=(\theta,\alpha)^{\mathsf T}$, and physical state $x=(\theta,\alpha,\omega,\nu)^{\mathsf T}$. These benchmark coordinates are defined here, not in the generic Problem Formulation. Retain the unwrapped arm angle, define the pendulum angle convention and positive torque direction, and specialize the generic limits and task error before using them later.
- **Third sentence:** State the ideal-joint, lossless rigid-body assumptions and introduce kinetic energy $K=\tfrac12\dot q^{\mathsf T}M(\alpha)\dot q$, potential energy $V=G(1-\cos\alpha)$, and Lagrangian $\mathcal L=K-V$. Define $M$ as the coupled inertia matrix and $G$ as the gravitational energy scale computed from the model's mass, gravity, and pendulum center-of-mass distance.
- **Fourth sentence:** Interpret the mixed kinetic-energy term: arm motion carries the pendulum hinge and therefore contributes to pendulum motion. This explains why the next paragraph derives both acceleration equations together.

**Equation placement:** Keep the energy identities inline; reserve the first display for the governing equation in A2. Give the exact entries of $M$ in the same display as that equation. The [evidence ledger](machine-scannables/evidence_ledger.md) retains the geometric derivation of its coefficients.

### A2. How the mechanical model predicts the physical motion

**Purpose:** Connect the energy model to the nonlinear dynamics used for prediction.

- **Opening sentence:** Explain that differentiating the Lagrangian converts the kinetic coupling and gravity into the accelerations produced by arm torque.
- **Second sentence:** Apply the forced Euler–Lagrange relation, stated inline as $\frac{\mathrm d}{\mathrm dt}\frac{\partial\mathcal L}{\partial\dot q}-\frac{\partial\mathcal L}{\partial q}=(u,0)^{\mathsf T}$. Here $t$ is continuous time, dots denote its derivatives, and $u$ is the motor torque; the zero second component expresses the passive joint.
- **Third sentence:** Present the resulting coupled equation and define every coefficient immediately.
- **Fourth sentence:** Interpret inertia coupling, velocity-dependent terms, and gravity, then explain that solving for the accelerations gives the physical-state vector field $\dot x=f(x,u)$. Define $f$ as that vector field; its integration supplies the later torque predictions.

The first display is the compact governing equation

$$
M(\alpha)\ddot q+
\begin{bmatrix}
A'(\alpha)\omega\nu+C'(\alpha)\nu^2\\
-\tfrac12 A'(\alpha)\omega^2+G\sin\alpha
\end{bmatrix}
=\begin{bmatrix}u\\0\end{bmatrix},\qquad
M(\alpha)=\begin{bmatrix}A(\alpha)&C(\alpha)\\C(\alpha)&J\end{bmatrix}.
$$

Immediately define $A(\alpha)=J_0+J\sin^2\alpha$, $C(\alpha)=b\cos\alpha$, and primes as differentiation with respect to $\alpha$. The fixed physical coefficients $J_0,J,b$ are the baseline arm-axis inertia, pendulum hinge inertia, and inertial coupling coefficient, respectively, derived from the rigid-body model. Explain that the off-diagonal entries transmit the effect of arm torque to pendulum acceleration. Keep the complete symbolic geometric definitions in the ledger; omit a second expanded four-state matrix and the intermediate derivative algebra from the main narrative.

### A3. Why raising total energy alone does not specify upright rest

**Purpose:** Introduce energy distribution as the physical motivation for the teacher.

- **Opening sentence:** Explain that reaching upright rest requires energy to become gravitational potential rather than remain kinetic motion.
- **Second sentence:** Define arm-body kinetic energy $K_a=\tfrac12I_a\omega^2$, where $I_a$ is the arm's own inertia about its motor axis, and pendulum-body kinetic energy $K_p=K-K_a$, which includes the moving hinge and coupling contribution.
- **Third sentence:** Introduce the energy representation and upright-rest energy using the second display.
- **Fourth sentence:** Explain immediately that raising $V$ toward the upright-rest energy while bringing $E$ to the same value removes the remaining kinetic-energy budget. Present this as the teacher's motivation.

The second display is

$$
z=H(x)=(K_a,K_p,V)^{\mathsf T},\qquad
E=K_a+K_p+V,\qquad E_\star=2G.
$$

Define $H$ as the physical-state-to-energy map, $z$ as its energy representation, $E$ as total mechanical energy, and $E_\star$ as the energy of upright rest. The identity $K_a+K_p=E-V$ explains the intended energy redistribution without a separate displayed target vector. Reserve $e$ for the later physical-state error, keeping the two representations distinct.

### A4. Why the energy controller also needs physical phase

**Purpose:** Explain the division between energy feedback and physical-state prediction.

- **Opening sentence:** Explain that total energy determines how much work is needed, while the signed physical motion determines how a torque transfers that work.
- **Second sentence:** Differentiate $E=K+V$ along the lossless dynamics; the internal exchange cancels, leaving motor power. Differentiate $V$ to show why pendulum motion direction matters.
- **Third sentence:** Interpret the third display immediately: motor torque changes total energy through arm speed, while pendulum direction determines whether potential energy is rising or falling.
- **Fourth sentence:** Use the example of $\alpha$ and $-\alpha$ with the same signed velocities: they give the same $z$ but opposite potential-energy rates whenever $\sin\alpha\,\nu\ne0$. Conclude that the teacher combines energy feedback with full-state prediction.

The third display is

$$
\dot E=u\omega,\qquad \dot V=G\sin\alpha\,\nu.
$$

All symbols are already defined. Explain physical phase as the pendulum orientation and signed velocities needed to predict the direction of energy exchange. Omit the full Jacobian of $H$ and its componentwise multiplication: the two power identities and the phase example supply the necessary reasoning.

## B. Mechanical teacher and demonstration trajectories

### B1. How the energy requirement becomes a nominal torque

**Purpose:** Explain the teacher's action construction at the level needed to understand its reference trajectories.

- **Opening sentence:** Explain that the teacher requests a bounded amount of work from the difference between current and upright-rest energy.
- **Second sentence:** Define the requested work and the work delivered by a predicted constant torque using the fourth display.
- **Third sentence:** Interpret the sign of the request and the dependence of delivered work on the predicted arm displacement; this connects the energy requirement to the phase argument in A4.
- **Fourth sentence:** Explain that the request fixes a total-energy budget, whereas bounded torque and the current state determine the predicted energy endpoints. This is the compressed remnant of the old energy-surface argument.
- **Fifth sentence:** State that the teacher selects among predicted bounded-torque actions using potential-energy and arm-excursion penalties, with work-based candidates included in that comparison.

The fourth display is

$$
w=\operatorname{clip}\bigl(\kappa(E_\star-E),-\bar w,\bar w\bigr),\qquad
\mathcal W(x,u)=\int_0^\Delta u\omega(s)\,\mathrm ds
=u\bigl[\theta(\Delta)-\theta(0)\bigr].
$$

Here $w$ is requested work, $\kappa>0$ is a chosen feedback gain, $\bar w>0$ is the chosen work-request bound, and $\operatorname{clip}$ restricts its first argument to the stated bounds. The function $\mathcal W(x,u)$ is delivered work predicted from initial state $x$ when torque $u$ is held for the fixed duration $\Delta>0$; $s$ is elapsed time and $\theta(s),\omega(s)$ are the corresponding predicted arm angle and speed. The integral becomes a displacement because the torque is constant. Present the candidate comparison as a penalized choice rather than an imposed equality $\mathcal W=w$; keep its work weight and all search settings in the source record.

### B2. Why the completed teacher supplies the learning reference

**Purpose:** Connect the mechanical controller to the subsequent teacher–student learning formulation.

- **Opening sentence:** Explain that the student needs reference trajectories that include both energy-increasing motion and continued upright control.
- **Second sentence:** State that the demonstration teacher combines the nominal energy action with local linear balancing feedback and predictive arm braking; name the role of each addition without introducing its threshold or another optimization problem.
- **Third sentence:** Explain that qualifying trajectories supply state–action examples and approximate return labels for initialization, and remain available through demonstration replay during learning.
- **Fourth sentence:** End by introducing the student's role: learn a stationary torque policy and continuation-value estimates from these references and its own transitions. Leave the loss functions and continuing-return definition to the learning subsection.

**Equation placement:** No new display. The learning subsection defines the student policy, value functions, and training objective. Teacher performance and trajectory counts belong in **Experiments and Results**.

## Scope, validation, and editing anchors

| Writing milestone | Validation | Success flag |
|---|---|---|
| Paragraph architecture | Read only the six opening ideas. | The argument progresses continuously from physical coupling to learning references; each paragraph has one distinct job. |
| Mathematical preparation | Check A2 against the current dynamics and B1 against the energy policy/decoder. | Signs, coordinate conventions, power identities, and teacher additions match the source; every symbol has a definition and immediate interpretation. |
| Later prose revision | Write each paragraph in three to five sentences and check the joint word/display budget. | No configured values, unit inventories, empirical results, redundant equations, or implementation search details enter these subsections. |
| Later paper integration | Check the links to the problem formulation and learning subsection; compile and inspect the revised paper. | Shared definitions are reused, the teacher remains a supporting construction, and the overall three-page non-figure target is checked in the rendered draft. |

The six paragraphs replace the content at `main.tex` lines 123–348. The valid mechanics and energy reasoning are preserved in shorter form. Expanded physical-energy-region algebra, numerical root finding, bisection complexity, and component Jacobians remain in supporting records. The source and exact parameter relations are retained in [the evidence ledger](machine-scannables/evidence_ledger.md); all performance results remain assigned to **Experiments and Results**.

This is planning-only work. New runtime classes, objects, functions, methods, data classes, wrappers, helpers, configuration parameters, and CLI options: **zero**. Keep the repository's compact continuous logic and minimum-realization discipline; add no fallback behavior or configuration layers. `main.tex`, the controller, and the training formulation remain unchanged.
