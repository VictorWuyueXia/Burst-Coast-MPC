# Section III.B: dynamics-led narrative plan

- Scope: a new paragraph and equation plan, dated 2026-10-12. The current manuscript remains unchanged. This plan replaces the earlier six-paragraph outline for the next revision.
- Writing route: Nature writing → control academic writing → Nature polishing → academic humanizer; English algorithmic Methods for IEEE L-CSS and control readers encountering this system for the first time.
- Central argument: request a correction in total energy, predict how bounded torque changes the coupled motion, and select an input using the resulting energy distribution and arm excursion.
- Proposed title: **Work selection and phase-dependent torque decoding**. Remove “Hierarchical” and describe energy feedback and torque decoding by their functions. A layered architecture is unnecessary to explain this controller.
- Organization: seven paragraphs under the existing subsection, with eight existing equation groups. Aim for roughly the current section length by replacing repeated explanations. The added emphasis is the causal connection between the mechanics, energy geometry, and input selection.

## The actuator distinction that governs the narrative

- The current implementation already requests signed **work** and searches over bounded **torque**. The work-request bound is a controller setting. It is neither a power limit nor a guarantee that the requested energy change is achievable.
- The mechanical input is $u$, the torque applied to the arm. Its power is $P_{\mathrm{mech}}=u\omega$, where $\omega$ is the signed arm velocity. A torque bound $|u|\le U$ implies $|P_{\mathrm{mech}}|\le U|\omega|$. Thus mechanical power changes with motion even when torque is held constant.
- Quanser's QUBE-Servo 2 and 3 interfaces command motor voltage; the QUBE-Servo 3 also supports a PWM command. Available torque depends on voltage, current, speed, and thermal operating limits. A fixed maximum-power command is not the documented actuator interface. See the official [QUBE-Servo 2 interface](https://docs.quanser.com/quarc/documentation/qube_servo2_usb.html) and [QUBE-Servo 3 interface](https://docs.quanser.com/quarc/documentation/qube_servo3_usb.html).
- A basic direct-drive DC motor model gives $u\approx k_t i$ and $v_m\approx Ri+k_e\omega$ when electrical transients and motor mechanical losses are neglected. Here $i$ is motor current, $v_m$ is motor terminal voltage, and $k_t$, $R$, and $k_e$ are the torque constant, winding resistance, and back-emf constant, obtained from specifications or identification. Voltage and current constraints therefore induce a speed-dependent admissible torque range. This is a model-based conclusion consistent with Quanser's [motor-model derivation, pp. 40–41](https://www.quanser.com/wp-content/uploads/2017/03/Quarc-Matlab-Simulink-Primer.pdf).
- Retain the paper's ideal bounded-torque model in this writing plan. A hardware-specific revision would change the actuator model or admissible input set used in prediction. The proportional work request and the plane–curve interpretation can remain. Constant voltage and constant torque are different action parameterizations.
- Electrical input power is $P_{\mathrm{elec}}=v_m i$. The present lossless mechanical model gives $\dot E=P_{\mathrm{mech}}$ and contains no electrical losses. It therefore cannot establish electrical-to-mechanical efficiency. Configuration-dependent conversion among kinetic and potential energy is a different phenomenon; potential-energy growth can draw on stored kinetic energy.
- The exact Quanser model has not yet been identified by the author. Do not insert a numerical hardware limit or treat a current-sensor range as a permissible continuous motor current. No numerical hardware rating is needed for the proposed mathematical narrative.

## 1. Open with the control construction implied by III.A

- **Paragraph job:** explain how the derived dynamics lead to the control law before introducing the work request.
- Start with III.A's two relevant conclusions: total energy changes through motor work, and the same energy state can correspond to different motions. The energy error supplies a scalar correction, while the observed physical state supplies the information needed to realize it.
- Name the two operations directly: energy feedback requests work, and torque decoding predicts which bounded input can deliver it. Do not introduce a separate architectural vocabulary.
- Connect the symbols to their established roles: $x_k=(\theta_k,\alpha_k,\omega_k,\nu_k)^{\mathsf T}$ is the observed physical state at update $k$, $e_k=H(x_k)$ is its computed energy state, and $E_k$ is the sum of the three energies. The upright-rest target is $E_\star=2G$.
- Introduce proportional feedback in the existing **eq:workpolicy** group:

$$
E_k=K_{a,k}+K_{p,k}+V_k,
\qquad
w_k=\operatorname{clip}\bigl(\kappa(E_\star-E_k),-W_{\max},W_{\max}\bigr).
$$

- Define $w_k$ as requested signed work over the next action, $\kappa>0$ as the dimensionless gain, and $W_{\max}>0$ as the work-request bound. The gain and bound are design choices; clip restricts its argument to the stated interval.
- Immediately interpret the sign: energy deficit requests addition, and surplus requests removal. A request changes the desired total-energy budget, but does not prescribe the energy distribution or establish input feasibility.
- Carry the goal from III.A into the next paragraph: upright rest requires both $E$ and $V$ to approach $E_\star$, so regulating their sum alone leaves a control task unresolved.

## 2. Establish the energy plane before any intersection argument

- **Paragraph job:** turn the scalar work request into an intuitive geometric target.
- First identify the three axes: arm-body kinetic energy, pendulum-body kinetic energy, and gravitational potential energy. An endpoint is a point $z=(z_a,z_p,z_V)^{\mathsf T}$ in this three-dimensional space.
- Explain that all endpoints delivering the same requested work have the same total energy. The single linear equation $z_a+z_p+z_V=E_k+w_k$ therefore defines a plane.
- Interpret the plane before discussing sets: increasing one component requires decreasing the sum of the other two. For fixed $E_k$, changing $w_k$ moves the plane parallel to itself. The plane is positioned by the requested correction; it need not pass through $e_\star$ until the request equals the full energy error.
- Define $\mathcal E=\{H(x):x\text{ is a physical state}\}$ as the physically realizable energy triples. Use **eq:surface** for the physically possible part of the plane:

$$
\mathcal S(e_k,w_k)
=\{z\in\mathcal E:z_a+z_p+z_V=E_k+w_k\}.
$$

- Distinguish the full plane from $\mathcal S$: the linear equality defines the plane, while membership in $\mathcal E$ removes impossible combinations. Do not call the restricted set the entire plane.
- Explain its control role: it expresses the requested total energy while allowing different kinetic–potential distributions. The decoder will compare the endpoints that torque can produce, rather than choose all three coordinates independently.

## 3. Use the mechanical energy expression to restrict possible endpoints

- **Paragraph job:** explain which parts of the plane can represent the physical pendulum, preserving the author's requested derivation.
- Connect directly to III.A's $K$ and $K_a$. Rotating the arm moves the pendulum body, so two arbitrary nonnegative kinetic-energy values need not describe any physical configuration and velocity.
- Retain **eq:energy-feasibility**, including $K_p=K-K_a$, subtraction of $I_a\omega^2/2$, addition and subtraction of $C^2\omega^2/(2J)$, and the resulting square:

$$
K_p=\frac J2\left(\nu+\frac CJ\omega\right)^2
+\frac{A-I_a-C^2/J}{I_a}K_a,
\qquad
K_a\ge0,\quad 0\le V\le2G,\quad
K_p\ge\frac{A-I_a-C^2/J}{I_a}K_a.
$$

- Immediately identify the bound as the minimum pendulum kinetic energy compatible with the specified $V$ and $K_a$. The minimum occurs at $\nu=-C\omega/J$; the potential range corresponds to downward through upright elevation.
- Retain **eq:energy-inertias** to compute $C=b(1-V/G)$ and $A=I_a+m_pr^2+J[1-(1-V/G)^2]$. These follow from $\cos\alpha=1-V/G$ and III.A's physical parameters. The multiplier depends on $V$ without requiring the sign of $\alpha$.
- Retain **eq:energy-realization** with both velocity formulas. Explain the construction in order: the potential supplies a real angle, $K_a$ supplies an arm-speed magnitude, and energy above the lower bound supplies a real pendulum velocity. Arm position may be chosen within its limits because it does not enter $H$.
- State the conclusion precisely: these conditions characterize the existence of a physical state with the given energies when velocities are unrestricted. They do not establish reachability under bounded torque over the next action.
- Clarify the computational role: these bounds explain the energy geometry. The implemented decoder obtains physically consistent candidates by predicting $x$ and evaluating $H(x)$; it does not independently sample energy triples and filter them using this proof.

## 4. Derive the finite-action predictor from III.A's governing equations

- **Paragraph job:** show how $f$, $H$, and $DH$ determine the predicted energy change for a trial input.
- Fix the observed state $x=x_k$ and a prescribed duration $\Delta=t_{k+1}-t_k>0$. Hold a trial torque $u\in[-U,U]$ over that duration, where $U$ is the imposed torque bound already defined in Section II.
- Integrate the governing equation $\dot x=f(x,u)$ from III.A. Denote the resulting state at elapsed time $s$ by $x(s)=F_s(x,u)$, with $x(0)=x$ and $0\le s\le\Delta$. This defines the discrete transition $x_{k+1}=F_\Delta(x_k,u_k)$.
- Expand the existing **eq:work** group with the finite-time form of III.A's chain rule:

$$
H(F_\Delta(x,u))-H(x)
=\int_0^\Delta DH(x(s))f(x(s),u)\,\mathrm ds.
$$

- Interpret this identity immediately: $f$ predicts physical motion, $DH$ converts that motion into instantaneous changes of each energy component, and integration accumulates those changes over the action.
- Then sum the three energy rates to recover the scalar work relation in the same equation group:

$$
\begin{aligned}
W_\Delta(x,u)
&=\int_0^\Delta u\,\omega(s)\,\mathrm ds\\
&=u\,[\theta(F_\Delta(x,u))-\theta(x)],\\
E(F_\Delta(x,u))&=E(x)+W_\Delta(x,u).
\end{aligned}
$$

- Here $\omega(s)$ is the arm velocity along the predicted motion, $\theta(\cdot)$ extracts unwrapped arm position, and $E(\cdot)$ evaluates total mechanical energy. Torque times angular displacement gives the delivered work.
- Use III.A's $\dot V=G\sin\alpha\,\nu$ to explain the need for prediction: torque affects accelerations and subsequent energy transfer. Potential-energy rate is not a freely commanded input. At rest, a finite torque can initiate motion although instantaneous mechanical power is zero.
- State the implementation faithfully: the numerical predictor integrates $f$ and evaluates $H$ at the endpoint. The explicit Jacobian is the analytical basis of the energy-rate identity; it is not inverted or evaluated as a separate online search operation.

## 5. Construct the torque-response curve from predicted endpoints

- **Paragraph job:** explain the curve as a consequence of varying one bounded input.
- Keep the same initial state and duration for every trial. One torque produces one predicted physical endpoint; applying $H$ produces one point in the three-dimensional energy space.
- Varying the scalar torque moves this endpoint point and traces a parameterized curve. Retain **eq:curve**:

$$
\mathcal C_x
=\{H(F_\Delta(x,u)):u\in[-U,U]\}.
$$

- Explain what changes and what stays fixed: $u$ parameterizes the curve, while $x$ and $\Delta$ are fixed. The curve records endpoints from different torque trials, not the time history of one trial.
- State its location accurately: the curve lies in the physical energy set $\mathcal E$, but generally does not lie in the requested energy plane. Each torque can deliver a different amount of work.
- Explain phase dependence through the dynamics already established: configuration-dependent inertia and signed velocities change the predicted accelerations, arm displacement, and kinetic–potential exchange. Equal initial energies therefore need not generate the same curve.
- Establish the next question: which points on this curve have the total energy requested by the feedback law?

## 6. Explain why intersections matter and how the scalar search finds them

- **Paragraph job:** connect the geometric intersection to a computable input condition.
- An intersection has two properties at once: a bounded torque produces the endpoint over $\Delta$, and its total energy matches the requested budget.
- Use the energy balance to explain the equivalence, preferably inline beside the curve discussion in the manuscript:

$$
H(F_\Delta(x,u))\in\mathcal S(H(x),w)
\quad\Longleftrightarrow\quad
W_\Delta(x,u)=w.
$$

- Here $w=w_k$ denotes the current request. Multiple intersections give multiple work-matching torques with potentially different endpoint potential and arm excursion. No intersection means no held torque in the imposed interval delivers that request over this duration. This does not imply infeasibility for every time-varying input or longer duration.
- Introduce hats only for numerical predictions. $\widehat F_\Delta$ is computed by fourth-order Runge–Kutta integration; $\widehat W_\Delta$ follows from its predicted arm displacement using the same work formula.
- Evaluate the scalar error $\widehat W_\Delta(x,u)-w$ on a uniform torque grid. Opposite error signs at adjacent torques place their endpoints on opposite sides of the requested total-energy level, up to numerical energy-balance error. Bisection refines such intervals, including an endpoint with zero error.
- Retain both grid torques and valid refined torques in the existing finite set $\mathcal U_{\mathrm{test}}(x,w)\subseteq[-U,U]$. The procedure need not find tangential intersections or every root between sampled torques.
- End with the remaining control requirement: intersection identifies agreement with the scalar work request, but does not select the best kinetic–potential distribution or control arm excursion. The implemented score compares all candidates together.

## 7. Select the bounded torque using energy distribution and arm motion

- **Paragraph job:** turn the predicted candidates into an executed input and close the feedback loop.
- Use the same predicted trajectories to compute $\widehat V^+(x,u)$, the endpoint potential, and $\widehat\Theta(x,u)$, the maximum absolute arm position over the initial state and integration samples. The prescribed arm range is $|\theta|\le\theta_{\max}$.
- Explain why endpoint energy and endpoint arm position are insufficient for arm-range assessment: $H$ omits absolute arm position, and intermediate motion can exceed the arm range before returning.
- Retain the existing score and minimizer in **eq:score**:

$$
\begin{aligned}
\mathcal J(x,u;w)
&=\left(\frac{E_\star-\widehat V^+(x,u)}{E_\star}\right)^2
+\lambda_W\left(\frac{\widehat W_\Delta(x,u)-w}{W_{\max}}\right)^2\\
&\quad+\lambda_\theta
\left(\frac{\max\{\widehat\Theta(x,u)-\theta_{\max},0\}}{\theta_{\max}}\right)^2,\\
u^\star(x,w)&\in\underset{u\in\mathcal U_{\mathrm{test}}(x,w)}{\arg\min}\ \mathcal J(x,u;w).
\end{aligned}
$$

- Immediately interpret the terms in order: approach upright potential energy, deliver the requested work, and reduce excursion beyond the arm range. The first term complements total-energy feedback because $K_a+K_p=E-V$. The positive dimensionless weights $\lambda_W$ and $\lambda_\theta$ are design choices; all denominator scales are already defined.
- State what the rule enforces: candidate torques obey the input bound, while requested work and arm position enter as finite penalties. A selected torque may miss an available intersection or exceed the arm range. Checking predicted samples does not establish containment between samples.
- Apply $u_k=u^\star(x_k,w_k)$ for $\Delta$, observe the resulting state, and repeat. Fresh physical-state observations update both the energy request and the predicted torque response.
- End with that feedback operation. No electrical-efficiency, stability, global-optimality, or hard arm-constraint claim follows from the current formulation.

## Notation and equation-use audit

| Object retained from III.A | Role in the revised III.B |
| --- | --- |
| $f(x,u)$ and the coupled inertia equation | Predict physical trajectories for each trial torque, including the changing coupling during an action. |
| $H(x)$ | Compute the initial energy error and map every predicted endpoint to energy coordinates. |
| $DH(x)$ | Derive the energy rates and their finite-time integral; it is not an inverse map or an input-sensitivity matrix. |
| $\dot E=u\omega$ | Derive work delivery and the energy-plane intersection condition. |
| $\dot V=G\sin\alpha\,\nu$ | Explain why existing motion and future acceleration matter for potential-energy transfer. |
| $K$, $K_a$, and the physical inertia terms | Restrict the plane to energy triples that a physical state can realize. |
| $F_\Delta(x,u)$ | Represent the action-duration integration used to construct endpoints, work, and arm excursion. |

- Terminology: retain physical state, energy state, requested work, phase-dependent torque decoding, arm excursion, and control update. Define the energy plane before the torque-response curve and their intersection.
- Notation budget: retain $x,e,u,H,DH,f,K_a,K_p,V,E,E_\star,I_a,J,A,C,b,G,z,\mathcal E,\mathcal S,\mathcal C_x,w,\kappa,W_{\max},U,\theta_{\max},\Delta,F_\Delta,W_\Delta,\mathcal U_{\mathrm{test}},\widehat V^+,\widehat\Theta,\mathcal J,u^\star,\lambda_W,\lambda_\theta$ and their established meanings. The elapsed-time variable $s$ and notation $x(s)=F_s(x,u)$ make the integral explicit; they introduce no new state, controller parameter, or phase estimator.
- Structure budget: seven paragraphs, eight existing numbered groups, no added theorem, algorithm block, controller layer, or manuscript figure in this revision. The finite-time identity expands eq:work; other identities can remain inline where they explain existing displays.
- Software budget: zero new classes, functions, methods, data classes, wrappers, helpers, tunable quantities, command-line options, configuration entries, or runtime edits. Preserve the project's minimum-realization, continuous-logic style without additional abstractions or fallback behavior.
- Scope: preserve III.A and the current scientific equations. Identify wording dependencies in III.A separately rather than rewriting them during a III.B plan. In particular, the loss of phase prevents a deterministic energy-only evolution law in general; it is not merely a failure to find a closed-form inverse.

## Writing milestones and success conditions

| Milestone | Validation procedure | Success condition |
| --- | --- | --- |
| Actuation and model interpretation | Compare official Quanser interface documentation with current energy policy, decoder, and physics configuration. | Work request, torque bound, mechanical power, and electrical input power remain distinct; unknown hardware ratings are not inferred. |
| Narrative plan | Trace physical dynamics → energy correction → plane → physical bounds → predicted endpoints → curve → intersections → scored input. | Each object appears after its motivation, and every object from III.A has an explicit analytical or computational role. |
| Manuscript drafting after this plan | Replace the III.B narrative using the seven paragraphs and retained labels; apply all four skills in order. | Connected control-theory exposition, immediately interpreted equations, approximately unchanged length, no prohibited wording or unsupported performance claims. |
| Verification after drafting | Compare source outside III.B, check labels and definitions, compile, and inspect equations in the two-column layout. | Scope preserved; numerical settings stay outside the mathematical narrative; compilation and presentation limits are reported accurately. |

The proposed changes concern exposition of the existing proof-of-concept controller. Adding motor electrical dynamics or testing electrical efficiency would be separate modeling and experimental work.
