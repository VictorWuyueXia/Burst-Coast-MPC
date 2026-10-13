# Section III.B: paragraph and equation plan

- Scope: plan only. Preserve the current main.tex, including the author's latest III.A, and prompts.md. No manuscript prose is drafted in this step. The current III.A ends with the need for phase information; the finite-action integration remains in III.B.
- Route: English algorithmic Methods writing for IEEE L-CSS and control readers. Apply Nature writing → control academic writing → Nature polishing → academic humanizer.
- Argument: proportional total-energy feedback requests motor work; physical-state prediction translates that request into torque candidates; a common score balances elevation, work agreement, and arm excursion.
- Organization: three conceptual components and six paragraphs, without adding subsection headings inside III.B. Keep one continuous sequence from observed state to executed input.
- Evidence boundary: the current controller enforces a torque bound and uses a soft arm-position penalty. It does not guarantee exact work delivery, continuous-time arm containment, convergence, or global optimality.

## 1. Proportional energy tracking — paragraphs 1–2

**Paragraph 1: why the upper layer requests work.**

- Start from the power identity $\dot E=u\omega$ in III.A. Its time integral identifies motor work with the change in total mechanical energy, motivating a proportional energy correction. Derive the finite-action work formula in paragraph 3.
- Reuse the observed physical state $x_k$ and its encoded energy $e_k=H(x_k)$ at control update $k$. The components $K_{a,k}$, $K_{p,k}$, and $V_k$ are the computed arm kinetic, pendulum kinetic, and potential energies. Define their sum as the computed total energy $E_k$.
- Present the proportional request immediately after the motivation, using the existing label eq:workpolicy:

$$
E_k=K_{a,k}+K_{p,k}+V_k,
\qquad
w_k=\operatorname{clip}\!\left(\kappa(E_\star-E_k),-W_{\max},W_{\max}\right).
$$

- Define $w_k$ as requested signed motor work over the next action, not a measured energy increment or a torque. The target $E_\star=2G$ is the upright-rest energy derived in III.A.
- Define $\kappa>0$ as a dimensionless feedback gain and $W_{\max}>0$ as a prescribed work-request bound. Both are design choices. Define clip as restricting its first argument to the stated interval.
- Explain immediately: positive error requests energy addition; negative error requests removal. The gain scales the correction, and clipping limits each request. Neither operation establishes that the requested work is attainable.
- Call this proportional feedback on **total energy**. The upper layer does not independently regulate all three energy components.
- Explain why total energy is insufficient: upright rest requires the energy to be in potential energy. The torque score introduced later supplies that preference.
- Keep an ideal error-contraction derivation in the supporting audit rather than the main text. Exact work delivery and inactive clipping would be additional assumptions, not properties of the implemented loop.

**Paragraph 2: which endpoint energies can satisfy the request.**

- Motivate the candidate set: the scalar request fixes a total-energy budget but leaves its distribution unspecified. This gives the decoder possible outcomes to compare.
- Define $\mathcal E=\{H(x):x\text{ is a physical state}\}$ as the set of physically realizable energy triples. This is a mathematical description of the energy image, not a new input variable or a set searched independently by the implementation.
- Let $e$ denote the current energy state and $E=K_a+K_p+V$ its total energy. Let $w$ denote its work request. Define a candidate endpoint energy as $z=(z_a,z_p,z_V)^{\mathsf T}$, with arm kinetic, pendulum kinetic, and potential components respectively.
- Place the existing work-budget surface, eq:surface, here:

$$
\mathcal S(e,w)=\{z\in\mathcal E:z_a+z_p+z_V=E+w\}.
$$

- Explain immediately: every point on this set has the requested total energy, but different points represent different divisions between motion and elevation. The set can be empty; membership does not imply reachability from the current physical state.
- Follow this set definition with the existing physical-feasibility derivation, preserving the intermediate algebra requested in the preceding revisions. Its sole job is to characterize $\mathcal E$.
- Keep the subtraction $K_p=K-K_a$ and the addition/subtraction used to complete the square in eq:energy-feasibility. The essential result is

$$
K_p=\frac J2\left(\nu+\frac CJ\omega\right)^2
+\frac{A-I_a-C^2/J}{I_a}K_a,
\qquad
K_a\ge0,\quad 0\le V\le2G,\quad
K_p\ge\frac{A-I_a-C^2/J}{I_a}K_a.
$$

- Reuse III.A's definitions: $I_a$ is arm inertia, $J$ is pendulum hinge inertia, $A$ is configuration-dependent inertia associated with arm rotation, $C$ is coupling inertia, and $\omega,\nu$ are the signed angular velocities. $G$ is the gravitational energy scale. These are physical quantities computed from the model, not tuning weights.
- Explain why the bound matters: the moving arm carries the pendulum body, so the two kinetic energies cannot take arbitrary independent values. The nonnegative square gives the minimum compatible pendulum kinetic energy.
- Retain eq:energy-inertias to show how the prescribed potential determines the coefficients through $\cos\alpha=1-V/G$. Here $\alpha$ is pendulum angle; both angle signs give the same bound.
- Retain eq:energy-realization as the short existence argument. The potential range supplies a real angle; the arm energy supplies a speed magnitude; the remaining nonnegative energy supplies a real pendulum velocity. Arm position can be chosen within its limits because it does not enter $H$.
- State exactly what sufficiency means: a physical state with those energies exists when velocities are unrestricted. It does not show that an admissible torque can reach that state over the next action.
- End by posing the next required operation: use the observed physical state to determine which outcomes a torque can actually produce.

## 2. Torque decoding from the current motion — paragraphs 3–4

**Paragraph 3: predict the finite-duration effect of each torque.**

- Begin from III.A's information limit: equal energies can correspond to different motion directions and coupling. The work request therefore does not specify a unique torque.
- Retain the observed $x=(\theta,\alpha,\omega,\nu)^{\mathsf T}$ in the predictor. The coordinates describe arm position, pendulum configuration, and signed velocities; no additional phase angle or phase estimator is needed.
- Define $\Delta=t_{k+1}-t_k>0$ as the prescribed duration for holding a constant torque. Its value is an experiment setting, not part of the transferable derivation.
- Define $F_\Delta(x,u)$ as the exact state obtained by integrating III.A's governing dynamics from $x$ under constant torque $u$ for duration $\Delta$. Present the existing eq:work here:

$$
x_{k+1}=F_\Delta(x_k,u_k),\qquad
W_\Delta(x,u)=u\,[\theta(F_\Delta(x,u))-\theta(x)],
\qquad
E(F_\Delta(x,u))=E(x)+W_\Delta(x,u).
$$

- Define $\theta(\cdot)$ as extraction of the unwrapped arm coordinate and $W_\Delta$ as predicted signed motor work. $E(\cdot)$ denotes total mechanical energy evaluated at a physical state.
- Explain the equations immediately: torque times angular displacement is the accumulated work, which changes total energy in the lossless model. Positive torque need not imply positive work because displacement can be negative.
- Explain why integration is needed: it predicts the accumulated effect of an action. Initially zero angular velocity gives zero instantaneous motor power but does not prevent a finite torque from creating motion and performing work.
- Keep physical energy redistribution distinct from net work: zero torque conserves total energy while the components can change.

**Paragraph 4: find and compare torques that realize the work request.**

- With $x$ and $\Delta$ fixed, vary only the held torque $u$ within the usable torque interval $[-U,U]$, where $U>0$ is the imposed input limit from Section II.
- Retain eq:curve as a compact explanation of the outcomes generated by that search:

$$
\mathcal C_x=\{H(F_\Delta(x,u)):u\in[-U,U]\}.
$$

- Explain immediately: $\mathcal C_x$ is parameterized by torque at the current physical state. Its intersections with $\mathcal S(e,w)$ correspond to exact work matching, $W_\Delta(x,u)=w$, in the lossless model.
- State that the equation can have several solutions or none. Configuration and signed velocities change this relation through the mechanical dynamics. Do not suggest an inverse energy map or a direct division of requested work by the initial velocity.
- Describe the numerical realization in prose: integrate a finite torque grid; refine intervals whose work errors bracket zero; retain both the original grid and the refined candidates.
- Introduce a hat only when numerical predictions are first used. $\widehat F_\Delta$ approximates the exact state transition; $\widehat W_\Delta$ is computed from its predicted displacement. Finite integration error means the exact energy balance is not an exact numerical identity.
- Denote the finite set of evaluated torque candidates with valid predictions by the existing $\mathcal U_{\mathrm{test}}(x,w)$. It lies inside $[-U,U]$ and is not restricted to exact work-matching roots.
- State that finite sampling and refinement need not identify every root. Keep search counts and integration settings in the implementation record.
- Transition to ranking: even a work-matching torque may distribute energy poorly or cause excessive arm excursion. Hence matching work alone does not select the action.

## 3. Admissible torque and arm-range treatment — paragraphs 5–6

**Paragraph 5: select the action using the same predicted trajectory.**

- Separate three meanings of feasibility: an energy triple may be physically possible, a torque may satisfy the input bound, and its trajectory may or may not remain within the arm-position bounds. None should be used as a synonym for the others.
- Explain why the arm check uses intermediate predicted states: an endpoint inside the permitted range can follow an intermediate crossing.
- Define $\widehat V^+(x,u)$ as predicted endpoint potential energy. Define $\widehat\Theta(x,u)$ as the maximum absolute arm angle among the initial state and the integration samples used to predict that action. Both are computed outputs of the same rollout.
- Reuse the desired arm limit $\theta_{\max}>0$ from Section II. Define $\max(\widehat\Theta-\theta_{\max},0)$ as predicted excess arm excursion; it is zero for sampled motion within the range.
- Introduce the existing score and minimization together in eq:score:

$$
\begin{aligned}
\mathcal J(x,u;w)
&=\left(\frac{E_\star-\widehat V^+(x,u)}{E_\star}\right)^2
+\lambda_W\left(\frac{\widehat W_\Delta(x,u)-w}{W_{\max}}\right)^2\\
&\quad+\lambda_\theta\left(\frac{\max\{\widehat\Theta(x,u)-\theta_{\max},0\}}{\theta_{\max}}\right)^2,\\
u^\star(x,w)&\in\underset{u\in\mathcal U_{\mathrm{test}}(x,w)}{\arg\min}\;\mathcal J(x,u;w).
\end{aligned}
$$

- Explain the three terms immediately and in the same order: favor elevation toward upright, favor delivery of the requested work, and penalize arm excursions beyond the prescribed range.
- Define $\lambda_W,\lambda_\theta>0$ as dimensionless, tunable penalty weights. The first term sets the reference weight. The three denominators make the residuals dimensionless using already defined energy, work, and angle scales; no additional normalization parameter is needed.
- Keep these short explicit ratios rather than introducing an undefined normalize operation. The score is a dimensionless control objective, not a modified physical energy.
- Explain the joint selection rule: grid candidates and refined work-matching candidates compete under the same score. The chosen action can miss the requested work even when a tested work-matching candidate exists.
- State the constraint boundary beside the score: torque bounds are enforced by the candidate interval, while a finite arm penalty can still select a violating trajectory. A sampled check also cannot guarantee containment between samples.
- Do not describe the method as projection onto a hard feasible set, as exact root selection followed by arm correction, or as guaranteed arm-range enforcement. Those would be different control laws.

**Paragraph 6: close the feedback loop.**

- Apply $u_k=u^\star(x_k,w_k)$ over the prescribed action duration, observe the resulting state, recompute its energy, and repeat. Use “control update” or “time step.”
- Explain that new observations correct the next work request and torque prediction; this is the operational feedback loop, not a claim of stability or convergence.
- Keep tie-breaking and treatment of nonfinite predictions in the implementation record. If mentioned in the paper, report the actual behavior without inventing a fallback controller.
- End with the role of the two layers: the upper layer sets a total-energy correction; the lower layer uses the current motion to balance that correction against elevation and arm excursion.
- Add no numerical settings, experiment termination rules, unrelated future learning objectives, or new displayed equation here.

## Structure and notation budget

| Category | Planned objects and roles |
| --- | --- |
| Paragraphs and displays | Six paragraphs under three conceptual components; retain the eight existing III.B equation groups. Add no theorem or separate algorithm block. |
| Physical quantities | Reuse $x,u,H,K_a,K_p,V,E,E_\star,I_a,J,A,C,G,\theta,\alpha,\omega,\nu$ and their III.A meanings. $E_k$ is the existing total energy evaluated at update $k$. |
| Work feedback | Reuse $w,\kappa,W_{\max}$; requested work, gain, and request bound. |
| Energy geometry | Reuse $z,\mathcal E,\mathcal S,\mathcal C_x$; candidate energy, physically possible energies, work-budget set, and torque-response curve. |
| Finite action | Reuse $\Delta,F_\Delta,W_\Delta$ and their numerical predictions; action duration, state transition, and accumulated work. |
| Candidate ranking | Reuse $\mathcal U_{\mathrm{test}},\widehat V^+,\widehat\Theta,\mathcal J,u^\star$; finite candidates, endpoint elevation, sampled arm excursion, score, and selected torque. |
| Limits and weights | Reuse $U,\theta_{\max},\lambda_W,\lambda_\theta$; torque bound, desired arm range, and score weights. No new adjustable quantities. |
| Independent inputs | The observed $x$ is fixed during candidate evaluation; $u$ is the trial scalar input; $w$ is supplied by the energy law. Candidate energy components are outputs, not independent actuator inputs. |
| Software structure | Zero new classes, functions, methods, data classes, helpers, wrappers, CLI options, configuration values, or runtime edits. Keep continuous task-oriented logic; no extra abstractions or fallback behavior. |

- Preserve the established terms: physical state, energy state, requested work, state-transition map, phase-dependent torque decoding, arm excursion, and control update.
- Preserve existing equation labels. In the plan, eq:energy-feasibility, eq:energy-inertias, and eq:energy-realization remain subordinate to the definition of the candidate energy set, as the user previously requested.
- Remove repeated explanations of non-invertibility already established in III.A. Use one linking sentence to motivate retaining the physical state.
- Remove numerical parameter values, unit-specific action durations, torque-grid counts, and refinement counts from the main mathematical narrative. Record them separately for reproducibility; do not imply their values are universal.

## Writing milestones and validation

| Stage | Validation procedure | Success flag |
| --- | --- | --- |
| Structure plan, this task | Follow each paragraph's input and output through the six-paragraph sequence. | Each of the three requested components is explicit; no manuscript or runtime source changes. |
| Mathematical preparation | Check work dimensions, exact-versus-numerical balance, energy-image conditions, and the common candidate score against current code. | All symbols have definitions; requested work, physical realizability, and dynamic reachability remain distinct. |
| Section drafting, when requested | Apply the four writing passes to III.B only; compare against the current III.A. | Each display follows its motivation and is immediately interpreted; earlier sections and terminology are preserved. |
| Final verification after drafting | Check labels and source diff, compile, inspect two-column equations, and verify claim strength. | No added numerical claims or unsupported constraint/convergence guarantees; mathematical expressions remain legible. |

This is a theory-oriented proof-of-concept exposition. The plan does not add a hard state-constraint controller, change the implementation, run new experiments, or establish publication readiness for the full paper.
