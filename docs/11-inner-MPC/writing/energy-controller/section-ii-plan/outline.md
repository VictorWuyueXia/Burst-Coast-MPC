# Section II: problem formulation outline

This is an argument and equation-placement plan, not replacement manuscript prose. Preserve the author's revised Section I. Use three paragraphs of roughly the current section's length, retaining only equations that define the task or disambiguate a metric. Apply nature-writing, control-academic-writing, nature-polishing, and academic-humanizer in that order.

## Paragraph 1: geometry and nomenclature

- Introduce a motor-driven arm rotating about a vertical axis, with an unactuated pendulum hinged at its tip. The pendulum swings in the vertical plane perpendicular to the arm. Explain that this arrangement requires the motor to influence pendulum motion through the arm.
- Define the unwrapped arm angle theta relative to a fixed horizontal reference. Keeping complete revolutions distinguishes arm excursion from a periodic orientation.
- Define the pendulum angle alpha relative to downward, represented on the half-open interval from minus pi to pi. Upright is the same physical orientation at either angular boundary; its stored representative is minus pi.
- Define omega and nu as the signed arm and pendulum angular velocities. Angular velocity describes continuous physical motion, not the derivative of a coordinate jump at the wrapping boundary.
- Define u as the motor torque about the arm axis, positive in the direction of increasing theta. No torque acts directly at the pendulum hinge.
- Use inline notation in the manuscript; no displayed geometry equation. Leave masses, inertias, energy variables, and the equations of motion to III.A.

## Paragraph 2: state, input, initial conditions, and objective

- Gather the coordinates into the physical state and state the full-state observation assumption. At decision time t_k, the controller observes x(t_k) and chooses a torque held until the next decision. The nonnegative integer k indexes decisions; the times need not be equally spaced in the general formulation.
- Present the state, initial-set membership, and torque bound together:

$$
x=(\theta,\alpha,\omega,\nu)^{\mathsf T},\qquad
x(0)\in\mathcal X_0\subset\mathbb R\times[-\pi,\pi)\times\mathbb R^2,
\qquad |u(t)|\le U.
$$

- The state x collects the four quantities defined in paragraph 1; the superscript denotes transpose. The time t is elapsed physical time and U is the prescribed positive torque limit. The bounded initial set X_0 and the sampling distribution on it are declared by the experiment. Give the classes in words here: downward, moving, near-upright, and tightly near-upright starts. Leave their configured ranges and sampling rules in the experiment table. Do not imply that all physical states are feasible starts.
- Define the wrapped upright error before using it in the target:

$$
\beta=\operatorname{wrap}(\alpha-\pi),\qquad
\mathcal X_\star=\{x:\beta=0,\ \omega=\nu=0,\ |\theta|<\theta_{\max}\}.
$$

- The operator wrap returns the unique angular representative in the interval from minus pi to pi. Thus beta is the shortest signed angular displacement from upright and remains suitable near the coordinate boundary. The prescribed positive theta_max defines the desired arm envelope. The target X_star describes upright rest without requiring a particular arm position within that envelope.
- Explain the design objective immediately: drive the pendulum toward upright rest while respecting the torque limit and limiting arm excursion. Torque saturation is a hard input constraint; the present controller treats the arm envelope as a soft constraint, so containment is evaluated rather than guaranteed.
- State that capture is an evaluation event: the upright error stays within a prescribed positive angular tolerance epsilon_alpha for a prescribed number of consecutive recorded samples. The experiment fixes the tolerance, sample spacing, and count. This angle-only event does not establish rest or continued holding. Do not add a speed condition to the existing labels.
- Define coasting compactly as an interval of zero motor torque. Defer the future timing optimization and prediction-horizon variables to the method that introduces them.

## Paragraph 3: performance and intervention metrics

- Start with acquisition: report the fraction of trials that capture before a declared deadline and their capture times, with failures reported separately. Initial-state distributions, deadlines, and capture definitions must match across controller comparisons.
- For continued trajectories, define a common protocol-specified observation interval I=[t_0,t_0+T]. Here t_0 is its start and T is its positive duration. The protocol fixes whether the window begins at initialization or at capture. Apply the same anchoring and duration across controllers.
- Retain the two complementary angular metrics in one display:

$$
\beta_{\mathrm{rms}}=\left(\frac{1}{T}\int_{\mathcal I}\beta(t)^2\,\mathrm dt\right)^{1/2},
\qquad
\rho_\alpha=\frac{1}{T}\int_{\mathcal I}\mathbf 1_{\{|\beta(t)|\le\varepsilon_\alpha\}}\,\mathrm dt.
$$

- The indicator equals one when its condition holds and zero otherwise. The computed beta_rms measures typical angular deviation; the computed rho_alpha measures the fraction of the window near upright. Neither alone establishes rest. Summarize angular-speed magnitudes and arm-envelope violations in words, without another family of equations. Use speed RMS and the fraction of trials crossing the arm boundary as compact diagnostics.
- Retain physical effort and torque-on time in a second display, using the same observation window:

$$
W_{\mathrm{abs}}=\int_{\mathcal I}|u(t)\omega(t)|\,\mathrm dt,
\qquad
T_{\mathrm{act}}=\int_{\mathcal I}\mathbf 1_{\{u(t)\ne0\}}\,\mathrm dt.
$$

- The computed absolute shaft work W_abs counts energy injection and removal without cancellation; it is not electrical consumption. The computed T_act measures the time for which applied torque is nonzero. Retain the explicit normalization and integral definitions, because they define the metrics and need no extra notation.
- For future act-coast control, add the number of scheduled actuation periods and the durations of scheduled coasting periods. Count an initially active period once. Count whole interventions rather than individual torque updates; an active period may contain several commands. Report computation time per decision and total computation time over the evaluation window.
- Do not equate torque-on time with the duration of an active scheduler mode: an active period can contain a zero-torque command. Do not infer suspended computation or sensing from zero torque. Derive scheduled coasting metrics from the act-coast schedule when it exists.
- Current trials stop at capture. Post-capture error, occupancy, and speed metrics require continued trajectories; report their definitions here and their availability with the experiments. Avoid presenting them as existing measured results.

## Structure and notation budget

- Three manuscript paragraphs; at most four displayed equation groups as specified above. Equations may break lines to fit the column.
- Reuse theta, alpha, omega, nu, x, u, k, t_k, U, beta, theta_max, epsilon_alpha, X_star, beta_rms, rho_alpha, W_abs, and T_act. Introduce only the bounded initial set X_0 and the explicit observation-window quantities I, t_0, and T where missing or inconsistent. The wrap operator receives its complete definition at first use.
- No dynamic-horizon symbol, burst scheduler variable, tolerance for each velocity, optimization objective, or additional metric acronym in this section. Discuss the few additional diagnostics in words.
- No code is planned: zero new classes, functions, data classes, helpers, wrappers, runtime parameters, or configuration options. Preserve the project's minimal continuous logic and avoid fallback behavior or unnecessary abstractions.

## Writing milestones and checks

| Milestone | Validation | Success condition |
| --- | --- | --- |
| Argument structure | Read paragraph roles in order without equations. | Geometry explains indirect actuation; the task follows from the coordinates; metrics answer whether and at what cost it is achieved. |
| Mathematical setup | Check domains, wrapping, symbol definitions, and constraints against the current model. | The upright target belongs to the stated domain; all symbols have one role; initial conditions and hard versus soft constraints are explicit. |
| Evidence alignment | Compare success wording with the environment and evaluation loop. | Capture remains angle-only; no sustained-holding or scheduled-coasting result is implied. |
| Prose and length | Apply polishing and humanizer passes after the first two skills; compare prose length and display count with current Section II. | Three connected paragraphs, roughly unchanged length, no undefined shorthand, unnecessary values, author-name citation phrases, or duplicated method derivations. |

## Terminology decisions

| Term | Meaning in this section |
| --- | --- |
| Act-coast control | The author's preferred name from the revised Section I. |
| Upright rest | Desired angular position with both angular velocities zero. |
| Capture | The existing sampled angle-residence event. |
| Arm envelope | A desired range treated through a soft penalty by the present controller. |
| Intervention | One scheduled actuation period, possibly containing multiple commands. |
| Coasting | Zero motor torque; scheduled coasting is identified from the future controller's schedule. |
