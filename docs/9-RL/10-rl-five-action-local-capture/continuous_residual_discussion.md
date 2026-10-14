# Continuous residual control: discussion, not a frozen development plan

## Human quick reading

Continuous torque and an energy-guided residual policy are reasonable next experiments. They give the learner finer corrections and a structured starting policy. They do not by themselves resolve the value-learning and local action-ranking failures found in the dense-Q campaign.

Two clarifications are now confirmed: **90 degrees means a horizontal pendulum**, and **inward braking is allowed when zero torque would not avoid an arm excursion**. Three corrections follow from the current simulator:

1. The arm motor cannot statically hold this underactuated pendulum exactly horizontal. There is no finite horizontal-holding arm torque from which to derive the policy cap. The pendulum hinge's peak gravity torque is 0.01518588 N m, but that hinge is unactuated; this value is also larger than the current 0.00918 N m pump action.
2. Desired energy change divided by time is power, not torque. Converting it to shaft torque requires velocity and the arm–pendulum coupling, including their signs. Dividing by a near-zero velocity without regularization is unsuitable.
3. Zero torque permits coasting; it does not guarantee arm-limit compliance. Check the combined heuristic-plus-residual command, then use zero only when its predicted motion is acceptable. Otherwise use inward braking. Large initial momentum may make a crossing unavoidable within a small torque cap; report that honestly and keep the existing nonterminal semantics.

Keep 100 ms decisions, 20 ms physics, the 20 s deadline, the physical clip, and the five-term reward while testing this change. Calibrate a smaller policy cap through local recovery and swing-up tests. Candidate caps of 0.001, 0.002 and 0.004 N m are a proposed screening range, not validated choices. Their smallness relative to the old pump does not establish sufficient control authority.

Prefer an **action residual with a full-return continuous critic**: the heuristic proposes torque; the actor corrects it; the critic learns the same task return under the resulting total torque. Learning a residual action does not require defining a residual reward or subtracting an unknown heuristic Q function. Adding a learned correction to a conventional controller is the core residual-control construction in [Johannink et al.](https://arxiv.org/abs/1812.03201).

A reasonable first continuous learner is TD3: one deterministic residual actor and two action-conditioned critics, with target copies. Its clipped double-critic targets and delayed actor updates address approximation-error problems in continuous actor–critic learning; this is a design rationale, not a guarantee for our task. See the [original TD3 paper](https://proceedings.mlr.press/v80/fujimoto18a.html). Our earlier Q/rollout discrepancy is not itself proof of the specific bias mechanism in that paper.

This document records a proposal; no controller code or training is changed. The checks below use the existing NumPy dynamics on CPU. A development structure budget should be frozen after the torque-cap and residual-authority decisions; production-code changes in this discussion have a budget of zero.

## Model-derived details

### State, model constants, and horizontal holding

The simulated state is

$$
x=(\theta,\alpha,\omega,\nu),\qquad \omega=\dot\theta,\quad \nu=\dot\alpha.
$$

The simulator measures arm angle theta and pendulum angle alpha in radians, and velocities omega and nu in rad/s. Alpha is measured from downward: zero is down, pi/2 is horizontal, and pi is upright. Motor torque u in N m acts on theta, not directly on alpha. Time t and consecutive goal-hold count d remain part of the Markov decision state s=(x,t,d), as in v3.

The existing pendulum equation is

$$
J_p\ddot\alpha+B\cos\alpha\,\ddot\theta
=-b_p\nu+J_p\sin\alpha\cos\alpha\,\omega^2-G\sin\alpha.
$$

Constants are computed from `configs/physics.yaml` by `derive_model`: J_p=0.000133128 kg m² is pendulum inertia about its hinge; B=m_p r l_c=0.00013158 kg m² is coupling inertia; m_p=0.024 kg is pendulum mass; r=0.085 m is arm length; l_c=0.0645 m is pendulum center-of-mass distance; G=m_p g l_c=0.01518588 N m is the gravitational coefficient with g=9.81 m/s². Pendulum damping b_p=0.00005 N m s is configured. Radians are dimensionless for these units; G is also the coefficient of potential energy in joules.

At horizontal rest, alpha=pi/2 and nu=0, so both cosine-dependent terms vanish regardless of arm velocity or finite arm torque:

$$
\ddot\alpha=-G/J_p=-114.06976744\;\mathrm{rad/s^2}.
$$

An acceleration of zero is required to keep the angle fixed. Therefore a finite static horizontal-holding arm torque does not exist in this model. CPU evaluations at arm torques −0.0204, zero and +0.0204 N m all give the same pendulum acceleration at that state. The value G would be the required direct hinge torque at horizontal if that hinge had its own motor; our plant does not have that motor.

### Choosing a policy torque cap

Let U>0 be a new fixed policy torque cap in N m, strictly below the physical clip of 0.0204 N m. The actuator remains physically unchanged. U should be selected from task performance and motion scales, rather than equated with G.

The following deterministic CPU check starts exactly upright at rest, applies a positive constant shaft torque for 0.1 s, and performs five existing 0.02 s RK4 steps without episode termination. Beta is the resulting upright angle error alpha−pi. This is a response-scale check, not a recovery test or training result.

| Torque [N m] | Arm displacement [rad] | Beta [rad] | Arm speed [rad/s] | Pendulum speed [rad/s] |
|---:|---:|---:|---:|---:|
| 0.000408 | 0.006925 | 0.007439 | 0.1365 | 0.1598 |
| 0.001 | 0.016972 | 0.018232 | 0.3345 | 0.3915 |
| 0.002 | 0.033937 | 0.036461 | 0.6685 | 0.7828 |
| 0.004 | 0.067821 | 0.072893 | 1.3338 | 1.5637 |
| Current pump: 0.00918 | 0.154985 | 0.166926 | 3.0195 | 3.5645 |
| Hinge gravity reference: 0.01518588 | 0.254113 | 0.274903 | 4.8570 | 5.8169 |

These values show why the gravity reference would not give the desired smaller action scale. A continuous policy can use values much smaller than its cap near upright; the cap itself need not keep the exact-rest response inside the goal. Conversely, reducing U too far can remove the authority needed to brake or reach upright within 20 s. Screen both local recovery and energy pumping before selecting it. Keep the current torque reward normalization fixed during the cap comparison; dividing by each candidate U would also change the reward per physical torque and confound the experiment.

### Correcting energy-to-torque conversion

Keep the same pendulum-relative energy used in the reward:

$$
E_p(x)=\tfrac12J_p\nu^2+G(1-\cos\alpha),\qquad E_\star=2G=0.03037176\;\mathrm J.
$$

E_p is computed from simulated angle and speed; E_star is the defined upright-rest target. E_p excludes arm kinetic energy and the kinetic cross term. Define desired power P_star and a tunable energy-correction timescale tau_E in seconds:

$$
P_\star(x)=\frac{E_\star-E_p(x)}{\tau_E}.
$$

P_star is positive when energy should be added and negative when it should be removed; its unit is J/s (watts), not N m. Setting tau_E equal to the decision period Delta=0.1 s expresses an attempt to remove the entire energy error in one action. This can saturate the controller repeatedly. A larger tau_E requests a partial correction; its value remains unfrozen. There is no claim that the requested power is achievable everywhere.

For comparison, full coupled mechanical energy E_total satisfies

$$
\dot E_{\rm total}=u\omega-b_r\omega^2-b_p\nu^2,
$$

where b_r=0.001 N m s is configured arm damping. Motor power is u times arm speed omega. Even this total-energy identity cannot be inverted safely at zero arm speed, and it is not the energy quantity in the current reward.

For the pendulum-relative energy, differentiating the model gives

$$
\dot E_p=-B\cos\alpha\,\nu\,\ddot\theta
+J_p\sin\alpha\cos\alpha\,\omega^2\nu-b_p\nu^2
=a(x)+b(x)u.
$$

Here a(x) is the model-computed pendulum energy rate at zero shaft torque, in watts; b(x) is the signed sensitivity of that rate to shaft torque, in inverse seconds. These are derived functions, not learned or tunable parameters. Explicitly define the arm-side inertia A, coupling C, determinant D, and zero-input generalized forces f_theta and f_alpha:

$$
A=J_0+J_p\sin^2\alpha,\quad C=B\cos\alpha,\quad D=AJ_p-C^2,
$$
$$
f_\theta=-b_r\omega-2J_p\sin\alpha\cos\alpha\,\omega\nu+B\sin\alpha\,\nu^2,
\quad f_\alpha=-b_p\nu+J_p\sin\alpha\cos\alpha\,\omega^2-G\sin\alpha,
$$
$$
a(x)=\nu\left[J_p\frac{Af_\alpha-Cf_\theta}{D}+G\sin\alpha\right],\qquad
b(x)=-\frac{J_p C\nu}{D}.
$$

J_0 is `MODEL.base_inertia_kg_m2`, derived from configured arm and pendulum geometry. A and C have units kg m², D has units kg² m⁴, and both f terms are torques in N m. D is positive for this configured plant. The equations match `jax_dynamics.state_derivative`; the symbols here are explanatory and do not propose separate one-use code helpers.

A compact analytic heuristic is the regularized inverse

$$
u_h(x)=\operatorname{clip}\left(
\frac{b(x)[P_\star(x)-a(x)]}{b(x)^2+b_0^2},-U,U\right).
$$

The positive tunable sensitivity scale b_0, in inverse seconds, suppresses division by weak coupling. Clip limits the heuristic shaft torque to the policy interval. This expression is the bounded minimizer of squared power-tracking error plus b_0² times torque squared, before the arm filter. Its sign follows swing phase and coupling, not just the sign of the energy deficit.

At zero pendulum speed or exactly horizontal, b(x)=0; this instantaneous heuristic cannot initiate energy transfer there. In particular it returns zero at exact downward rest. RL exploration can supply a starting perturbation, but a standalone heuristic benchmark must explicitly disclose this limitation. Do not hide it with an arbitrary velocity denominator.

A model-based alternative is scalar finite-step inversion. Define F_Delta(x,u) as the existing five-step RK4 prediction under one constant torque, then choose torque to match a reachable fraction of the energy error:

$$
\eta=1-e^{-\Delta/\tau_E},\qquad E_{\rm target}(x)=E_p(x)+\eta[E_\star-E_p(x)],
$$
$$
u_h(x)\in\arg\min_{|u|\le U}
\left[\frac{E_p(F_\Delta(x,u))-E_{\rm target}(x)}{E_\star}\right]^2.
$$

Eta is a computed dimensionless correction fraction in (0,1); the scalar objective is dimensionless. Equal minima require an explicit deterministic tie rule, for example smallest absolute torque followed by positive sign. This finite-step approach can account for a torque starting motion from rest, unlike instantaneous power inversion; it is still approximate numerical minimization of a potentially nonconvex function. Batched torque sampling followed by local scalar refinement is sufficient for the proof of concept. This is an alternative baseline, not a requirement to implement both heuristics immediately. I favor testing this alternative because the accepted action lasts a finite 100 ms and the predictor already exists; the analytic form remains the clean reference derivation.

Neither energy-only heuristic uniquely identifies upright capture. Upright rest and a fast downward passage share energy; energy error is also weak near the desired equilibrium. With an energy-only baseline, RL must learn capture, phase-sensitive correction, damping and arm centering. A local stabilizing baseline could be added later if those demands defeat small residuals, but that would be an explicit additional design choice.

### Residual, filtering, and critic semantics

Let pi_phi(s) be the actor's dimensionless residual output in [−1,1], with trainable parameters phi. Let rho>0 be a tunable dimensionless residual authority factor, and delta_u its torque correction:

$$
\delta u=\rho U\,\pi_\phi(s),\qquad
u_{\rm proposed}=\operatorname{clip}(u_h(x)+\delta u,-U,U),\qquad
u_{\rm applied}=\mathcal F(x,u_{\rm proposed}).
$$

The deterministic arm filter F uses current state and forward prediction and returns applied shaft torque. At zero actor output, the policy follows the filtered heuristic. Initialize the residual near zero and initially explore with small perturbations. The final residual range remains open: rho=1 can cancel a saturated heuristic but cannot reverse it to the opposite cap; rho=2 can reach any total torque in [−U,U] from any heuristic value. Large residual authority need not mean large initial perturbations. Clipping creates redundant/saturated residual commands, so log saturation and check learning gradients before freezing the parameterization.

Filter order is essential: first combine the heuristic and residual, then enforce the cap and arm prediction. Filtering only the heuristic would allow the residual to undo the limit handling. Preserve the physical clip in the environment.

The proposed arm-filter behavior is explicit:

- Accept the combined command if the predicted arm trajectory is admissible.
- When it is not admissible, use zero if the predicted coast trajectory is admissible.
- Otherwise use an inward braking command, selected using the same dynamics and cap.
- If no command avoids an excursion, select the command minimizing predicted outward excursion, record that condition, and continue under the existing arm-position penalty. Do not claim guaranteed constraint satisfaction.

Check all five physics samples, not only the 100 ms endpoint. Near the boundary, also inspect the ability to stop under subsequent capped braking: one-step coasting can look acceptable yet leave unavoidable momentum for the next step. The exact braking prediction length remains to be frozen. Sampling-based checking is a simulator-level guard, not a continuous-time safety proof. The arm threshold remains ±pi/2 rad; an already-outside state needs recovery behavior rather than an endlessly repeated zero command. Do not add a reward penalty or terminal failure merely because the filter intervenes.

Use critics Q_i(s,u), i=1,2, for the full undiscounted finite-deadline reward, conditioned on actual total shaft torque. The actor chooses a residual, but its proposed torque and every bootstrap target must pass through the same action composition/filter. Replay should retain proposed torque, applied torque and intervention status; fit the critic to the applied action that generated the observed transition. Hard filtering has flat/discontinuous regions, so actor-gradient handling there must be explicitly specified in the development plan. An alternative critic over residual commands is valid, but it would not directly be the physical-action Q prior wanted by future MPC.

The seven existing Markov features remain sufficient for a memoryless heuristic and filter; the heuristic torque may be supplied as an additional deterministic feature if useful, but is not required for Markovianity. Last-action memory is unnecessary unless switching costs or action-history dependence are introduced. A provisional small learner is a 128×128 SiLU actor (seven state inputs, one residual output) and two 128×128 SiLU critics (seven state inputs plus one normalized applied torque, one value output each), with corresponding target networks. These sizes are starting choices, not established optima; no software class or method structure is frozen here.

## Proposed milestones and open decisions

| Stage | Work | Validation and success flag |
|---|---|---|
| 1: physics and cap | Choose the energy inversion, screen smaller caps with the heuristic alone | Finite commands at rest/horizontal, correct pumping/removal where controllable, documented behavior at arm boundaries; show measured 20 s reachability before claiming the cap sufficient |
| 2: residual interface | Implement the agreed continuous learner and consistent torque filter | Zero residual reproduces the filtered baseline; cancellation/reversal matches the agreed authority; replay and bootstrap actions match executed physics |
| 3: controlled comparison | Heuristic alone, continuous actor from scratch, and residual actor at the same cap/reward/budget | Residual improves held-out capture/return across paired seeds, with state, action, energy and override plots; do not infer success from value loss |
| 4: task gate | Reuse tight/near and eventual broader validation | Existing 90% tight / 80% near gates twice; preserve the final set until the stage gates pass |

Keep the implementation compact: reuse the predictor, reward, collector, evaluator and workflow; avoid thin wrappers, unnecessary configuration and unrelated abstractions. Before implementation, freeze a handful of task-oriented objects/functions and their parameters, each with a concrete responsibility and files of 40–300 lines. No such code changes or training begin in this discussion.

Remaining decisions are (1) replacing the impossible horizontal-hold cap definition with measured recovery calibration; (2) whether residual authority may cancel and reverse a saturated heuristic or must remain a strict small correction; and (3) analytic power inversion versus a small finite-step scalar solve for the baseline. Inward braking is already accepted. The preferred discussion starting point is recovery-calibrated U, finite-step energy inversion, and small initial residuals with eventual cancellation/reversal authority.
