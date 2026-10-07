# Dense swing-up and hold reward — version 3 contract

## Human quick reading

Status: accepted and implemented as `rotary-q-prior-v3`, reward revision 1; not trained. This supersedes the phase-10 version-2 reward and termination semantics. The implementation plan is in `dense_reward_development.md`; the complete baseline experiment is in `machine-scannables/dense_experiment.json`.

The proposed five terms better express sustained control: penalize energy error, torque magnitude, delayed recovery, and arm displacement; reward upright residence. The old reward is already Markovian. Its limitation is that energy/capture potential differences mostly redistribute learning feedback while retaining the sparse terminal objective. Changing the running reward deliberately changes what the controller optimizes.

Accepted user decisions: arm-boundary crossings do not terminate or trigger a separate failure penalty; the existing arm-displacement term continues to grow. Smooth partial upright credit is used for learning. Successful sustained holding ends the episode early, avoiding subsequent running costs. These decisions supersede the earlier fixed-duration proposal.

With variable episode duration, an unconditional increasing time penalty encourages earlier completion. Its minimum rate is greater than the maximum upright-credit rate, so remaining active always incurs a net cost. Otherwise a controller could earn positive credit near upright while avoiding the strict success condition. This preserves exactly five terms. The accepted success hold duration is 0.1 s: five consecutive physics samples.

Use normalized energy **error**, not positive normalized energy. Upright at rest and a fast downward passage can have the same pendulum energy. The upright reward must distinguish them using pendulum angle and speed. Including arm speed in that same reward discourages an upright pendulum carried by a rapidly moving arm.

A smooth upright score provides credit near the target. It is a reward surrogate, not proof of holding. Use hard tolerance and dwell measurements for validation and successful termination, without a hard success reward. Accumulate reward at each existing 20 ms physics sample while holding an action for up to 100 ms, so a good endpoint cannot hide poor behavior during the action.

## Machine-scannable formulation

### D1. State, constants, and provenance

The physical state, time-augmented decision state, and upright error are

$$
x=(\theta,\alpha,\omega,\nu),\qquad s=(x,t,d),\qquad
\beta=\operatorname{atan2}(\sin(\alpha-\pi),\cos(\alpha-\pi)).
$$

The simulator computes all quantities: arm angle theta and pendulum angle alpha in radians; arm speed omega and pendulum speed nu in rad/s; episode elapsed time t in seconds. Theta is unwrapped and centered at zero. Alpha is measured from downward; beta is wrapped error from upright. The counter d is the number of consecutive inside-goal physics samples: increment after an inside sample, reset to zero after any outside sample, initialize to zero. It must be observed because termination depends on it. No last-action memory or arm-failure state is needed for these five terms.

| Symbol | Value | Source and meaning |
|---|---:|---|
| T | 20 s | Existing task horizon |
| delta | 0.02 s | Existing physics sampling interval |
| Delta | 0.10 s | Existing decision interval; five physics samples |
| J_p | 0.000133128 kg m² | Derived model pendulum inertia |
| G | 0.01518588 J | Derived gravitational energy coefficient |
| E_star | 0.03037176 J | Computed as 2G; upright-rest swing energy |
| u_p | 0.00918 N m | Existing pump action magnitude |
| u_f | 0.000408 N m | Existing fine action magnitude |
| theta_lim | pi/2 rad | Arm penalty scale and reporting threshold; not a termination boundary |
| b | 0.16 rad | Proposed tunable upright-score width, twice current angle tolerance |
| v | 0.40 rad/s | Proposed tunable pendulum-speed width, twice current tolerance |
| w | 0.30 rad/s | Proposed tunable arm-speed width, twice current tolerance |

Action order remains zero, negative pump, positive pump, negative fine, positive fine. Physical torque clipping remains at 0.0204 N m. Widths are heuristic starting choices, not new success tolerances or experimentally validated values.

### D2. Energy error and upright score

$$
E(x)=\tfrac12J_p\nu^2+G(1-\cos\alpha),\qquad
e(x)=\frac{E(x)-E_\star}{E_\star},
$$

$$
h(x)=\exp\left[-\tfrac12\left(
(\beta/b)^2+(\nu/v)^2+(\omega/w)^2\right)\right].
$$

E is pendulum-relative swing energy, not total coupled mechanism energy. The dimensionless signed error e is zero at the desired energy, -1 at downward rest, and positive above target energy. Its absolute value penalizes equally sized deficits and excesses. This avoids the saturation of the old bounded potential and avoids squaring energy error, which can scale as the fourth power of speed.

The dimensionless score h lies between zero and one. Its maximum requires upright pendulum and zero pendulum/arm speed. Arm centering is handled separately. At one width in one coordinate and zero in the others, h is about 0.607; at two widths it is about 0.135. This is partial upright-quality credit, not a binary success declaration. It is continuous across the beta branch cut because beta is squared, although its derivative need not be continuous there.

Energy alone is ambiguous: at alpha=0 and nu=sqrt(4G/J_p), approximately 21.36 rad/s, E equals E_star despite the pendulum pointing downward. Near upright rest, energy error is also second order in angle and speed and cannot provide independent feedback for each of them.

### D3. Exactly five running reward components

Define the reward rate, in reward units per second, as

$$
\ell(x,u,t)=
-c_E|e(x)|
-c_u\frac{|u|}{u_p}
-c_t\left(1+\frac{t}{T}\right)
-c_\theta\left(\frac{\theta}{\theta_{\rm lim}}\right)^2
+c_Hh(x).
$$

All c coefficients are nonnegative tunable reward rates. Proposed starting values, not a frozen or optimized setup:

| Coefficient | Initial rate | Role |
|---|---:|---|
| c_E | 1.0 | Ongoing absolute normalized energy-error cost |
| c_u | 0.02 | Small magnitude-weighted actuation cost |
| c_t | 1.05 | Increasing elapsed-time cost; proposed minimum exceeds c_H |
| c_theta | 0.20 | Smooth arm displacement cost |
| c_H | 1.0 | Upright-quality credit per second |

The time cost grows linearly and doubles over 20 s. Successful trajectories can now end at different times, so their total time penalties differ. The proposed coefficient condition c_t > c_H guarantees that every active sample has negative net reward, because all other terms are nonpositive and h <= 1:

$$
\ell(x,u,t)\leq c_H-c_t(1+t/T)<0.
$$

The upright component itself remains positive; better upright quality reduces the total cost. The condition prevents collecting positive net reward by lingering near success. It does not guarantee that every faster trajectory outranks every slower one: energy error, torque, and arm displacement remain real tradeoffs. No minimum-time optimality claim is made.

Torque cost is proportional to absolute magnitude: one pump second costs 0.02; one fine second costs about 0.000889. It is a control-effort preference, not mechanical work or electrical energy. Quadratic torque cost is possible but initially unnecessary. Absolute value is continuous, though not differentiable at zero; discrete-action Q learning does not require differentiating the reward with respect to action.

At ideal upright with centered stationary arm and zero torque, reward rate is -0.05 initially and -1.10 at 20 s. At downward rest with centered arm and zero torque, it is approximately -2.05 initially and -3.10 at 20 s. At half the arm reporting boundary, the arm term is -0.05 per second; at the boundary it is -0.20 and at twice the boundary -0.80. The arm term is not clipped at the boundary. These comparisons explain coefficient scale; they do not establish optimal control behavior.

For decision k, action u_k is held constant. Sum over its m_k active post-integration samples, where m_k is at most five and can be shorter at successful termination or the horizon:

$$
r_k=\sum_{j=1}^{m_k}\delta\,\ell(x_{k,j},u_k,t_k+j\delta).
$$

Here k indexes decisions; j indexes physics samples within the decision; x_k,j is the resulting sampled state; t_k is time at decision start. Endpoint quadrature is the specified numerical convention. Include the sample completing the hold or reaching the horizon; later absorbing samples contribute zero reward and zero elapsed time. Arm crossings do not shorten an action. An alternative midpoint convention is not implicitly allowed.

Use undiscounted finite-horizon return initially: gamma=1, where gamma is the RL discount factor. The learner observes time already via normalized remaining time. No potential difference, binary actuator-on cost, switch penalty, or timeout bonus is included in this candidate formula.

### D4. Accepted episode semantics

- Accepted: arm reporting-boundary crossings are nonterminal and add no separate penalty. Continue the same dynamics with unwrapped theta and the quadratic displacement cost; do not clamp or wrap theta at the boundary. Numerical invalidity, if encountered, is a simulator diagnostic error, not invented task success or failure.
- Accepted: smooth partial upright credit is the fifth reward component. Hard criteria define validation and the successful-stop event, not an extra reward term.
- Accepted: stop immediately after completing the required consecutive hold, or at T if it has not been completed. Success at the final physics sample counts as success. Neither event adds a terminal reward or penalty.
- The proposed termination and validation box is absolute theta/beta/omega/nu <= [0.08 rad, 0.08 rad, 0.15 rad/s, 0.20 rad/s], retaining the existing tolerances. Reward widths do not change these limits.
- Required hold duration H=0.1 s, equivalent to five consecutive physics samples. Observe d/5. Leaving the box resets d. Initial membership alone is not success.
- Record success fraction, completion time, decision-sampled hold count, integrated absolute torque, arm excursion fraction and maximum angle, decision-endpoint-sampled excursion duration, and all five reward components. Excursion duration weights each decision-endpoint predicate by its actual duration; it is not a continuous crossing-time estimate. Boundary excursions are metrics, not failure labels. Do not present an early-stopped trace as evidence of continued balancing.
- Enforce c_t > c_H as the coefficient constraint for strictly negative active reward. Baseline coefficients satisfy it without a sixth component.

### D5. Why potential shaping is optional

Potential shaping adds a difference of a chosen state score Phi:

$$
r'_k=r_k+\gamma\Phi(s_{k+1})-\Phi(s_k).
$$

Here r is the underlying task reward, r-prime is the shaped learning reward, and Phi is an arbitrary scalar state potential. This mathematical potential is different from gravitational potential energy. With gamma=1 and terminal potential zero, the sum of the added terms is minus the initial potential. For a fixed initial state, it does not change full-episode policy ranking. Appropriate potential shaping was chosen to supply intermediate feedback while preserving the previous base objective. [Primary reference: Ng, Harada, Russell (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

For this redesign we want to change the objective: a large energy error should be costly every second it persists. Direct negative energy error does that; it need not be converted to a potential. Positive E/E_star would instead reward excessive energy. A capture potential is also optional; the direct upright-quality reward supplies angle and speed information missing from energy alone.

Both the old and proposed rewards are Markovian when episode time and any reward/termination counters are included in state. Markovian does not mean continuous, dense, or effective. No evidence establishes that changing reward alone fixes the observed training failure.

### D6. Validation before a new campaign

1. Formulation milestone: H=0.1 s and c_t > c_H are fixed. Success flag: every transition and stopping event has a defined reward and dwell condition.
2. Reward milestone: check energy-error monotonicity, torque symmetry and magnitude ranking, increasing time cost, arm-displacement monotonicity beyond the reporting boundary, and negative active reward. Success flag: all analytical cases match computed components; same physical trajectory has consistent reward when regrouped into decisions; arm crossings continue and completed holds stop at the correct physics sample.
3. Controller diagnostic milestone: compare zero torque and short-horizon planning from fixed tight, near, and downward starts. Success flag: inspect component returns and trajectories; reject configurations that prefer lingering near success, violent energy overshoot, or upright fly-throughs over feasible sustained hold.
4. Learning milestone: train only after the preceding checks, using physical GPUs 0-3 only and at most four concurrent GPU trials. Plot all state coordinates including unwrapped alpha, torque, five reward components, dwell and return after each training task; inspect before iterating. Confirm per-lane rollout exposure extends through the full horizon. Success flag: improvement over baselines in hard hold metrics across seeds; report arm excursions separately; numerical promotion thresholds must be frozen before training.

Implementation structure is enumerated in `dense_reward_development.md`. Existing checkpoints and replay targets are not interchangeable with the new reward. Base-return and train-return fields are retained as equal direct returns for reporting; there is no potential offset. Initial rollout lanes start from the near reset stratum; stage-0 automatic resets use the configured tight fraction. The baseline uses 4,096 lanes so the stage-0 budget covers 2,048 decisions per lane, allowing complete 20 s episodes. No training is included in this implementation.
