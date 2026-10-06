# Phase 9 Formulation: Three-Action Q Prior

## Human quick reading

The agent chooses one torque every 100 ms: full negative, off, or full positive. It observes the rotary mechanism, time remaining, and progress through the required upright hold. It learns how useful each action is when followed by its learned continuation policy.

The physical objective is to capture upright before the 20-second deadline while using little powered time. A weak time cost discourages unnecessary delay. Energy guidance rewards movement toward the upright kinetic-plus-potential energy level; a local capture term also distinguishes slow, centered arrival from a fast upright crossing. Success still requires all four physical coordinates inside the existing tolerances for 100 ms.

We learn complete Q-values from scratch. One experiment uses ordinary random exploration; another sometimes uses a one-step energy/capture proposal during exploration. Both use the same learning target, action set, reward, and evaluation. The heuristic can accelerate experience collection without becoming a compulsory control law.

The learned network estimates the shaped training return. Before comparing values at different predicted states, remove the known shaping offset. Before comparing switch times, also include the powered/coast costs already incurred to reach each candidate. These distinctions make the prior interpretable in later planning.

This phase includes a short, fixed-horizon planning comparison. Continuous torque, switching penalties, optimized switch times, and joint controller learning begin only after the current gates pass. The exact implementation and experiment schedule are in [development_plan.md](development_plan.md). [NN-design.md](NN-design.md) specifies the network, dedicated workflow, and removal of the previous rotary PPO implementation before building this learner from scratch.

## Machine-scannable specification

### F0. Authority, status, and change policy

`contract_id = rotary-q-prior-v1`. This is a frozen initial formulation, not a claim of implemented or demonstrated behavior. Fixed semantics are normative. The explicitly identified coefficients are initial tunable choices; every run saves their resolved values and a reward revision. Changing the action set, clocks, observation semantics, terminal rules, or mathematical reward structure requires a new contract revision and separate results.

### F1. Physical state, action, and terminal events

The physical state is computed by the existing nominal simulator:

$$
x=[\theta,\alpha,\omega,\nu]^T.
$$

Here theta is the unwrapped arm angle in radians; alpha is the unwrapped pendulum angle in radians measured from downward; omega and nu are the corresponding angular velocities in radians per second. All are observed exactly. Nominal masses, lengths, gravity, damping, and torque limit come from [physics.yaml](../../src/rotary_pendulum/configs/physics.yaml); derived quantities come from `derive_model`. No physical randomization or observation noise is introduced.

| Contract key | Fixed value / interpretation |
|---|---|
| `physics_dt_s` | `0.02`, one existing RK4 physics interval |
| `hold_physics_steps` | `5`, one action lasts at most `0.10 s` |
| `max_physics_steps` | `1000`, an episode lasts at most `20 s` |
| `torque_limit_nm` | Read from physics configuration; current value `0.0204` |
| `action_torques_nm` | Ordered array `[0, -torque_limit_nm, +torque_limit_nm]` |
| `action_index` | Integer `0`, `1`, or `2`; order gives off-first tie breaking |
| `goal_tolerances` | Existing four tolerances, each currently `0.05` in its state coordinate's unit |
| `goal_hold_steps` | Existing `5` consecutive inside-goal physics samples |
| `arm_limit_rad` | Existing unwrapped absolute arm-angle boundary `pi/2` |
| `event_precedence` | Arm violation, then success, then timeout |
| `terminal_bootstrap` | Zero on success, arm violation, and timeout |
| `post_terminal_step` | No state change and no new reward; do not insert repeated terminal transitions |

The set is the requested negative-full/off/positive-full action space; its storage order is only an interface convention. Greedy and heuristic ties choose the lowest index exactly; no near-tie tolerance is introduced. No small actions, action interpolation, switching charge, reversal charge, or total powered-time constraint exists in version 1.

Use `step` for reward-bearing transitions and event checks. Use `rk4_step` only for plant-only diagnostics. The inherited arm check is sampled at 50 Hz and is not a continuous-time safety certificate.

### F2. Markov state and observation

Let n be `physics_steps`, c be `goal_count`, and z be the augmented state `(x,n,c)`. The existing terminal flags accompany z for masking; they are not extra network inputs. Define the upright error beta and the seven-element observation o:

$$
\begin{aligned}
\beta&=\operatorname{atan2}(\sin(\alpha-\pi),\cos(\alpha-\pi)),\\
o(z)&=[\theta/\theta_{\max},\sin\alpha,\cos\alpha,
\omega/s_\omega,\nu/s_\nu,(1000-n)/1000,c/5],\\
s_\omega&=\theta_{\max}\omega_0,\qquad s_\nu=2\sqrt{G/J_p}.
\end{aligned}
$$

Theta_max is the fixed arm limit; omega_0 is `MODEL.natural_frequency_rad_s`, computed by `derive_model`; s_omega and s_nu are fixed velocity scales in radians per second. J_p is `MODEL.pendulum_inertia_kg_m2`; G is `MODEL.gravity_torque_nm`, numerically the gravity-energy coefficient in joules. Their physical definitions appear in F3. Observation order and scales are checkpoint metadata. Do not clip or adaptively normalize these features. Pendulum periodicity is represented by sine/cosine; the arm is never wrapped.

Previous action is absent because neither dynamics nor version-1 reward depends on it. Episode totals are logging state, not policy inputs. Reset sets n and c to zero, including resets that happen to start inside the goal.

### F3. Energy and capture potential

For the nominal uniform pendulum, m_p is configured pendulum mass in kilograms, L_p is its full length in meters, g is configured gravity in meters per second squared, and l is its computed center-of-mass distance. Define:

$$
l=L_p/2,\qquad J_p=m_pL_p^2/3,\qquad G=m_pgl,
\qquad E_s(x)=\tfrac12J_p\nu^2+G(1-\cos\alpha),\qquad E_\star=2G.
$$

E_s is the computed pendulum-relative swing energy in joules, consisting of kinetic and downward-referenced potential energy. E_star is upright-rest swing energy. This diagnostic excludes the full arm/pendulum kinetic coupling; it is intentionally not the complete mechanism's mechanical energy. Use the existing derived constants rather than maintaining a second parameter calculation in the learner.

Let epsilon_theta, epsilon_beta, epsilon_omega, and epsilon_nu be the existing goal tolerances. Define dimensionless energy error e_E, mean squared tolerance-scaled capture error q_c, and bounded cost potential C:

$$
\begin{aligned}
e_E(x)&=(E_s(x)-E_\star)/E_\star,\\
q_c(x)&=\tfrac14\left[(\theta/\epsilon_\theta)^2+(\beta/\epsilon_\beta)^2
+(\omega/\epsilon_\omega)^2+(\nu/\epsilon_\nu)^2\right],\\
C(x)&=\frac{e_E(x)^2}{1+e_E(x)^2}
+w_c\frac{q_c(x)}{1+q_c(x)}.
\end{aligned}
$$

The initial capture weight w_c is `1.0`, a dimensionless tuning choice. The first term values approaching the target energy from either side. The second resolves errors relative to the actual capture tolerances and is bounded even far from upright. Both terms remain active; no energy gate or phase-dependent reward is introduced. C is dimensionless, nonnegative, and less than `1 + w_c`. Its mean squared capture error is guidance only; success uses all four separate tolerances and the dwell counter.

Define the reward potential Phi as negative C on active states and exactly zero on every terminal state, with a fixed conversion of one reward unit per unit of C:

$$
\Phi(z)=\begin{cases}-C(x),&z\text{ active},\\0,&z\text{ terminal}.\end{cases}
$$

Do not reward positive energy gain alone, clip negative progress to zero, or retain a nonzero potential at timeout/failure. An energy cycle cannot accumulate net shaping reward when its endpoints have equal potential.

### F4. Base reward, shaping, and duration accounting

Decision j begins in active state z_j, selects torque u_j from the fixed action array, and ends at z_(j+1). Let delta_t_j be the actual elapsed seconds, calculated from the change in physics-step count. Let I_s, I_f, and I_t indicate newly reached success, arm failure, and timeout, respectively; only one can be true. Let the indicator `1{condition}` equal one when its condition is true and zero otherwise. Then:

$$
\begin{aligned}
\delta t_j&=0.02(n_{j+1}-n_j),\\
r_j^{\rm base}&=R_s I_s-R_f I_f-R_t I_t
-\lambda_{\rm on}\delta t_j\mathbf1\{u_j\ne0\}-\lambda_{\rm time}\delta t_j,\\
r_j^{\rm train}&=r_j^{\rm base}+\Phi(z_{j+1})-\Phi(z_j).
\end{aligned}
$$

| Coefficient | Initial value | Unit / role |
|---|---:|---|
| R_s / `success_reward` | `5.0` | Reward units awarded once at success |
| R_f / `arm_failure_cost` | `5.0` | Reward units subtracted once at arm failure |
| R_t / `timeout_cost` | `2.0` | Reward units subtracted once at timeout |
| lambda_on / `on_cost_per_s` | `0.05` | Reward units per actual powered second; tunable |
| lambda_time / `time_cost_per_s` | `0.005` | Reward units per actual elapsed second; tunable |

Terminal amounts are fixed for initial experiments; among reward settings the development plan permits tuning only the two rate costs and the F3 capture weight. The maximum initial duration charge over 20 seconds is `1.1`. Successful base episode returns therefore lie within `[3.9,5.0]`, arm-failure returns within `[-6.1,-5.0]`, and timeout returns within `[-3.1,-2.1]`. These are full-episode inclusive bounds from a fresh reset, not assertions that every endpoint is attainable. Shorter continuation returns use their own remaining time. Scalar expected-return optimization is not strict lexicographic optimization of success probability.

A failure after two physics intervals incurs `0.04 s` of time and, if powered, on-time, rather than a full `0.10 s`. Component logging order is `[energy_shaping, capture_shaping, on_cost, time_cost, terminal]`; the first two are potential differences of the corresponding F3 terms, with both terminal potentials zero. Their sum is r_train. Also log r_base separately.

Shaping telescopes over every complete episode: total training return equals total base return minus the initial Phi. Thus it changes learning guidance while preserving full-episode action preferences from a given start. This uses the potential-difference construction of [Ng, Harada, and Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf); the finite deadline guarantees termination here. The chosen energy and capture functions are this project's design, not results claimed by that paper.

### F5. Value meaning and Double DQN update

The discount gamma is fixed at `1.0`. For policy pi, Q_base^pi is the expected sum of remaining base rewards after choosing the specified first action and following pi. Q_train^pi uses training rewards instead. The finite deadline is part of z. An asterisk denotes the optimal value over permitted future actions, which learning attempts to approximate:

$$
Q_{\rm base}^{\pi}(z,a)=\mathbb E_\pi\!\left[\sum_{j=0}^{T-1}r_j^{\rm base}\mid z_0=z,a_0=a\right],
\qquad Q_{\rm train}^{\pi}(z,a)=Q_{\rm base}^{\pi}(z,a)-\Phi(z).
$$

Here a is an action index and T is the number of decisions until the first terminal event; partial final holds count as one decision. The expectation concerns policy/randomness; the nominal plant is deterministic. The neural network outputs three Q_train estimates. There is no separate actor or value network.

Let w be the learned online network weights, w_minus the target copy, d the transition's terminal flag, and `(o,a,r_train,o_next,d)` a replay sample. Define a_star as the maximizing online next action and y as the stopped-gradient target:

$$
a_\star=\arg\max_b Q_w(o_{\rm next},b),\qquad
y=r^{\rm train}+(1-d)Q_{w^-}(o_{\rm next},a_\star).
$$

The index b ranges over the three actions. On terminal samples implement y as r_train without relying on multiplying an invalid next value by zero. Optimize the minibatch mean Huber loss of `Q_w(o,a) - y`, with Huber threshold `1.0` reward unit. The loss is half squared error below that absolute threshold and absolute error minus one-half above it. Gradient norm is clipped before Adam; exact settings appear in D3 of the development plan. Double DQN separates target action selection and evaluation as described by [van Hasselt, Guez, and Silver](https://arxiv.org/abs/1509.06461).

For active z, the export conversion is the first expression below. The second defines the terminal state value; all exported action values at a terminal z are also defined as zero:

$$
\widehat Q_{\rm base}(z,a)=Q_w(o(z),a)+\Phi(z),\qquad
\widehat V_{\rm base}(z)=\begin{cases}\max_a\widehat Q_{\rm base}(z,a),&z\text{ active},\\0,&z\text{ terminal}.\end{cases}
$$

Hats denote learned estimates, not certified values. Since Phi is action-independent, conversion preserves action ranking at one state. It is essential for comparisons across different states. Terminal network outputs are not trained as independent state values; always apply the terminal mask.

### F6. Exploration and replay semantics

At each decision choose the greedy online action with probability `1 - epsilon`. In the exploration branch choose the heuristic with probability h and a uniformly random action with probability `1 - h`. Epsilon is the scheduled exploration probability; h is `0` for ordinary exploration and `0.5` for guided exploration. Both values are experiment definitions, not learned quantities.

The heuristic evaluates all three candidate actions for one existing environment decision and chooses the largest immediate r_train from F4. Compute candidates in a separate batch axis, retain event semantics, and do not reset the hypothetical candidates. This is an energy/capture-guided proposal including duration and terminal costs. It does not estimate a long-horizon Q-value. Compute it only in the guided family; evaluation never uses exploratory proposals.

Only the chosen real transition enters replay. Candidate branches are not extra training samples. The learner target maximizes over all three actions regardless of the proposal. No heuristic pretraining, residual critic, teacher loss, prioritized replay, or multi-step return is included.

Store the actual terminal next observation before automatic reset. Afterwards reset completed lanes and clear their episode statistics. The next rollout iteration uses the reset observation; the stored transition never connects the old episode to its successor. Uniform replay samples initialized slots independently with replacement.

### F7. Deadline and present deployment contract

Success and failure end the task immediately; timeout is terminal at 20 seconds. Do not bootstrap beyond it or reset remaining time within an episode. Longer physical diagnostics may be labeled separately in a later revision but do not contribute version-1 targets or success statistics.

The present deployment experiment enumerates all `3^3 = 27` three-decision sequences, a maximum `0.30 s` lookahead, and replans after executing the first action. It uses the same event-aware transition, augmented state, and reward. For a sequence A of length H=3, let z_k be its predicted augmented state after k decisions. Its score is:

$$
S(A)=\sum_{k=0}^{H-1}r_k^{\rm base}+B(z_H).
$$

B is the selected terminal score: zero, Phi for the handcrafted-potential comparator, or V_hat_base for the learned-prior planner. All are zero at a terminal state. No rewards accrue after a predicted terminal event; duplicate terminated branches are allowed and tied sequences use lexicographic action-index order. Enumerate future indices in the F1 storage order. Include greedy Q as a fourth controller. This is a fixed-horizon discrete planning test, not the later burst/coast switching optimizer. Measured benefit must include the handcrafted-potential comparison, not only the weak zero-terminal baseline.

### F8. After this phase succeeds

The next task is a richer or continuous torque critic. Preserve duration accounting and return units while extending action representation; a three-output network cannot be treated as a continuous-action critic without retraining or distillation. Retest local capture and the value audit.

Then define switch-on count separately from powered duration. If a switching cost depends on prior torque, append previous applied torque as the last observation element and define near-zero/on thresholds explicitly. Any hard total-use budget additionally requires remaining budget in the state. These changes need a new reward/observation contract.

The desired later controller has an outer search over bang-off and off-bang times, with an inner optimization over feasible near-future torque sequences. For a candidate switch boundary k, let P_k be the predicted base reward accumulated before that boundary and z_k its augmented state. Values for the first action after switching enter as:

$$
S_{\rm bang\to off}(k)=P_k+\widehat Q_{\rm base}(z_k,0),\qquad
S_{\rm off\to bang}(k)=P_k+\max_{a\in\{1,2\}}\widehat Q_{\rm base}(z_k,a).
$$

Indices refer to F1's discrete array only as an illustration; later continuous actions require the revised critic. P_k contains all costs before the switch, while Q contains the first post-switch action and continuation; do not count that action twice. These expressions provide candidate scores under the learned continuation, not exact values of future mandatory burst/coast schedules. Enforcing those schedules requires compatible continuation rollouts or a schedule-conditioned value. Never choose a switch by comparing raw shaped Q across states without the prefix and shaping conversion.

Finally, collect transitions from the sampled MPC controller, fine-tune the compatible critic, and compare frozen-prior versus jointly trained control under the same physical task and compute budgets. Benchmark nonlinear JAX prediction first; adopt local affine approximations only if measured speed and error justify them. None of these extensions is implemented in phase 9.
