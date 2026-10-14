# Phase 10 Formulation: Five-Action Local Capture

Version-2 historical formulation below. The current accepted reward and termination contract is [version 3](dense_reward_proposal.md): five direct dense terms, nonterminal arm excursions, and early success after 100 ms. Action magnitudes and goal tolerances remain as recorded here.

## Human quick reading

Phase 9 showed that the three actions `off`, `negative full`, and `positive full` are too coarse at the 100 ms decision cadence. The strongest diagnostic learner captured only one of 512 tight starts, while representative trajectories accumulated many pendulum rotations under repeated maximum torque.

This revision retains the physical maximum as a safety clip but does not expose it as a learner action. The indices are ordered as `off`, `negative pump`, `positive pump`, `negative fine`, and `positive fine`. Pump torque is 45% of the physical limit; fine torque is 2%. Existing sign indices remain stable, while old three-output checkpoints are explicitly incompatible with the new network.

The success box is widened enough to accommodate one measured fine pulse lasting a complete 100 ms decision. Success still requires five consecutive 20 ms samples inside the box. This is a show-of-concept capture criterion rather than a hardware safety claim.

“Tight” continues to mean the fixed local reset/evaluation distribution around upright. It is larger than the success box and therefore does not make capture automatic. A “wave” continues to mean one parallel experiment round, normally eight independent trials on eight GPUs with one controlled parameter family under comparison.

## Machine-scannable contract

### F10.0 Revision identity

| Key | Fixed value |
|---|---|
| `contract_id` | `rotary-q-prior-v2` |
| `parent_contract_id` | `rotary-q-prior-v1` |
| `network_id` | `mlp-7-128-128-5-tanh-v2` |
| `decision_period_s` | `0.10` |
| `physics_period_s` | `0.02` |
| `deadline_s` | `20.0` |
| `goal_hold_steps` | `5` physics samples, equal to `0.10 s` |

All phase-9 results remain evidence about version 1. They must not be pooled with version-2 training results.

### F10.1 Action set

Let $u_{\max}$ be the unchanged physical clip, currently $0.0204\ \mathrm{N\,m}$. Define pump fraction $p=0.45$, fine fraction $f=0.02$, pump torque $u_p=p u_{\max}=0.00918\ \mathrm{N\,m}$, and fine torque $u_f=f u_{\max}=0.000408\ \mathrm{N\,m}$. The ordered learner action array is

$$
\mathcal U=[0,-u_p,+u_p,-u_f,+u_f].
$$

| Index | Name | Normalized torque | Current torque |
|---:|---|---:|---:|
| `0` | `off` | `0` | `0 N m` |
| `1` | `negative_pump` | `-0.45` | `-0.00918 N m` |
| `2` | `positive_pump` | `+0.45` | `+0.00918 N m` |
| `3` | `negative_fine` | `-0.02` | `-0.000408 N m` |
| `4` | `positive_fine` | `+0.02` | `+0.000408 N m` |

Random exploration is uniform over all five indices. The one-step guided proposal scores all five. Greedy ties retain lowest-index selection. Powered duration counts every nonzero action equally; this phase does not introduce magnitude-weighted energy cost.

A reversal occurs only when two consecutive nonzero torque values have opposite signs. A change between pump and fine torque of the same sign is not a reversal. Off-to-on retains its version-1 definition.

The pump fraction comes from a fixed 256-start downward-swing diagnostic using the same three-decision potential planner later used for comparison. At `0.45`, mean peak swing-energy ratio was `0.919`, mean peak absolute unwrapped pendulum angle was `1.827 rad`, and no arm violation occurred. At `0.60`, mean and 95th-percentile peak pendulum magnitudes rose to `10.54 rad` and `24.88 rad`, indicating renewed multi-rotation behavior. These measurements select a show-of-concept action magnitude; they do not prove that 45% is globally optimal.

### F10.2 Success box

Let $\beta=\operatorname{atan2}(\sin(\alpha-\pi),\cos(\alpha-\pi))$ be wrapped pendulum error from upright. A physics sample is inside the revised goal exactly when

$$
|\theta|\leq0.08\ \mathrm{rad},\quad
|\beta|\leq0.08\ \mathrm{rad},\quad
|\omega|\leq0.15\ \mathrm{rad/s},\quad
|\nu|\leq0.20\ \mathrm{rad/s}.
$$

The limits are configured task constants. They were selected from the 100 ms action-resolution study, not learned. From exact upright, one positive fine action produced

$$
(\Delta\theta,\Delta\beta,\omega,\nu)
=(0.006925,0.007439,0.136490,0.159761),
$$

in radians and radians per second as appropriate. Thus the fine pulse remains inside the revised box at its endpoint. A 10% pulse produced velocities near `0.682` and `0.798 rad/s`, so normalized `0.1` is too coarse for this role.

The tight distribution remains uniform over upright deviations

$$
(\theta,\beta,\omega,\nu)\in
[-0.08,0.08]\times[-0.12,0.12]\times[-0.15,0.15]\times[-0.30,0.30].
$$

It is an evaluation/reset box, not the success box. Its beta and nu ranges exceed the goal, so a tight reset is not automatically successful.

### F10.3 Unchanged learning semantics

The seven observation features, finite 20-second deadline, energy-plus-capture potential, base reward, potential-difference shaping, Double DQN target, uniform replay, curriculum, and validation gates retain the version-1 definitions. The capture potential automatically uses the revised four goal tolerances. The network now returns five shaped finite-deadline action values.

Value audit evaluates every first action. The fixed three-decision planner enumerates $5^3=125$ sequences. It executes the first action and replans. The planner horizon remains `0.30 s`; only branching changes.

### F10.4 Evidence boundary

The local action-resolution study enumerates every five-decision sequence from 128 fixed tight starts and 128 fixed near-upright starts for each tested fine fraction, using the selected 45% pump pair. The pump-resolution study evaluates 256 fixed downward starts for each candidate pump fraction. These are deterministic controller diagnostics. They do not prove trainability, global optimality, or real actuator resolution.

Version-1 checkpoints have three output values and cannot be loaded as version-2 checkpoints. Version 2 starts from fresh network, optimizer, and replay state. No output padding or checkpoint fallback is permitted.

### F10.5 After local capture succeeds

First repeat the phase-9 curriculum and deployment comparisons under version 2. If the five-action agent passes local and benchmark gates, use it as the discrete Q prior for the first sampled MPC experiments. Then add previous applied torque to the state before introducing switch-dependent costs or learned on/off switch values. Continuous action Q and joint Q/MPC training remain later contract revisions.
