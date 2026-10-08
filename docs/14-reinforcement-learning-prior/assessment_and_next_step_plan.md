# Analytical controller as a reinforcement-learning prior

## Assessment

**Yes: the present controller is a strong behavioral prior for learning swing-up, and a useful reference policy against which to measure improvement.** Here “prior” means a controller that supplies the initial action behavior or demonstration data; it does not mean a Bayesian probability distribution or a proven optimal value function.

The selected controller captured all 4,096 random starts in two final validation seeds, including 1,024 downward starts without arm-limit crossings. Its physics-based energy calculation and fixed 100-millisecond action give a compact, interpretable starting point. This is stronger evidence for initialization than an untested action heuristic. See the [measured analysis](../13-single-action-controller/controller_analysis.md).

The limits matter: six moving-start validation episodes crossed the arm soft limit; capture takes about 11.1 seconds for the median downward case; and episodes stop after five angle-qualified samples. We have not established sustained balancing, robustness to model mismatch, or any sleep behavior. A controller that is good at reaching the goal is not automatically a good teacher for remaining upright with infrequent intervention.

Learning a correction to a conventional controller is supported as a method by [Johannink et al., Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201). [Silver et al., Residual Policy Learning](https://arxiv.org/abs/1812.06298) also studies corrections to hand-designed and model-predictive policies. These papers motivate the approach; they do not establish that this pendulum controller will improve with learning. That is a proposed experiment.

## Recommended next experiment

First learn a **torque correction at the existing 100-millisecond decision interval**, while freezing the analytical controller. Start with zero correction, so the initial learned controller exactly reproduces the analytical controller. This makes the role of the prior directly testable and preserves an interpretable comparison. It is an intermediate experiment toward the project's later goal of learning prediction horizons and terminal values, not a replacement for that goal.

Capture success alone offers little improvement signal on the tested initial-state classes because it is already saturated. Keep capture as the primary outcome, then assess shorter time to capture and fewer arm-limit crossings. Report motor work, peak arm angle and terminal velocities separately. Do not relabel a weighted reward as physical energy or silently treat the soft limit as a hard constraint.

Before training, specify the time, crossing and work terms and their normalizations in the experiment formulation. Their relative weights are tunable preferences, not physical constants. Do not add a speed condition to the existing angle-only goal merely to obtain a different learning signal. If sustained balancing is chosen instead, define a separate task that continues after first capture and measure time spent in the upright band.

The existing [residual control module](../../src/rotary_pendulum/RL/jax_residual_control.py) is **not an adapter for the current controller**. It still contains the older hinge-relative energy heuristic, a smaller policy torque cap, and an appended braking prediction. Reusing that module unchanged would test a different prior. Any future implementation must call the current energy policy and single-action decoder explicitly, and test zero-correction equivalence.

## Proposed action and observation formulation

Let x be the measured simulator state: arm angle, pendulum position, arm angular velocity and pendulum angular velocity, in radians and radians per second. Let uₕ(x) be the torque computed by the frozen analytical controller. Let zψ(o) be the scalar output of a learned network with parameters ψ, evaluated on observation o. Let b be the chosen positive residual-torque scale in newton metres, to be declared before training. Let U = 0.01836 newton metres be the existing usable torque magnitude. The proposed applied torque is:

$$
u(x)=\operatorname{clip}\bigl(u_h(x)+b\tanh(z_\psi(o)),-U,U\bigr).
$$

Here `clip` enforces the existing actuator command range and `tanh` bounds the correction. Initialize the network's output to zero; then its torque equals uₕ at every state, including saturation. Hold the resulting torque constant for 100 milliseconds. Keep the 20-millisecond plant integration and existing angle-only capture rule. The correction scale b is a future design choice; no value was selected or tested in this task. A saturated prior can still be reduced or reversed if the correction permits it, but cannot receive additional outward authority beyond U.

The observation should include arm angle, sine and cosine of pendulum position, both signed angular velocities, current goal-streak count, and remaining episode time. The trigonometric pair avoids an artificial input discontinuity at the 0/360-degree wrap. Goal streak is the number of consecutive angle-qualified samples so far; remaining time matters for the finite deadline. Declare all observation scales. Energies may be added as derived features, but energies alone omit phase, direction and arm position and are insufficient to choose torque. Logged pendulum position remains in the full-turn range.

A neural terminal value is a separate later option. Returns measured under this controller estimate the value of following this controller under a specified task and cost; they are not optimal returns. Do not train a terminal value on energy distance and present it as expected task return. First define the return, termination rule, time units and state information, then collect sufficient continuation data under the policies being evaluated.

## Development milestones and success checks

| Stage | Work | Validation and completion flag |
| --- | --- | --- |
| Preserve the reference | Keep current controller, reset distributions and physical clocks fixed; retain the portable examples linked below. | Exported samples match original logs, goals and timing; references remain reproducible. Completed for the animation bundle. |
| Define the learning task | Choose capture improvement or sustained balancing; specify reward, correction scale, observations and termination. | A written definition separates measured physical quantities from tunable costs. Proposed; training has not started. |
| Connect the current prior | Replace the old residual module's action source with the current controller in a separately scoped implementation. | Zero correction reproduces the current torque and complete closed-loop trajectories on paired states. No hidden braking prediction or speed cap. |
| Compare learning methods | Compare frozen analytical control, zero-initialized residual learning and learning without the prior, using equal environment-interaction budgets and reset distributions. | Report learning curves over at least three training seeds, capture and crossing counts, capture times and computation cost. A prior benefit means faster learning or better final behavior, not merely imitation loss. |
| Validate the chosen setting | Use initial states and random seeds not used for training or selection. Existing validation seeds are now inspected and should not be the only final test. | Preserve downward capture reliability and show the intended improvement without hiding regressions in other state classes. Check an independently finer plant integration step. Report uncertainty rather than claiming a universal guarantee. |
| Study the long-term objective | After a successful fixed-interval study, define the actuator behavior during sleep and assess sustained upright behavior before learning horizon or terminal value. | Sleep duration, action schedule, control effort and balancing criteria are explicit; new results are separate from current capture evidence. |

When training becomes an authorized task, prepare paired runs for parallel execution, check shared-GPU occupancy, and iterate after inspecting failures. A few attractive trajectories cannot validate an RL prior. The eight selected episodes are for interpretation and animation, **not an adequate training dataset**. Do not use the selected validation trajectories for training while continuing to call their original seeds held-out validation.

## Portable animation records and scope of this change

The [portable bundle interpretation](../../artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/portable-animation-records/interpretation_summary.md) explains eight selected episodes and their animation use. The [manifest](../../artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/portable-animation-records/records/manifest.json) specifies original seeds, lane indices, selection rules, outcomes and checksums. A lane index is the episode's position in the original parallel rollout arrays.

Only the portable subset is deliberately force-added under the ignored artifact path. The full experiment archive remains ignored. CSV files contain every active 20-millisecond sample plus the initial state, applied interval torque, unweighted physical energies and 3D geometry. NumPy archives preserve every logged per-decision field for the selected episodes, including terminal padding and its activity mask. Complete episode tables for the two source validation runs are included to make the selection auditable.

No animation, new RL training or controller change is implemented here. On the other machine, animate the saved coordinates directly and use the recorded timestamps; no GPU or simulator rerun is needed. Stop at capture rather than depicting a frozen terminal pose as evidence of balancing.

Implementation structure for this task: one new script, [export_single_action_representatives.py](../../scripts/export_single_action_representatives.py), with one `main()` function; no classes, dataclasses, wrappers, helper functions, CLI switches or controller parameters. Local workflow data are source/output paths, episode tables, selection tuples, state/torque/energy/geometry arrays and a manifest. Selection is deterministic; state and geometry processing is vectorized. This keeps reproduction explicit and within the project's compact-code discipline.
