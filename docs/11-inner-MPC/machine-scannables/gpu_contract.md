# GPU solver contract: centering and batched optimization

This revision explicitly changes the running cost and the local solver implementation.
The frozen 4.59 mN m actor and twin critics, dynamics, discount and hard limits remain the same.
SciPy SLSQP is retained as historical reference code; no CPU solver benchmark is part of this revision.

## Objective and finite-grid feasibility

For decision index k, a_k is optimized torque divided by the checkpoint torque cap.
N is the selected horizon in 100 ms decisions. Delta=0.1 s and delta=0.002 s are fixed time steps.
Theta_k,j is the predicted arm angle after RK4 microstep j, computed from the measured initial state.
Theta_max=3*pi/4 rad is the existing arm limit. Gamma=0.995 is the trained discount.
W_u=2 cost units/s is the retained effort rate. W_theta is the new nonnegative tunable centering rate.
F_Q is the negative minimum of the two frozen critics evaluated at the terminal state and frozen actor action.

$$
J=\sum_{k=0}^{N-1}\gamma^k\left[\Delta w_u a_k^2+
\delta w_\theta\sum_{j=1}^{50}\left(\frac{\theta_{k,j}}{\theta_{\max}}\right)^2\right]
+\gamma^N F_Q(x_N).
$$

The centering term is a right-endpoint quadrature on every 2 ms node, not a boundary hinge.
It is returned separately as `centering`; `effort` remains input effort only.
The reward-consistent diagnostic still uses the original RL reward without adding this new term twice.
A zero centering weight reproduces the original numerical primary objective.

## Native SQP, distinct from SciPy SLSQP

`gpu_sqp.solve_gpu` batches measured states and three initial plans in one compiled scan.
The variable z contains N torque fractions and the terminal epigraph scalar t.
The objective f(z) replaces F_Q by t; two constraints require t to exceed each negative critic.
All constraint residuals c(z) use the nonnegative feasible convention.
For each decision the upper and lower arm residuals are their worst microstep residuals.
This preserves the entire discrete feasible set while introducing kinks when the extremizing node changes.
Torque bounds contribute another 2N residuals: total 4N+2, plus numerical step bounds in the QP.
Automatic forward derivatives retain the full actor-through-critic chain.

The computed gradient g and constraint Jacobian A linearize the current subproblem.
B is a positive-definite Hessian estimate, initially identity and updated by Powell-damped BFGS on Lagrangian gradients.
The step p solves approximately:

$$
\min_p\ \tfrac12p^\top Bp+g^\top p,\qquad Ap+c\ge0.
$$

The numerical trust box caps each torque step by `step_limit` and the epigraph step by 10 cost units.
Rows are divided by the maximum of their Euclidean norm and one; this does not change the QP feasible set.
ADMM uses `rho` and `qp_iterations`, a Cholesky factorization and nonnegative slack projection.
Its primal/stationarity residual is recorded as `qp_error`; the fixed budget is not a convergence certificate.
This compact method follows the splitting principle described in the [OSQP solver documentation](https://osqp.org/docs/solver/index.html); it does not implement all OSQP features or use its package.
BFGS uses damping threshold 0.2 and updates only above curvature 1e-14; 1e-20 denominator guards prevent undefined inactive arithmetic.
These numerical constants do not modify the MPC cost or physical constraints.

Eight trial lengths, 1 through 1/128, and the unchanged iterate are evaluated together.
Trial torque coordinates are projected onto their declared bounds before nonlinear evaluation.
The exact L1 merit weight is the maximum of 10 and twice the largest absolute QP multiplier.
Choose the smallest finite merit; retain every better nominally feasible trial as a best-so-far candidate.
This discrete trial search is not an Armijo convergence proof.
The recorded `stationarity` describes the current iterate and approximate multipliers, not necessarily the retained incumbent.

## Compiled CEM and common acceptance

`gpu_cem.initial_plans` generates the actor rollout and constant +0.5/-0.5 plans.
`sample_gpu` samples clipped diagonal Gaussian populations per seed, selects only feasible elites and updates distributions on-device.
Insufficient feasible elites or zero variance stop that distribution; there is no variance floor or penalty surrogate.
Keys, means, spreads, feasible counts and best scores are saved each generation; full populations can be regenerated.
All runs use float64 physical states and promoted checkpoint weights.
The final plan for each seed is independently replayed on a 1 ms grid.
Strict initial/path arm bounds, finite trajectories, actuator bounds and the existing refinement tolerances remain required.
A finite-budget feasible incumbent is accepted without claiming SQP convergence, identically to sampling's best-so-far semantics.
If all three final incumbents fail replay, the episode stops. No alternative-controller or earlier-plan fallback is applied.

## Implementation budget and experiment protocol

No new classes, data classes, wrappers or dependencies. Two functions in each GPU solver file plus their algorithmic scan bodies.
One `main` in `gpu_run.py`; preparation and analysis use straight-line scripts.
Data groups: states/plans; networks/plant; optimizer carry/history; settings; recorded outputs.
Additional parameters: `centering_cost_per_s`; SQP `iterations`, `qp_iterations`, `step_limit`, `rho`.
CEM retains `population`, `elite_count`, `generations`, `initial_std`.
GPU occupancy is checked before importing JAX. Prefer idle devices 6–7; refuse more than four occupied devices.
The runner requires an actual GPU backend, float64, exact checkpoint/physics/source hashes, and a fresh result directory.
Compile and warm the full batch before timing. Synchronize completion as required by [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).
`elapsed_s` is whole-batch optimization plus audit latency; `throughput_s` divides it by batch size and is not command latency.
Seed time is separate; archive transfers and disk I/O are outside the solver timer. Closed-loop timing retains inactive lanes.
Pair objective comparisons only at the same measured state, horizon and centering weight; different weights define different objectives.
For receding-horizon tests use physical executed arm/upright metrics and completion counts, not pooled surrogate objectives.


## Local tracking extension (2026-10-20)

The optional extension is specified in [weak tracking](../human-readables/weak_tracking.md).
`tracking_cost_per_s` is the explicitly supplied nonnegative rate; zero recovers the previous primary objective. `tracking_width_rad` is the explicitly supplied positive angular width (5 degrees converted to radians in this campaign). Both fields are present even in a zero-rate baseline; there is no missing-key fallback. The existing `centering_cost_per_s` stays at 10 for the paired long study.

Evaluate the periodic exponential proximity on every nominal 2 ms rollout node, multiply each stage's nodes by the same per-100 ms discount power used for effort, then sum the added pendulum and arm contributions separately. Add both to `objective`, `sqp`, `score`, and primary `audit`/`rl_objective` paths. When `rl_cost=True`, retain the original full RL running objective instead; added components remain diagnostic only. Policy/value-only evaluation is unchanged. Constraints and finer physical acceptance are unchanged.

`tracking_run.py` archives one file per ten decisions. The leading axes of `executed` are decision, episode, 100 integration samples, four physical state components. `plans` contains ten normalized torques and the terminal epigraph value. `plan_end` is the selected one-second refined terminal state. Cost components, twin negative critic values, arm margin and maximum refinement errors have decision and episode axes. SQP's outer-iteration diagnostics are stored per selected seed; CEM stores feasible population counts and final sampling spreads. Full populations are not transferred to the CPU. `solve_s` includes seed generation, solver, selection and fine audit in the synchronized compiled batch, but excludes compilation, warmup and archive I/O.

Initial states and frozen weight checksum match the prior closed-loop campaign. CEM uses 1024 samples per seed, 32 elites, 16 generations, spread 0.5 and three seeds; SQP uses 20 outer iterations, 320 QP iterations, step cap 0.5 and penalty 10. A shifted previous plan replaces the actor seed after the first decision, exactly as before. No success/quality failure latch is present. All requested decisions are recorded unless no physically accepted plan exists, in which case the runner explicitly errors and preserves diagnostics; no unapproved fallback is executed.

The report's one-/ten-second hold completion is the beginning of a consecutive in-band sample run plus the requested hold duration. Sample-run lengths count points times 1 ms and may differ from elapsed first-to-last sample time by 1 ms. Final holds require every point in the last 1000/10000 samples to have wrapped upright error strictly below 5 degrees. These are positional scores, not invariant-set or velocity conditions. Counts-over-time plots sample archived states every 100 ms; representative traces use every 1 ms point.

Additional long-run scores: `upright_fraction_after_10s` is the fraction of all 1 ms samples from 10 s through the recorded endpoint with error below 5 degrees. The aggregate is the mean across episodes. `escaped_after_first_10s_hold` is true when any subsequent exit from the band occurs after the first completed ten-second hold; it is null when no such hold exists. `median_longest_dwell_s` summarizes each episode's longest uninterrupted in-band sample run. These descriptive measures were added to distinguish occupancy, temporary capture and retained balance; none changes the controller or stopping policy.

## 2026-10-20 speed-cost replacement

The angle-tracking extension above is historical. Current required settings are `speed_cost_per_s` and `speed_width_rad`; the removed `tracking_*` settings are not used. See [the fully defined speed objective](../human-readables/weak_speed.md). Audit fields are `speed_arm` and `speed_pendulum`, each in objective units. `speed_run.py screen RATE` and `speed_run.py confirm RATE` run CEM only for 400 decisions (40 seconds), using 8 and 32 starts respectively. Each archived `executed` array has shape (decisions in chunk, sessions, 100, 4); states have order arm angle, pendulum angle, arm speed, pendulum speed, in radians and rad/s. No initial sample is included in time accounting. `speed_report.py STAGE` checks completion, source hashes, finite states, torque/arm bounds and integration refinement, then writes per-session scores at 2°, 5°, 10°, paired differences and full trajectories. English and Chinese conceptual definitions are in `weak_speed.md` and `weak_speed-CN.md`.
