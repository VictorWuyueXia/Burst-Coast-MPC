# Weak angle tracking near upright and longer evaluation

Historical angle-cost experiment. The current implementation removes these additions and uses the [weak speed experiment](weak_speed.md). Reproduce the old experiment using its archived source snapshots, not the current model.

The original energy-phase capture controller had a 20-second limit and could terminate on brief capture; its deadline is distinct from the later stationary teacher validation.

The original five-second MPC plots are a short diagnostic, not a meaningful timeout for every swing-up. The selected stationary TD3 uses 120-second collection segments with continuing bootstrap targets; its actor and augmented heuristic teacher were independently tested for 600 seconds. Success does not terminate those continuing evaluations. The earlier finite weak-motor capture study used a 40-second deadline. Prediction horizon, training reset interval and evaluation duration are different quantities: the current MPC still predicts one second and replans every 100 ms.

The new paired study screens two weak scales for 40 seconds, then compares the chosen scale against zero added tracking for 120 seconds, using both GPU solvers and the same archived initial states. Failure or success scores never terminate sessions. A genuine inability to produce a physically accepted plan is reported as an execution error, not silently replaced with a different controller.

## A small local preference, rather than a large swing-up penalty

Simply multiplying squared pendulum error by a small constant would still give its largest force far from upright. Instead we use a smooth saturated angle penalty whose gradient is concentrated near upright. Define alpha as the simulated pendulum angle in radians (zero downward), theta as simulated arm angle in radians (desired value zero), theta_max as the existing 135° arm limit, sigma as an explicitly chosen 5° angular width converted to radians, and epsilon as the tunable nonnegative tracking cost rate, in objective units per second. Compute proximity g and the added rate ell_track at every 2 ms prediction node:

$$
g(\alpha)=\exp\!\left[-\frac{1+\cos\alpha}{\sigma^2}\right],\qquad
\ell_{\mathrm{track}}(\theta,\alpha)=
\epsilon\left[1-g(\alpha)+g(\alpha)\left(\frac{\theta}{\theta_{\max}}\right)^2\right],
\qquad \sigma=5\pi/180.
$$

Near upright, let beta be the wrapped angular error from upright, in radians. Since one plus cos(alpha) is approximately beta squared divided by two, the pendulum term is locally quadratic:

$$
\ell_{\mathrm{track}}\approx
\epsilon\left[\frac{\beta^2}{2\sigma^2}
+\left(\frac{\theta}{\theta_{\max}}\right)^2\right]
\quad\text{near }\beta=0,\ \theta=0.
$$

The pendulum target is upright, the arm target is zero. No angular-velocity term or switch is added. Far from upright, g is almost zero, the added cost approaches epsilon, and its state gradient approaches zero. For feasible arm angles the added rate is between zero and epsilon. At 15° pendulum error g is about 0.0114; at 30° it is about 2.3e-8. This provides a near-upright tie-breaker, not a guarantee that distant actions never change: a plan that can reach upright may receive up to epsilon times its duration of additional preference.

Use epsilon = 0.01 and 0.1 in screening, with epsilon = 0 as the paired baseline. At zero arm error and 1° pendulum error, the 0.1 setting adds approximately 0.00198 cost units/s. At exact upright and 1° arm error, it adds approximately 0.00000549 units/s. Existing torque cost rate 2 and broad arm-centering rate 10 remain unchanged. Thus the extra arm term is deliberately much weaker than existing centering; this experiment does not claim to isolate a large additional arm-centering effect.

Let k index planned 100 ms control intervals, j index their fifty 2 ms nodes, gamma = 0.995 be the inherited per-interval discount, and N = 10 be the fixed prediction length. With theta_kj and alpha_kj computed by the existing rollout, the new objective is:

$$
J_{\mathrm{new}}=J_{\mathrm{previous}}+
\sum_{k=0}^{N-1}\gamma^k\sum_{j=1}^{50}
0.002\,\ell_{\mathrm{track}}(\theta_{k,j},\alpha_{k,j}).
$$

The previous objective contains normalized torque effort, squared normalized arm centering and the unchanged frozen twin-Q terminal cost. Both SQP's differentiable epigraph objective and CEM's sampled score use the same added term. Audits record its pendulum and arm components separately. The original full RL cost remains a separate diagnostic and is not silently substituted.

## Development and evidence contract

Implementation remains in the existing `evaluate` function. New parameters are `tracking_cost_per_s` and `tracking_width_rad`; preparation and tests declare them explicitly. The compact `tracking_run.py` has one compiled `advance` function for a full GPU batch. It archives selected actions, every executed 1 ms state, plan endpoints, cost components, refinement discrepancies and solver diagnostics in one-second chunks, with progress output. `tracking_report.py` computes session statistics and figures from those records. No new classes or controller fallbacks are introduced.

Validation milestones: (1) independently reconstruct the tracking cost and check finite-difference gradients on GPU; zero rate recovers the prior objective; (2) confirm every requested decision is recorded and physically accepted; (3) compare first arrival, first continuous one-/ten-second hold, final ten-second hold, subsequent departures, arm error and batch solve time on paired starts. Screening uses eight starts (two per original group), confirmation uses all 32. Screening and confirmation CEM draws differ because batch shape changes; neither cohort is a new independently sampled test set.

Source checkpoint directory is now `artifacts/rotary_pendulum/1-TD3_20261016`; its weight checksum is checked against the exported manifest. No Q retraining, CPU solver benchmark, dynamics change, or new project stage is introduced.
