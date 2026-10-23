# Test the LQR stage cost inside nonlinear CEM MPC

**Official baseline, 2026-10-23: [MPC+RL](mpc_rl_baseline.md).** The 16-generation CEM controller with exact LQR stage weights and the frozen RL terminal value is now the development baseline. RL remains enabled after capture; no local handoff is used. Earlier study descriptions below are historical.

The retained hybrid uses CEM for swing-up and the old LQR after temporary stabilization. This experiment separately asks whether replacing CEM's running cost with the **exact discrete LQR quadratic cost**, while retaining the frozen RL terminal value and physical constraints, can also sustain balance. No local-controller handoff is used in the new experiment.

## The common cost and the remaining differences

The physical state is arm angle theta, pendulum angle alpha (zero downward), arm speed omega and pendulum speed nu, with angles in radians and speeds in rad/s. The arm target and both target speeds are zero. Define wrapped pendulum upright error beta and local error vector e:

$$
\beta=\operatorname{atan2}(-\sin\alpha,-\cos\alpha),\qquad e=(\theta,\beta,\omega,\nu)^\mathsf{T}.
$$

The original LQR uses the following fixed, designer-chosen weights. The physical arm boundary theta_max is 135 degrees converted to radians. W is the diagonal state-weight matrix; R is the normalized-input weight. The 0.15 rad scale determines upright-error emphasis. Speed weights have inverse squared rad/s units:

$$
W=\operatorname{diag}(\theta_{\max}^{-2},\ 0.15^{-2},\ 0.25,\ 0.25),\qquad R=1.
$$

Normalized input a is commanded torque divided by u_max = 0.00459 N m. The new optimization has N = 10 decisions, each lasting 0.1 seconds, hence a one-second prediction horizon. Index k runs from zero at the measured state to N at the final predicted state. The nonlinear simulator propagates each state under the constant input for that interval. The objective is:

$$
\min_{a_0,\ldots,a_{N-1}}
\sum_{k=0}^{N-1}\left(e_k^\mathsf{T}We_k+a_k^2\right)+V_{\rm RL}(x_N).
$$

Here x_N is the final physical predicted state. The frozen terminal cost V_RL is the existing conservative conversion of the two TD3 reward critics into a cost, evaluated at the frozen actor's action. Q_1 and Q_2 are the saved critics and pi is the saved actor, all receiving the existing normalized observation encoding:

$$
V_{\rm RL}(x) = \max_{i\in\{1,2\}}[-Q_i(x,\pi(x))].
$$

There is **no multiplication by 0.1 seconds, no discounting, and no additional terminal multiplier**. The LQR design is a discrete undiscounted sum; importing its coefficients into the old quadrature formula would change their strength relative to the RL terminal value. The state terms use x_0 through x_(N−1), not post-action states x_1 through x_N. The final state receives the RL value rather than another stage penalty. The current-state penalty is constant during one optimization, but retained for exact accounting.

The old effort/centering rates and weak speed cost do not contribute in this explicit cost mode. This is a replacement, not an added weak regularizer. The constraints remain normalized torque magnitude ≤1, arm magnitude strictly below 135 degrees at the prediction/audit nodes, finite numerical values, and the existing integration accuracy tolerances.

LQR is related to this MPC, but it solves an infinite-horizon **linear** unconstrained quadratic problem with an exact Riccati solution. This test uses the **nonlinear** simulator, a finite horizon, a sampled optimizer, wrapped angle error away from upright, hard constraints, and a learned continuation value. Equal stage weights alone do not make the two policies equivalent. The RL value was trained with a different reward and discount; it is deliberately reused without retraining or fitted scaling, as requested.

## Numerical resolution and experiment

The initial 2 ms prediction / 1 ms verification attempt rejected a plan at 0.4 s for integration disagreement, while its arm margin was still positive. The quadratic cost drove a rapid swing. Its maximum pendulum-angle disagreement was 0.001655 rad and pendulum-speed disagreement 0.021480 rad/s, above the unchanged 0.001 rad and 0.01 rad/s limits. These partial runs are preserved as diagnostics and excluded from complete-session metrics.

The corrected experiment uses 1 ms prediction and independent 0.5 ms verification. Actual motion follows the finer simulation; every second refined sample is archived at 1 ms for the same hold-time scoring as the previous experiments. Prediction-node feasibility and finer-grid feasibility are both required. This changes numerical resolution, not the physical model, cost weights, or rejection thresholds. The existing hybrid retains its original integration path.

The main run has the same 32 benchmark initial states and completes 400 decisions, or 40 seconds. The four initial-state groups are downward, moving swing, near upright and outward arm, with eight starts per group; the eight-start pilots contain two per group. CEM uses 1,024 samples per seed, 32 elites, 16 generations, normalized initial spread 0.5, and three existing seeds: shifted actor/warm-start plan and constant ±0.5 controls. The eight-start optimization-budget check compares 16 and 32 generations using the same batch shape and random seed. We do not tune the requested cost weights. Higher-budget samples only diagnose whether additional optimization changes the closed-loop behavior; optimizer score alone is not the selection criterion.

Compare total time and longest uninterrupted time within ±5° of upright over the entire 40 seconds. The initial point earns no extra sample and failures do not bridge hold intervals. The old CEM and retained hybrid are historical matched-start references. Their trajectories need not match the new runs before reaching upright, and their integration resolution differs. There is no early success or task-failure-score cutoff.

## Structure budget and validation

Follow the compact line-of-logic discipline: extend the existing `evaluate` function; add one runner `lqr_cost_run.py` with one compiled `advance` function; add one substantive test `test_lqr_stage_objective`; and add one straight-line `lqr_cost_report.py` for audit, scores and plots. No new classes, dataclasses, wrappers, fallback controllers or configuration layers.

The only runner choices are stage (eight or 32 starts) and CEM generations. Existing plant/network objects are reused. The added `lqr_state_weights` vector explicitly selects the replacement cost. Audit state consists of selected plans, nominal decision states, cost components, terminal costs, executed refined trajectory, feasibility and refinement records. Fixed weights and numerical resolutions are archived, not tuned silently.

Milestone 1: GPU checks reconstruct the exact discrete cost, its one-step current-state convention, periodic angle error and frozen terminal addition; legacy cost tests must still pass. Milestone 2: complete the full sessions with all accepted actions and save every decision. Milestone 3: independently reconstruct each recorded objective from saved decision states and actions, verify state continuity and feasibility, then compare both hold metrics and representative full trajectories. Human summaries and PNG/PDF/SVG figures are separate from machine-readable CSV/JSON/NPZ and source snapshots.

## Local optimization diagnostic

One additional straight-line GPU script, `lqr_terminal_probe.py`, inspects local behavior without changing the controllers. It has no new functions or adjustable inputs. First it evaluates 241 stationary upright states with arm angles from −60° to +60° in 0.5° increments, using ten zero actions and the same total objective. This is a restricted candidate family, not a solved MPC. Its minimum occurs near +2.5°, with only about 0.00557 lower total cost than zero arm angle; this static preference alone does not explain the pilot's roughly 20° arm RMS.

Next it takes all eight final-decision states from the completed 16-generation pilot and generates ten successive existing-LQR commands using the fine nonlinear predictor. Both the saved CEM plan and the LQR-generated plan are audited under exactly the new running cost, terminal value and constraints. The terminal epigraph scalar is set to the evaluated critic cost. All eight LQR sequences are feasible and have lower objective than the saved CEM sequences (mean 3.445 versus 5.093). This directly establishes a local search gap at those states; it does not prove global optimality of LQR or identify every cause of residual motion. LQR candidates are not injected into the experiment. A later explicit warm-start comparison would be a suitable follow-up.

The completed main run also nearly activates the arm bound. The probe replays all four executed intervals whose stored arm magnitude exceeds 134° at 0.125 ms and 0.0625 ms. Both finer grids stay below 135°, with a worst peak of 134.999925°. The roughly 0.000075° clearance is not a practical uncertainty margin; finite-grid acceptance is not a continuous-time safety certificate. No extra reserve was inserted into the requested constraint.
