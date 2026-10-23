# Exact LQR running-cost experiment contract

**Official baseline, 2026-10-23: [MPC+RL](../human-readables/mpc_rl_baseline.md).** The 16-generation CEM controller with exact LQR stage weights and the frozen RL terminal value is now the development baseline. RL remains enabled after capture; no local handoff is used. Earlier study descriptions below are historical.

The full mathematical definition and symbol units are in [the human formulation](../human-readables/lqr_stage_cost.md). This contract describes the executable selection, records and checks.

## Cost dispatch

Presence of `lqr_state_weights` in the plant dictionary explicitly selects the replacement quadratic cost in the existing `evaluate` function. Historical plant dictionaries omit this key and retain the original quadrature objective. The experiment sets the vector to `[1/arm_limit_rad**2, 1/0.15**2, 0.25, 0.25]`, `discount=1`, and `rl_cost=False`. The input coefficient is exactly one on normalized action squared. Old torque/centering/speed cost rates are set to zero for clarity and ignored by this branch. No learned terminal multiplier is fitted.

`objective` uses the free epigraph scalar for terminal cost; `score` and `audit` use the maximum of the two negative reward critics, evaluated with the frozen actor. CEM sets the plan's last scalar to that maximum. The actor and critic parameter arrays, observation scales, physical configuration and checkpoint hash are the existing TD3 bundle.

For N actions, the stage-state array concatenates current x_0 with the nominal predicted endpoints x_1 through x_(N−1). It has shape (..., N, 4). Wrapped upright error replaces its pendulum-angle component only for cost evaluation; the simulator still carries its physical angle coordinate. Sum squared errors times the four weights over the decision dimension. The returned components are `centering`, `upright_tracking`, `speed_arm`, `speed_pendulum`; `effort` is the sum of squared normalized actions. Their sum plus maximum negative critic is the objective. All terms are undiscounted discrete costs, not cost rates.

## Dynamics and feasibility

The LQR-cost branch uses 20 RK4 substeps per 20 ms block: five blocks per 100 ms decision, hence 100 nominal nodes. Verification uses 40 substeps per block, hence 200 refined nodes and 0.5 ms resolution. Historical objectives retain 10/20 substeps and 2/1 ms grids. These are compile-time structural choices from the explicit cost mode, not a runtime numerical fallback.

Torque bound and all nominal/refined arm checks are unchanged. The normalized action must lie in [-1,1]; current and predicted arm angles must be strictly inside the physical arm limit. Every required state and terminal value must be finite. Refined-versus-nominal disagreement uses corresponding nodes, wraps the pendulum-angle difference, and must be ≤[0.001 rad, 0.001 rad, 0.01 rad/s, 0.01 rad/s] componentwise. A candidate is accepted only if the existing physical audit passes.

The runner executes the first refined interval of the selected accepted plan. Saving its second, fourth, ... refined nodes yields 100 post-integration samples at 1 ms; the final sample becomes the next measured state. This is the same dense-grid scoring interval as the historical studies, although execution is now integrated more finely. Prediction-stage states for cost reconstruction remain on the nominal grid. No future-stage state is substituted for the next actual measured state.

Three initial plans remain actor/shifted warm start and constants +0.5 and -0.5. The shifted plan appends the frozen actor command at its predicted tail. These seed-generation rollouts use the unchanged helper; the objective and feasibility of every seed and sampled plan are evaluated on the new fine prediction grid. CEM still chooses feasible incumbents among three seeds with 1,024 samples, 32 elites, initial standard deviation 0.5 and the declared generation count. LQR feedback is never executed or injected as a new seed.

## Run parameters and source preservation

`lqr_cost_run.py screen 16` and `screen 32` compare eight starts. `confirm 16` uses 32 starts. All have 400 decisions and a fixed one-second horizon. The code permits 64 generations for a subsequent explicit budget extension; it does not silently increase the solve budget. The same stored initial states and random seed are used, but an eight-lane random batch is not the same random draw layout as the first eight lanes of a 32-lane batch.

The corrected output root is `artifacts/rotary_pendulum/inner-mpc/20261022-lqr-cost-fine-{stage}-{generations}`. The earlier roots without `fine` contain incomplete coarse-grid diagnostics, including an explicit incomplete marker. They are excluded by the report's path selection. Completed studies are not overwritten.

`design.json` records the cost definition, weights, torque coefficient, stage, population, elites, generation count, spread, seed, physical settings, checkpoint metadata, device, time resolutions and source hashes. `sources/` preserves exact copies of runner, model, CEM and physical dynamics. `starts.npz` stores physical initial states and group labels. `completed.json` is written only after all 400 decisions. Partial runs do not count as complete sessions.

## Decision records

Each compressed chunk contains ten decisions, or the shorter saved diagnostic prefix if no accepted plan remains. Leading dimensions are decision and independent session. `plans` adds N+1 elements: N actions and one terminal epigraph scalar. `selected` identifies the seed lane supplying the selected incumbent.

`stage_states` adds dimensions (N,4), nominal decision-start states including the measured state. `executed` adds (100,4), the selected first refined interval sampled at 1 ms. `plan_end` stores the refined terminal state. Cost components, objective, two terminal costs, arm margin, four-component refinement error and the accepted flag are saved. The selected distribution's last sampling spread and per-generation feasible counts are also recorded. `solve_s` is synchronized wall time per GPU batch, excluding file I/O.

A numerical/physical inability to return an accepted plan is explicitly diagnosed and preserved; it is not relabeled as task failure or hidden by another controller. The experiment does not stop because an episode succeeds, fails to reach upright, or loses balance. All complete-run scores use the full 40 seconds.

## Independent analysis

`lqr_cost_report.py` reconstructs all four state-cost components from `stage_states`, effort from selected actions, and terminal addition from stored critics. It checks equality with recorded objective, current-state equality with the preceding executed endpoint, initial states, source hashes, finite states, action bounds, positive arm margin and the unchanged refinement tolerance. Every complete run must contain exactly 400 decisions per session.

For each episode it computes total and longest continuous time inside ±5°, final ten-second continuous holding, first upright time, final angle error and arm angle. Time is measured using exactly 40,000 post-integration samples, without gap bridging. For confirmation it loads the saved original CEM and hybrid complete runs as matched-start historical references; it verifies their archived sources rather than incorrectly requiring old snapshots to equal the now-edited model.

Machine outputs are per-session CSV, aggregate/paired JSON, complete trajectory NPZ, quadratic cost-component NPZ, report source and rendered alignment audit JSON. Human outputs are the interpretation summary and full-session/hold-time PNG/PDF/SVG figures. The report uses fixed episode indices 0 and 2 for examples, not score-based cherry-picking. No claim of robustness, global optimality, real-time feasibility or terminal-value consistency is inferred solely from these nominal trajectories.

## Read-only local probe

`lqr_terminal_probe.py` is an additional straight-line GPU analysis script, with no new functions or tunable input. It checks occupancy before CUDA, loads the same frozen checkpoint and cost, scans stationary upright arm angles, then reads all eight states at decision 399 from the completed 16-generation pilot. Its temporal loop generates ten clipped existing-LQR actions on the 1 ms nonlinear prediction grid, batching the eight independent states. It does not execute these plans in the closed-loop experiment. Their terminal epigraph is set to the evaluated conservative critic before the final full audit.

The confirmation summary's machine folder stores `stationary_arm_value.csv/json`, `lqr_plan_comparison.json/npz` and the probe source. The comparison re-evaluates the saved CEM objective and verifies it against the original recorded number. LQR candidates must pass physical refinement and epigraph constraints. These records diagnose local optimization quality; they are neither additional 40-second sessions nor new controller settings.

The same probe additionally reads the completed main run, selects all executed intervals whose stored arm magnitude exceeds 134°, and replays each from its saved measured start under its actual first command at 160/320 substeps per 20 ms block (0.125/0.0625 ms). `boundary_replay.json` identifies decision/lane pairs and the maximum/crossing count on each grid. This is interval-local numerical verification, not an open-loop replay of the entire unstable 40-second trajectory or a robustness margin.
