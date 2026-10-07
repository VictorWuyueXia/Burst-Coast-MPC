# Dense reward implementation

## Human quick reading

Implement direct energy-error, torque-magnitude, increasing time, arm-position, and smooth upright rewards. Success requires the accepted 100 ms continuous hold, then terminates immediately. Arm excursions are penalized by position only and never terminate. Reward is integrated at 20 ms; actions remain 100 ms. Train fresh under `rotary-q-prior-v3`; no training is part of this change. Only physical GPUs 0–3 may be used.

Keep compact task-oriented files, no thin wrappers, unnecessary CLI/configuration layers, fallback behavior, or try/with/except. Preserve historical artifacts. Network and optimizer algorithms are unchanged.

## Machine-scannable structure budget

No new classes or dataclasses. Three new task-oriented functions (plotting, experiment validation, and a standalone untrained diagnostic workflow); one nested physics-reward scan body. All other changes modify existing functions. Substantive Python files remain 40–300 lines. Extract existing experiment checks alongside the new reward checks into one validation operation so the training workflow stays below 300 lines without compressing its layout.

| File / object | Exact changes and role |
|---|---|
| `jax_environment.step`, nested `advance_one` | Add keyword `physics_steps=5`, allowing a single 20 ms sample for reward integration. Arm flag becomes cumulative diagnostic only; success/timeout alone stop; retain five-sample dwell. |
| `jax_task.observe(env_state)` | Return seven state/time/dwell features only; remove potentials and experiment argument. |
| `jax_task.transition`, new nested `advance_reward` | Scan five single physics steps; sum ordered energy/torque/time/arm/upright components; actual active-time masking; preserve existing six-result tuple with base/train both equal direct reward. |
| `jax_task.collect`, existing nested `advance` | Update observation calls, terminal mask, component names, excursion metrics; preserve exploration/replay behavior. |
| `jax_evaluation.evaluate`, existing nested `advance` | Remove potential planner/conversion; greedy, zero-terminal lookahead, Q-terminal lookahead, value audit use direct reward; save actual times, dwell and five components; report excursions independently of outcomes. |
| `jax_artifacts.write_artifacts` | Version-3 metadata, direct reward parameters, explicit sample-axis selection, revised labels; delegate substantive trajectory plotting. |
| New `jax_validation_plots.plot_validation(human_dir, trajectories)` | Plot physical trajectories including unwrapped alpha, torque, dwell, and five rewards for tight/near; use actual times and valid masks. |
| New `jax_experiment.validate_experiment(experiment)` | Validate reward identity, coefficients/widths, existing replay/update/budget/cadence constraints; return integer updates per collection. No defaults or configuration merging. |
| `rotary_q_workflow.train` | Call experiment validation; remove excursion failure gates, rank by success then direct return then powered time; retain excursion histories. |
| `launch_rotary_q_wave.main` | Preflight all GPU assignments before file creation or process launch; require distinct indices from 0–3. |
| New `scripts/validate_dense_q_reward.main()` | Fixed CPU-only, untrained zero-torque baseline and three-decision planning on 16 tight and 16 near starts; one parameter-free workflow writes CSV/NPZ and the shared human plots, printing progress before each evaluation. No fitting or checkpoint promotion. |

New tunable experiment parameters, replacing all six old reward fields: `energy_cost_per_s=1`, `torque_cost_per_s=0.02`, `time_cost_per_s=1.05`, `arm_cost_per_s=0.2`, `upright_reward_per_s=1`, `upright_widths=[0.16,0.4,0.3]` (angle rad, pendulum speed rad/s, arm speed rad/s). Five coefficients are finite, nonnegative; time rate strictly exceeds upright rate; widths finite and positive. `contract_id=rotary-q-prior-v3`; `reward_revision=1`. Existing network ID is unchanged because architecture is unchanged. Existing physical constants, actions, tolerances, 20 s horizon, and five-sample success hold are retained.

Runtime data: `reward_components[...,5]` in energy/torque/time/arm/upright order; observation length seven; diagnostic `arm_violation` records any boundary excursion and is not a terminal bit. Evaluator adds time, hold count, integrated absolute torque and beyond-boundary time. All new working arrays serve these existing workflows; no public helper state container.

## Stages, tests, and completion flags

1. Contract: update proposal and supply one complete version-3 example experiment. Pass: hold is exactly five samples, coefficient domains and terminal events unambiguous; old artifacts remain historical.
2. Physics/reward: exercise continued boundary crossings, clipping, dwell reset, early success inside an action, timeout, absorbing zero reward, direct five-component sum, symmetry, increasing time cost, growing arm cost, and scalar/batch/JIT agreement. Pass: semantic tests pass, including an independently accumulated physics-sample reference.
3. Integration: exercise replay terminal masks, all three controllers and value audit, actual trajectory times, component metadata/plots, and launcher rejection of forbidden or duplicated GPUs. Pass: no potential conversion or arm-failure termination in the live RL path; old configuration rejected before rollout.
4. Regression: run focused tests, repository suite and Ruff; inspect generated validation figures. Pass: all checks pass, finite diagnostics and readable plots, no training launched. Record evidence in interpretation summary.

Stage gates retain success thresholds (tight 90%, near 80%; expansion also downward/moving 30%) and two consecutive passes, but remove arm-excursion vetoes. Excursions remain reported. Success is capture for 100 ms, not evidence of indefinite stabilization. Later training requires inspecting figures after every task before iteration.

## Verification evidence

The full CPU repository suite passed: 100 tests, with two existing empty-slice warnings in unrelated offline-training diagnostics. Focused environment/Q tests passed 27 checks. The untrained CPU diagnostic completed finite rollouts for zero torque and three-decision planning from 16 tight and 16 near starts. Both state and reward figures were visually inspected; successful single-decision dwell/reward samples use markers. Reports and plots are under `artifacts/rotary_pendulum/q-prior-v3/implementation-check/human-readables`, and CSV/NPZ/JSON under its separate `machine-scannables` directory. No fitting or training was run.
