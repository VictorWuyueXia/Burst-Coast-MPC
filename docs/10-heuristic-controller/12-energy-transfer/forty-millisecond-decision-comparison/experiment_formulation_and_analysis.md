# 40 ms decision experiment contract

> Archive note: the method and commands below describe the controller at the time of the experiment. Reproduction requires its original source snapshot; the current `src/` has since changed. Collected figures, records and logs are in [experiment artifacts](../../../artifacts/rotary_pendulum/experiment-results/forty-millisecond-decision-comparison/).

This addendum changes the analytical controller's decision/torque-hold interval from 100 ms to 40 ms. The existing simulator's RK4 physics step remains 20 ms. The ±15° angle-only goal still requires five consecutive physics samples, nominally 100 ms, with no velocity or arm-centering requirement. Goal counts continue across decision boundaries; they do not reset at each new torque command. The deadline is 20 s, requiring 500 decisions instead of 200. The earlier 10 ms integration experiment is separate and is not used as this comparison's baseline.

## Variables and preserved quantities

| Quantity | Baseline | Added test | Source/meaning |
| --- | --- | --- | --- |
| Decision/held torque interval | 100 ms | 40 ms | Requested experimental variable |
| RK4 physics interval | 20 ms | 20 ms | Existing simulator, unchanged |
| Goal hold | Five samples / nominal 100 ms | Same | Existing goal duration |
| Angle threshold | ±15° | Same | User-defined angle-only target |
| Recovery horizon | 20 samples / 0.4 s | Same | Existing decoder prediction duration |
| Predicted brake refresh | 100 ms | 40 ms | Aligned with the action grid |
| Braking time constant | 100 ms | Same | Existing speed-to-torque calculation |
| Deadline | 20 s | Same | Same physical observation duration |
| Maximum decisions | 200 | 500 | Computed from deadline and decision interval |

The six cases are half/base/double usable torque at ±90°/±180° arm bounds. Base torque is ±0.00918 N m, giving ±0.00459 and ±0.01836 N m for half/double. The physical actuator clip is ±0.0204 N m. Damping stays zero, arm angle remains unwrapped, and bounds remain soft/nonterminal. Fixed numerical choices include 33 torque grid points, ten bisections, and 0.05 rad inner boundary margin.

Policy gain 0.04, kinetic weight 1, work-override penalty 1, and the existing per-decision work cap are unchanged. Thus this is an unchanged-policy, faster-decision experiment; it does not hold requested work per second constant. There are 2.5 times as many potential decisions. No sleep schedule, sustained balancing test, gain normalization or new controller structure is added.

Independent experimental variables are decision interval, torque multiplier, arm range, and seeded sample set. The pilot uses seed 20261011 with 16 starts per stratum. Confirmation uses seeds 20261008–20261010 with 64 starts per stratum per seed. Fresh confirmation uses 20261012–20261014 for original torque/±90° and double torque/±180°. The four strata and four fixed probes per seed are unchanged; see the [angle-goal contract](../angle-only-goal-and-integration-step-comparison/experiment_formulation_and_analysis.md). All comparisons assert exact initial-state equality to the archived 100 ms runs, not merely matching seed numbers.

## Implementation budget and validation

Keep compact task-oriented files, no classes/dataclasses, thin wrappers, fallback behavior, try/with/except, added dependencies or unnecessary configuration. Two new workflow functions are budgeted: `prepare_decisions.main()` stages explicit source changes, writes configurations and performs CPU behavioral checks; `summarize_decisions.main()` compares the old/new runs, independently reconstructs goals, verifies timing/range/torque and plots results. Existing driver, model, policy, decoder, environment and rollout functions are reused. JIT closures directly invoke existing computations. Both new scripts are between 40 and 300 lines.

Local objects are source-replacement dictionaries, case/settings dictionaries, batched physical state/torque arrays, active-sample and angle-streak arrays, episode records and figures. `state` contains unwrapped arm angle, downward-referenced pendulum angle and the two speeds (rad, rad, rad/s, rad/s). `decision` selects the comparison's decision interval in milliseconds; `dt` is the fixed 20 ms integration interval. `inside` is the active angle-only goal mask; `streak` counts consecutive true physics samples; `captured` checks whether five samples were reached. No measured speed participates in this goal check.

Milestones and flags: (1) upright-rest traces show goal counts 2, 4, 5 and elapsed times 40, 80, 100 ms over three decisions; moving-upright capture confirms no velocity requirement, and hanging rest fails; (2) decoder predicted work equals the two-substep applied pulse's work; (3) a noncapturing trajectory reaches timeout at 20 s after 500 decisions; (4) full pilots precede three-seed confirmation and representative plot inspection; (5) fresh seeds verify the principal comparison; (6) independent full-trace capture/time/range/torque/reset checks, Ruff/format, bilingual reports, separate numerical and visual artifacts.

The 2 ms NumPy single-action audit now uses twenty substeps for a 40 ms action, rather than fifty substeps for 100 ms. Only complete actions are included. Partial terminal actions preserve their true duration and work. The plot timestamps and complete-action work metrics also use 40 ms. Goal checks remain at the unchanged 20 ms sample clock. The audit is not an independent fine-plant closed-loop validation.

## Reproduction and artifacts

From a clean destination, prepare sources and run CPU checks:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/experiments/forty-millisecond-decision-comparison/prepare_decisions.py
```

After inspecting shared-server GPU occupancy, run the confirmation outside the sandbox:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 JAX_ENABLE_X64=true MPLCONFIGDIR=/tmp/bcmpc-decision40-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python artifacts/rotary_pendulum/experiment-results/forty-millisecond-decision-comparison/raw-runs/prepared/scripts/experiments/torque-and-arm-angle-range-comparison/run_study.py --config artifacts/rotary_pendulum/experiment-results/forty-millisecond-decision-comparison/records/confirmation.json --output artifacts/rotary_pendulum/experiment-results/forty-millisecond-decision-comparison/raw-runs/confirmation
.venv/bin/python scripts/experiments/forty-millisecond-decision-comparison/summarize_decisions.py --stage confirmation
```

Preparation/run destinations must not exist; the summarizer explicitly refreshes its records. The original study driver rechecks all-server/selected-device GPU occupancy before each run and emits regular progress/heartbeats. Independent cases are distributed over idle GPUs 0–3. The summarizer reads the corresponding existing `angle-goal-study/dt20_*` runs; reproduce those first from their saved configuration if absent. For `fresh`, it selects only the two matching original cases.

`decision40.patch` records the exact source edits; each run additionally preserves its torque/range patch, source snapshot and hashes. `goal_clock_checks.json` records behavioral assertions. `records/` contains comparison CSV, action audits, settings and validation flags; human PNGs are under `human-readables/decision40_study`. Full trajectories, logs and isolated sources stay in `artifacts/rotary_pendulum/experiment-results/forty-millisecond-decision-comparison/raw-runs/`. Runtime defaults are unchanged. Chinese documents remain ignored by the existing `*-CN.*` rule.

Captures and clean captures remain distinct; excursions use each case's own bound before capture/timeout, so they are not equal-exposure indefinite containment tests. Terminal speeds remain available as diagnostics. Legacy `approach` is a separate metric, not the success rule. Representative lanes use the median minimum-energy-cost episode in each stratum, not a selected success.
