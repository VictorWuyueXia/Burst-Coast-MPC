# Angle-only goal, torque and integration-step study

> Archive note: the method and commands below describe the controller at the time of the experiment. Reproduction requires its original source snapshot; the current `src/` has since changed. Collected figures, records and logs are in [experiment artifacts](../../../artifacts/rotary_pendulum/experiment-results/angle-only-goal-and-integration-step-comparison/).

## Goal and physical clocks

The goal is the wrapped distance from pendulum upright at most 15 degrees. There is no arm-angle condition and no arm- or pendulum-speed condition in the goal. Capture still requires a nominal 100 ms sampled hold: five consecutive samples at 20 ms or ten at 10 ms. A failed angle sample resets the count. Successful and timed-out lanes freeze, as in the original evaluator. This is a sampled capture criterion, not a continuous-time or sustained-balancing guarantee.

“Half step size” is interpreted as halving RK4 physics integration from 20 ms to 10 ms, in both the plant and decoder predictions. The torque update/hold stays 100 ms. The physical deadline stays 20 s. Recovery horizons stay fixed in seconds, so halving the step doubles their sample counts. The hypothetical braking continuation still refreshes every 100 ms. Its terminal arm-speed check remains an internal decoder criterion; only the episode goal loses velocity requirements.

The original upper energy-work policy, zero physical damping, 33-point torque grid, ten work-root bisections, 0.05 rad inner arm margin, and physical actuator ceiling of ±0.0204 N m remain fixed. Policy work gain and arm-kinetic weight remain 0.04 and 1. Increasing or decreasing the torque cap also changes the spacing of the fixed-size torque grid. The environment arm angle remains unwrapped and soft-limit crossings remain nonterminal.

## Cases and independent variables

| Torque multiplier relative to original decoder | Controller cap | Arm soft ranges | Physics steps |
| --- | --- | --- | --- |
| 0.5 | ±0.00459 N m | ±90°, ±180° | 20 ms, 10 ms |
| 1 | ±0.00918 N m | ±90°, ±180° | 20 ms, 10 ms |
| 2 | ±0.01836 N m | ±90°, ±180° | 20 ms, 10 ms |

The main 12 cases use a 0.4 s total recovery prediction and work-mismatch weight 1. Names encode integration step (`dt20` or `dt10`), torque multiplier (`t0.5`, `t1`, `t2`), and arm range (`a90`, `a180`). Per-case JSON `recovery_steps` is a sample count: 20 or 40 for 0.4 s.

Pilot seed 20261011 uses 16 random states in each of downward, moving, near-upright and tight-upright classes, plus four fixed probes. Confirmation seeds 20261008–20261010 use 64 per class per seed, or 768 random episodes per case plus 12 probes. All cases and both time steps share exactly identical initial arrays. Reset definitions are unchanged from the [previous contract](../torque-and-arm-angle-range-comparison/experiment_formulation_and_analysis.md).

The previous eight tuning candidates plus work weight 0.01 are repeated at both integration steps on the pilot seed. Recovery horizons of 0.2, 0.3, 0.6 and 0.8 s use work weight 1; the 0.4 s horizon tests weights 0, 0.01, 0.1, 10 and 100. These cases use double torque and ±180°. Historical `h` name components describe the original 20 ms sample count; use JSON and the physics step to recover the actual horizon. The later fresh-seed configuration explicitly records the selected candidates and unchanged baseline.

Independent variables are torque cap, arm range, physics step, recovery horizon, work weight, and seeded sample set. The angle threshold and nominal hold duration are user-requested/retained experimental definitions. Torques and physical states are computed by the existing simulation; no fitted physical parameters or new optimization objective is introduced.

## Structure budget and reproducibility

Use compact task-oriented logic, no thin wrappers, unnecessary CLI/configuration, classes, dataclasses, fallback behavior, try/with/except, or dependencies. Exactly three new workflow functions are budgeted:

- `prepare_study.main()`: copy the unchanged runtime sources and existing study driver; apply explicit goal/timing edits with presence assertions; save the patch and trial configurations.
- `validate_goal.main()`: verify the goal with moving upright/outside-angle/off-center probes, the physical clocks, predicted held-action work and partial-action time accounting.
- `summarize_study.main()`: independently reconstruct every capture from physics samples, verify timing/range/torque and paired states, archive compact records and render readable figures.

The existing study driver's `main()`, environment reset/step, energy encoder/policy, decoder and evaluator are reused. JIT closures in validation directly call these existing computations. New data objects are snapshot edit dictionaries, case/study dictionaries, paired initial-state arrays, sampled angle/active/streak arrays, result rows and figure objects. They are local workflow data, with no new public container type. CLI additions are only the summarizer's required `--stage` choice; the existing runner retains `--config` and `--output`.

`interval` is the selected physics step; `hold` is computed as the integer number of samples in 100 ms. The physical state order is arm angle, downward-referenced pendulum angle, arm speed, pendulum speed, with units rad, rad, rad/s, rad/s. `inside` is the independently recomputed active angle-condition mask. `streak` counts consecutive true samples using the index of the latest failed sample. `captured` reports whether any streak reached the required sample count. No velocity enters that computation. `final` is the last active physical sample, used to report terminal speeds.

Source isolation is under `artifacts/rotary_pendulum/experiment-results/angle-only-goal-and-integration-step-comparison/raw-runs/prepared/dt20` and `dt10`. Each contains the original runner at its expected relative path. The preparation patches are retained in the experiment’s artifact records. Each driver case then records its torque/range patch and each rollout records source snapshots and SHA-256 hashes. Timing and goal definitions are included in rollout provenance. Runtime defaults in `src/` are not edited.

From a clean checkout with the existing environment:

```bash
.venv/bin/python scripts/experiments/angle-only-goal-and-integration-step-comparison/prepare_study.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=true PYTHONPATH=artifacts/rotary_pendulum/experiment-results/angle-only-goal-and-integration-step-comparison/raw-runs/prepared/dt10/src .venv/bin/python scripts/experiments/angle-only-goal-and-integration-step-comparison/validate_goal.py
```

After checking GPU occupancy outside the sandbox, run one confirmation as follows. The 10 ms study uses its corresponding prepared root, configuration and output; independent studies used GPUs 0–1 and 2–3 respectively. The driver checks all-server and selected-device occupancy before every launch and reports every simulated second, with a five-second process heartbeat during compilation/plotting.

```bash
CUDA_VISIBLE_DEVICES=0,1 JAX_ENABLE_X64=true MPLCONFIGDIR=/tmp/bcmpc-angle-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python artifacts/rotary_pendulum/experiment-results/angle-only-goal-and-integration-step-comparison/raw-runs/prepared/dt20/scripts/experiments/torque-and-arm-angle-range-comparison/run_study.py --config artifacts/rotary_pendulum/experiment-results/angle-only-goal-and-integration-step-comparison/records/integration_20_milliseconds_confirmation.json --output artifacts/rotary_pendulum/experiment-results/angle-only-goal-and-integration-step-comparison/raw-runs/dt20_confirmation
.venv/bin/python scripts/experiments/angle-only-goal-and-integration-step-comparison/summarize_study.py --stage confirmation
```

Preparation and experiment output directories must not already exist; summarization explicitly refreshes its archive directory. Summarization requires both integration-step studies for the selected stage to have completed. `smoke` configurations are retained for optional short reproductions; executed validation used the direct goal checks and full-duration pilots.

## Validation and data meanings

Milestones and success flags: (1) moving upright probe succeeds, outside-angle probes fail, off-center arm does not block capture; (2) 100 ms action/goal clocks and 20 s deadline match; (3) pilot trajectories are finite, cap/range assertions and independent sampled-goal reconstruction pass; (4) three-seed comparison uses identical initial arrays across all cases and both step sizes; (5) tuning is inspected before fresh-seed selection; (6) reports/plots and machine records are separate, with English/Chinese interpretation.

`records/` retains stage-level `episodes.csv`, `summary.csv`, independent action `integration_audits.json`, settings and validation flags. Each episode includes final arm/pendulum speed and whether a new-goal capture exceeds either old speed limit. The old speed limits are used only for this diagnostic. Captures, clean captures, and excursions are separate. Excursion rates use each case's own arm bound and stop accumulating at capture/timeout, so easier earlier termination also shortens exposure to possible excursions.

The existing runner independently replays up to 256 complete 100 ms actions at 2 ms using NumPy. This is an action-level numerical audit, not a full independent fine-plant closed-loop validation. Main 10 ms experiments change both the decoder and plant integration. Representative plots use physics samples for angle, speed and arm position, and the applied held torque, so one-action captures remain visible. A heatmap reports percentages separately by initial-state class; numerical tables retain counts.

Other inherited episode fields keep their previous meanings: maximum potential fraction is sampled at decision endpoints; absolute work is summed absolute net work per action, not electrical energy; approach is a separate legacy proximity diagnostic, not the new capture rule. Source snapshots and complete NPZ trajectories remain in the gitignored artifacts tree; compact records and human figures are retained under docs. Chinese companions are gitignored by the existing `*-CN.*` rule.
