# Heuristic implementation and validation plan

## Scope and discipline

Implement the analytical encoder → three-energy work policy → full-state work decoder in `src/rotary_pendulum/heuristic/`. The authorized recovery revision includes nonzero zero-work roots, predictive braking priority and phase-aware selection with explicit work overrides. Keep zero damping, no arm centering, and a nonterminal ±90° environment limit. Capture still uses upright angle and both speeds for five 20 ms samples.

Use the simplest compact task-oriented implementation: substantive code files of 40–300 lines, no thin wrappers, unnecessary CLI/configuration, try/with/except, new classes or dependencies. Do not add hidden coast fallbacks. Batch independent episodes, torque candidates and trial settings; only time integration and root refinement remain sequential. The function budget below is retained from the original implementation.

Only physical GPUs 0–3 may be used. The initial recovery pilot completed before this restriction; all subsequent pilots, confirmations and fine replay complied. Allocation is validated before JAX initialization. Report progress at least every simulated second and monitor elapsed runtime.

## Structure budget

| File / workflow | Functions and roles |
| --- | --- |
| `heuristic/energy.py` | `encode(x)` computes the three physical body energies; `policy(energy, work_gain, kinetic_weight)` returns signed work using those energies only |
| `heuristic/decoder.py` | `decode(x, requested_work, work_weight, recovery_steps)` selects torque and diagnostics; nested `predict(torques, steps)` batches pulse-plus-braking prediction, `advance(carry,index)` integrates one sample and tracks peak angle, `bisect(index,brackets)` refines parallel work roots |
| `heuristic/evaluation.py` | `evaluate(initial, parameters, recovery_steps, decisions, mode)` runs one chunk; nested `advance(state,unused)` handles decisions and `physics(state,unused)` preserves 20 ms terminal semantics and actual work |
| `heuristic/artifacts.py` | `write_artifacts(output,traces,labels,settings)` writes episode tables, numerical trajectories and representative figures |
| `scripts/validate_rotary_heuristic.py` | `main()` validates a campaign and allocation, prepares paired resets, distributes trial lanes, saves provenance and audits complete-action work |
| `machine-scannables/audit_decoder_recovery.py` | Existing `main()` classifies archived crossings, evaluates zero-work and braking probes, and runs paired 20 ms/2 ms full closed-loop replay |
| `tests/test_energy_heuristic.py` | Four tests: encoder/policy information contract; mirrored zero-work reversal and singular states; lossless/soft-boundary/capture and early braking; rollout terminal accounting and artifacts |

Existing objects: physical configuration and model, `EnvState`, batched state/energy/torque arrays, root brackets and masks, predicted endpoint/peak/work arrays, recovery masks and mode flags, trial-parameter arrays, trace dictionaries, table rows and figures. No new classes, dataclasses, wrappers or public helper containers. The fine audit extends its existing main rather than adding another script or abstraction.

## Independent variables and parameters

Measured state `x` contains arm angle, downward-referenced pendulum angle and their signed velocities. `energy` contains arm kinetic, full pendulum-body kinetic and pendulum potential energy in joules; `requested_work` is the upper command in joules. Selected torque is in N m. The upper policy does not receive phase, position, velocity signs or history.

Four controller tunables: dimensionless `work_gain` and `kinetic_weight` in the policy; dimensionless `work_weight` for normalized work mismatch; integer `recovery_steps` for total prediction samples including the five-sample held pulse. It must exceed five and be divisible by five. Selected values are respectively 0.04, 1, 1 and 20. Decoder `work_weight,recovery_steps` replace the old `arm_weight,coast_steps` rather than expanding the budget.

Fixed numerical choices: 100 ms control hold; 20 ms RK4; 33 torque grid points within ±0.00918 N m; ten bisections; work tolerance 0.0001 times upright-rest energy; work cap 0.04 times that energy; inner angle margin 0.05 rad; final predicted arm speed at most 0.15 rad/s. Nominal masses, inertias and gravity come from existing physics. These constants are declared and audited, not additional swept policy parameters.

Campaign variables are seed, episodes per stratum, decision/chunk counts, controller mode and output path. CLI remains only `--config` and `--output`. Runtime device and precision settings are provenance. No training or deployment is included.

## Stages, validation and status

| Stage | Procedure and success flag | Outcome |
| --- | --- | --- |
| 1. Physical/interface contract | Verify zero damping, exact energy definition, energy-only upper input, off-center capture and continued boundary crossing | Passed behavioral and repository regressions |
| 2. Decoder behavior | Fine-step mirrored zero-work reversal; positive-work request overridden by early braking; unresolved recovery explicitly marked; exact upright rest preserved; actual work includes partial terminal actions | Passed; numerical failure returns nonfinite command and is rejected by the runner |
| 3. Parallel pilots | Inspect trajectories between horizon/weight waves; measure reversals, potential gain, mismatch, capture and excursions separately | 20 settings evaluated; 0.4 s horizon selected; removing terminal-speed gate degraded capture and was reverted |
| 4. Paired confirmation | Verify initial-array equality against old decoder; three seeds × 64 starts/stratum; audit 256 complete actions/seed | Capture and range control improved; zero downward captures and 24/768 random excursions remain, so general swing-up/range guarantee did not pass |
| 5. Closed-loop refinement | Same 16 states/stratum plus four probes; independent 20 ms and 2 ms NumPy plants, same full-state JAX decoder and capture clock | Broad improvement persists; individual outcomes vary; report both rather than claiming integration independence |
| 6. Handoff | Full suite, Ruff/format, Mypy; human report and plot separate from machine records; English/Chinese documentation | 115 tests passed; style/type checks passed; bilingual results written |

The upper policy was fixed during the recovery experiments to isolate this decoder revision. Earlier soft-penalty experiments remain historical and require archived code/config schemas. Pilot refinements stopped once repeated parameter changes failed to produce downward swing-up. A two-pulse transfer-aware decoder is a suggested next experiment, outside this implementation budget.

## Reporting contract

The [formulation](formulation_plan.md) defines energy and actual work. The [recovery contract](formulation/decoder_recovery_contract.md) specifies the current algorithm. Re-encode actual physical motion every step. Log actual minus requested work, mode, root/recovery counts and predicted terminal speed. Never claim an override fulfilled the original energy plane. No hidden work-debt state is added.

Range recovery is a finite sampled prediction with a simple backup controller, not a hard invariant guarantee. The environment stays soft and nonterminal. Report any-excursion episodes, in-bound endpoint capture and capture without earlier excursion separately. Capture ends the episode and is not sustained hold.

Human reports and PNGs are separate from machine NPZ/CSV/JSON and source snapshots. The [interpretation summary](../../artifacts/rotary_pendulum/experiment-results/physical-energy-transfer-controller/interpretations/interpretation_summary.md) states conclusions, limitations and reproduction commands. Compact records are retained in the documentation; full runtime artifacts are gitignored. Chinese companions use `-CN` and are gitignored as requested.
