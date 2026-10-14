# Torque and arm-range study: reproducibility contract

> Archive note: the method and commands below describe the controller at the time of the experiment. Reproduction requires its original source snapshot; the current `src/` has since changed. Collected figures, records and logs are in [experiment artifacts](../../../artifacts/rotary_pendulum/experiment-results/torque-and-arm-angle-range-comparison/).

The controller still replans every 100 ms. It has no sleep schedule, skipped control updates, or committed zero-torque interval. The braking continuation is a prediction only. Occasional zero-torque actions are reported as coast fraction; they do not represent controller sleep. Capture freezes the original simulation and is not evidence of continued balance.

## Cases and fixed choices

| Case | Controller torque cap | Soft arm range |
| --- | --- | --- |
| baseline | ±0.00918 N m | ±90 degrees |
| torque_2x | ±0.01836 N m | ±90 degrees |
| range_180 | ±0.00918 N m | ±180 degrees |
| both | ±0.01836 N m | ±180 degrees |

The physical actuator clip stays ±0.0204 N m, both physical damping coefficients stay zero, and arm angles remain unwrapped. Widening the range changes the decoder recovery threshold, environment excursion flag, artifact metrics, and plot lines together. The fixed inner margin remains 0.05 rad. The 33-point torque grid spans the selected cap, so its spacing doubles in the larger-torque cases. The baseline grid size, ten root bisections, 20 ms RK4 integration, 100 ms torque hold, 20 s deadline, and five-sample capture condition are otherwise unchanged.

The upper policy gain is 0.04 and its arm-kinetic weight is 1. The requested-work cap remains 0.0012148704 J. The initial comparison uses work-mismatch weight 1 and 20 recovery samples, which give a 0.4 s total prediction. These are declared heuristic/numerical settings, not fitted physical constants.

## Execution and source isolation

`run_study.py` contains one `main()` for source preparation, process management, paired-data checks, and report tables/plots. No classes, dataclasses, or helper functions are added. Its two arguments use the existing validation convention: `--config` and `--output`. Each case uses an isolated copy of the source tree, including an exact `experiment.patch`. The checked-in runtime defaults are not edited. The existing validation runner also saves source snapshots and hashes in every run.

The process receives an explicit `CUDA_VISIBLE_DEVICES`. Empty means CPU with `JAX_PLATFORMS=cpu`; GPU execution uses physical GPUs 0–3 and rechecks occupancy before each launch. It stops if more than four GPUs are occupied or a selected device is busy, following the current AGENTS.md allocation.

`smoke.json` checks all four cases on eight lanes for 0.2 s. `pilot.json` uses seed 20261011 and 16 random starts per stratum, plus four deterministic probes. `confirmation.json` uses seeds 20261008, 20261009, and 20261010, with 64 starts per stratum per seed. These confirmation seeds are the historical benchmark, so subsequent selected-configuration confirmation should use fresh seeds. All cases within a study share exactly equal initial arrays; the driver asserts equality.

Example CPU pilot, from the repository root:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=true MPLCONFIGDIR=/tmp/bcmpc-study-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-7 .venv/bin/python scripts/experiments/torque-and-arm-angle-range-comparison/run_study.py --config artifacts/rotary_pendulum/experiment-results/torque-and-arm-angle-range-comparison/records/initial_parameter_screening.json --output artifacts/rotary_pendulum/experiment-results/torque-and-arm-angle-range-comparison/raw-runs/pilot
```

The output directory must not already exist. Independent episode and torque candidates are batched; permitted independent GPU jobs run concurrently. CPU work is affinity-limited. Each physics chunk reports progress; the driver additionally reports process elapsed time during compilation and plotting.

## Data and validation

Collected human-readable plots are under the experiment’s `figures/`; numerical records and campaign settings are under `records/`. Original run logs and source copies are under `raw-runs/`, whose internal historical directory names are preserved. `episodes.csv` preserves every run's per-episode records with case and seed identifiers. `summary.csv` aggregates random strata separately from deterministic probes. `integration_audits.json` contains independent 2 ms NumPy replay of up to 256 complete held actions per run. These action audits are not full closed-loop integration refinement.

Capture is five consecutive 20 ms samples with upright-angle error at most 0.08 rad, arm speed at most 0.15 rad/s, and pendulum speed at most 0.20 rad/s. Arm position does not gate capture. Clean capture additionally requires no earlier excursion beyond that case's soft limit. Excursion fractions across different bounds describe different allowed ranges; they are not a common-bound comparison.

`maximum_potential_fraction` is the maximum pendulum potential divided by upright-rest potential, sampled at 100 ms decision endpoints. It is not a 20 ms continuous-trajectory maximum. `absolute_work_j` is summed absolute net work per held action, not electrical consumption or the integral of absolute instantaneous power. `max_work_error_j` concerns complete exact-work actions only; overrides and partial terminal actions are excluded. `max_energy_balance_j` covers active actions and measures total mechanical energy change minus delivered work. Phase-dependent overrides remain part of the controller.

Validation milestones: (1) finite CPU smoke, matching initial arrays, torque-cap and excursion assertions; (2) inspect full pilot figures and separate capture from containment; (3) repeat the four-way comparison across three seeds; (4) inspect any focused tuning wave before selecting another; (5) report numerical sensitivity and preserve English/Chinese interpretations. The experiment is a proof of concept and does not establish invariant containment or sustained balance.

## Completed extension and structure budget

The current source/controller reference is commit `8f3a4c58b53fdee7fe6919ed46252b9f4c2518de`; HEAD before this experiment was `dc2f15ee7dae3a59b390b54b1374c40bb90c2d16`. Runtime sources under `src/` match between those commits. Follow the simplest compact task-oriented implementation, with no new classes, dataclasses, wrappers, fallback behavior, try/with/except, or dependencies. The structure consists of the reused `run_study.main()` and one added `audit_closed_loop.main()` for independent NumPy-plant refinement. The audit's JIT closure directly invokes the existing decoder and policy. Both scripts remain between 40 and 300 lines. Existing encoder, policy, decoder, evaluator, reset generator, artifact writer, and physical configuration/model objects are reused.

Task data objects are study/case dictionaries and pending process records in the driver; batched physical states, torques, capture counters, peaks, and histories in the audit; and existing CSV rows/NPZ arrays/figure objects for delivery. No public data container is introduced. `occupancy` holds measured GPU memory/use pairs; `devices` is the explicit physical allocation; `pending` and `active` are queued/running subprocess records. These are execution bookkeeping, not controller parameters.

Experimental independent variables are controller torque multiplier, soft arm angle, recovery sample count, work-mismatch weight, seed, and sample count. Physical actuator limit, policy work gain/kinetic weight, torque-grid count, integration step, decision clock, and episode deadline remain fixed. The added audit varies only plant integration step (20 ms or 2 ms), holding capture and excursion sampling at 20 ms. `state` holds each plant's independently evolving physical state; `torque` is computed from that state; `success`, `count`, `elapsed`, and `peak` track episode completion, consecutive goal samples, elapsed active samples, and maximum absolute arm angle. The arrays are initialized directly from paired archived resets or zero counters.

The physical state has four measured/simulated components, in order: arm angle (unwrapped radians), downward-referenced pendulum angle (radians), arm angular speed (rad/s), and pendulum angular speed (rad/s). Reset boxes are fixed benchmark definitions, sampled uniformly by the existing seeded JAX generator:

| Stratum | Arm angle | Pendulum angle | Arm speed | Pendulum speed |
| --- | --- | --- | --- | --- |
| downward | −0.20 to 0.20 | −0.20 to 0.20 | −0.5 to 0.5 | −0.5 to 0.5 |
| moving | −0.50 to 0.50 | magnitude 0.40 to 2.60, random sign | −2 to 2 | −6 to 6 |
| near | −0.25 to 0.25 | upright ±0.25 | −1 to 1 | −1 to 1 |
| tight | −0.08 to 0.08 | upright ±0.12 | −0.15 to 0.15 | −0.30 to 0.30 |

The four probes are hanging rest, upright rest, and mirrored states with arm angle ±1.45 rad, pendulum angle ±0.4 rad, outward arm speed ±2 rad/s, and zero pendulum speed. They are unchanged when the soft boundary expands; they are not new-boundary probes.

`tuning.json` tests eight combined-setting variants on pilot seed 20261011: recovery samples 10/15/30/40 at weight 1, or weight 0/0.1/10/100 at 20 samples. `fresh_confirmation.json` compares baseline, both changes with weight 1, and both changes with weights 0 and 0.01 on fresh seeds 20261012–20261014. The upper policy is fixed throughout. No torque-grid-resolution adjustment is included.

`records/` retains per-episode tables, aggregate summaries and integration audits by study. Full source snapshots, initial arrays, trajectories and driver logs remain under `artifacts/rotary_pendulum/experiment-results/torque-and-arm-angle-range-comparison/raw-runs/`. Each experiment snapshot includes the exact torque/range patch and each trial includes SHA-256 source hashes. The driver can regenerate them from the unchanged runtime sources and saved campaign JSON. The closed-loop audit consumes the `confirmation` output and imports its `both` source snapshot; its `settings.json` identifies that source and numerical settings.

After inspecting occupancy outside the sandbox, run the four-case confirmation using:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 JAX_ENABLE_X64=true MPLCONFIGDIR=/tmp/bcmpc-study-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/experiments/torque-and-arm-angle-range-comparison/run_study.py --config artifacts/rotary_pendulum/experiment-results/torque-and-arm-angle-range-comparison/records/confirmation.json --output artifacts/rotary_pendulum/experiment-results/torque-and-arm-angle-range-comparison/raw-runs/confirmation
```

For the closed-loop audit, select an idle permitted GPU and use the combined-setting snapshot:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=true PYTHONPATH=artifacts/rotary_pendulum/experiment-results/torque-and-arm-angle-range-comparison/raw-runs/confirmation/machine-scannables/sources/both/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/experiments/torque-and-arm-angle-range-comparison/audit_closed_loop.py
```

Both commands require output destinations not to exist. The audit asserts the imported torque cap and arm limit, uses 780 initial states and two independently evolving plants, and reports every simulated second. Repeated decisions are sequential; independent lanes and plants are batched. Its retained NumPy 20 ms run is an implementation-sensitivity control, not expected to match JAX after many discrete root/candidate choices.

Milestone success flags: all finite values; exact paired-reset equality; torque/soft-bound assertions; historical baseline count reproduction; completed pilot, tuning, and fresh-seed runs; independent fine-plant completion; four controller tests plus script Ruff/format checks. Behavioral success is reported separately: no general downward swing-up or sustained-balance claim. English and Chinese summaries distinguish containment, capture, zero torque, terminal freezing, and actual scheduled sleep.
