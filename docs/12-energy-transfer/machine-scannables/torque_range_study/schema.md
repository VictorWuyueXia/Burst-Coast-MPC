# Torque and arm-range study: reproducibility contract

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

The process receives an explicit `CUDA_VISIBLE_DEVICES`. Empty means CPU with `JAX_PLATFORMS=cpu`; GPU execution accepts only 6 or 7 and rechecks the selected device before each launch. This experiment follows the current AGENTS.md allocation, superseding the historical runner's 0–3 restriction only in its isolated copies. A busy-server GPU pause still requires a user decision; the occupancy check is not authorization to override it.

`smoke.json` checks all four cases on eight lanes for 0.2 s. `pilot.json` uses seed 20261011 and 16 random starts per stratum, plus four deterministic probes. `confirmation.json` uses seeds 20261008, 20261009, and 20261010, with 64 starts per stratum per seed. These confirmation seeds are the historical benchmark, so subsequent selected-configuration confirmation should use fresh seeds. All cases within a study share exactly equal initial arrays; the driver asserts equality.

Example CPU pilot, from the repository root:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=true MPLCONFIGDIR=/tmp/bcmpc-study-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-7 .venv/bin/python docs/12-energy-transfer/machine-scannables/torque_range_study/run_study.py --config docs/12-energy-transfer/machine-scannables/torque_range_study/pilot.json --output artifacts/rotary_pendulum/torque-range-study/pilot
```

The output directory must not already exist. Independent episode and torque candidates are batched; permitted independent GPU jobs run concurrently. CPU work is affinity-limited. Each physics chunk reports progress; the driver additionally reports process elapsed time during compilation and plotting.

## Data and validation

Human plots are under `human-readables/`; all numerical records, logs, campaign settings, and source copies are under `machine-scannables/`. `episodes.csv` preserves every run's per-episode records with case and seed identifiers. `summary.csv` aggregates random strata separately from deterministic probes. `integration_audits.json` contains independent 2 ms NumPy replay of up to 256 complete held actions per run. These action audits are not full closed-loop integration refinement.

Capture is five consecutive 20 ms samples with upright-angle error at most 0.08 rad, arm speed at most 0.15 rad/s, and pendulum speed at most 0.20 rad/s. Arm position does not gate capture. Clean capture additionally requires no earlier excursion beyond that case's soft limit. Excursion fractions across different bounds describe different allowed ranges; they are not a common-bound comparison.

`maximum_potential_fraction` is the maximum pendulum potential divided by upright-rest potential, sampled at 100 ms decision endpoints. It is not a 20 ms continuous-trajectory maximum. `absolute_work_j` is summed absolute net work per held action, not electrical consumption or the integral of absolute instantaneous power. `max_work_error_j` concerns complete exact-work actions only; overrides and partial terminal actions are excluded. `max_energy_balance_j` covers active actions and measures total mechanical energy change minus delivered work. Phase-dependent overrides remain part of the controller.

Validation milestones: (1) finite CPU smoke, matching initial arrays, torque-cap and excursion assertions; (2) inspect full pilot figures and separate capture from containment; (3) repeat the four-way comparison across three seeds; (4) inspect any focused tuning wave before selecting another; (5) report numerical sensitivity and preserve English/Chinese interpretations. The experiment is a proof of concept and does not establish invariant containment or sustained balance.
