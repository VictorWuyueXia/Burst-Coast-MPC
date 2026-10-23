# GPU-only inner MPC experiments

**Official baseline, 2026-10-23: [MPC+RL](mpc_rl_baseline.md).** The 16-generation CEM controller with exact LQR stage weights and the frozen RL terminal value is now the development baseline. RL remains enabled after capture; no local handoff is used. Earlier study descriptions below are historical.

Separate requested test: [exact LQR stage cost + frozen RL terminal value](lqr_stage_cost.md). Run `lqr_cost_run.py confirm 16` or matched pilots `screen 16` / `screen 32` under the CUDA environment below. This test uses CEM throughout; the retained hybrid remains unchanged. The corrected integration grids are 1 ms prediction / 0.5 ms verification. Run `lqr_cost_report.py screen` or `confirm` only after all corresponding runs complete. See the [record contract](../machine-scannables/lqr_stage_cost_contract.md).

Current study: **CEM swing-up with the existing local LQR**, all sessions **40 seconds**, scored by total upright time and longest continuous hold inside ±5°. Weak angle and speed additions are disabled; the broad arm-centering cost remains. See [LQR construction and handoff](lqr_handoff.md). SQP and weak-speed studies below are historical reference material.

From the repository root, first check `nvidia-smi` and select an idle GPU (prefer 6–7; do not launch if more than four devices are occupied). The runner also checks occupancy. GPU execution needs the approved host terminal, outside the filesystem sandbox.


Run the hybrid with `scripts/experiments/inner-mpc/hybrid_run.py confirm 0.5` under the same CUDA environment below; use dwell `0` for the matched CEM baseline. Screening uses `screen 0.5` and `screen 1`. All runs preserve full 40-second trajectories. `hybrid_report.py screen` or `confirm` requires all runs in that stage to have completed and uses the same panel-alignment environment as the older report. The dated output directories are never overwritten: reproducing a completed run requires an explicitly new destination. No CPU solver or SQP comparison is launched.

```bash
CUDA_VISIBLE_DEVICES=6 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/experiments/inner-mpc/speed_run.py confirm 0.03
```

Use `screen` for eight starts or `confirm` for 32. Both execute 400 decisions, with no performance-based early stop. Strengths in the current confirmation are 0, 0.01, 0.03 and 0.1. Existing output directories are deliberately not overwritten. To recreate a completed historical run, use its archived sources and an explicitly new output destination.

`speed_report.py confirm` analyzes completed runs, produces CSV/JSON/NPZ data in `machine-scannables/`, and figures in `human-readables/`. Its Python environment must provide the nature-figure panel-alignment checker on `PYTHONPATH`, as in the preceding report workflow. Actual simulation, optimization and rollout run on GPU; plotting and archival arithmetic do not consume solver CPU benchmarks.


The current running cost includes squared torque and squared normalized arm angle.
The new centering term encourages returning toward zero before the hard arm boundary becomes active.
The terminal value remains the selected frozen TD3 twin-critic continuation value.
See the [formulation](formulation.md) and [precise GPU contract](../machine-scannables/gpu_contract.md).

## Which solvers now run

The historical comparison used compiled batched CEM and a JAX-native SQP implementation.
The latter is a new member of the SQP family, not SciPy/Kraft SLSQP running magically on CUDA.
It uses damped BFGS, linearized constraints, ADMM quadratic subproblems and batched trial steps.
The historical SciPy implementation remains available, but the present experiment does not benchmark it or recommend CPU deployment.
The reason for replacing its execution path is that GPU model evaluations alone leave its optimizer and host/device synchronization loop on the CPU.
See [SciPy's SLSQP interface](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).

Both new solvers batch independent simulations and their three initial guesses on one GPU.
CEM additionally batches candidate populations; SQP additionally batches differentiation and trial steps.
Sequential physical time and optimizer iterations remain sequential on the device.
No CasADi export, added dependency, controller wrapper, boundary reserve or fallback controller is introduced.
Both return the best evaluated feasible candidate within their numerical budget and independently replay it at 1 ms.
Acceptance means an executable audited plan, not a convergence, stability or recursive-feasibility certificate.

## Milestones and validation

| Milestone | Check | Success flag |
|---|---|---|
| Centering formulation | Direct quadrature, zero-weight equivalence and directional finite differences | Cost and derivative tests pass |
| GPU SQP | Known constrained quadratic program, multiplier sign and exact arm-grid compression | Analytic QP and constraint tests pass |
| Common GPU execution | Frozen-network parity, repeated CEM key, invalid-state rejection | GPU tests pass |
| Larger paired study | Same starts, checkpoint and physical audit at each solver setting | Full solve records including failures |
| Numerical tuning | Compare populations, generations, QP accuracy and iteration budgets | Explicit cost/time changes recorded |
| Replanning | Shift accepted plans, execute first 100 ms, stop rejected episodes | State/plan transitions audited |

Structure follows the project's compact line-of-logic rule: two numerical functions in each GPU solver file, their scan bodies, one runner `main`, and a straight-line comparison script.
No classes, dataclasses, thin wrappers, fallback values or configuration framework.
Data and parameter roles are enumerated in the machine contract.
The project remains a theoretical simulation benchmark, not a real-time controller release.

## Prepare and execute

From the repository root, prepare a new design without initializing CUDA:

```bash
.venv/bin/python scripts/experiments/inner-mpc/prepare.py \
  --output artifacts/rotary_pendulum/inner-mpc/my-gpu-study \
  --study gpu --decisions 1 --seed 20261019
```

This prepares 36 settings and 64 paired physical starts, including mirrored states.
Horizons are 0.5, 1 and 2 s; centering rates are 0.1, 1 and 10 cost units/s.
SQP compares 20/40 iterations; CEM compares 256/1024 samples per initial plan.
The four initial-state groups are downward, moving swing-up, near-upright and outward arm motion.
They are controlled benchmark neighborhoods, not a globally representative state distribution.
For a custom follow-up, explicitly edit the prepared `design.json` and record the rationale before running.

Choose an idle GPU after occupancy inspection, then run through the escalated terminal:

```bash
CUDA_VISIBLE_DEVICES=6 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/inner-mpc-mpl PYTHONPATH=src \
.venv/bin/python -u scripts/experiments/inner-mpc/gpu_run.py \
  artifacts/rotary_pendulum/inner-mpc/my-gpu-study
```

The runner checks GPU occupation before importing JAX and rejects occupied requested devices or more than four occupied GPUs.
Run disjoint designs on GPUs 6 and 7 to use both; one batch is not automatically split across exposed devices.
A source change requires a freshly prepared directory. Existing result files cannot be overwritten.
All optimization, physical rollout, derivatives, candidate selection and audit use GPU arrays.
The host handles preparation, file I/O, progress and small summary statistics.
Compilation and a full numerical warm-up are outside the measured solve interval.

## Read the artifacts

`machine-scannables/design.json` freezes settings, model metadata, source hashes and device identity.
Each configuration/decision stores `input.npz`, `plans.npz` and `history.npz`.
The final plans include every initial guess, nominal/refined trajectories, costs and physical checks.
CEM histories store distributions and keys instead of expensive per-generation population transfers.
SQP histories include QP residuals, stationarity diagnostics, trial lengths and best-so-far costs.
`solves.csv` retains rejected attempts; rejection ends that episode without another controller taking over.

`elapsed_s` measures whole-batch solve and audit latency. `throughput_s` is that time divided by the batch size.
A 64-state batch taking one second has 15.6 ms amortized throughput, but every state waits about one second.
Neither number includes disk output; seed preparation is recorded separately.
Compare objectives only within the same horizon, centering weight and initial state.
Across weights and after trajectories diverge, compare physical behavior and completion counts instead.

Run `compare_gpu.py OUTPUT_DIRECTORY RUN_DIRECTORY ...` to audit accepted plans and shifted replans, regenerate paired comparisons and plot the cost/time tradeoff.
Human figures go to `human-readables`; detailed configuration, paired differences and audit records go to `machine-scannables`.
The current results and best-so-far judgment are in the [artifact interpretation](../../../artifacts/rotary_pendulum/inner-mpc/interpretation_summary.md).
