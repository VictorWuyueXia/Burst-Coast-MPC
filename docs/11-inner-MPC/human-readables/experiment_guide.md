> Current revision: arm-centering cost and GPU-native SQP/CEM are implemented. Use the [GPU experiment guide](gpu_experiment_guide.md). The initial SciPy/CPU comparison below is historical.

# Running the two inner-MPC solvers

The inner controller is implemented with constrained SLSQP and batched cross-entropy sampling (CEM).
Both optimize direct 100 ms torque commands, with squared effort and the frozen twin-critic terminal cost.
Both check the same 2 ms arm-path constraints and independently replay candidate plans on a 1 ms grid.
The selected portable 4.59 mN m model is used; the heuristic does not restrict the optimized controls.

## Implementation milestones and structure

| Milestone | Concrete validation | Success flag |
|---|---|---|
| Shared model | Compare frozen network outputs with exported reference vectors, RK4 with NumPy, and reward with the existing stationary transition | Numerical parity tests pass |
| Optimization | Compare automatic derivatives with finite differences; reject a trajectory crossing between decision endpoints | Derivative and acceptance tests pass |
| Two solvers | Exercise converged, infeasible and iteration-limited cases; repeat CEM with the same random key | Explicit acceptance/failure and reproducibility checks pass |
| Experiment preparation | Save paired states, parameter designs, model hashes and source snapshots before execution | Prepared design exists and source verification passes |
| Receding-horizon smoke | Run several decisions, shift previous plans, preserve failed episodes in the records | Complete records and comparison figures exist |

The implementation follows the compact line-of-logic rule: no new classes, dataclasses, controller wrappers, fallback controls, CLI framework or dependencies. Four numerical functions retain the structure budget: `rollout`, `evaluate`, `solve_inner`, `sample_inner`. `rollout` has one scan body; the existing RK4 integration body now optionally records microsteps and supports the finer audit grid. JIT gradient/Jacobian transforms are library callables, not new wrapper functions.

The experiment uses one `main` function. Preparation and reporting are separate straight-line scripts, grouped by their actual tasks. Three tests cover parity, derivatives and feasibility. Source files remain between 40 and 300 lines. Independent data remain grouped as state/actions/epigraph, networks/settings, seeds/key, trajectory/values/Jacobian, and results/records. The public inputs and output records are specified in the [implementation contract](../machine-scannables/implementation_contract.md).

## Prepare without running an experiment

Run from the repository root using the existing virtual environment:

```bash
.venv/bin/python scripts/experiments/inner-mpc/prepare.py \
  --output artifacts/rotary_pendulum/inner-mpc/my-parameter-study \
  --study parameters --decisions 1 --seed 20261019
```

This creates 32 configurations on 32 paired starting states. The four physical groups are downward, moving swing-up, near-upright and outward motion near the arm boundary. They are explicitly constructed in physical coordinates. The present distributions are narrow numerical benchmark neighborhoods, not a representative global control-performance test.

| Parameter | Prepared comparison |
|---|---|
| Prediction length, counted in 100 ms decisions | 5, 10, 20, 40: 0.5, 1, 2, 4 seconds |
| SLSQP iteration cap | 100 versus 200 |
| SLSQP tolerance | 1e-7 versus 1e-6 |
| CEM population per initial seed | 1024 versus 4096 |
| CEM distribution-update count | 5 versus 10 |
| CEM initial standard deviation in normalized torque | 0.5 versus 0.25 |
| CEM feasible elite count | 64 versus 128 |

The parameter design changes one solver setting at a time from its baseline, separately at each horizon. Horizon changes alter the objective itself, so compare solvers within a horizon. Each solver receives the same three initial sequences: actor rollout, constant +0.5 torque fraction, and constant −0.5 fraction. Later replans replace the actor sequence with the shifted previously selected plan and append the actor's terminal command.

`--study smoke` prepares eight configurations and four paired starts. `--decisions 3` exercises three replans, or 0.3 seconds of executed control. Increasing `--decisions` enables longer receding-horizon runs, but the current reports are solver diagnostics rather than a complete capture/balance evaluation campaign. The larger parameter study is prepared separately from its execution.

## Execute a prepared design

For CPU execution:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/inner-mpc-mpl \
PYTHONPATH=src .venv/bin/python scripts/experiments/inner-mpc/run.py \
  artifacts/rotary_pendulum/inner-mpc/my-parameter-study
```

For GPU execution, use the escalated terminal after the shared-server occupancy check. The current default device preference is 6–7; choose an idle device, for example:

```bash
CUDA_VISIBLE_DEVICES=6 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/inner-mpc-mpl PYTHONPATH=src \
.venv/bin/python scripts/experiments/inner-mpc/run.py \
  artifacts/rotary_pendulum/inner-mpc/my-parameter-study
```

Before importing JAX, the GPU runner checks occupancy and refuses an occupied requested device or a server with more than four occupied GPUs. Run independent designs on different idle devices; merely exposing two GPUs does not shard this runner. Four threads execute independent solves; CEM batches all candidate trajectories within each solve on its selected device. Time recursion remains sequential.

The runner verifies the exported weight hash and current physics, and rejects changed sources since preparation. Prepare a new directory after modifying code. Completed or partially recorded runs cannot be overwritten by another execution. Progress appears at compilation, each local-solver iteration, each CEM generation and each completed solve.

## Read the results correctly

`human-readables/solver_comparison.png` shows acceptance and solver-plus-validation timing. `machine-scannables/summary.json` includes attempted solves, accepted solves and episodes completing every requested decision. `solves.csv` records individual comparisons. A rejected episode stops and remains in the denominator; no zero-action or teacher fallback is applied.

Per-step subdirectories retain initial states, initial sequences, all final candidate plans, 2 ms predictions, 1 ms audit trajectories, critic values, margins, solver reasons and CEM populations/random keys. `design.json` records the checkpoint, source hashes, numerical settings and hardware; `sources/` preserves the implementation. `warmup.json` records compilation/warm-up time separately. Reports can be regenerated with `report.py RUN_DIRECTORY`.

Solve timing includes optimization, transfers and the finer-grid audit. Shared initial-sequence preparation is recorded separately as `seed_batch_s`; queueing and disk output are excluded. These concurrent timings are not end-to-end real-time guarantees. CEM timing currently includes preserving population records in host memory. Neither fixed iteration counts nor fixed sample counts impose equal wall-clock budgets; use the measured tradeoff for later matched-budget selection.

For multiple replans, solver trajectories diverge after the first decision. Compare first-decision objectives on identical states; do not treat pooled later objectives, or medians over different accepted subsets, as a paired solver-quality ranking. `predicted_first_rl_cost` is the nominal prediction diagnostic, not an independently accumulated long-run return.

The first smoke evidence is linked in the [package summary](../interpretation_summary.md). Critic calibration, a long closed-loop task comparison, equal-latency tuning and the wider parameter matrix remain separate experiments. Solver convergence does not establish critic accuracy or task success. A converged boundary solution can still fail the finer-grid audit; report that discrepancy instead of silently adding a boundary margin or accepting an excursion.
