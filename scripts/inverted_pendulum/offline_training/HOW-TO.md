# Offline Structured Critic Training

This folder trains the structured residual critic from Monte Carlo `rl_steps.csv`.
It does not call the simulator, MPC solver, `Coordinator`, or `burst-coast-mpc` CLI.

## 0. Prepare The Environment

Use the project environment on the GPU workstation:

```bash
conda env update -f environment.yml
conda activate burst-coast-mpc
```

Install the CUDA-enabled PyTorch build that matches the NVIDIA 4070 workstation
driver if the default resolver does not select it. This Mac environment is only
suitable for source checks.

## 1. Generate Or Select Data

Generate Monte Carlo data with the existing command:

```bash
burst-coast-mpc --inverted-pendulum montecarlo --epochs 40
```

The trainer reads artifact folders under `artifacts/MonteCarloData` by default,
and each selected run must contain `rl_steps.csv` plus `config.json`.

The loader uses the logged `return-cost` target and the real logged
`solve-time-s`. There is no fitted solve-time model in this training pass.

## 2. Edit The Minimum Config

Training config lives at `scripts/inverted_pendulum/offline_training/config.yaml`.

Keep the file small: `data-roots`, `output-root`, run name, seed, split, batch,
learning-rate, epoch count, and plotting-grid count only.

## 3. Run Training

Launch from the repository root:

```bash
python scripts/inverted_pendulum/offline_training/train.py
```

```text
offline-training load ...
offline-training train ...
offline-training finish ...
offline-training snapshot_dir=...
```

Training is hard-coded to one GPU with `accelerator="gpu"`, `devices=1`, and
`precision="32-true"`. If CUDA is absent, the command should fail directly.

## 4. Inspect The Snapshot

Snapshots are written to:

```bash
artifacts/model-snapshots/<run-name>_<timestamp>/
```

Core reload artifacts:

- `config.json`: resolved data roots, split seed, training hyperparameters, and physical constants.
- `model_structure.txt`: residual MLP architecture used for this snapshot.
- `normalization.json`: train-split feature order, mean, and standard deviation.
- `lambda.json`: learned nonnegative physical weights for time, compute, and burst effort.
- `critic_state_dict.pt` and `checkpoints/`: final and Lightning checkpoint weights.

Training and validation artifacts:

- `training_curve.csv`: one row per epoch with `train_mse` and `val_mse`. A useful run has validation MSE falling early, then flattening. Stop adding epochs when validation MSE no longer improves; add data or change features when both train and validation MSE flatten high.
- Terminal output: each epoch prints `offline-training epoch epoch=... train_mse=... val_mse=...`.
- `metrics.json`: final train MSE, validation MSE, validation MAE, physical-only validation MSE, split row counts, and learned lambdas. The critic is useful only if validation MSE is clearly below `physical_validation_mse` and the RMSE is small relative to the return-cost scale.
- `region_metrics.csv`: validation error split by coarse action and state regions. Regions with few rows or much larger MSE point to missing data coverage.
- `observed_policy_sanity.csv`: held-out logged actions sorted by predicted Q. This ranks only actions that were actually sampled in the Monte Carlo logs; it is not a guarantee that a dense action-grid policy is good.

Q-surface plots:

- `q_surface_downward.png`: predicted Q over `(bbar, hbar)` for the held-out state with the lowest cosine, usually the most downward state.
- `q_surface_high_energy.png`: predicted Q over `(bbar, hbar)` for the held-out state with the largest normalized energy error.
- `q_surface_near_upright.png`: predicted Q over `(bbar, hbar)` for the held-out state closest to upright by cosine.
- `q_surface_terminal.png`: predicted Q over `(bbar, hbar)` for one held-out terminal row.

In every Q-surface, x is `bbar`, y is `hbar`, color is predicted Q, and the white marker is the logged action from the selected validation row. Smooth surfaces are expected from the MLP. Treat sharp ridges, edge-only minima, or nearly identical surfaces across very different states as warning signs.

## 5. Judge Whether The Critic Is Good Enough

Use these checks before putting the critic in a controller loop:

- Learning curve: validation MSE should decrease and then plateau. A widening train-validation gap means overfitting; both curves stuck high means the data or features are insufficient.
- Baseline lift: `validation_mse` should beat `physical_validation_mse` by a meaningful margin. A tiny improvement means the residual network has learned little beyond the physical terms.
- Error scale: compare validation RMSE to the validation return-cost standard deviation and to the action-ranking margins that matter for control.
- Region coverage: every region in `region_metrics.csv` should have enough rows to be meaningful, and no deployment-important region should dominate the error.
- Policy sanity: low predicted-Q rows in `observed_policy_sanity.csv` should usually have low return cost. Large underpredictions on expensive rows mean the critic may select bad actions.
- Surface sanity: Q-surface minima should move with state and should not always live on the same boundary unless the real MPC behavior supports that.

For the `structured-critic_20260614T012645` snapshot, the data is not enough for a reliable critic yet. It used 60 runs, 950 rows, 746 train rows, and 204 validation rows. Validation MSE was about 666,991 versus 694,193 for physical-only, only about a 4 percent improvement. Validation RMSE was about 817 while the validation return-cost standard deviation was about 515, so the residual critic is still worse than a rough mean-scale predictor. Several validation regions have only 25 rows, and the observed policy sanity file shows many expensive rows predicted far too cheaply.

A practical next target is 300 total Monte Carlo runs, about 240 more than this snapshot, which should give roughly 4,750 rows at the current 15.8 rows per run. That gives the residual model several rows per trainable weight and makes each coarse validation region less fragile. If the 300-run learning curve still has high validation MSE and weak baseline lift, move to 600 total runs before changing the model architecture.
