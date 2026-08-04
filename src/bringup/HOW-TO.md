# Burst-Coast MPC HOW-TO

This package provides the command surface for inverted-pendulum MPC, Monte Carlo
data generation, critic deployment, and online critic updates. RL never outputs
torque directly: it scores burst-horizon candidates, then MPC solves the selected
candidate and emits the torque sequence.

## Environment

```bash
conda env update -f environment.yml
conda activate burst-coast-mpc
pip install -e .
```

Run commands from the repository root so artifact paths in the configs resolve
to the intended tracked and ignored directories.

## Rotary-Pendulum Physics Simulation

```bash
bcmpc --rotary-pendulum
```

This runs the coupled QUBE-Servo 3 torque-driven model at 500 Hz and opens the
combined 3D and time-series monitor. Until its MPC and RL paths are implemented,
each replan samples a constant signed torque magnitude and a duration of up to
three coupled natural periods.

## MPC Baseline

```bash
bcmpc --inverted-pendulum mpc-only --no-visual
```

This loads `src/inverted_pendulum/configs/default-config.yaml`, runs the fixed
natural-period MPC controller, and writes dense step artifacts under
`artifacts/inverted_pendulum/experiments` when artifact recording is enabled.

## Monte Carlo Data

```bash
bcmpc --inverted-pendulum montecarlo --epochs 40
```

This loads `data-generation-config.yaml`, samples initial states and normalized
actions, solves one MPC plan per sampled action, and writes `rl_steps.csv` with
return-cost labels. The action realization is fixed:

```text
H = max(1, round(hbar * Hmax))
B = max(1, round(0.5 * bbar * H))
C = H - B
```

## Offline Critic Snapshot

```bash
python src/inverted_pendulum/training_scripts/offline_training/train.py
```

This script trains the structured residual critic from Monte Carlo
`rl_steps.csv` files. It is intentionally outside the runtime package and is
hard-coded for one CUDA GPU. Snapshots are written under
`artifacts/inverted_pendulum/model-snapshots`.

## Compute-Time Fit

```bash
python src/inverted_pendulum/training_scripts/cmp-time-fitting/fit_cmp_time_model.py
```

The deployed critic uses a fitted nonnegative solve-time model over realized
MPC dimensions `H`, `B`, and `B * H`. Copy the selected generated artifact into
the configured frozen time-model directory before running intelligent mode.

## Intelligent Deployment

```bash
bcmpc --inverted-pendulum intelligent --no-visual
```

This loads `intelligent-config.yaml`, evaluates the frozen critic over the full
normalized action grid, selects the minimum predicted cost, solves that MPC
candidate once, and executes the resulting burst-coast torque plan. Use
`--with-exploration` only for stochastic grid-sampling checks.

## Online Training Mode

```bash
bcmpc --inverted-pendulum train --no-visual
```

This loads `online-training-config.yaml`, samples grid actions with critic
cost-softmax exploration, records executed RL segments, then applies fitted-Q
updates to the loaded critic and saves an online snapshot.

## Artifact Checks

Each successful runtime command should leave `config.json`, `metadata.json`,
`manifest.json`, `run.log`, and mode-specific CSV or figure artifacts. Treat
missing config fields, unsupported time-model terms, negative solve-time
predictions, solver failures, and empty RL transition lists as hard errors.
