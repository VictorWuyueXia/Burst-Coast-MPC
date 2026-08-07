# Burst-Coast MPC HOW-TO

This package provides the command surface for rotary and inverted-pendulum MPC,
Monte Carlo data generation, critic deployment, and online critic updates. RL
never outputs torque directly: it scores burst-horizon candidates, then MPC
solves the selected candidate and emits the torque sequence.

## Environment

```bash
conda env update -f environment.yml
conda activate burst-coast-mpc
pip install -e .
```

Run commands from the repository root so artifact paths in the configs resolve
to the intended tracked and ignored directories.

## Rotary-Pendulum MPC

```bash
bcmpc --rotary-pendulum
bcmpc --rotary-pendulum mpc-only --no-visual
```

The default command runs the coupled QUBE-Servo 3 model at 500 Hz and opens the
combined 3D and time-series monitor. At each replan, the controller solves every
configured split over `H = ceil(2 Tn / Ts)`, selects the minimum objective, and
executes the complete burst and exact zero-coast sequence before solving again.
The headless form skips realtime plotting but still writes static figures. Rotary
RL paths remain intentionally empty.

Rotary runs compose `physics.yaml`, `mission.yaml`, one runtime policy,
`mpc.yaml`, and `artifacts.yaml`; interactive runs additionally load
`visual.yaml`. Each session writes `steps.csv`, candidate-level `plans.csv`, five
figures, and `interpretation_summary.md` under
`artifacts/rotary_pendulum/experiments`.

## MPC Baseline

```bash
bcmpc --inverted-pendulum mpc-only --no-visual
```

This composes the inverted physics, mission, runtime, MPC, and artifact domains,
runs the fixed natural-period controller, and writes dense step artifacts under
`artifacts/inverted_pendulum/experiments` when artifact recording is enabled.

## Monte Carlo Data

```bash
bcmpc --inverted-pendulum montecarlo --epochs 40
```

This composes the data-generation mission, runtime, MPC override, artifact, and
sampling domains, solves one MPC plan per sampled action, and writes
`rl_steps.csv` with return-cost labels. The action realization is fixed:

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

This composes the intelligent mission, runtime, artifact, and RL overrides with
the physical MPC domains, evaluates the frozen critic over the full normalized
action grid, selects the minimum predicted cost, solves that MPC candidate once,
and executes the resulting burst-coast torque plan. Use `--with-exploration`
only for stochastic grid-sampling checks.

## Online Training Mode

```bash
bcmpc --inverted-pendulum train --no-visual
```

This composes the online-training overrides with the physical MPC domains,
samples grid actions with critic cost-softmax exploration, records executed RL
segments, then applies fitted-Q updates to the loaded critic and saves an online
snapshot.

## Artifact Checks

Each successful runtime command should leave `config.json`, `metadata.json`,
`manifest.json`, `run.log`, and mode-specific CSV or figure artifacts. Treat
missing config fields, unsupported time-model terms, negative solve-time
predictions, solver failures, and empty RL transition lists as hard errors.
