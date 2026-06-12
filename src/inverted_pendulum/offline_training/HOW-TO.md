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
burst-coast-mpc --inverted-pendulum generate-montecarlo-data --epochs 40
```

The trainer reads artifact folders under `artifacts/MonteCarloData` by default,
and each selected run must contain `rl_steps.csv` plus `config.json`.

The loader uses the logged `return-cost` target and the real logged
`solve-time-s`. There is no fitted solve-time model in this training pass.

## 2. Edit The Minimum Config

Training config lives at `src/inverted_pendulum/offline_training/config.yaml`.

Keep the file small: `data-roots`, `output-root`, run name, seed, split, batch,
learning-rate, epoch count, and plotting-grid count only.

## 3. Run Training

Launch from the repository root:

```bash
python src/inverted_pendulum/offline_training/train.py
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

- `config.json`
- `model_structure.txt`
- `normalization.json`
- `lambda.json`
- `metrics.json`
- `critic_state_dict.pt`
- `checkpoints/`
- `region_metrics.csv`
- `observed_policy_sanity.csv`
- `q_surface_*.png`

The observed policy sanity CSV ranks held-out logged actions only. Full
compute-aware grid policy validation is deferred until solve-time fitting is
implemented.
