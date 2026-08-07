# Burst-Coast MPC

Wake-sleep model predictive control experiments.

This repository provides domain-composed configuration, a CLI entry point,
nonlinear rotary and inverted-pendulum environments, and burst-coast MPC.

## Environment

Use the explicit Windows package spec for the fastest clone-to-run setup:

```powershell
conda create -n burst-coast-mpc --file conda-win-64.lock
conda activate burst-coast-mpc
python -m pip install -e .

bcmpc --inverted-pendulum mpc-only --no-visual
bcmpc --rotary-pendulum mpc-only --no-visual
```

Use the human-maintained environment file only when changing dependencies:

```powershell
conda env update -n burst-coast-mpc -f environment.yml --prune
conda activate burst-coast-mpc
python -m pip install -e .

bcmpc --inverted-pendulum mpc-only --no-visual
bcmpc --rotary-pendulum mpc-only --no-visual
```
