# Burst-Coast MPC

Wake-sleep model predictive control experiments.

This repository provides domain-composed configuration, a CLI entry point,
nonlinear rotary and inverted-pendulum environments, and burst-coast MPC.

## Current rotary-pendulum controller

The runnable JAX energy-work controller uses **±0.01836 N m torque,
±180° soft arm limits and 100 ms decisions**. Physics integration remains 20 ms.
Capture requires the pendulum to remain within ±15° of upright for five consecutive
physics samples; arm position and angular velocities are absent from the capture rule.
Episodes stop at capture or 20 s. There is no scheduled sleep phase.
The controller predicts only the next constant-torque 100 ms action. It regulates
unweighted physical mechanical energy and rewards increased pendulum potential,
with a soft penalty beginning at the actual ±180° arm limit. There is no braking
continuation, arm-speed cap or inward margin.
See the [method and validation](docs/13-single-action-controller/controller_analysis.md).

Pendulum position uses **0–360°**, with 0° downward and 180° upright; 360° wraps
to 0°. Stored state angles use the equivalent range in radians. The goal band
is 165–195°. Angular velocities and arm positions remain signed. See the
[angle convention and validation](artifacts/rotary_pendulum/experiment-results/full-turn-pendulum-angle-representation/interpretations/interpretation_summary.md).

Run the default validation campaign with the existing Linux virtual environment:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  .venv/bin/python scripts/validate_rotary_heuristic.py \
  --output artifacts/rotary_pendulum/heuristic-default
```

The output directory must be new. The runner uses
[heuristic.json](src/rotary_pendulum/configs/heuristic.json) unless `--config` is supplied.
GPU studies check occupancy before every launch and use idle GPUs 0–3.

The [downward-start study](artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/interpretations/interpretation_summary.md)
explains 83,968 randomized episode evaluations, the dominant swing-up failure,
and paired tests of potential fixes in the previous controller. That controller
is now superseded by the 100 ms action method; its archives retain the original
recovery-speed and energy-weight experiments.
The older NumPy/CasADi MPC and TD3 prior remain separate implementations. The shared
mission configuration supplies the 15° angle tolerance, but the older NumPy goal
still uses its arm-position and velocity conditions.

The [JAX environment design](docs/8-jax-rotary-environment/interpretation_summary.md)
and [energy-controller study](artifacts/rotary_pendulum/experiment-results/physical-energy-transfer-controller/interpretations/interpretation_summary.md)
provide the earlier formulation and validation history.

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

Recent experiment figures, logs and short interpretations are indexed in the
[experiment results](artifacts/rotary_pendulum/experiment-results/README.md).
Plans, formulations and detailed analyses remain in `docs/`; reproduction scripts
are in `scripts/experiments/`.

For the latest operations, working-tree state and pending decisions, read the
[development history and agent handoff](docs/recent_development_history.md).

The [RL prior assessment and next-step plan](docs/14-reinforcement-learning-prior/assessment_and_next_step_plan.md)
and [portable animation records](artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/portable-animation-records/interpretation_summary.md)
cover learning from the current controller and replaying selected validation episodes.
