# Burst-Coast MPC

Wake-sleep model predictive control experiments.

This repository provides domain-composed configuration, a CLI entry point,
nonlinear rotary and inverted-pendulum environments, and burst-coast MPC.

## Current rotary-pendulum controller

The [stationary balancing study](docs/9-RL/14-reinforcement-learning-prior/20-stationary-balancing/interpretation_summary.md)
trains physical-state-only, continuing TD3 actors using capture, braking and recovery
examples from the unchanged augmented teacher. Across 512 new starts per condition,
selected actors balance through 600 seconds at 4.59 and 2.295 mN m. Nominal full-duration
arm compliance is 99.6% and 96.5%; final torque RMS is 0.0693 and 0.00412 mN m.
Explicit disturbance tests and remaining soft-constraint violations are reported separately.
These are experimental neural actors; the default heuristic remains below.

The [sustained-balance study](docs/9-RL/14-reinforcement-learning-prior/19-sustained-balance/interpretation_summary.md)
sets the soft arm range to **±135°** and compares live balancing for 120–600 seconds,
with weaker motors, extended teacher demonstrations, and two eight-GPU TD3 rounds.
The earlier [torque audit](docs/9-RL/14-reinforcement-learning-prior/18-weak-motor-study/interpretation_summary.md)
explains why the previous 2.36 s result used 18.36 mN m, twice the earlier TD3 limit.

The runnable JAX energy-work controller uses **±0.01836 N m torque and 100 ms decisions**.
Each 20 ms observation interval now contains ten 2 ms RK4 integration steps.
Default capture requires strictly less than 5° upright error for one consecutive second;
arm position and angular velocities are absent from this angle-only success rule.
Default episodes stop at capture or 40 s. Sustained-balance experiments explicitly
train for ten-second holds with a 120 s deadline and evaluate continued live control.
There is no scheduled sleep phase.

The original heuristic predicts the next constant-torque 100 ms action. Its physical-energy
policy and decoder remain in place, with work-matching weight zero and arm weight one.
The soft arm penalty now starts at 108° to warn before the 135° boundary. It cannot
guarantee compliance. The sustained-balance study separately tests an augmentation with
local linear balancing feedback and predictive braking; this is an experimental teacher,
not the default heuristic. Original archives retain their historical settings.
See the [earlier method and validation](docs/10-heuristic-controller/13-single-action-controller/controller_analysis.md).

Pendulum position uses **[−180°, 180°)**: 0° is downward and ±180° is upright.
Stored state angles use [−π, π) radians, with exact +180° represented as −180°.
The goal bands are [−180°, −175°) and (175°, 180°). Arm position and angular
velocities remain signed. See the
[signed-angle convention and validation](docs/15-signed-pendulum-angle/representation_and_validation.md).

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
mission configuration supplies the 5° angle tolerance, but the older NumPy goal
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

The [RL prior assessment and next-step plan](docs/9-RL/14-reinforcement-learning-prior/assessment_and_next_step_plan.md)
and [portable animation records](artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/portable-animation-records/interpretation_summary.md)
cover learning from the current controller and replaying selected validation episodes.

The [completed TD3 diagnosis](docs/9-RL/14-reinforcement-learning-prior/16-td3-training-diagnosis/interpretation_summary.md)
reports 32 historical training runs on eight GPUs: retained policies solve the earlier
15°/100 ms angle-only capture task, while strict position-and-velocity capture still fails.
It includes held-out evaluations, checkpoints, learning diagnostics and the next
local-learning experiment. These runs learn torque control, not sleep schedules
or MPC horizon/terminal-value selection.

The [sustained-capture demonstration study](docs/9-RL/14-reinforcement-learning-prior/17-heuristic-demonstrations/interpretation_summary.md)
uses the new 5°/one-second goal, matched teacher/student torque, successful heuristic
trajectories, behavior cloning and retained demonstration replay.
