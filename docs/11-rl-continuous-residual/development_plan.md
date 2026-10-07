# Continuous residual control: implementation and heuristic validation

## Quick reading

Build a continuous residual TD3 module inside the existing installable project,
under `src/rotary_pendulum/RL`, controlled by a dedicated bringup workflow. Keep
the discrete implementation available for reproducing the earlier experiments.
Do not train during this phase. First measure what the analytical controller
can accomplish by itself, including failures, at 20, 40 and 60 seconds.

The policy cap remains 0.00918 N m; the physical cap remains 0.0204 N m.
Decisions remain 100 ms, integrated at 20 ms. The heuristic requests a desired
pendulum-energy change, analytically converts power into torque, and regularizes
states with little instantaneous control authority. A small explicit startup
pulse avoids the exactly motionless downward equilibrium. A predictive arm
filter accepts the request, coasts, or brakes. Arm excursions remain nonterminal.
The learned residual can cancel or reverse the entire heuristic request.

Keep compact task-oriented files of 40–300 lines, no thin wrappers, unnecessary
configuration/CLI, fallback behavior, or try/with/except. Use batched JAX
operations for independent environments. Only GPUs 0–3 may be used.

## Frozen structure budget

Existing environment: extend `step` with `max_physics_steps`, defaulting to its
existing 1000 steps. No other existing interface changes are planned.

| File | Planned public objects/functions | Responsibility |
| --- | --- | --- |
| `RL/jax_residual_control.py` | `heuristic_torque`, `filter_torque`, `residual_action` | Analytical inverse, predictive arm filter, observation-to-applied-action composition |
| `RL/jax_residual_task.py` | `observe`, `transition`, `collect` | Eight-feature observation, continuous dense reward, collection and replay insertion |
| `RL/jax_td3.py` | `Actor.__call__`, `Critic.__call__`, `initialize`, `update` | Zero residual initialization, twin critics, delayed actor/target updates |
| `RL/jax_residual_evaluation.py` | `validation_resets`, `evaluate`, `write_validation` | Shared deterministic reset fixture, rollouts, machine records and human plots |
| `bringup/rotary_residual_workflow.py` | `train` | Explicit future training entry point, uniform replay, progress, validation and checkpoints |
| `scripts/validate_rotary_residual.py` | `main` | Heuristic-only parameter trials and matched horizon/reset comparisons |
| `scripts/summarize_rotary_residual.py` | `main` | Reproducible cross-wave comparison, energy-transfer diagnostic and selected plots |
| `tests/test_jax_residual.py` | At most 12 tests | Mathematical, terminal, gradient, replay and integration checks |

No dataclass, wrapper or additional public class is planned. Nested functions
are limited to JAX scan bodies, critic/actor losses and the delayed-update branch.
The learner, replay, rollout and experiment are dictionaries of arrays/settings;
the existing `EnvState` remains the only physical episode state object.

Independent quantities: physical state `(theta, alpha, omega, nu)`, physics-step
count, hold count, deadline; random keys; replay position/size; optimizer-update
count. Network parameters, optimizer moments and target parameters belong to
the learner. Replay stores observation, applied normalized torque, reward,
next observation and terminal flag. Collection diagnostics separately record
heuristic/proposed/applied torques and arm-filter decisions.

Heuristic tunables are `energy_time_s`, `sensitivity_floor_per_s`, and
`filter_steps` (20 ms prediction steps, at least five). Policy cap and residual
range are fixed. Initial trials use energy times 0.5, 1 and 2 s and sensitivity
floors 0.5 and 2 /s. Adapt these only after inspecting measured trajectories.

TD3 fixed defaults: two 128-unit SiLU hidden layers; actor zero output layer;
independent scalar critics; Adam learning rate 0.0003; gradient norm cap 10;
MSE critic loss; discount 1; actor/target update every two critic updates;
target Polyak coefficient 0.005; target noise standard deviation 0.2 and clip
0.5 in units of policy torque cap; behavior noise standard deviation 0.1.
These are recorded in experiment metadata, not spread across CLI switches.

Workflow settings: seed, run directory, deadline seconds, total transitions,
parallel environments, replay capacity, minibatch size, warmup transitions,
evaluation interval, validation seed and episodes per stratum. One transition
per lane is collected per iteration; one gradient update follows each iteration
after warmup. Deadline may be 20, 40 or 60 s; every checkpoint records it.

Reward parameters retain the accepted dense contract: energy 1, torque 0.02,
time 1.05, arm 0.2, upright 1 per second; upright widths `(0.16 rad, 0.4 rad/s,
0.3 rad/s)` for pendulum angle, pendulum speed and arm speed respectively.

## Stages, validation and success flags

1. **Controller/task:** verify analytical energy derivative against the physical
   ODE, finite torques at singular states, policy cap, exact-rest startup,
   reversal/cancellation, arm-filter braking and nonterminal arm excursions.
   Verify 20/40/60 s timeouts and unchanged 100 ms early success. Pass means all
   assertions hold; a predictor is not claimed to guarantee constraint safety.
2. **TD3 plumbing:** verify zero-initialized actor reproduces the heuristic,
   twin targets mask both success and deadline termination, delayed updates,
   finite gradients and replay consistency. Small synthetic update tests are
   permitted; no training campaign or learned checkpoint is produced now.
3. **Heuristic experiment:** paired fixed seeds, downward/moving/near/tight
   reset strata plus exact-rest/upright/boundary probes. Include zero torque and
   unfiltered heuristic controls. Batch independent resets and horizons. Save
   every trial's config, per-episode outcomes and compressed numerical traces;
   plot representative trajectories and inspect them before choosing another
   trial. Pass means finite bounded controls, reproducible data and an honest
   measurement of energy regulation, arm excursions and strict hold success.
   No artificial minimum success rate is imposed on the energy-only heuristic.
4. **Handoff:** run relevant and existing tests/lint; save human interpretation
   and validation plots separately from machine records. Clearly distinguish
   energy pumping from full swing-up-and-hold success. Report limitations and
   the next change suggested by trajectories. Training remains unexecuted.

The user has now conditionally authorized autonomous training once the heuristic
is reasonably acceptable; see [training_plan.md](training_plan.md). That condition
is not met by the current validation. Once it is met, run short TD3 trials at all three deadlines,
compare against identical heuristic-only seeds, inspect validation plots after
each iteration, then tune training settings. Only later add horizon mixing,
switch objectives or MPC joint training. A reliable local stabilizer is a
possible follow-up if the energy heuristic cannot capture the upright state.
