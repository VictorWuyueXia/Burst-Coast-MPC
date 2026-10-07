# Implemented continuous residual controller: heuristic-only result

The continuous controller, predictive arm filter and TD3 learner are implemented.
**No training was executed.** The analytical heuristic pumps energy, but it does
not solve swing-up or recover upright balance in the tested resets. The arm
filter substantially improves arm behavior. Longer deadlines do not cure the
energy controller's persistent oscillation.

Start with the [campaign report](../../artifacts/rotary_pendulum/continuous-residual/handoff/human-readables/interpretation_summary.md),
[overview](../../artifacts/rotary_pendulum/continuous-residual/handoff/human-readables/campaign_overview.png)
and [100 ms energy-transfer diagnostic](../../artifacts/rotary_pendulum/continuous-residual/handoff/human-readables/energy_transfer_detail.png).
Plots were visually inspected between parameter sweeps and again for the final handoff.

## What was implemented

- Current policy cap ±0.00918 N m, unchanged physical clip ±0.0204 N m.
- Analytical regularized pendulum-energy controller, with a small explicit
  downward-rest startup pulse; no action optimization in the heuristic.
- Predictive arm filter: accept, coast, brake or least predicted excursion.
- Continuous residual spanning twice the policy cap, allowing cancellation
  and reversal; zero-initialized actor reproduces the heuristic plus filter.
- TD3 twin physical-action critics, delayed actor updates and target smoothing.
- Eight Markov observation features, including elapsed and remaining time;
  20, 40 and 60 s episodes retain the 100 ms decision interval.
- Existing five dense reward terms, nonterminal arm excursions, and early
  success after 100 ms of strict holding.
- Explicit future training workflow with replay, progress, checkpoints and
  validation plots. It was not invoked; only synthetic learner unit tests ran.

## Numerical outcome

Three sequential sweeps tested 18 filtered parameter pairs, three unfiltered
controls and zero torque: 22 trials × 780 episodes = 17,160 rollouts. Each
trial reuses 64 resets in each of four strata plus four deterministic probes,
paired across three deadlines. These are not 17,160 independent initial states.

| Random reset stratum | 20 s strict success | 40 s | 60 s |
| --- | --- | --- | --- |
| Downward | 0/64 | 0/64 | 0/64 |
| Moving | 0/64 | 0/64 | 0/64 |
| Near upright | 0/64 | 0/64 | 0/64 |
| Tight upright | 9/64 | 9/64 | 9/64 |

These counts are identical for all 22 trials. The nine tight successes finish
in 0.10–0.12 s and also succeed with zero torque. They establish favorable
initial conditions, not acquired capture. No downward trajectory reaches 90%
of upright target energy in any tested configuration.

At energy correction time 0.1 s and sensitivity floor 0.5 /s, the filter reduces
60 s arm excursions from 100% to 0% for downward/moving/near resets, and from
85.94% to 0% for tight resets. Some other filtered configurations still cross
the limit in one of 64 cases: the filter is not a guarantee.

The lowest downward mean absolute normalized energy error is 0.4089 at energy
time 0.05 s and sensitivity floor 2 /s. Its late mean energy is around 60% of
target, but capture is still zero. This is a best energy-pumping diagnostic,
not a deployable or successfully trained controller.

## Interpretation and next step

The exact instantaneous energy derivative is correct, as verified against the
physical ODE. It does not predict work accurately across a held 100 ms action.
On a representative settled oscillation, instantaneous power averages +0.00597 W
while actual interval-averaged power is approximately −0.000572 W over 10–20 s;
34% of decisions start with positive power but yield negative interval work.
The slightly negative window average also depends on its oscillation phase.
This supports a sampled-control limitation; it is not a proof that every
possible analytical energy controller must fail at this decision interval.

Keep 100 ms and the current cap. Before a long TD3 campaign, improve the
analytical heuristic's estimate of **energy change over the whole held action**,
for example using a closed-form local phase/work approximation. Validate it
on the same resets. Near upright, energy alone also does not specify the arm
position and both velocities required for capture: a compact local balancing
law or the residual must supply that behavior. More gain sweeps of this same
instantaneous inverse are not supported by the results so far.

## Code, contracts and reproducibility

- [Formulation](formulation_plan.md): exact equations, units, reset and reward definitions.
- [Development plan](development_plan.md): structure budget, milestones and test criteria.
- Core modules: `src/rotary_pendulum/RL/jax_residual_control.py`,
  `jax_residual_task.py`, `jax_td3.py`, `jax_residual_evaluation.py`.
- Future entry point: `src/bringup/rotary_residual_workflow.py::train`.
- [Future template](machine-scannables/training_template.json) is unexecuted.
  Change deadline, evaluation deadline and output directory together for 40/60 s.
- Three heuristic campaign JSONs in `machine-scannables/` reproduce all trials
  using `scripts/validate_rotary_residual.py --config PATH --output NEW_DIRECTORY`.
  Set `CUDA_VISIBLE_DEVICES=0`; no GPU outside 0–3 is permitted. Validation used GPU0.
- Recreate the handoff with `scripts/summarize_rotary_residual.py --root
  artifacts/rotary_pendulum/continuous-residual` on CPU or an allowed GPU.
- Trial directories contain `machine-scannables/{settings.json,episodes.csv,
  trajectories.npz,plot_selection.json}` and human plots. Each sweep stores
  source snapshots and hashes. Artifacts remain gitignored.

Validation: existing full suite plus the first eight new tests passed (108 tests);
all eleven final residual tests passed, including three additional checks for
twin-min targets, terminal replay/reset separation and actor gradients. Ruff
passes; strict mypy passes for all five new runtime modules. The full-suite
run reports two existing empty-slice warnings in offline diagnostics.
