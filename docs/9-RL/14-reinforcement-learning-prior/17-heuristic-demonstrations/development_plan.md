# Sustained upright capture and heuristic demonstrations

Use compact task-oriented logic, no new classes, wrappers, fallback behaviors, or dependencies. Keep edited/new Python files within 40–300 lines where feasible; reuse the existing simulator, decoder, replay, TD3, launcher, and artifact writer. The old campaign has source snapshots and must remain distinguishable from this changed task.

## Stages and validation milestones

1. **Task contract.** Set the angle-only goal to strictly less than 5 degrees for 50 consecutive physics samples (1 second), default deadline 40 seconds, compare 30 seconds. Preserve five physics steps per action (100 ms). Match the student's torque cap to the actual heuristic's 90% physical cap. Pass: boundary/reset/terminal/time-limit tests and matched applied torques.
2. **Teacher qualification.** Run eight paired heuristic parameter trials on available GPUs, inspect trajectories, then generate independent training and validation records. Only trajectories satisfying the new sustained goal qualify as successful demonstrations. Pass: finite traces, independently recounted uninterrupted holds, no split leakage, explicit success and arm-excursion counts. If the teacher fails, investigate its limits before claiming useful demonstrations.
3. **Demonstration learning.** Train the existing actor by behavior cloning, optionally fit critics to complete demonstrated returns, then compare pure TD3, initialization from demonstrations, and continued demonstration replay/actor supervision. Use paired seeds, matched budgets and reward/action contracts. Pass: held-out supervised error plus actual closed-loop success; a small action error alone does not qualify.
4. **Iterate and evaluate.** Inspect each wave before selecting the next bounded changes in model size, supervision strength, or exploration. Compare 30/40 seconds on unseen starts, independently verify holds, and report what improved and what still fails. Pass: completed checkpoints, clear human summary/plots, reproducible machine arrays/settings/source snapshots, bilingual documentation, relevant tests/lint.

## Structure budget

No new classes, data classes, wrappers, or framework objects. Reuse `EnvState`, `Actor`, `Critic`, Adam, replay dictionaries and the current launcher. Local arrays and loop indices stay within the task that computes them.

- **Contract workflow:** edit existing `step`, `transition`, `observe`, `evaluate`, `write_validation`, and validation script `main`; constants `GOAL`, `MAX_PHYSICS_STEPS`, `POLICY_TORQUE_NM` define goal/deadline/torque. Retain explicit historical strict-goal comparison, label its distinct hold length.
- **Data workflow:** one new script `prepare.py`, one `main` function; objects `traces`, `initial`, `settings`, `dataset`, `manifest` hold teacher trajectories, initial states, task settings, transition arrays, and provenance. Recompute rewards/observations with the learner's task and preserve episode identity, sampled starts and train/validation split. Existing heuristic `evaluate`/`decode` generate trajectories; existing validation script handles eight GPU teacher sweeps.
- **Learner workflow:** one new module `jax_demonstrations.py` with `pretrain` and its two substantive actor/critic loss closures; edit existing TD3 `update` and workflow `train`. Objects `demonstrations`, `learner`, `replay`, `metrics` carry immutable teacher transitions, trainable state, online experience, and measured losses. Reuse network architecture and optimizer implementation.
- **Experiment workflow:** reuse existing `launch.py` and `audit.py`; at most one new analysis script `analyze.py` with `main`, for aggregate tables, learning curves and hold audits. Config files declare each wave; no new command framework. Tests: up to four new task-oriented tests plus adapting affected existing assertions.

## Explicit experiment parameters

Task: angle tolerance 5 degrees, hold 50 samples, deadlines 30/40 seconds, unchanged physics 20 ms/action 100 ms, torque cap 0.9 times physical limit. Teacher tuning: existing work gain, work weight, arm-limit weight only. Learner tuning: existing hidden widths, learning rate, exploration/target noise, update reuse and reward weights; new demonstration settings are `demonstration_path`, `pretrain_steps`, `demonstration_fraction`, `cloning_weight`, and `critic_pretrain` (enable/disable complete-return regression). No other new tunable parameters planned. Seeds, dataset sizes, and compute budgets are recorded per wave, not hidden defaults.

Human reports and figures are separate from machine-scannables. The latter retain all settings, arrays, episode membership, exact sources and metrics. Report simulated task success separately from arm excursions and from real hardware claims.
