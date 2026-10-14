# TD3 training diagnosis

The user explicitly authorized training, iteration and all eight GPUs on 2026-10-13.
This replaces the older training hold and four-GPU restriction for this campaign.
Check occupancy before each wave; exclude occupied devices and stop if more than
four are occupied. Run independent paired-seed trials, one per available GPU.

Keep compact task-oriented code, 40–300 lines per file, without thin wrappers,
unnecessary CLI/configuration, fallback behavior or try/with/except. This is a
proof-of-concept diagnosis, not a production controller. No new dependencies.

## Stages and completion flags

1. Freeze task and record source/package versions. Verify goal/terminal semantics,
   zero residual equivalence, direct-policy gradients and variable network shapes.
   Flag: semantic tests and lint pass before full training.
2. Compare residual/direct policies and ordinary/increased update reuse with two
   seeds each. Start with bounded budgets; verify finite losses, completed episodes,
   checkpoints and reproducible validation. Flag: eight completed runs with evidence.
3. Inspect learning and trajectory plots; choose architecture, exploration or reset
   changes from observed failures. Run at least two additional hypothesis-led rounds
   when useful. Flag: improvements replicated or a documented tuning plateau.
4. Evaluate retained checkpoints on a fresh seed and audit value/action behavior.
   Flag: held-out outcomes, numerical source data, readable plots and bilingual
   interpretation distinguish observed causes from unresolved hypotheses.

## Structure budget

Retain Actor and Critic classes and their __call__ methods; initialize and update
stay in jax_td3.py. Add hidden_widths to both classes, with existing two-layer
defaults. The existing nested critic_loss, actor_loss and delayed branches remain.
No additional learner classes, data classes or wrappers.

Retain residual_action, observe, transition, collect, validation_resets, evaluate,
write_validation and train in their existing task-oriented files. Add one nested
scan body inside train to batch repeated learner updates. Extend the existing
transition scan with an explicit strict/angle goal; do not change the shared plant.

New scripts: launch.py has main for occupancy-checked process launch and progress;
analyze.py has main for campaign tables and plots; audit.py has main for checkpoint
evaluation, value calibration and action sensitivity. Maximum three new public
functions, one in each script. Tests: at most three new semantic tests, in a separate
40–150 line test file if the existing file would exceed 300 lines.

Independent runtime objects remain settings, learner, rollout, replay and validation
traces. Launcher adds a plan, child processes and log streams. Analysis adds episode
rows and histories; audit adds paired initial states and critic predictions.

New experiment controls, grouped by purpose:

- Architecture: hidden_widths, a tuple of hidden layer sizes for actor and critics.
- Action: action_mode, one of residual, direct, residual_unfiltered or direct_unfiltered.
- Collection: reset_mode, mixed or local; local samples tight/near upright starts.
- Learning: updates_per_collection, positive integer repeated gradient updates.
- Task: goal_mode, strict or angle; fixed 100 ms decisions and 20 ms physics.

Existing settings retain seed, validation_seed, output path, lane/replay/batch sizes,
warmup, budget/evaluation cadence, learning rate, exploration noise, deadline,
heuristic settings and the five dense reward terms. No additional tunable controls
are planned. Derived diagnostics include saturation, filter intervention, nonzero
action sensitivity, success/terminal replay occupancy and value calibration.

The strict goal uses tolerances 0.08 rad for arm and upright error, 0.15 rad/s for
arm speed and 0.20 rad/s for pendulum speed, for five physics samples. The angle goal
uses the current configured 15-degree tolerance and five samples. Arm excursions
are nonterminal at the current 180-degree limit. Old campaigns used different arm
limits; new outcomes are not a matched historical replication.

Save human reports/figures in human-readables; detailed configurations, histories,
checkpoints, arrays, source snapshots and environment versions in machine-scannables.
The campaign root is artifacts/rotary_pendulum/td3-diagnosis-20261013.

## Decisions after the first round

The user chose current angle-only capture as primary, with strict capture as a
comparison. Four updates per collection produced successful angle capture in
both residual seeds. Keep actual learning outcomes rather than presuming failure.
Second round compares strict mixed/local training and wider/deeper angle networks.
The launcher plan declares operation=train or audit; audit settings add the explicit
checkpoint_path. These are execution inputs, not extra training hyperparameters.

Add one fixed local linear-quadratic regulator benchmark inside audit.main. Derive
its sampled dynamics from the same simulator and its state penalty from strict
goal tolerances; normalize effort by the existing policy cap. No extra function,
class or tunable setting. evaluate gains a linear mode with its existing actor
argument carrying the gain vector. This tests local physical reachability without
claiming a swing-up solution. Save the derived matrices and gain in machine records.

Third-round evidence: strict local trials still oscillate, with almost no arm-filter
intervention. A fixed linear controller succeeds on all 64 tight validation starts.
Its global application can become numerically unstable, so its admissible benchmark
is explicitly tight starts only; global trial traces are excluded from conclusions.
Add one learning parameter, target_noise, for a final targeted smoothing comparison.
It is the standard deviation in normalized policy-cap units; clipping is 2.5 times
that value, reproducing the previous 0.2 standard deviation / 0.5 clip exactly.
Test 0.02 and zero, plus paired low behavior/target noise with increased updates,
and a deeper strict network. This expands the planned setting budget to six controls
without introducing any function, class or wrapper.
