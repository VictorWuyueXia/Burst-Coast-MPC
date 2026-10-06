# Phase 9 Lightweight Autonomous Training Plan

## Human quick reading

This plan lets a future training session continue through several evidence-driven experiment rounds without waiting for the user after every result. Once the user asks to implement or start phase-9 training, the agent may run the baseline, inspect its artifacts, change approved training/model choices, launch the next comparison, and repeat until a validated winner or a documented plateau is reached.

The first run remains the documented Double DQN baseline: seven inputs, two 128-neuron tanh layers, three action values, and Huber loss. Later rounds may test smaller or larger networks, tanh/SiLU/ReLU activations, Huber or mean-squared loss, and the approved optimizer, replay, target-update, exploration, curriculum, and reward settings below. Every variant learns the same three-action finite-deadline task and is evaluated on the same saved validation states.

Use the eight GPUs for independent trials. Screen several hypotheses in parallel, promote only supported candidates, then confirm finalists over multiple seeds. Never select a model from training return or one lucky seed. Inspect capture, per-stratum outcomes, arm violations, powered time, Q calibration, action regret, learning stability, throughput, and representative trajectories between waves.

Stop when a candidate passes all phase gates, when further tuning reaches the declared plateau, or when the allowed experiment budget is exhausted. Deliver the best fully validated checkpoint and resolved setup. If nothing works, deliver the strongest partial checkpoint plus logs, comparisons, failure classification, and the smallest justified next formulation change.

This document authorizes iteration only after the user starts the implementation/training phase. It does not launch training by itself. Host GPU execution can still require the platform's command approval at launch; after that approval, ordinary in-scope experiment waves should proceed without repeated design questions.

## Machine-scannable training contract

### T0. Authority boundary

| Key | Rule |
|---|---|
| `authority_activation` | A user instruction to implement, train, run, continue, or babysit phase 9 |
| `autonomous_actions` | Edit phase-9 training/model configuration, run tests, launch/monitor trials, inspect artifacts, select the next approved comparison, checkpoint, and report |
| `fixed_task` | JAX plant, action set `{negative full, off, positive full}` with stored indices `[off, negative full, positive full]`, 100 ms decision, 20 s deadline, goal/dwell, arm boundary, terminal precedence |
| `fixed_evaluation` | Saved validation/final initial states, metrics, M2–M6 gates, no tuning on final test |
| `forbidden_changes` | Relaxing success, changing action space/clock/physics, adding observations, redefining terminal events, or using final-test results for tuning |
| `maximum_waves` | Six eight-process GPU waves after the single-GPU pilot |
| `maximum_trials` | Forty-eight full trials; at most two infrastructure retries per identical setup, which do not count as model trials |
| `user_interruption` | A new user instruction overrides this standing plan |

Changing a fixed-task item requires a new formulation discussion. Approved training changes receive a new `experiment_id`; reward coefficient changes also increment `reward_revision`, and architecture/input changes increment `network_id`. Do not compare raw returns across different reward revisions as though they share a scale.

### T1. Baseline and experiment record

The baseline is exactly `mlp-7-128-128-3-tanh-v1` with the D3 settings in [development_plan.md](development_plan.md). Run semantic tests and the single-GPU batch pilot first. The first full wave compares ordinary and energy-guided exploration using four seeds each.

Every trial record must contain: experiment/network/reward IDs; parent experiment; one-sentence hypothesis; all resolved parameters; seed and PRNG domains; git revision and dirty-patch identity; dependency/device information; transition/update budgets; stop reason; validation history; checkpoint choice; artifact paths; and pass/fail state for M2–M6. A variant changes only the named parameter family from its parent.

### T2. Approved search space

These values are bounded choices, not a Cartesian sweep. Select a comparison only when the preceding artifacts support its hypothesis.

| Parameter family | Approved alternatives |
|---|---|
| Hidden widths | `[64,64]`, baseline `[128,128]`, `[256,256]`, or `[128,128,64]` |
| Hidden activation | `tanh`, `silu`, or `relu`; output remains unrestricted linear |
| TD loss | Huber with delta `0.5`, `1.0`, or `2.0`; mean squared error only when targets/outliers are demonstrably controlled |
| Learning rate | `1e-4`, `3e-4`, or `1e-3`; constant within a trial |
| Gradient norm limit | `1.0`, `5.0`, or `10.0` |
| Target update | Hard copy every `128`, `256`, or `1024` updates; optional Polyak coefficient `0.005` as one documented scheme |
| Minibatch size | `2048`, `4096`, or `8192`, subject to measured memory and whole-block arithmetic |
| Replay capacity | `524288`, `1048576`, or `2097152` transitions |
| Sample reuse ratio | `0.5`, `1.0`, or `2.0` sampled items per new transition |
| Exploration end | Epsilon `0.02`, `0.05`, or `0.10` |
| Exploration decay | `2^21`, `2^22`, or `2^23` transitions per stage |
| Guided fraction | `0.0`, `0.5`, or `1.0` within the exploration branch |
| Parallel environments | Pilot-approved `1024`, `8192`, or `65536` |
| Curriculum | Existing three stages; stage mixtures/budgets may vary while every batch retains at least 25% capture-capable resets after stage 0 |
| Reward coefficients | `capture_weight`, `on_cost_per_s`, and `time_cost_per_s` at half, baseline, or double their current values |

The optimizer remains Adam, the replay remains uniform, gamma remains one, and Double DQN remains the learning target during phase 9. Do not add prioritized replay, ensembles, distributional heads, recurrent state, auxiliary losses, or a new algorithm during unattended tuning. Such changes expand code structure and require a revised development plan.

Model variants use the existing `QNetwork` class with explicit `hidden_widths` and activation choice; do not create one class per architecture. Loss choice and target-update scheme are explicit experiment settings inside the existing `update` logic. These additions do not authorize new wrapper/helper functions, data classes, files, or parameters outside this table.

### T3. Autonomous iteration loop

1. Establish baseline correctness and throughput. Do not tune around a semantic, nonfinite, import, or artifact failure.
2. Run the baseline exploration wave. Evaluate every scheduled checkpoint on the fixed validation sets and classify the dominant failure using actual traces and metrics.
3. Form at most four candidates addressing one parameter family. Screen them as four configurations with two common seeds across eight GPUs. Use equal transition budgets and shared validation states.
4. Rank candidates lexicographically: gate eligibility; minimum stratum success; overall success; arm-failure rate; Q action regret; paired-success powered duration; then warm throughput. A candidate with nonfinite learning or a regressed capture gate is ineligible.
5. Confirm at most two finalists using four common seeds each in the next wave. A finalist replaces its parent only when its median improves the diagnosed metric and does not violate an already passed gate.
6. Inspect the new evidence, then either promote the winner, test the next supported parameter family, run final M4–M6 evaluation, or stop under T5. Never chain speculative changes without an intervening evaluation.

Architecture/activation comparisons come after the baseline objective and data pipeline learn stable finite values. Loss/optimizer/target changes address divergence, oscillation, overestimation, or calibration. Exploration/replay/curriculum changes address coverage or forgetting. Reward changes address behavior that optimizes the stated return but misses capture/on-time intent. This ordering is diagnostic guidance; clear evidence may justify another approved family, recorded in the experiment hypothesis.

### T4. Monitoring and recovery

Emit a flushed progress record and host heartbeat at least every 30 seconds as specified in the development plan. Monitor finite states/rewards/Q values/loss/gradients, GPU utilization and memory, throughput, replay fill, action frequencies, terminal outcomes, and evaluation gates. Save latest and eligible-best checkpoints before each scheduled evaluation.

Stop a trial immediately for nonfinite computation, corrupted/missing artifacts, device loss, or repeated process failure. Diagnose and fix implementation/infrastructure faults, rerun semantic tests, then repeat the identical experiment ID with a retry suffix. Do not reinterpret a crashed run as poor model performance or silently change its settings. An out-of-memory pilot/configuration is a measured rejected configuration; choose another approved batch setting and record it.

Independent trials may finish at different times. Reuse a free GPU for the next already-selected trial only after all evidence required to define that trial exists. Do not let early results from one seed alter still-running peers in the same comparison.

### T5. Stop conditions and final selection

Stop successfully when one family has at least three of four confirmation seeds passing M2–M6 and its selected checkpoint passes the untouched final test. Select the final setup by the T3 ranking across eligible confirmed families, never by a single seed. Deliver each passing seed checkpoint and identify the representative checkpoint whose validation metrics are closest to the family median.

Declare a tuning plateau when two consecutive evidence-targeted waves improve the failing success metric by less than two percentage points and its associated calibration/on-time metric by less than ten percent. Also stop at six waves, forty-eight full trials, or when all remaining hypotheses require a forbidden task/algorithm change. Do not spend the remaining budget merely because it exists.

If stopped without full success, select the partial setup by deepest milestone reached, then minimum-stratum success, violations, action regret, and paired-success powered duration. Label it unusable for any failed downstream milestone. State whether evidence supports an action-resolution, reward, exploration, optimization, approximation, or planning-interface limitation.

### T6. Required unattended handoff

The handoff must be understandable without reading live console output. Provide:

- `interpretation_summary.md` stating the best setup, passed/failed gates, confidence across seeds, and next decision;
- a compact experiment lineage table from baseline through every promoted/rejected candidate;
- resolved configuration and checkpoint metadata for the selected setup;
- all histories, evaluation tables, value audits, timing, logs, and representative success/failure trajectories required by D8;
- a comparison figure for each changed parameter family and a final controller comparison;
- the exact reason every stopped/crashed/ineligible trial was excluded;
- either the fully validated result or a failure analysis tied to recorded evidence and one smallest recommended formulation change.

Do not claim that a lower TD loss proves a better controller, that a passing policy has a calibrated Q prior, or that a best observed seed represents the family. Preserve failed runs and negative results in the aggregate tables.
