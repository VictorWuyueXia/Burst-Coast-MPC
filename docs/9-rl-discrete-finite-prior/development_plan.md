# Phase 9 Development: Training and Deployment Validation

## Human quick reading

Implement the [formulation](formulation_plan.md) using the existing JAX plant and episode logic. First remove the old rotary PPO code/configuration and build the Q learner from scratch according to [NN-design.md](NN-design.md). Then check reward and terminal accounting and prove local upright capture. Expand training only when capture works. Compare ordinary exploration with energy-guided exploration, inspect numerical action values, and finish with a small planning experiment that uses the same reward units.

Use one independent process per GPU. The eight GPUs support two exploration families with four independent seeds each; they do not need a synchronized distributed learner. First measure one complete learner's throughput and memory. The simulator's very large batch capacity is not evidence that the largest environment batch trains best.

Keep the code compact and task-oriented. Use no thin wrappers, unnecessary CLI/configuration system, hidden defaults or fallbacks, or `try`/`with`/`except` blocks. Keep each substantive source file between 40 and 300 lines through proper grouping, not compressed formatting. The symbol and state budgets below are ceilings; revise this plan before exceeding them.

Inspect artifacts between experiment rounds. [training-plan.md](training-plan.md) governs bounded autonomous comparisons after the baseline, including training settings, model size, activation, and loss. If accurate capture remains inaccessible with 100 ms full-torque pulses, report the limitation and propose a richer action set in the next contract. Do not lower the goal or redefine success to finish this phase.

After success, extend the critic to richer/continuous torque, add explicit switching semantics and previous-action state, then develop nested switch-time/action-sequence optimization and joint sampled-MPC training. These later tasks are outlined in F8 of the formulation and are not part of the present implementation budget.

## Machine-scannable development contract

### D0. Fixed scope and repository discipline

Contract: `rotary-q-prior-v1`, initial `reward_revision=1`. This document freezes the initial implementation plan; no phase-9 training result exists yet. F0–F8 in the formulation define task semantics. D3 settings are explicit initial engineering choices. D5 thresholds are declared proof-of-concept acceptance criteria, not measured capabilities.

Reuse `EnvState`, `reset`, `step`, and nominal dynamics. Remove rotary PPO components, configuration, exports, and tests as enumerated in NN-design N5; this supersedes the earlier PPO-preservation requirement. Keep legacy NumPy/CasADi MPC and inverted-pendulum workflows runnable. Shared PyTorch/Lightning dependencies remain while those workflows require them. Do not port the old PPO reward, add a Gym adapter, add new physical configuration, or modify the JAX environment's action clock or terminal rules. Add Flax and Optax to a named optional dependency group in the existing root `pyproject.toml`; inspect the `.venv`, select compatible versions, validate imports plus one update, and pin the actually tested versions. Preserve existing JAX pins and record a Linux package snapshot with artifacts.

Keep one installable project with lightweight learning modules. No separate RL distribution, standalone CLI, experiment-manager class, configuration dataclass, network wrapper, or deployment service is planned. Runs call `bringup.rotary_q_workflow.train(experiment)` from a short launch command or temporary batch-launch script. That dedicated workflow owns substantive orchestration and imports environment/RL computations. Resolve one complete experiment dictionary before calling it; missing keys are errors. Save that resolved dictionary with the run. The tables here define initial settings; old PPO YAML keys are removed rather than translated or used as defaults.

### D1. Exact code structure and symbol budget

Four new computation files are under `src/rotary_pendulum/RL/`; the fifth is the dedicated workflow under `src/bringup/`. This budget replaces the earlier all-RL-file sketch. NN-design N1/N2 fixes package boundaries and network structure, while this table owns the complete callable/file budget.

| File | Planned symbol / signature | Responsibility | File budget |
|---|---|---|---|
| `jax_task.py` | `observe(env_state, experiment)` | Return seven observations and the two terminal-masked potential components; used by collection, value conversion, and evaluation | 150–280 lines |
| same | `transition(env_state, action_index, experiment)` | Advance batched or candidate-batched states using the existing step; return next state, rewards, components, and done | included above |
| same | `collect(learner, rollout, replay, experiment)` | Scan a short collection block; exploration, replay insertion, autoreset, and episode statistics | included above |
| `jax_q.py` | `QNetwork(hidden_widths, activation_name)` and `__call__(observation)` | One Flax module with explicit width/activation fields, seven inputs, and three Q_train outputs; no handwritten constructor | 90–180 lines |
| same | `update(learner, replay, experiment)` | Sample replay, compute Double DQN loss, update weights and target copy | included above |
| `src/bringup/rotary_q_workflow.py` | `train(experiment)` | Import environment/RL functions; initialize, run gated curriculum, emit progress, call evaluation, and select/save checkpoints | 150–280 lines |
| `jax_evaluation.py` | `evaluate(learner, initial_states, mode, experiment)` | Execute greedy/planning/value-audit batches and return metrics plus selected traces | 180–300 lines |
| `jax_artifacts.py` | `write_artifacts(run_dir, metrics, trajectories, checkpoint, experiment)` | Write structured records, checkpoint files, and the fixed figure set | 100–220 lines |

Budget: one new module class, one method, seven module-level functions, and no new dataclasses/NamedTuples. Four substantive nested callbacks are allowed: `collect.advance` and `evaluate.advance` for decision scans; `update.update_one` for the update scan; and its `loss_fn` for differentiation. Their arguments are scan carry/input or trial network parameters respectively. No other wrappers/helpers are budgeted. Straight-line algebraic temporary variables are local computations, not additional independent parameters. Do not also create `RL/jax_training.py`; the workflow is the single owner of the training schedule. Existing `RL/__init__.py` becomes passive package metadata.

Two test files are allowed, each 80–250 lines and at most four parametrized test functions: `tests/test_jax_q_task.py` covers observation/potential, reward/termination, exploration, and replay/reset accounting; `tests/test_jax_q_learning.py` covers hand-computed targets/gradients, target-copy cadence, checkpoint/model/import reproduction, and planner/value conversion. Delete PPO-only `tests/test_rotary_rl.py`. Include NN-design N6 checks in these planned tests and the cleanup audit; tests must exercise semantic failures, not merely mirror code statements.

### D2. Persistent objects, independent inputs, and array contracts

Use ordinary dictionaries as JAX pytrees where needed. A dictionary is not permission to introduce unlisted hidden state. Existing Flax/Optax library objects may be used directly; they do not require project wrapper classes.

| Object | Exact owned fields / role |
|---|---|
| `learner` | `params`, `target_params`, `opt_state`, `key`, `updates`; weights, optimizer state, explicit learner PRNG, integer update count |
| `rollout` | `env_state`, `key`, `episode_totals`, `completed_totals`; current batched EnvState, collection PRNG, per-lane and aggregate logging statistics |
| `replay` | `observation`, `action`, `reward`, `next_observation`, `done`, `position`, `size`; fixed arrays and ring-buffer bookkeeping |
| `experiment` | Fully resolved fixed contract, reward revision, D3/D4 settings, seed, trial identifier, stage, stage and total transition counts, run directory; host control record |

`episode_totals` contains base return, training return, powered seconds, elapsed seconds, off-to-on count, sign-reversal count, previous action index, maximum energy ratio, and peak absolute arm angle. Previous action is used only for diagnostics. `completed_totals` contains completed episode count, three outcome counts, and sums of those episode statistics; reset aggregate totals after each emitted log record. Intermediate reward components are aggregated per collection block, without storing full training histories on device.

Let N be parallel environments, C be replay capacity, and B be sampled minibatch size. Their values are D3 settings. Replay arrays have shapes `(C,7)`, `(C,)`, `(C,)`, `(C,7)`, `(C,)` respectively and dtypes float32, int32, float32, float32, bool. Reward means r_train from F4. Position and size are int32 scalars. Ring insertion uses distinct modulo-C slots; require each collection block's transition count not to exceed C. The sampled batch has the corresponding leading size B. Only indices below size are sampled.

Every terminal transition retains its pre-reset next observation. Episode outcome totals are updated before reset; new lanes start with zero totals and action index zero for logging. Keep PRNG streams separate for model initialization, behavior, reset sampling, replay, and evaluation using named fold-in integer domains `0,1,2,3,4` in that order. `rollout.key` contains the behavior and reset keys; `learner.key` is the replay key. Model initialization and evaluation keys are explicit workflow-local inputs derived from their roots.

| Call | Exact returned objects |
|---|---|
| `observe` | `(observation, potential_components)`, trailing shapes `(7,)` and `(2,)`; component order energy then capture, including negative signs and terminal masks |
| `transition` | `(next_env_state, observation, base_reward, train_reward, reward_components, done)`; reward-component trailing shape `(5,)` in F4 order |
| `collect` | `(rollout, replay, metrics)`; updated keys remain inside rollout |
| `update` | `(learner, metrics)`; updated replay key remains inside learner |
| `train` | Run-directory `Path`; writes its selected/latest checkpoint and reports gate status |
| `evaluate` | `(metrics, trajectories)`; value-audit arrays are named entries of the second object |
| `write_artifacts` | `None`; materializes the supplied payloads without deriving a new training objective |

Independent entry inputs are the experiment record, evaluation initial EnvStates, evaluation mode, output path, and artifact payloads named in D1. Local `network` and `optimizer` objects are the direct QNetwork instance and Optax transformation; construct them from explicit settings during initialization or tracing, without extra factories or persistent wrappers. Bind only stable experiment settings as compiled configuration; changing counters and stage inputs are scalar arrays. Do not pass paths, strings, or the entire changing host record through a JIT boundary. `observe` receives the same explicit potential coefficients as collection and evaluation; no hidden module default is permitted.

### D3. Initial numeric experiment settings

All settings below define the required baseline. Values labeled fixed stay fixed within that baseline; values labeled tunable may change between documented rounds. After the baseline is measured, training-plan T2 authorizes its bounded architecture, activation, loss, and training alternatives while task/formulation semantics remain fixed.

| Group | Key | Initial value / meaning |
|---|---|---|
| Network | `hidden_widths`, `activation` | Fixed `[128,128]`, `tanh`; linear three-output head |
| Network | `kernel_init`, `bias_init`, `dtype` | Fixed Glorot uniform, zeros, float32; independent kernels; no dropout or normalization layers |
| Optimization | `learning_rate` | Tunable `0.0003`, constant within a trial |
| Optimization | `adam_betas`, `adam_epsilon` | Fixed `[0.9,0.999]`, `1e-8`; no weight decay |
| Optimization | `huber_delta`, `gradient_norm_limit` | Fixed `1.0`, tunable `10.0`; clip global norm before Adam |
| Optimization | `target_copy_updates` | Tunable `256`; initialize target equal to online, copy after each positive multiple of this count |
| Data | `parallel_environments` | Initial `8192`; select from the D4 pilot before full runs |
| Data | `collection_decisions` | Fixed `8` consecutive decisions per lane per collection call |
| Data | `replay_capacity`, `minibatch_size` | Tunable `1048576`, `4096` transitions |
| Data | `warmup_transitions` | Fixed `262144`; collect with epsilon `1.0`, do not update until reached |
| Data | `sample_reuse_ratio` | Tunable `1.0`; sampled replay items per newly collected transition after warmup |
| Exploration | `epsilon_start`, `epsilon_end` | Tunable `1.0`, `0.05` |
| Exploration | `epsilon_decay_transitions` | Tunable `4194304`; linear decay by transitions collected within each stage, restart at stage change |
| Exploration | `heuristic_fraction` | Family definition: `0.0` ordinary or `0.5` guided, conditional on exploration |
| Reward | `success_reward`, `arm_failure_cost`, `timeout_cost` | Fixed `5.0`, `5.0`, `2.0` |
| Reward | `on_cost_per_s`, `time_cost_per_s` | Tunable `0.05`, `0.005` |
| Potential | `capture_weight` | Tunable `1.0`; both F3 potential terms always active |
| Run | `stage_transition_budgets` | Initial `[8388608,16777216,33554432]`; maxima per stage, including warmup in stage 0 |
| Run | `evaluation_every_transitions` | Fixed `1048576`; also evaluate at every stage end |
| Run | `progress_interval_s` | Fixed maximum `30` seconds between progress messages |
| Planning | `lookahead_decisions` | Fixed `3`; all `27` sequences, F7 terminal-score modes |

One collection block holds the online parameters fixed while collecting. Then perform `sample_reuse_ratio * N * collection_decisions / B` optimizer updates against current replay; require that result to be an integer. Every pilot choice satisfies this at the initial settings. Blocks ending below the warmup threshold produce no updates; the first update follows the block that reaches or exceeds it. Whole-block collection can round warmup upward; record the actual count. All warmup decisions use epsilon `1.0`. Target-copy cadence counts optimizer updates, not environment steps. Log both sampled-item reuse and optimizer updates per collected transition.

Let s be the completed transition count in the current stage and D be `epsilon_decay_transitions`. Outside warmup, epsilon at a collection block's start is `epsilon_end + (epsilon_start - epsilon_end) * max(0, 1 - s/D)` and stays fixed for that block. All run/evaluation budgets are whole multiples of the selected block size; reject an incompatible tuned configuration before starting. Evaluation checks use total collected transitions and are also triggered at every stage end.

Loss uses stopped targets, averages over B, and never clips rewards or Q-values. Record predicted/target minima, maxima, mean absolute TD error, loss, and pre-clipping gradient norm. A nonfinite state, reward, value, loss, or gradient is a failed run to diagnose; do not substitute zero actions, clamp NaNs, or silently restart.

### D4. Parallel execution and curriculum

First inspect the actual `.venv` and available devices. Run CPU semantic tests, then request GPU execution through `sandbox_permissions="require_escalated"` as required by this host. Use one process and one explicit `CUDA_VISIBLE_DEVICES` value per GPU, set before importing JAX. Pin `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80` for isolated one-process-per-card runs; do not colocate default-allocator training processes.

The single-GPU pilot tries N in `[1024,8192,65536]` sequentially on one card, for both exploration families. Each trial completes warmup, compilation, and 100 timed collection/update blocks. Synchronize outputs before timing. Record collection throughput, update throughput, end-to-end transitions per second, peak device allocation, and driver-reserved memory. Choose the N with the largest minimum throughput across the two families among choices below 80% active-memory use; ties within 5% choose smaller N. A pilot memory failure is recorded and excluded, never a silent runtime batch-size fallback. Keep the chosen N identical across comparison families.

| Stage | Reset distribution | Advance rule |
|---|---|---|
| `0: capture` | Half tight capture box below; half existing near-upright stratum | Individual seed passes M2 twice consecutively |
| `1: expansion` | Half existing near-upright, one-quarter moving swing, one-quarter downward | Individual seed passes M3 twice consecutively |
| `2: benchmark` | Existing downward/moving/near-upright strata with equal probability | Train to its budget, retaining best eligible checkpoint |

The tight box samples theta, beta, omega, nu independently and uniformly within `±[0.08 rad,0.12 rad,0.15 rad/s,0.30 rad/s]`, then sets alpha to pi plus beta and all episode memory/flags to reset values. This is an explicit RL curriculum distribution; it does not change the environment's three benchmark reset strata. Implement its sampling inline in collection and evaluation setup, without adding an environment reset mode or helper.

At a stage change keep weights, optimizer, target, replay, active episodes, and their clocks. Only subsequent resets use the new mixture; restart the exploration schedule and stage transition counter. No deadline extension or curriculum advancement without the gate is permitted. A stage that exhausts its budget without passing stops that trial with a failed gate.

Initial sweep: two families times four seeds across eight GPUs. Family IDs are `0=ordinary`, `1=guided`; seed labels are `2026100600` through `2026100603`. Fold the family ID into each seed's root key so all eight training roots differ. Save root and stream identities. Validation root is `2026100700`; final test root is `2026100800`. Use identical saved validation/test initial states across all models and controllers. Evaluation randomness never shares training keys.

Within each seed, choose checkpoints by highest minimum benchmark-stratum success rate, then highest overall success, then lowest arm-failure rate, then lowest success-conditional powered time; ties keep the earlier checkpoint. During stage 0 use tight/wide capture rates in place of benchmark-stratum rates. Only checkpoints satisfying the already-required capture gates are eligible for final selection. If none is eligible, report failure and retain latest weights for diagnosis; do not label them a selected passing model. Missing success-conditional duration is unavailable, not zero. Compare families lexicographically by medians of the same metrics across all four seeds, reporting every failed seed. Do not select a method from its single best seed.

### D5. Milestones, validation procedures, and success flags

Validation sets are generated once and saved: 512 tight-box starts and 1024 starts from each existing benchmark stratum. A separate final test uses the same counts and its distinct root. Greedy evaluation has zero exploration, preserves the 20-second deadline and existing goal dwell, and uses initial goal count zero. Gate percentages are empirical point estimates; also report 95% Wilson intervals for success/failure proportions. They are not safety or optimality guarantees.

| Milestone | Required procedure | Success flag |
|---|---|---|
| M0: baseline and PPO retirement | Record revision/dependencies and passing pre-removal tests; perform NN-design N5 cleanup; audit imports and run retained tests | No rotary PPO runtime/configuration remains; retained JAX environment, MPC, and inverted-pendulum behavior passes |
| M1: contracts | Test all reward terms, early terminal durations, potential telescoping, observation order, replay wrap/reset, handcrafted Bellman targets, and target copies | Every semantic test passes on CPU; float32 shaping-sum error at most `1e-4` reward unit over 200 decisions |
| M2: local capture | Evaluate tight and original near-upright starts every scheduled check | At least `90%` tight success, `80%` wide success, zero tight violations, and at most `1%` wide violations; two consecutive checks per advancing seed |
| M3: expansion | Evaluate all three benchmark strata plus tight regression set | Tight/wide M2 retained; at least `30%` downward and `30%` moving success, at most `10%` violations in each of those strata; two consecutive checks |
| M4: learned policy | Finish balanced training; evaluate selected checkpoints on untouched final test | At least three of four seeds in one family: overall success at least `80%`, each stratum at least `70%`, each stratum arm failures at most `5%`, and both tight/wide M2 capture thresholds retained |
| M5: value prior | Perform D6 continuation-return audit per passing checkpoint | D6 calibration and action-regret gates pass for at least three seeds; final results report all four |
| M6: deployment | Run all four D6 controllers on shared final starts; record warm wall time and outcomes | D6 controller-improvement gate passes for the same three seeds; complete matched report |

Boundary/terminal tests include all three action indices; terminal at the first, middle, and fifth physics sample; simultaneous timeout/success precedence; repeated terminal calls; and batched `jit`/`vmap` execution. Reward tests verify lower energy error helps when capture term is held equal, target-energy overshoot is not rewarded as gain, rotating-at-target-energy is not success, and equal-potential cycles have zero net shaping. Check finite-difference reward examples and analytical sums rather than claiming every one-step heuristic action is globally correct.

At M1 also test zero-torque behavior from the tight set and enumerate all three-decision action sequences there. These are local diagnostics of pulse resolution, not a proof that longer action sequences can or cannot capture. If learning fails, inspect traces and short-sequence reachable outcomes before assigning the cause to network capacity.

### D6. Value audit and deployment metrics

For each checkpoint, take the first 256 evaluation resets in each benchmark stratum. From their greedy trajectories, additionally select one active pre-action boundary per episode uniformly using the evaluation stream, including the initial boundary among eligible choices. Also enumerate the F7 planning branches from those resets and sample 256 active depth-three endpoints per stratum uniformly without replacement from the indexed candidate pool. If a stratum has fewer than 256 active endpoints, record insufficient audit coverage and do not pass the gate. Preserve physical state, remaining time, and hold count for every start.

This yields 2304 audit starts: 768 resets, 768 greedy-trajectory states, and 768 planner endpoints. Duplicate physical states reached by distinct candidate sequences retain distinct identities. For each start, force each of the three first actions and then follow the frozen greedy policy to the task terminal event. Retain all start/action identifiers. Development audits use validation resets; the final audit uses the untouched final-test resets. Never tune a checkpoint on the final audit.

Compare exported Q_base with the realized complete base return of each branch. This measures consistency with the deployed greedy continuation, not error against an independently known optimal Q-function. RMSE is the square root of mean squared prediction error; signed bias is mean predicted minus realized return. Also compute the fraction with overprediction greater than `2.0` reward units, and action regret: the largest of the three measured branch returns minus the branch return selected by predicted Q. Audit gates: RMSE at most `1.0`, overprediction fraction at most `5%`, mean regret at most `0.25`, and 90th-percentile regret at most `1.0`. Require these separately for the three audit cohorts within each stratum; do not hide a failing cohort in a global average.

Evaluation modes are exactly `greedy`, `lookahead_zero`, `lookahead_potential`, `lookahead_q`, and `value_audit`. F7 defines the three planner scores. Simulate candidate and environment axes in parallel; sequential time remains an event-aware scan. No heuristic action masking is applied to planning or evaluation. The hypothetical planner states are never inserted into training replay in this phase.

For each controller report per-stratum and overall success, arm failure, timeout, powered duration, completion time, off-to-on count, reversal count, peak energy ratio, and maximum arm angle. Overall means the equal-size union of the three benchmark strata, excluding the separate tight-box diagnostic. Off-to-on means previous index zero and current index nonzero, including the first action after reset. A reversal means two consecutive nonzero torques with opposite signs; a sign change across an intervening off action is not counted as a reversal. Energy ratio is E_s/E_star. Energy and angle peaks are sampled at decision boundaries including the final state, not inferred between samples.

Report durations both over all episodes and conditional on success, with cohort counts and unavailable values for empty cohorts. Powered duration is not switch-on count. For resource comparisons use states that both compared controllers solve; report that paired cohort's size so failures cannot make a controller appear economical. Freeze the evaluation batch size at `1024`, with explicit valid-lane masking for a smaller final chunk; batch-1 timing is a separate measurement.

The deployment gate compares learned-prior planning with handcrafted-potential planning on the same starts. For each of the three passing seeds require either: (a) overall success improves by at least `5` percentage points, or (b) success is no more than `1` percentage point lower and mean powered duration on paired successful episodes falls by at least `10%`. In both cases require arm-failure rate to rise by no more than `1` percentage point in any stratum, and paired-success coverage at least `50%` of all starts for criterion (b). These tolerances define an empirical non-regression margin, not a claim of identical risk. Also show zero-terminal and greedy-Q results; report timing overhead even when the gate passes.

Measure batch-1 decision latency and fixed evaluation-batch throughput after compilation, including candidate generation, prediction, network calls, and selection. Synchronize results before stopping timers. Report compilation separately, with median and 95th-percentile latency over at least 1000 completed decisions. No deployment-speed threshold is claimed for this proof of concept.

### D7. Tuning rounds and stop decisions

1. Complete semantic tests and the single-card pilot, then run the eight independent initial trials. Every progress record names phase, stage, transitions, updates, elapsed time, throughput, outcomes, and current evaluation status. During long compilation/evaluation the temporary batch launcher polls subprocess status and emits a host heartbeat at most every 30 seconds, distinguishing alive/waiting from completed progress. This is top-level launch orchestration, not another project helper. If a process is silent for more than 60 seconds, stop it, add progress reporting, and rerun the affected stage.
2. Inspect per-seed histories and representative successes/failures before the next round. Classify failure as insufficient capture resolution, insufficient exploration/coverage, energy/capture shaping imbalance, excessive use cost, or unstable value fitting; cite actual metrics/traces. A classification is a hypothesis until its controlled comparison supports it.
3. Change at most one parameter family per round: exploration; the two duration costs; capture weight; or learning/replay settings. Start from half/base/double of the diagnosed coefficient, retaining valid probability bounds. Keep four seeds per compared setting, using successive eight-GPU waves if needed. Keep validation sets fixed and label all revisions. Changing reward weights starts fresh replay/training; it does not reuse stored rewards from another objective.
4. Judge tuned improvements on validation only. Once a complete candidate is selected, run its frozen final test and D6 audit. Do not tune on final-test failures; any later selection round requires a newly seeded final holdout and must retain the old result as prior evidence.
5. Continue while a controlled adjustment improves the currently failing gate. After two successive targeted rounds produce less than `2` percentage points improvement in the failing success rate and less than `10%` improvement in its associated value error or on-time metric, record a tuning plateau. If a gate still fails, report this phase as incomplete, state the supported diagnosis, and propose the smallest formulation change. Do not launch unbounded sweeps or claim success from training return alone.

For manual focused rounds, use the D3 tunable keys. For an activated unattended session, training-plan T2 is the complete additional search space and T3–T5 govern evidence, promotion, and stopping. Action values, clock, goal, gamma, terminal amounts, input features, and planner horizon remain fixed. No synchronous multi-GPU learner or algorithm outside Double DQN is authorized.

### D8. Artifact and checkpoint contract

Each trial writes under `artifacts/rotary_pendulum/q-prior/<round>/<family>/<seed>/`. Keep artifacts in the existing ignored runtime tree; do not commit model tensors or trajectory archives under docs. The development report links them and distinguishes failed, policy-ready, value-ready, and deployment-demonstrated gates.

| Location | Required contents |
|---|---|
| `interpretation_summary.md` | Human-written concise outcome, passed/failed gates, failure interpretation, and next decision |
| `human-readables/` | `learning.png`, `outcomes.png`, `trajectories.png`, `q_calibration.png`, `deployment.png`; readable units, short defined legends |
| `machine-scannables/metadata.json` | Contract/reward revisions, git revision and dirty patch identity, resolved experiment, seeds, device, dependency versions, clocks and physical-config hashes |
| `machine-scannables/history.csv` | Progress, optimizer diagnostics, decomposed rewards, training outcome totals, stage and counters |
| `machine-scannables/evaluation.csv` | Checkpoint/controller/stratum counts, rates, intervals, duration and switching metrics |
| `machine-scannables/initial_states.npz` | Validation/final EnvState arrays, split identity, seeds; no training samples mislabeled as holdout |
| `machine-scannables/trajectories.npz` | Selected physical/augmented histories, action indices, actual durations, reward components, terminal reasons, valid-step masks |
| `machine-scannables/value_audit.npz` | All audit starts, three predicted/realized returns, chosen actions, regret, and cohort identifiers |
| `machine-scannables/timing.csv` | Pilot and evaluation timing, memory, warm/compile distinction, batch size, hardware |
| `machine-scannables/checkpoints/` | Selected and latest Flax weights/target/optimizer/key state plus complete export metadata |
| `machine-scannables/run.log` | Flushed console progress, compilation milestones, and explicit stop/failure reasons |

Only write figures relevant to completed milestones; report later figures as not produced rather than inventing empty results. Save the first five successes and first five failures by evaluation index in each stratum/controller when available; absence is recorded, without replacing them with cherry-picked examples. Episode histories have a valid mask and retain the final pre-reset state. No Chinese companion documents are generated for this requested phase-9 document set.

Checkpoint metadata must state input feature order/scales, action order/units, architecture, shaped-output meaning, Phi formula/coefficients, base-reward coefficients, gamma, terminal masking, deadline, physical/task hashes, training stage, and validation gates. Serialize optimizer and PRNG state for inspection/fine-tuning, but do not claim bitwise training resume without replay and rollout state. Version 1 checkpoints are inference/fine-tuning artifacts; fresh fine-tuning replay is explicit. Test reloaded float32 predictions against the saved batch with absolute tolerance `1e-6`.

Round summaries aggregate every trial, including failed seeds, into separate machine CSVs and human comparison figures. Update dependency records alongside code changes. Run focused checks after each semantic milestone and the full repository test suite before final handoff; do not repeat unchanged successful checks without a new reason.

### D9. Handoff after success

Deliver the frozen checkpoint, its export contract, complete failed/successful-seed evidence, value audit, and matched deployment comparison. Phase 9 is complete only when M0–M6 pass. A usable policy with a failed value/deployment gate is a partial result and must be labeled accordingly.

Next phase order is fixed at the conceptual level: richer/continuous torque critic; switching-aware reward with previous applied torque last in the observation; outer switch-time search containing inner action-sequence optimization; then joint training from sampled-MPC experience. Freeze the detailed formulation and structure budget of each extension separately. Retain phase 9 as a reproducible baseline, and never reinterpret its discrete shaped outputs as arbitrary continuous-action or schedule-conditioned values.
