# Phase 9 Neural Network and Module Design

## Human quick reading

Build the discrete Q learner from scratch after removing the old rotary PPO implementation and its configuration. The validated JAX plant and episode contract remain the simulation foundation. Earlier PPO results remain historical evidence about capture difficulty; their actor, critic, reward, optimizer state, and checkpoints are not starting points for this agent.

Keep one installable repository. The learning code is a lightweight subpackage under `src/rotary_pendulum/RL/`, composed of four task-oriented modules. A dedicated `src/bringup/rotary_q_workflow.py` file imports the environment and learning functions and owns the experiment sequence. It performs actual initialization, training scheduling, curriculum decisions, evaluation, and checkpoint selection; it is not a wrapper around a second training workflow.

Use a small fully connected network: seven input features, two hidden layers of 128 neurons with tanh activations, and three unrestricted output values. Each output estimates the remaining shaped return after choosing one of the three permitted torques. One forward pass compares off, negative full torque, and positive full torque. The network does not predict torque probabilities or physical energy.

The inputs preserve the complete observed task state: arm position, periodic pendulum orientation, both velocities, time remaining, and progress through the upright hold. Equal-energy states can require different actions, so energy alone is insufficient. An explicit energy input is a possible later comparison, not part of this baseline.

Double DQN maintains an online parameter set and a periodically copied target set for the same architecture. It does not add a separate policy network or independently trained second critic. Energy-guided exploration changes the actions used to gather experience; it does not change the network architecture or turn the learned value into a heuristic residual.

Start with this model and judge it through capture, action ranking, and complete-return calibration. A larger network is not the first response to a failed controller. After the fixed baseline is measured, [training-plan.md](training-plan.md) authorizes bounded, evidence-driven comparisons of model size, activation, and loss without changing the task. After this phase succeeds, consider an action-conditioned critic for richer torque choices, then switching-aware state and joint MPC training.

Read [formulation_plan.md](formulation_plan.md) for physical/reward semantics and [development_plan.md](development_plan.md) for initial settings, training gates, and artifacts. This document fixes the baseline network, import boundaries, and PPO removal scope. The training plan owns permitted post-baseline variants. The implementation and PPO retirement are complete; training remains intentionally pending.

## Machine-scannable specification

### N0. Authority and implementation order

| Key | Required value |
|---|---|
| `contract_id` | `rotary-q-prior-v1` |
| `network_id` | `mlp-7-128-128-3-tanh-v1` |
| `status` | Implemented and CPU-validated; no phase-9 training or trained weights yet |
| `initialization_source` | Fresh seeded initialization, never a PPO checkpoint |
| `distribution` | Existing root `burst-coast-mpc` project only |
| `learning_package` | `rotary_pendulum.RL` |
| `workflow_module` | `bringup.rotary_q_workflow` |
| `workflow_entry` | `train(experiment)` |
| `implementation_order` | Baseline checks, PPO removal, fresh Q modules/workflow, semantic tests, training and deployment gates |
| `documentation_language` | English only for this phase-9 document set |

The latest user instruction supersedes the old requirement to preserve rotary PPO. The NumPy/CasADi plant and MPC and the inverted-pendulum workflows remain in scope for regression checks. The task, reward, and observation definitions are unchanged, so no new return-contract identifier is required for this packaging/removal revision. N2 owns the baseline structure; development-plan D3 owns baseline optimizer/settings; training-plan T2 owns bounded post-baseline variants and their identifiers.

### N1. Package boundary and workflow ownership

The root `pyproject.toml` already discovers `rotary_pendulum*` and `bringup*`. Do not create another `pyproject.toml`, `setup.py`, distribution name, package version, or installation step under RL. Install the repository once into the existing `.venv`; add and pin the tested Flax/Optax optional dependencies in the root manifest when implementing.

| Path relative to `src/` | Owned computation | Public symbols |
|---|---|---|
| `rotary_pendulum/RL/jax_task.py` | Observation/potential, rewarded transition, exploration, replay collection | `observe`, `transition`, `collect` |
| `rotary_pendulum/RL/jax_q.py` | Network and Double DQN update | `QNetwork`, `update` |
| `rotary_pendulum/RL/jax_evaluation.py` | Greedy control, value audit, three planning comparators | `evaluate` |
| `rotary_pendulum/RL/jax_artifacts.py` | Checkpoint/data/figure writing | `write_artifacts` |
| `bringup/rotary_q_workflow.py` | Initialize and control one complete training/evaluation trial | `train` |

The workflow imports the existing JAX environment and these learning modules. Learning modules must not import the workflow or root CLI. The task module may import the Q model for action selection; the Q model/update module must not import task or workflow code. Evaluation may import task and model code. Artifacts consume supplied arrays and metadata without controlling training. This dependency direction avoids circular imports.

Keep `rotary_pendulum/RL/__init__.py` passive: package documentation only, with no eager model, environment, optimizer, or framework imports. Use explicit submodule imports at call sites. Importing the Q path must not import the old PPO modules, PyTorch, or Lightning. Importing reusable modules must not start training, launch processes, write artifacts, or read PPO configuration. Existing physics/mission loading by the JAX environment remains its established behavior.

The workflow owns explicit experiment resolution, seeded initialization, replay allocation, curriculum gates, collection/update ordering, progress, evaluation cadence, and checkpoint selection. GPU visibility is set by the process launcher before importing JAX. Each process calls this same workflow with one resolved trial record. A temporary launcher may schedule independent trials and report process heartbeats; it must not duplicate learner or curriculum logic.

The proposed `RL/jax_training.py` is replaced by the workflow file; do not implement both. There is no new CLI command, general experiment manager, workflow class, actor wrapper, or configuration dataclass.

### N2. Forward architecture and learned parameter budget

Let o be one seven-element observation defined in N3. Let h1 and h2 be computed hidden feature vectors of length 128, and q be the three computed output values. W1, W2, W3 are learned dense weight matrices; b1, b2, b3 are learned bias vectors. The forward map is:

$$
\begin{aligned}
h_1&=\tanh(oW_1+b_1),\\
h_2&=\tanh(h_1W_2+b_2),\\
q&=h_2W_3+b_3.
\end{aligned}
$$

The tanh function acts elementwise. Its bounded hidden outputs are a chosen baseline, not a guarantee that the learned value matches sharp success/failure boundaries. The final layer is linear and permits both positive and negative returns.

| Layer name | Input/output widths | Learned array shapes | Parameter count |
|---|---|---|---:|
| `hidden_0` | `7 -> 128` | W1 `(7,128)`, b1 `(128,)` | `1024` |
| `hidden_1` | `128 -> 128` | W2 `(128,128)`, b2 `(128,)` | `16512` |
| `q_values` | `128 -> 3` | W3 `(128,3)`, b3 `(3,)` | `387` |
| Total | One online model | Six learned arrays | `17923` |

These are Flax parameter leaf names `kernel` and `bias` under the listed layer names; mathematical W/b notation refers to those leaves. Instantiate the baseline as `QNetwork(hidden_widths=(128,128), activation_name="tanh")` with one `__call__(observation)` method. `hidden_widths` selects the dense hidden widths and `activation_name` selects exactly `tanh`, `silu`, or `relu` as authorized by training-plan T2. Use the Flax Linen compact module pattern and its standard `init`/`apply` interface. No handwritten constructor, custom layer class, or encoder/head wrapper is planned. Input width seven and output width three are fixed by this contract.

All arrays and forward computation use float32. Input shape `(...,7)` maps to output shape `(...,3)`, preserving any leading batch/candidate axes. An individual observation of shape `(7,)` returns `(3,)`. No softmax, sigmoid, softplus, output clipping, batch/layer normalization, dropout, recurrence, skip connection, attention, or auxiliary prediction head is included.

For each layer, initialize its kernel with Glorot uniform and its bias with zeros. If n_in and n_out are that layer's input and output widths, computed from the table, the sampling bound b_init and initialization distribution are:

$$
b_{\rm init}=\sqrt{\frac{6}{n_{\rm in}+n_{\rm out}}},\qquad
W_{ij}\sim\operatorname{Uniform}(-b_{\rm init},b_{\rm init}).
$$

Indices i and j select input/output coordinates of a kernel. The distribution is a seeded initialization choice, not a physical prior. Use independent random draws for the three kernels from the model-initialization PRNG domain specified in D2. Initialize target parameters as an exact copy of online parameters, not as another random network. Record both parameter counts; there are 17923 trainable scalars and another 17923 target scalars, excluding optimizer storage.

### N3. Input features and action-output meaning

The following table restates F2's order without changing it. Physical coordinates and counters are computed by the existing environment; scales are fixed from physical/task configuration, not fitted from replay.

| Index | Feature | Definition and interpretation |
|---:|---|---|
| `0` | Arm position | Unwrapped arm angle divided by the configured arm travel limit |
| `1` | Pendulum sine | Sine of its downward-referenced angle, representing periodic orientation |
| `2` | Pendulum cosine | Cosine of the same angle; together with sine distinguishes all orientations |
| `3` | Arm velocity | Arm angular velocity divided by arm limit times the derived natural angular frequency |
| `4` | Pendulum velocity | Pendulum angular velocity divided by the upright-energy speed scale defined in F2 |
| `5` | Time remaining | Remaining physics steps divided by 1000; one at reset, zero at deadline |
| `6` | Goal-hold progress | Consecutive inside-goal physics samples divided by five |

Do not clip features or replace them by an energy-only state. Remaining time and hold progress are required for this task's Markov observation. Previous action is excluded from the network because version-1 reward/dynamics do not use it. Diagnostic switching counts may retain previous action separately.

| Output index | First action | Meaning |
|---:|---|---|
| `0` | Zero torque | Predicted remaining shaped return after one off decision and learned continuation |
| `1` | Negative full torque | Same return definition with negative full torque as the first action |
| `2` | Positive full torque | Same return definition with positive full torque as the first action |

The full torque magnitude is read from the physical configuration, currently `0.0204 N m`. Greedy selection takes the largest output; exact ties use the lowest index. Output units are reward units, not joules, seconds, probabilities, or motor torque. The network has no action input: action conditioning is represented by the three output columns.

The output estimates the shaped training return from F4/F5, including the actual remaining deadline and goal hold. Use F5's known potential correction to export base-return estimates. Apply explicit zero continuation at terminal states. The network itself does not enforce terminal values, and raw shaped values at different states cannot directly be compared as switch-time scores.

### N4. Learning state and training responsibilities

Use the existing D2 object budget: `learner`, `rollout`, `replay`, and `experiment`. The learner stores exactly `params`, `target_params`, `opt_state`, `key`, and `updates`. Local `network` is the Flax module instance; local `optimizer` is the direct Optax transformation. Their construction requires no new project classes. Learned W/b arrays live inside params; h1, h2, and q are temporary computed values, not stored agent memory or tunable settings.

The online network chooses the next maximizing action; the target network supplies its value for a stopped-gradient Double DQN target. Optimize mean Huber error using the D3 Adam and gradient-clipping settings. Copy online parameters to target every 256 optimizer updates initially. There is one optimized critic, one copied target, no PPO policy likelihood, no action-distribution standard deviation, no advantage estimator, and no actor loss. The algorithm rationale is described in the [Double DQN paper](https://arxiv.org/abs/1509.06461); all task-specific choices are defined here and in the formulation.

Ordinary and heuristic-guided exploration use identical network structures, initialization rules, losses, and replay contracts. Only behavior-action proposals differ. Neither imports a trained PPO model or fits a residual to the energy heuristic. Do not load actor weights into the Q encoder or interpret an old PPO state-value output as the new action-value prior.

### N5. Exact rotary PPO retirement scope

Perform this cleanup before adding the fresh Q implementation. The following inventory was checked against the current source; rerun the reference search when implementing so newly introduced call sites are not missed.

| Existing path | Required action |
|---|---|
| `src/rotary_pendulum/RL/model.py` | Delete the PPO actor-critic and its Gaussian action-distribution code |
| `src/rotary_pendulum/RL/environment.py` | Delete the PPO-specific NumPy batching, curriculum, observation, and reward adapter |
| `src/rotary_pendulum/RL/training.py` | Delete PPO rollout buffers, GAE, clipped-policy updates, training loop, and module entry point |
| `src/rotary_pendulum/RL/evaluation.py` | Delete PPO policy evaluation and its different dwell/clock assumptions |
| `src/rotary_pendulum/RL/artifacts.py` | Delete the PPO-only artifact writer and plotting implementation |
| `src/rotary_pendulum/RL/__init__.py` | Replace PPO imports/exports and description with passive Q-package documentation |
| `src/rotary_pendulum/configs/rl.yaml` | Delete the entire PPO configuration file; do not rename its keys into a Q configuration |
| `src/rotary_pendulum/utils/config_schema.py` | Remove `PPO_CONFIG_PATH`, `PPOConfig` and its three validators, and `load_ppo_config`; retain shared configuration logic |
| `tests/test_rotary_rl.py` | Delete PPO-only tests; create the two Q test files already budgeted in D1 |

The schema validators to remove are `_curriculum_decisions_must_define_three_stages`, `_initialization_bounds_must_be_complete`, and `_minibatch_must_fit_rollout`. Keep shared imports if still used: `math`, `Any`, `cast`, and `field_validator` currently also serve retained configuration code.

Remove every key in the old `ppo` domain, including the following exhaustive top-level inventory. Identically named concepts needed by Q learning are new explicit D3 settings, not inherited PPO values or compatibility aliases.

| Old key group | Keys removed with `rl.yaml` |
|---|---|
| Collection/update | `action-repeat-steps`, `parallel-environments`, `rollout-steps`, `minibatch-size`, `optimization-epochs` |
| Curriculum | `curriculum-decisions`, `initialization-bounds`, `stage-b-downward-fraction`, `stage-c-exact-fraction`; also delete the nested `near-upright`, `downward-noise`, and `phase` bound entries |
| PPO objective | `discount-time-constant-s`, `gae-lambda`, `policy-clip`, `value-loss-weight`, `entropy-weight`, `target-kl`, `initial-log-standard-deviation` |
| Optimization | `initial-learning-rate`, `final-learning-rate`, `adam-epsilon`, `maximum-gradient-norm` |
| Reward | `phase-weight`, `torque-slew-weight`, `torque-effort-weight`, `arm-boundary-weight`, `success-bonus`, `arm-failure-penalty`, `timeout-penalty` |
| Evaluation/run | `evaluation-episodes`, `evaluation-hold-steps`, `evaluation-every-updates`, `random-seed`, `artifact-root`, `device` |

Check source, tests, active entry points, and active usage documentation for dangling PPO references. Current inspection found no dedicated PPO route in `bcmpc.py` or `src/bringup/`; the generic existing train command belongs to the inverted-pendulum workflow and must not be deleted as a presumed PPO route. Revise stale active usage statements when adding the new workflow.

PyTorch and PyTorch Lightning remain dependencies of retained inverted-pendulum critics and offline training. Do not uninstall or remove those shared dependencies merely because rotary PPO used PyTorch. The new rotary Q import path must be independent of them. Remove a dependency only if a whole-repository usage audit demonstrates that it was PPO-exclusive.

Keep historical PPO design/results documents and existing experiment artifacts as labeled historical records. They are not executable components and must not be treated as current instructions or Q checkpoints. Do not create a second archived executable PPO copy inside the package; source history provides the removed implementation. This document specifies future removals and does not claim they have already happened.

### N6. Structure discipline and validation milestones

The complete new implementation remains within D1's budget: four RL source files plus one workflow file; one project-defined Flax module class, one method, seven module-level functions, and four substantive nested callbacks. The only move is `train(experiment)` into `bringup/rotary_q_workflow.py`. There is no additional training function or wrapper. Two planned test files cover the following checks using their existing parametrized-function budget.

Keep substantive files 40–300 lines, use compact task-oriented logic and parallel batch computation, and avoid thin wrappers, fallback behavior, unnecessary configuration surfaces, and `try`/`with`/`except`. The existing passive package initializer needs no executable filler. The architecture adds no tunable quantities beyond the D3 settings; hidden activations and all reward/normalization intermediates are computed, not independent parameters.

| Milestone | Validation procedure | Success flag |
|---|---|---|
| N-A: retirement | Capture a passing baseline; remove N5 items; scan retained runtime/test references and run retained tests | No rotary PPO implementation/config/export/caller remains; JAX environment, MPC, and inverted-pendulum tests pass |
| N-B: imports | In a fresh Python process import the Q submodule and dedicated workflow after editable project installation | No PyTorch/Lightning or deleted PPO modules in imported modules; no automatic experiment or output creation |
| N-C: model | Check layer leaves, exact parameter count, single/batch/candidate shapes, seeded initialization, and online/target equality | Six online arrays totaling 17923 scalars; output trailing width three; deterministic replay of seed; finite float32 outputs |
| N-D: learning | Hand-check ordinary and terminal targets; test stopped target gradients, parameter updates, and target-copy cadence | Online weights update from a nonzero-loss batch; targets change only on scheduled copies; terminal target equals its immediate reward |
| N-E: checkpoint | Reload the saved architecture/weights and a recorded observation batch | Action order and features match metadata; absolute prediction error at most `1e-6` |
| N-F: behavior and prior | Apply development-plan M2–M6, including per-stratum capture and complete-return audit | Existing gates pass; shape correctness and lower TD loss alone do not establish success |

The baseline is recorded before deleting PPO tests. After removal, report the changed test inventory rather than demanding the historical test count remain unchanged. N-C adds its architecture checks to the planned checkpoint/model test coverage; N-D uses the planned target/gradient tests. Do not create a separate validation framework.

### N7. Changes after a successful baseline

An explicit energy-feature comparison changes the observation contract and therefore remains outside unattended phase-9 tuning. Controlled model-size, activation, and loss comparisons may proceed under training-plan T2 when the evidence calls for them; each receives a new network or experiment identifier.

For richer discrete torque, expand the action-output mapping and retrain/validate the changed critic. For continuous torque, supply torque as an input to an action-conditioned scalar head; a shared state encoder may be reused or distilled, but three learned action columns do not determine reliable values between those actions.

When switching costs are introduced, append previous applied torque as the final state feature and retrain the compatible value definition. Then implement the outer bang-off/off-bang switch-time search with inner action-sequence optimization, followed by joint training using sampled-MPC experience. Preserve deadline accounting, shaping conversion, prefix costs, and the distinction between powered duration and switch-on count described in formulation F8.
