# Wake Sleep MPC Engineering Development Plan

## 1. Engineering conclusion

Use a Python first research codebase with strict node interfaces.

Do not start with C++ or MATLAB. The first benchmark needs rapid iteration, custom objective construction, learned terminal costs, data logging, and ablation control. Python gives the shortest path to a working system. C++ becomes relevant only after the pendulum benchmark works and you need embedded style real time solve latency. MATLAB is useful only for quick control sanity checks, not for the main codebase.

The simplest workable architecture is:

1. A Python monorepo.
2. Three replaceable computational nodes: `simulation`, `mpc`, and `rl`.
3. A thin message interface shared by all nodes.
4. One coordinator node that owns experiment time.
5. A logger node that records every state, action, schedule, solve artifact, critic snapshot, and config hash.
6. In process execution first. Separate processes with ROS 2 or ZeroMQ only after the interfaces stabilize.

The implementation should preserve the theoretical structure: the MPC optimizes only the active burst, the learned Q critic scores the post burst coast, and the simulator remains the source of truth for executed state and action history.

## 2. Language choice

### 2.1 Recommended main stack

Use Python 3.12 on Ubuntu 24.04.

Reason:

1. ROS 2 Jazzy matches Ubuntu 24.04 if you later want true node separation.
2. CasADi, do mpc, PyTorch, Gymnasium, Hydra, h5py, Zarr, DuckDB, and plotting tools are all Python friendly.
3. Custom learned terminal costs are easier to debug in Python than in C++.
4. The pendulum benchmark is too small to justify early C++ complexity.
5. Python lets the same objects serve simulation, MPC prediction, RL replay, and offline analysis.

### 2.2 Where C++ fits later

Use C++ only for a later solver backend or deployment node.

Good future uses:

1. acados generated solver code.
2. A hard real time control bridge.
3. A compiled dynamics model if Python solve latency becomes the bottleneck.
4. A ROS 2 component node for real hardware integration.

Do not use C++ for the first pendulum benchmark.

### 2.3 Where MATLAB fits

Use MATLAB only as a scratchpad for checking classical pendulum control, LQR, or symbolic derivations.

Do not make MATLAB part of the main experiment pipeline. It will increase reproducibility friction and make RL integration worse.

## 3. Package choices

```text
core:
  python==3.12
  pydantic
  hydra-core
  omegaconf
  rich
  typer
  pytest
  ruff
  mypy

simulation:
  gymnasium
  scipy
  numba
  matplotlib
  plotly

mpc:
  casadi
  ipopt
  do-mpc
  acados_template
  scipy

rl:
  torch
  lightning
  stable-baselines3
  cleanrl
  tensorboard
  wandb

interfaces:
  rclpy
  pyzmq
  fastapi
  websockets

data:
  h5py
  zarr
  pyarrow
  duckdb
  pandas

monitoring:
  tensorboard
  plotly
  dash
  streamlit

experiment:
  mlflow
  wandb
  joblib
  optuna
```

Interpretation:

1. Use `casadi` plus IPOPT for the first nonlinear MPC prototype.
2. Keep `do-mpc` available for fast baseline construction and sanity checks.
3. Use `acados_template` only after the CasADi prototype works.
4. Use `torch` directly for the positive definite Q critic.
5. Use Stable Baselines3 only for baseline RL controllers, not for the core wake sleep method.
6. Use CleanRL as a style reference for compact custom training scripts.
7. Use `h5py` or Zarr for dense trajectory arrays.
8. Use Parquet plus DuckDB for episode summaries and queryable metrics.
9. Use Hydra for repeatable configuration sweeps.
10. Use TensorBoard first. Add Weights and Biases or MLflow only when experiment count becomes large.

## 4. Simplest node structure

### 4.1 First version

Use in process Python nodes.

Each node is a class with the same interface pattern:

```python
class Node:
    def configure(self, cfg): ...
    def reset(self, seed: int): ...
    def step(self, inbox, clock): ...
    def publish(self): ...
    def close(self): ...
```

The coordinator calls each node in a deterministic order:

```text
coordinator tick:
  1. simulation publishes current observation
  2. early wake monitor checks deviation and constraint margin
  3. mpc solves only if current time is a decision epoch
  4. action scheduler selects burst input or coast hold input
  5. simulation advances one integration step
  6. logger writes state, action, schedule, costs, flags, and artifacts
  7. episode manager checks termination and success
```

This gives deterministic replay and easier debugging.

### 4.2 Second version

Move nodes into separate processes only when the in process version is stable.

Use one of two options:

1. ROS 2 Jazzy if the project may later connect to robotics middleware, real sensors, or hardware style process isolation.
2. ZeroMQ if you only want lightweight research process separation without ROS message overhead.

Recommended path:

1. Start in process.
2. Add a message bus abstraction.
3. Implement a ROS 2 adapter later.
4. Keep the scientific node logic independent of ROS 2.

The important rule is that the computational nodes should not import each other. They should only exchange typed messages.

## 5. Required nodes

### 5.1 Simulation node

Purpose:

Run the continuous nonlinear inverted pendulum benchmark and publish observations.

Responsibilities:

1. Own the true state.
2. Integrate dynamics with fixed step RK4.
3. Apply the action selected by the action scheduler.
4. Support deterministic and noisy modes.
5. Publish state, observation, true action, energy, goal flag, and constraint status.
6. Save dense trajectory arrays to the experiment record.
7. Support real time pacing for visualization, but never let display timing alter simulation results.

Inputs:

1. `ActionCommand`
2. `ResetCommand`
3. `NoiseConfig`
4. `RealTimeConfig`

Outputs:

1. `StateObs`
2. `SimDiagnostics`
3. `StepRecord`

Hot swap targets:

1. Nonlinear pendulum.
2. Piece wise linear pendulum.
3. Cart pole.
4. Hill Clohessy Wiltshire relative motion.
5. Two body spacecraft relative motion.

Implementation notes:

1. The simulator must expose `step(x, u)` and `rollout(x0, U)`.
2. The same discrete dynamics function should be callable by the MPC prediction model.
3. Angle wrapping must be centralized. Never duplicate wrapping logic in separate nodes.
4. The simulator must record both commanded action and actually applied action.
5. The simulation node should not train RL and should not solve MPC.

### 5.2 MPC node

Purpose:

Solve the burst only wake sleep MPC problem at decision epochs.

Responsibilities:

1. Subscribe to the latest state observation.
2. Enumerate candidate pairs `(N_b, N_w)`.
3. Build and solve the nonlinear burst optimization problem.
4. Query the Q critic or Q cache for terminal coast score.
5. Select the best horizon pair and burst sequence.
6. Publish the planned burst sequence, coast command, predicted burst states, optional predicted coast rollout, and solver diagnostics.
7. Save optimization artifacts.

Inputs:

1. `StateObs`
2. `CriticQueryService`
3. `WarmStartCache`
4. `MPCConfig`

Outputs:

1. `PlanCommand`
2. `SolverDiagnostics`
3. `PredictedExecutionTrace`
4. `OptimizationArtifact`

Hot swap targets:

1. CasADi plus IPOPT nonlinear MPC.
2. do mpc baseline.
3. acados backend.
4. SciPy optimizer baseline.
5. Grid enumerated sign sequence baseline for scalar torque.
6. Fixed horizon MPC.
7. Explicit input blocking MPC.

Implementation notes:

1. Start with CasADi plus IPOPT.
2. Keep candidate horizon sets small first.
3. Store every failed solve, not only successful solves.
4. Warm start from the previous selected plan.
5. Separate objective terms in logs: burst state cost, saturation attraction, smoothness, terminal Q, constraint residuals.
6. Do not let the learned Q critic hide hard constraint violations.
7. Deterministic coast validation should be a separate function from the optimization problem.

### 5.3 RL node

Purpose:

Train the positive definite Q critic from decision epoch transitions and serve critic queries to MPC.

Responsibilities:

1. Read completed experiment records.
2. Build decision epoch replay items.
3. Train the Q critic after each episode or from an offline dataset.
4. Maintain a target critic.
5. Save model checkpoints and optimizer state.
6. Publish critic version and training diagnostics.
7. Serve `Q(x, u_h, N_c)` values to MPC.
8. Optionally precompute Q slices and Q series for caching.

Inputs:

1. `EpisodeRecord`
2. `DecisionTransitionRecord`
3. `CriticTrainConfig`
4. `CriticQuery`

Outputs:

1. `CriticValue`
2. `CriticSnapshot`
3. `TrainingDiagnostics`
4. `QContourArtifact`

Hot swap targets:

1. Hand designed Q terminal cost.
2. Positive definite neural Q critic.
3. Tabular Q over discretized state features.
4. Gaussian process terminal cost.
5. Nearest neighbor terminal cost from rollout library.

Implementation notes:

1. Implement the positive definite Q critic directly in PyTorch.
2. Do not use an actor in the core method.
3. Normalize state features, held command, and horizon index.
4. Clip TD targets before the neural network is stable.
5. Save critic snapshots with config hash and replay dataset hash.
6. Plot Q contours before trusting the critic inside MPC.
7. Keep the first network small.

## 6. Additional nodes needed

### 6.1 Coordinator node

Purpose:

Own the experiment clock and enforce deterministic ordering.

Why needed:

Without this node, asynchronous simulation, MPC, and RL messages can create ambiguous causality. The coordinator prevents hidden timing bugs.

Responsibilities:

1. Start and stop episodes.
2. Define decision epochs.
3. Trigger MPC solves.
4. Trigger RL updates after episodes.
5. Enforce synchronous mode.
6. Allow asynchronous mode later.
7. Own global random seeds.

### 6.2 Action scheduler node

Purpose:

Convert a solved plan into one action per simulator step.

Responsibilities:

1. Apply optimized burst inputs for `N_b` steps.
2. Apply configured coast command for the remaining sleep interval.
3. Stop the current plan when early wake fires.
4. Publish action source: burst, coast, fallback, safety override.
5. Handle solver failure with a predefined fallback policy.

This node should be separate from MPC because MPC plans; it should not directly own execution timing.

### 6.3 Early wake monitor node

Purpose:

Monitor whether sleep should be interrupted.

Responsibilities:

1. Compare observed state against predicted execution trace.
2. Check constraint margins.
3. Check Q degradation if a critic is available.
4. Publish `EarlyWakeEvent`.
5. Record trigger reason and local time index.

This node becomes the empirical safety mechanism before formal robustness is proven.

### 6.4 Data logger node

Purpose:

Make every experiment reproducible and queryable.

Responsibilities:

1. Save dense arrays to HDF5 or Zarr.
2. Save summaries to Parquet.
3. Save run metadata to JSON.
4. Save configs exactly as executed.
5. Save git commit hash if available.
6. Save solver artifacts.
7. Save critic snapshots.
8. Save failure traces.

Data layout:

```text
runs/
  2026_05_03_001_pendulum_wsmpc/
    config.yaml
    manifest.json
    trajectory.zarr
    decisions.parquet
    solver_artifacts/
      decision_000012.json
      decision_000013.json
    critic/
      qcritic_step_000100.pt
      qcritic_step_000100_meta.json
    plots/
      rollout_theta_omega_energy.html
      q_contour_epoch_000100.html
    logs/
      runtime.log
      tensorboard/
```

### 6.5 Critic cache node

Purpose:

Reduce MPC latency by caching repeated terminal Q calls.

Responsibilities:

1. Cache Q values for `(state_feature, held_command, coast_horizon)`.
2. Support nearest neighbor lookup for offline records.
3. Store Q series over candidate wake horizons.
4. Version cache entries by critic snapshot id.
5. Invalidate stale cache after critic update.

This should be optional at first. Add it after learned Q is stable.

### 6.6 Monitor node

Purpose:

Provide live visibility during experiments.

Minimum live plots:

1. Pendulum angle.
2. Angular velocity.
3. Energy error.
4. Applied torque.
5. Burst versus coast mode.
6. Current wake horizon.
7. MPC solve time.
8. Terminal Q value.
9. Constraint margin.
10. Early wake trigger status.

Use TensorBoard for scalar metrics and Plotly or Streamlit for trajectory views.

### 6.7 Baseline runner node

Purpose:

Keep baselines reproducible and comparable.

Required baselines:

1. Energy shaping swing up.
2. Fixed horizon MPC with L2 input cost.
3. Fixed horizon MPC with L1 input cost.
4. Fixed horizon MPC with explicit input blocking.
5. Wake sleep MPC with hand Q.
6. Wake sleep MPC with learned Q.
7. Wake sleep MPC with learned Q and early wake.

This node should run the same seed sets and write the same output schema for every controller.

## 7. Message interface

Use typed messages. Pydantic models are enough for the first version.

### 7.1 Core messages

```python
class StateObs(BaseModel):
    run_id: str
    episode_id: int
    t_index: int
    t_sec: float
    theta: float
    omega: float
    energy: float
    energy_error: float
    wrapped_angle_error: float
    constraint_margin: float
    goal_reached: bool

class ActionCommand(BaseModel):
    run_id: str
    episode_id: int
    t_index: int
    u: float
    source: str
    plan_id: str | None = None

class PlanCommand(BaseModel):
    run_id: str
    episode_id: int
    plan_id: str
    decision_t_index: int
    N_b: int
    N_w: int
    burst_u: list[float]
    u_hold: float
    predicted_states: list[list[float]]
    predicted_modes: list[str]
    objective_total: float
    objective_terms: dict[str, float]
    solver_status: str
    solve_time_ms: float

class EarlyWakeEvent(BaseModel):
    run_id: str
    episode_id: int
    t_index: int
    plan_id: str
    trigger_reason: str
    deviation_score: float | None = None
    constraint_margin: float | None = None
    q_degradation: float | None = None

class DecisionTransitionRecord(BaseModel):
    run_id: str
    episode_id: int
    decision_index: int
    t_start: int
    t_next: int
    x_start: list[float]
    x_post_burst: list[float]
    x_next: list[float]
    N_b: int
    N_w: int
    burst_u: list[float]
    u_hold: float
    realized_total_cost: float
    realized_coast_cost: float
    early_wake: bool
    critic_snapshot_id: str | None
```

### 7.2 Interface rule

All nodes should accept and emit only these messages or arrays stored by reference. Do not pass Python object handles between nodes. This is what makes hot swapping possible.

## 8. Synchronization modes

### 8.1 Synchronous mode

Use this for development and all reported benchmarks.

Behavior:

1. The coordinator owns time.
2. Simulation advances only when coordinator calls it.
3. MPC solves block execution at decision epochs.
4. RL trains only after an episode ends.
5. Logs are deterministic.

This mode is the correct mode for algorithm research.

### 8.2 Asynchronous mode

Use only after the synchronous system works.

Behavior:

1. Simulation publishes observations continuously.
2. MPC listens and publishes plans when available.
3. Action scheduler uses the latest valid plan.
4. RL trains in a background process from completed records.
5. Critic updates are versioned and atomically swapped.

Risk:

Asynchronous execution can contaminate results because solver latency becomes part of the control law. Treat it as a later real time engineering experiment, not as the first benchmark.

## 9. Data and caching plan

### 9.1 Experiment record

Store three levels of data.

Level 1: dense step record

```text
t_index
t_sec
theta
omega
energy
energy_error
u_commanded
u_applied
mode
plan_id
constraint_margin
goal_flag
early_wake_flag
```

Use Zarr or HDF5.

Level 2: decision record

```text
decision_index
t_start
t_next
N_b
N_w
u_hold
burst_cost
terminal_q
objective_total
solve_time_ms
solver_status
early_wake
realized_coast_cost
realized_total_cost
critic_snapshot_id
```

Use Parquet.

Level 3: artifact record

```text
solver_variable_values
constraint_residuals
warm_start
predicted_burst_trace
validated_coast_trace
q_query_inputs
q_query_outputs
critic_weights
normalization_stats
```

Use JSON for small artifacts and PyTorch files for model snapshots.

### 9.2 Cache types

Use three caches.

1. Warm start cache for MPC decision variables.
2. Critic query cache for repeated terminal Q calls.
3. Rollout cache for deterministic coast validation and DP wake scoring.

Each cache key must include:

```text
dynamics_id
cost_id
constraint_id
critic_snapshot_id
state_feature_quantization_id
coast_mode
horizon_set_id
```

Without these ids, stale cache contamination is likely.

## 10. Development stages

### Stage 0: Repository skeleton

Deliverables:

1. Package layout.
2. Hydra configs.
3. Pydantic messages.
4. Unit tests for message serialization.
5. One command that runs a dummy episode.

Exit criterion:

A dummy simulation, dummy controller, and logger produce a valid run folder.

### Stage 1: Verified pendulum simulation

Deliverables:

1. Nonlinear pendulum dynamics.
2. RK4 integrator.
3. Energy function.
4. State feature function.
5. Passive rollout tests.
6. Step level logging.

Exit criterion:

Energy is approximately conserved when damping is zero and decreases when damping is positive.

### Stage 2: Baseline controllers

Deliverables:

1. Energy shaping controller.
2. Fixed horizon MPC with L2 input penalty.
3. Fixed horizon MPC with L1 input penalty.
4. Fixed horizon MPC with explicit input blocking.

Exit criterion:

The explicit input blocking baseline runs on the same benchmark and logging schema as the proposed controller.

### Stage 3: Wake sleep MPC with hand Q

Deliverables:

1. Candidate horizon enumeration.
2. Burst only optimization.
3. Hand designed terminal Q.
4. Action scheduler.
5. Deterministic coast validation.
6. Solver artifact logging.

Exit criterion:

The controller reaches the upright target from selected deterministic initial states and produces compact burst segments.

### Stage 4: Replay builder

Deliverables:

1. Decision epoch replay extraction.
2. Coast cost computation.
3. TD target construction.
4. Dataset validation plots.

Exit criterion:

Every replay item can be traced back to a concrete episode segment.

### Stage 5: Learned positive definite Q critic

Deliverables:

1. PyTorch critic model.
2. Target critic.
3. Normalizer.
4. Training loop.
5. Checkpoint saving.
6. Q contour plots.

Exit criterion:

The learned critic is nonnegative by construction and improves at least one KPI relative to hand Q.

### Stage 6: DP Q series and wake selector

Deliverables:

1. Candidate coast rollout scoring.
2. Q series storage.
3. Best wake state recording.
4. Local minimum rule.

Exit criterion:

Chosen wake horizons vary with state and are explainable from stored score curves.

### Stage 7: Noise and early wake

Deliverables:

1. Process noise.
2. Model mismatch mode.
3. State deviation trigger.
4. Constraint margin trigger.
5. Q degradation trigger.
6. Early wake statistics.

Exit criterion:

Early wake reduces failures compared with fixed sleep execution under the same disturbance sweep.

## 11. Repository layout

```text
wake_sleep_mpc/
  pyproject.toml
  README.md
  configs/
    experiment/
      pendulum_hand_q.yaml
      pendulum_learned_q.yaml
    dynamics/
      pendulum.yaml
    mpc/
      casadi_ipopt.yaml
    critic/
      positive_definite_q.yaml
    logging/
      local.yaml
  src/
    wsmpc/
      interfaces/
        messages.py
        bus.py
      dynamics/
        pendulum.py
        rk4.py
        features.py
      simulation/
        sim_node.py
        gym_env.py
      mpc/
        mpc_node.py
        casadi_problem.py
        horizon_enum.py
        warm_start.py
        coast_validation.py
      rl/
        rl_node.py
        qcritic.py
        replay.py
        train_qcritic.py
        targets.py
        normalizer.py
      scheduler/
        action_scheduler.py
        dp_q_series.py
      safety/
        early_wake_monitor.py
      logging/
        run_logger.py
        artifact_writer.py
        schemas.py
      monitoring/
        dashboard.py
        tensorboard_writer.py
      eval/
        baselines.py
        metrics.py
        sweeps.py
      apps/
        run_episode.py
        train_critic.py
        run_sweep.py
        inspect_run.py
  tests/
    test_pendulum_energy.py
    test_message_roundtrip.py
    test_action_scheduler.py
    test_replay_targets.py
    test_qcritic_positive.py
```

## 12. Best first implementation choice for each node

### Simulation

Use custom Python, not Gymnasium as the core object.

Reason:

The simulator must publish rich diagnostics, handle real time pacing, and store full experiment records. Wrap it into a Gymnasium environment only for baseline RL comparisons.

### MPC

Use CasADi plus IPOPT first.

Reason:

The first optimization problem is nonlinear, small, and needs flexible objective terms with a learned terminal Q. CasADi is the lowest friction way to build and debug that.

Use do mpc for comparison baselines if it saves time.

Use acados only after the CasADi formulation is correct and you need solver speed.

### RL

Use PyTorch direct implementation.

Reason:

The critic is not a standard off policy actor critic module. It must expose differentiable terminal Q values and preserve the positive definite construction.

Use Stable Baselines3 only to compare against standard RL baselines.

Use CleanRL style scripts if you want minimal readable training loops.

### Communication

Use in process message passing first.

Then:

1. ROS 2 if you want robotics style nodes and future hardware integration.
2. ZeroMQ if you want lightweight process separation.

### Logging

Use Zarr or HDF5 for dense arrays.

Use Parquet for metrics and decision records.

Use TensorBoard for training and scalar monitoring.

Use Plotly or Streamlit for visual inspection.

## 13. Critical engineering policies

### 13.1 Solver failure policy

Define this before running experiments.

Recommended order:

1. Try previous warm start.
2. Try sign pattern warm starts.
3. Reduce candidate set.
4. Apply safe fallback controller for one short interval.
5. Mark the episode as solver failed if fallback persists.

Never silently reuse an old plan without logging it.

### 13.2 Critic version policy

Every MPC solve must record which critic snapshot produced the terminal Q values.

Required fields:

```text
critic_snapshot_id
critic_train_step
normalizer_id
replay_dataset_id
alpha_Q
```

Without this, learned Q results are not auditable.

### 13.3 Random seed policy

Every run must store:

```text
global_seed
sim_seed
noise_seed
optimizer_seed
torch_seed
initial_condition_seed
```

### 13.4 Time policy

Keep three times separate:

1. Simulator step index.
2. Simulated physical time.
3. Wall clock time.

Do not use wall clock time for algorithm decisions in the deterministic benchmark.

### 13.5 Constraint policy

Log hard constraint residuals at every simulator step and every predicted burst step.

The learned critic can guide terminal cost. It must not be treated as a hard safety certificate.

## 14. Questions you did not explicitly ask but should answer now

### 14.1 What is the minimum viable experiment?

One deterministic underpowered pendulum swing up from fixed initial states.

The first success should not include noise, piece wise linear models, DP wake selection, or learned Q. The first success is wake sleep MPC with explicit burst coast structure and hand Q.

### 14.2 What should be hot swapped?

Hot swap these interfaces:

1. Dynamics model.
2. Controller.
3. MPC solver backend.
4. Terminal Q provider.
5. Coast rule.
6. Early wake policy.
7. Logger backend.
8. Baseline controller.

Do not hot swap everything. Excess abstraction will slow development.

### 14.3 What is the exact action during coast?

Start with three selectable modes:

1. Hold last burst input.
2. Zero input.
3. Fixed configured coast input.

Record the selected mode in every run. The pendulum benchmark can start with hold last input. Spacecraft benchmarks should use zero thrust coast.

### 14.4 Should the RL train online during an episode?

No for the first benchmark.

Train after each completed episode. Freeze the critic during each episode. This prevents the controller from changing its terminal cost while one trajectory is being evaluated.

### 14.5 Should the MPC call the neural network inside the nonlinear optimizer?

First version: yes, but keep it simple.

Practical options:

1. Use a hand Q first.
2. Use a frozen Torch critic evaluated outside CasADi for candidate terminal states if needed.
3. Later export a CasADi compatible surrogate or use a differentiable wrapper.

Best first engineering compromise:

For each candidate burst solution, let CasADi optimize with hand Q or a simple differentiable critic approximation. Once learned Q is stable, test whether direct neural Q inside the optimizer is reliable. If not, use a table or interpolation cache.

### 14.6 How should horizon enumeration be controlled?

Use small discrete sets first:

```text
N_b = [2, 4, 6, 8]
N_w = [10, 20, 30, 40, 60]
```

The MPC node should solve all valid pairs where `N_b <= N_w`, then choose the minimum objective.

### 14.7 What are the first KPIs?

Use these for every run:

```text
success
time_to_goal
episode_cost
active_input_ratio
burst_saturation_ratio
longest_coast_ratio
number_of_mpc_solves
mean_solve_time_ms
max_solve_time_ms
constraint_violation_count
early_wake_count
solver_failure_count
```

### 14.8 What is the first ablation table?

Run the same initial condition seeds for:

1. Energy shaping.
2. Fixed horizon MPC with L2 input cost.
3. Fixed horizon MPC with L1 input cost.
4. Fixed horizon MPC with explicit input blocking.
5. Wake sleep MPC with hand Q.
6. Wake sleep MPC with learned Q.
7. Wake sleep MPC with learned Q and early wake.

The important comparison is against explicit input blocking MPC, not only against L1 MPC.

### 14.9 What should be cached?

Cache only after correctness is established.

First cache:

1. MPC warm starts.
2. Deterministic coast rollouts.
3. Critic Q values for repeated horizon queries.

Avoid premature caching before the logs prove the same query is being repeated often.

### 14.10 What should be monitored live?

At minimum:

1. State trajectory.
2. Energy error.
3. Torque.
4. Mode: burst or coast.
5. Wake horizon.
6. Current plan id.
7. Objective terms.
8. Solver status.
9. Solve time.
10. Constraint margin.
11. Q terminal value.
12. Early wake trigger.

## 15. Minimal commands the next coding agent should implement

```text
wsmpc run-episode experiment=pendulum_hand_q seed=1

wsmpc run-sweep experiment=pendulum_hand_q seeds=1,2,3,4,5

wsmpc train-critic run_group=pendulum_hand_q_v001

wsmpc run-episode experiment=pendulum_learned_q critic=latest seed=1

wsmpc inspect-run run_id=2026_05_03_001_pendulum_wsmpc

wsmpc compare run_group=ablation_v001 metric=success,time_to_goal,number_of_mpc_solves
```

## 16. Final handoff summary

Build the system as a Python research monorepo with strict typed interfaces. Keep simulation, MPC, and RL as replaceable nodes, but use an in process coordinator first. Add ROS 2 or ZeroMQ only after the interfaces and logs are stable.

The first engineering target is not the full theory. The first target is a deterministic pendulum benchmark where the simulator is verified, the explicit burst coast MPC works with a hand Q terminal cost, all artifacts are logged, and the result beats an explicit input blocking MPC baseline on at least one meaningful metric without violating constraints.

Only after that should learned Q, DP wake selection, noise, and asynchronous execution be added.
