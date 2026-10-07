# Phase 10 Training Plan: Five-Action Q Prior

## Human quick reading

Train a fresh five-output Double DQN under `rotary-q-prior-v2`. Start with a short guided pilot, then compare ordinary and guided exploration across common seeds on all eight GPUs. Inspect held-out capture, violations, action use, value calibration, and trajectories after every wave. Promote only a setting that improves validation behavior across seeds.

The first objective is local capture. Global swing-up expansion begins only after the tight and near-upright gates pass twice. Version-1 weights and replay are structurally incompatible and must never be loaded.

The agent may tune the approved exploration, replay, optimizer, reward, and network choices without waiting between waves. Stop with either a validated prior or a documented plateau supported by complete artifacts.

## Machine-scannable contract

### T10.0 Fixed task

| Key | Value |
|---|---|
| `contract_id` | `rotary-q-prior-v2` |
| `network_id_baseline` | `mlp-7-128-128-5-tanh-v2` |
| `reward_revision` | `2` |
| `action_order` | `[off, negative_pump, positive_pump, negative_fine, positive_fine]` |
| `action_fraction_of_limit` | `[0,-0.45,+0.45,-0.02,+0.02]` |
| `decision_period_s` | `0.10` |
| `deadline_s` | `20.0` |
| `goal_theta_beta_omega_nu` | `[0.08,0.08,0.15,0.20]` |
| `goal_hold_physics_steps` | `5` |
| `validation_seed` | `2026100700` |
| `final_seed` | `2026100800`; inaccessible for tuning |
| `artifact_root` | `artifacts/rotary_pendulum/q-prior-v2` |

`tight` means 512 fixed validation starts sampled independently and uniformly over
`theta ±0.08 rad`, wrapped upright error `beta ±0.12 rad`, arm speed `omega ±0.15 rad/s`,
and pendulum speed `nu ±0.30 rad/s`. `near` means 1,024 fixed starts over
`theta ±0.25 rad`, `beta ±0.25 rad`, and both speeds `±1 rad/s`. Neither term means
the state already satisfies the goal. The goal is `±[0.08 rad, 0.08 rad, 0.15 rad/s,
0.20 rad/s]` for five consecutive 20 ms physics steps.

A `wave` means eight independent trials launched concurrently, one per GPU. Under the
baseline stage-0 budget, each trial collects `8,388,608` transitions and validates every
`1,048,576`, giving eight scheduled validations per trial and `67,108,864` collected
transitions per complete wave.

Changing the action values, clocks, goal, terminal events, observation, or reward formula requires another formulation revision. Coefficient changes increment `reward_revision`; architecture changes increment `network_id`.

### T10.1 Baseline

The pilot and guided baseline use `[128,128]` tanh layers, Huber delta `1`, Adam learning rate `3e-4`, gradient limit `10`, minibatch `4096`, replay `1,048,576`, sample reuse `2`, hard target copies every `256` updates, all-tight stage-0 resets, heuristic fraction `1`, and epsilon `1.0` to `0.1`. Every trial starts from new parameters, optimizer, and replay.

The full first wave compares heuristic fractions `0` and `1` with four common seeds each. All other settings remain equal. The baseline stage-0 budget is `8,388,608` transitions with evaluation every `1,048,576` transitions.

### T10.2 Authorized iterations

After inspecting a completed wave, change one parameter family at a time:

| Family | Allowed values |
|---|---|
| Exploration | heuristic fraction `0`, `0.5`, or `1`; epsilon end `0.02`, `0.05`, or `0.1`; decay `2^21`, `2^22`, or `2^23` |
| Replay/update | reuse `1` or `2`; target copy `128`, `256`, or `1024`; replay `2^19`, `2^20`, or `2^21` |
| Optimization | learning rate `1e-4`, `3e-4`, or `1e-3`; Huber delta `0.5`, `1`, or `2`; gradient limit `1`, `5`, or `10` |
| Network | `[64,64]`, `[128,128]`, `[256,256]`, or `[128,128,64]`; tanh, SiLU, or ReLU |
| Reward coefficients | capture weight, powered-time rate, or weak-time rate at half, base, or double |
| Curriculum | tight fraction `0.5`, `0.75`, or `1`; stage budget base or double |

Do not add a new algorithm, replay scheme, observation, auxiliary loss, or compatibility adapter during this campaign.

### T10.3 Milestones and gates

| Milestone | Validation | Success flag |
|---|---|---|
| P10 pilot | One guided GPU trial | Finite training; five actions represented; complete artifacts; useful throughput |
| W10.1 exploration | Four ordinary and four guided seeds | Identify a family-level local-capture improvement without increased violations |
| M2 local capture | Fixed tight and near sets | Tight success `>=90%`, near success `>=80%`, tight violations `0`, near violations `<=1%`, twice consecutively |
| M3 expansion | Downward, moving, near, and tight sets | Retain M2; downward and moving success each `>=30%`, violations each `<=10%`, twice consecutively |
| M4 prior | Untouched final set after eligibility | At least three of four confirmation seeds: overall `>=80%`, every benchmark stratum `>=70%`, violations each `<=5%`, with M2 retained |

### T10.4 Ranking and stopping

Rank by deepest passed milestone, minimum required-stratum success, mean success, violation rate, Q action regret, and success-conditional powered time. Never select from TD loss or one seed.

Stop when a family passes confirmation, after at most six eight-GPU waves, or after two evidence-targeted waves improve the failing success metric by less than two percentage points and its associated calibration/on-time metric by less than ten percent. Preserve failed trials and state the smallest supported next change.

### T10.5 Artifact requirement

Each trial saves resolved configuration, metadata, tight/near success and violation history, evaluation tables, value audit, latest checkpoint, best held-out diagnostic checkpoint, and readable plots. The required validation plots are `learning.png`, with tight/near rates at every scheduled check, and `validation_trajectories.png`, with the first five successes and first five nonsuccesses by validation index from both tight and near sets. The agent must inspect these plots before choosing the next wave and state the observed trajectory failure mode in the wave interpretation. Only an eligible best checkpoint is named `selected`; otherwise it is named `diagnostic_best`. Each wave adds a machine-readable comparison and a short interpretation. The final handoff contains lineage, gates, selected or diagnostic checkpoint status, and exact reproduction paths.

### T10.6 Execution structure budget

The reusable multi-GPU launcher is `scripts/launch_rotary_q_wave.py`. It contains one `main` workflow and no class, dataclass, wrapper, or helper. Its only argument is the required wave-plan JSON path. The plan owns the base experiment mapping, trial-specific overrides, GPU index, run directory, configuration path, and log path. The launcher resolves and writes each trial configuration, starts one independent process per declared GPU, prints a ten-second heartbeat, closes logs, and fails when any child exits nonzero. It contains no training logic or fallback values; `bringup.rotary_q_workflow.train` remains the only training workflow.
