# JAX Rotary-Pendulum Environment: Implementation Plan

## Scope and repository shape

Keep **one installable project** with task-oriented modules under `src/rotary_pendulum/`, rather than separately installing an environment, agent, and controller. A top-level workflow can later compose those modules. In this phase, add only a JAX dynamics file, a JAX episode file, focused tests, and an optional JAX environment dependency; leave the NumPy/CasADi controller and its 2 ms benchmark path unchanged. This makes parity testing possible without a premature rewrite.

Follow the repository's strict code discipline: compact continuous logic; professional one-line comments at key blocks; no progress/version labels in code; no thin wrappers, defensive fallbacks, or `try`/`with`/`except`; no unnecessary CLI or runtime settings. Aim for 50–250 lines per source file and 20–100 lines per function. Do not exceed the structure budget below without first revising this plan with a concrete reason. Freeze physical and clock values as benchmark definitions; expose only genuinely varying experiment choices later.

### Exact proposed structure budget

| File | New object or function | Role | Budget |
|---|---|---|---:|
| `src/rotary_pendulum/environment/jax_dynamics.py` | `state_derivative(x, u)` | Evaluate the nominal four-state ODE with a direct symmetric $2\times2$ inertia solve; accept scalar or leading batched axes | 40–70 lines |
| same | `rk4_step(x, u)` | Advance exactly 20 ms under held shaft torque; remain free of episode logic for MPC/MPPI use | 20–35 lines |
| `src/rotary_pendulum/environment/jax_environment.py` | `EnvState` | One JAX-pytree `NamedTuple` carrying state and episode memory | 6 fields |
| same | `reset(key, stratum)` | Draw one continuous initial state for a specified reset stratum and clear episode memory | 25–45 lines |
| same | `step(env_state, u)` | Hold one bounded action for up to five physics steps, updating goal count and terminal flags after each step | 45–80 lines |
| inside `step` only | `advance_one(carry, _)` | Substantive `lax.scan` body that handles one physics step and early termination; not a standalone wrapper | 20–35 lines |

No new controller, network, reward, rollout class, parameter dataclass, CLI, general-purpose adapter, or fallback path is planned. `EnvState` is the only new class-like object. Existing `RotaryPendulumConfig`, `ModelConstants`, and `GoalConfig` remain the authoritative nominal parameter schemas. The JAX module reads their current YAML values once before tracing, converts them to fixed scalar constants, and does not introduce a second editable physical-parameter set. The new 20 ms clock and 1,000-step horizon are JAX benchmark constants; the old YAML's 2 ms and 5,000 steps remain the old-controller baseline until an intentional migration is justified.

Independent arguments and benchmark quantities are exhaustively listed here. The only per-call arguments are the four-state array $x$, torque $u$, reset random key `key`, reset `stratum`, and episode carry `env_state`. The fixed physical scalars are gravity, arm and pendulum mass and length, two viscous damping coefficients, and maximum shaft torque from `physics.yaml`; the derived inertias and gravity coupling come from the existing `derive_model`. The fixed task scalars are 20 ms per step, five physics steps per action, 1,000 physics steps per episode, ±$\pi/2$ arm angle, four current goal tolerances, and five consecutive goal samples. The only reset-design quantities are the three bound rows in [the formulation](formulation.md), their moving-swing sign, and the externally chosen stratum. No reward weight, training batch size, optimizer parameter, or MPC linearization-grid parameter belongs in this phase. Local algebraic scratch variables, RK4 stages, and scan carry fields are not independent parameters.

### Data and interaction contract

The JAX plant must accept an array with final axis length four and a torque broadcastable over leading axes. Its output has the same state shape. `reset(key, stratum)` is pure: the caller owns key splitting, which avoids hidden global RNG state. The three stratum IDs are 0 downward, 1 moving swing, and 2 near upright. `EnvState` stores physical state, elapsed physics-step count, consecutive goal count, success, arm violation, and timeout. `step` returns a new `EnvState`; after any terminal flag it advances neither time nor physics. Consumers take observations from `EnvState.x` and terminal outcomes from its flags. This is a functional reset/step interface without installing Gymnax or inventing a reward now.

Batch environments with `jax.vmap(reset)` and `jax.vmap(step)`; scan the decision horizon with `jax.lax.scan`. For MPC/MPPI, use `rk4_step` directly inside a scan of candidate torque sequences, without invoking reset, success, or termination logic. Gradients and Jacobians apply to this smooth raw transition, not to the discontinuous episode stop. [JAX's transformation and scan documentation](https://docs.jax.dev/en/latest/quickstart.html) supports this structure; [its random-key documentation](https://docs.jax.dev/en/latest/jax.random.html) motivates explicit keys.

## Milestones and success flags

| Milestone | Work | Validation procedure | Success flag |
|---|---|---|---|
| M0. Baseline | Record nominal parameter and old-step values; preserve old tests | Run existing rotary tests and check clean source diff before implementation | Existing suite passes; reference path unchanged |
| M1. Raw JAX plant | Implement exact nominal ODE and 20 ms RK4 | CPU float64 parity against the old NumPy ODE and same-20 ms RK4 across the three reset strata and torque extremes; compare 100 ms holds against a 2 ms reference | All outputs finite; componentwise parity and time-resolution gates in [validation matrix](../machine-scannables/validation_matrix.md) pass |
| M2. Episode contract | Add continuous reset sampling, five-step action hold, goal count, hard arm termination, timeout | Seed replay, empirical support checks, boundary/goal/timeout tests, and `jit`/`vmap`/`scan` tests | Every contract test passes; 1,000 steps equals 20 seconds; no hidden host-side episode loop |
| M3. Local portability | Add one optional JAX dependency path for CPU development; retain old dependencies | Fresh macOS install and `jax.devices()` smoke; complete M1–M2 tests | JAX CPU recognized and all tests pass without changing old MPC execution |
| M4. Server execution | Install the matching tested JAX release with NVIDIA support; benchmark one L40S before considering all eight | Confirm live GPU/driver discovery; warm compile; block on completed arrays; sweep fixed batch sizes and compare with CPU | One-card 200-decision rollout works at a useful batch size and exceeds CPU throughput there; no precision-gate regression |

The Ubuntu server snapshot supplied by the user is eight NVIDIA L40S GPUs (compute capability 8.9), driver 580.126.09, with roughly 45 GiB per card. That is **not** a live server check. As of this plan, the [official JAX installation page](https://docs.jax.dev/en/latest/installation.html) recommends pip-provided CUDA 13 wheels on Linux and requires driver 580 or newer for that route; the supplied snapshot appears compatible. It also says macOS GPU is unsupported, so local work uses JAX CPU. Do not mistake `nvidia-smi`'s “CUDA 13.0” for a locally installed toolkit version. Pin the actually tested JAX/JAXLIB/plugin combination after the server smoke test, not from this document alone.

For numerical parity, run a separate CPU process with JAX float64 enabled. For the intended accelerator workload, test default float32 and require its own accuracy gate; JAX defaults to 32-bit arrays according to its [dtype documentation](https://docs.jax.dev/en/latest/101/default_dtypes.html). Time only warm compiled calls and wait for completion with `block_until_ready`, as required by JAX's [benchmarking guidance](https://docs.jax.dev/en/latest/201/profiling.html). If float32 misses the physical gate, investigate numerical scaling and precision before claiming throughput. Multi-GPU sharding is not a milestone for an environment-only phase; it belongs when training or batch-prediction loads justify it.

## Explicit exclusions and next handoff

Do not add Flax, Optax, an RL framework, reward shaping, value networks, hard-coded action discretization, Jacobian lookup tables, or joint MPC training now. The environment must merely expose the nonlinear transition and a traceable interaction contract that make those later options possible. [Future RL work](future_rl.md) records the next decisions without treating them as settled implementation instructions. A later linearization phase can compare local $x^+=Ax+Bu$ batch approximations against the JAX nonlinear reference and keep them only where measured accuracy and speed justify them.
