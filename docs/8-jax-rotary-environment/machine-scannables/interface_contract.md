# JAX Rotary Environment: Machine-Scannable Contract

Status: implemented nominal JAX interface, updated for the version-3 dense Q reward. Arm excursions are diagnostic and nonterminal. The existing NumPy/CasADi route remains independently runnable.

## Fixed numeric contract

| Key | Value | Unit or shape | Source | Meaning |
|---|---:|---|---|---|
| `physics_dt_s` | `0.02` | s | new JAX benchmark definition | RK4 integration interval |
| `physics_hz` | `50` | Hz | reciprocal of `physics_dt_s` | sampled physical rate |
| `hold_physics_steps` | `5` | steps/action | agreed controller schedule | zero-order-held torque duration |
| `decision_dt_s` | `0.10` | s/action | product of first and third rows | 10 Hz decision interval |
| `max_physics_steps` | `1000` | steps | user decision | episode cap |
| `max_decisions` | `200` | actions | 1000/5 | complete-action cap without early termination |
| `max_duration_s` | `20` | s | 1000 × 0.02 | episode time cap |
| `goal_hold_steps` | `5` | physics samples | [mission.yaml](../../../src/rotary_pendulum/configs/mission.yaml) | 100 ms sampled goal dwell |
| `arm_limit_rad` | $\pi/2$ | rad | user decision | nonterminal reporting threshold and reward scale |
| `torque_limit_nm` | `0.0204` | N·m | [physics.yaml](../../../src/rotary_pendulum/configs/physics.yaml) | applied shaft-torque saturation |
| `goal_tolerances` | `[0.08,0.08,0.15,0.20]` | `[rad,rad,rad/s,rad/s]` | mission YAML | phase-10 absolute full-state thresholds |
| `state_order` | `[theta,alpha,omega,nu]` | `(4,)` | current plant | unwrapped angles and velocities |

The eight nominal physical primitives—gravity, arm mass and length, pendulum mass and length, two viscous damping coefficients, and torque limit—are read once from `physics.yaml`. The existing `derive_model` supplies derived inertia and gravity coefficients. No random physical parameters and no hidden runtime defaults are allowed. The JAX-only clock and episode cap do not alter the old YAML values, which define the 2 ms reference application.

## Proposed call signatures and tensor shapes

| Call | Inputs | Output | Transform behavior |
|---|---|---|---|
| `state_derivative(x, u)` | `x: (...,4)`, `u: (...)` or scalar | `dx: (...,4)` | pure, JIT and VMAP compatible |
| `rk4_step(x, u)` | same shapes | `x_next: (...,4)` | pure raw nonlinear map; no clipping or terminal logic |
| `reset(key, stratum)` | one JAX random key; scalar integer `0`, `1`, or `2` | `EnvState` with `x: (4,)` | pure; VMAP over distinct keys and strata |
| `step(env_state, u, *, physics_steps=5)` | one `EnvState`; commanded torque; static sample count | new `EnvState` | pure; per-physics-step terminal checks; RL integrates reward using one-step calls |

`EnvState` is one `NamedTuple` pytree with precisely these fields:

| Field | Type/shape | Reset value | Update rule |
|---|---|---|---|
| `x` | floating `(4,)` | sampled state | 20 ms RK4 step under clipped torque while active |
| `physics_steps` | integer scalar | `0` | increment once per actual physics step, never beyond 1000 |
| `goal_count` | integer scalar | `0` | increment if full-state goal is met, otherwise zero |
| `success` | boolean scalar | `false` | true at `goal_count >= 5` |
| `arm_violation` | boolean scalar | `false` | cumulative diagnostic: true after any sampled `abs(theta) >= pi/2`; never terminates |
| `timeout` | boolean scalar | `false` | true at 1000 steps if success has not occurred |

The derived `done` condition is `success | timeout`; it is not an extra stored field. Once done, `step` returns the same state even if called again. `step` may consume fewer than five physics steps if success or timeout occurs during the held action. An arm excursion neither stops nor clamps the state. The raw `rk4_step` intentionally does **not** clip torque: differentiable prediction uses its stated input, and callers requiring feasible actions must bound them. Environment `step` clips torque before integration. If a command is nonfinite, there is no fallback torque; host-side tests reject it and device-side validation must expose the invalid result rather than silently substitute zero. Finite behavior is validated over the benchmark domain; nonterminal excursions do not establish global numerical stability.

## Dynamics and event equations

Let $q=[\theta,\alpha]^T$ be the arm and pendulum angles, $v=[\omega,\nu]^T$ their rates, and $x=[q^T,v^T]^T$. Let $M(\alpha)$ be the $2\times2$ symmetric mass matrix, $h(x)$ the nonlinear velocity-and-gravity vector, $D$ the diagonal viscous-damping matrix, and $b=[1,0]^T$ the input-distribution vector. All four are specified by the [existing formulation](../../6.1-rotary-pendulum-MPC-formulation.md) and the frozen nominal parameter file. The JAX derivative must evaluate

$$
f(x,u)=\begin{bmatrix}v\\M(\alpha)^{-1}\{bu-h(x)-Dv\}\end{bmatrix}.
$$

The $2\times2$ inverse is evaluated by its analytic determinant formula rather than a per-sample general matrix factorization; this is algebraically identical to the current NumPy solve and will be parity-tested. The raw transition is $F_{0.02}(x,u)$ from the RK4 equation in [the physical formulation](../human-readables/formulation.md). The environment transition at one 10 Hz decision is a scan of up to five $F_{0.02}$ calls, checking the following after each call:

$$
\begin{aligned}
\beta&=\operatorname{atan2}(\sin(\alpha-\pi),\cos(\alpha-\pi)),\\
I_g&=(|\theta|\leq0.08)\land(|\beta|\leq0.08)\land(|\omega|\leq0.15)\land(|\nu|\leq0.20),\\
c^+&=\begin{cases}c+1,&I_g,\\0,&\text{otherwise},\end{cases}\\
I_a&=(|\theta|\geq\pi/2),\\
I_s&=(c^+\geq5),\\
I_t&=(n^+\geq1000)\land\neg I_s.
\end{aligned}
$$

$\beta$ is the wrapped upright error, $I_g$ means inside the full-state goal, $c$ is consecutive-goal count, $I_a$ is the arm-excursion diagnostic, $I_s$ is success, $I_t$ is timeout, and $n^+$ is the updated physics-step count. The arm predicate is evaluated on unwrapped $\theta$ and accumulated independently. Success has priority over timeout. This is a sampled simulator, not a continuous-time boundary detector.

## Initial-state sampler specification

`reset` consumes one caller-supplied key. Its returned state contains no key; the caller splits or folds keys for multiple resets. Each stratum uses independent uniform coordinates. The moving-swing angle additionally uses an independent fair sign. `stratum` values are explicit: `0=downward`, `1=moving_swing`, `2=near_upright`. A balanced experiment chooses strata with probabilities `[1/3,1/3,1/3]` outside `reset`, allowing controlled per-stratum evaluation. Numeric ranges are provisional and repeated below to make the sampler reproducible.

| Stratum | `theta` | `alpha` | `omega` | `nu` |
|---|---|---|---|---|
| 0 | `U(-0.20,0.20)` | `U(-0.20,0.20)` | `U(-0.5,0.5)` | `U(-0.5,0.5)` |
| 1 | `U(-0.50,0.50)` | `sign * U(0.40,2.60)`, `P(sign=+1)=1/2` | `U(-2,2)` | `U(-6,6)` |
| 2 | `U(-0.25,0.25)` | `pi + U(-0.25,0.25)` | `U(-1,1)` | `U(-1,1)` |

`U(a,b)` means continuous uniform sampling between fixed bounds $a$ and $b$. `sign` is independent of the four coordinate draws. These are proposed experiment-support ranges, not physical parameter variation and not a claim of uniform coverage of all reachable states.
