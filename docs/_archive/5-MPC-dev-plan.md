# Wake Sleep MPC Handoff Report

## 0. Purpose of this handoff

This document gives the current implementation plan for the wake sleep model predictive control benchmark. It should be treated as the practical engineering instruction for the next development stage.

The current target is an MPC only deterministic benchmark for an underpowered inverted pendulum. Reinforcement learning and learned Q terminal values are deferred. The controller should still preserve the wake sleep structure so that a learned terminal value can be added later without changing the simulator or controller interface.

The controller concept is:

1. At a decision epoch, solve a short active burst control sequence.
2. After the burst, apply a simple coast command.
3. During coast, do not run MPC unless an early wake trigger fires.
4. The controller chooses the split between burst and coast using a burst to whole horizon ratio.
5. The total prediction horizon is fixed for the current stage.

## 1. Current repository fit

The repository currently contains planning and specification documents, not a complete implemented source tree. The relevant documents are:

1. `docs/Plan.md`
2. `docs/wake_sleep_mpc_dev_spec.md`

The existing specification uses two horizon variables:

1. Burst horizon `N_b`
2. Wake horizon `N_w`

It also includes a future learned Q critic terminal term. For the current development stage, modify this into a simpler MPC only formulation:

1. Use one fixed total horizon `N`.
2. Use a burst to whole split ratio `lambda`.
3. Compute `N_b` and `N_c` from `N` and `lambda`.
4. Explicitly roll out both burst and coast inside the MPC candidate evaluation.
5. Do not use the learned Q critic yet.

This preserves the original wake sleep architecture while making the first benchmark easier to implement and debug.

## 2. Development objective

Build a deterministic underpowered pendulum swing up benchmark with explicit burst and coast behavior.

The benchmark should demonstrate:

1. Energy pumping under torque saturation.
2. Phase aware timing.
3. A compact burst followed by coast.
4. MPC horizon split selection without reinforcement learning.
5. Early wake when prediction error or constraint risk becomes too large.

The first implementation should not include model mismatch, process noise, learned Q values, neural networks, or piece wise linear prediction. These should be added after deterministic nonlinear MPC works.

## 3. State, dynamics, and convention

Use the physical state

```text
x = [theta, omega]^T
```

where:

1. `theta` is the pendulum angle in radians.
2. `omega` is the angular velocity in radians per second.

Use the convention:

```text
theta = 0      downward equilibrium
theta = pi     upright target
```

The upright target state is

```text
x_star = [pi, 0]^T
```

Use the nonlinear continuous dynamics

```text
dot(theta) = omega
I dot(omega) = - m g l sin(theta) - b omega + u
```

where:

1. `I` is the moment of inertia about the pivot.
2. `m` is the pendulum mass.
3. `g` is gravitational acceleration.
4. `l` is the distance from pivot to center of mass.
5. `b` is viscous damping.
6. `u` is the scalar torque input.

Important implementation note:

The current repo spec writes the angular acceleration equation with a positive gravity sign. With the energy definition `E = 0.5 I omega^2 + m g l (1 - cos(theta))` and the target `theta = pi`, the physically consistent sign is negative:

```text
I dot(omega) = - m g l sin(theta) - b omega + u
```

This should be corrected before simulation results are trusted.

Use RK4 for both simulator integration and MPC prediction in the first working version:

```text
x_next = f_d(x, u, dt) = RK4(x, u, dt)
```

where `dt` is the simulator step size.

## 4. Input limits and underpowered condition

The scalar input set is

```text
U = { u : |u| <= u_max }
```

where `u_max` is the maximum available torque.

The underpowered benchmark condition is

```text
u_max < m g l
```

This means the actuator cannot directly hold the pendulum statically at arbitrary angles against gravity. Successful swing up should require timing, energy pumping, and phase exploitation.

## 5. Energy and phase features

The raw dynamics state remains `theta, omega`. Energy and phase are feature maps used for cost, diagnostics, and later CLBF construction.

Define kinetic energy:

```text
K(x) = 0.5 I omega^2
```

Define potential energy, with zero potential at the downward equilibrium:

```text
V(x) = m g l (1 - cos(theta))
```

Define total mechanical energy:

```text
E(x) = K(x) + V(x)
```

The upright target energy is:

```text
E_star = 2 m g l
```

Define normalized energy error:

```text
e_E(x) = (E(x) - E_star) / E_star
```

Use normalized energy error instead of raw energy error because it improves solver scaling.

### 5.1 Phase proxy

Do not use true action angle coordinates in the first implementation. Use a phase proxy.

Choose a positive velocity scale `omega_s`. A practical choice is:

```text
omega_s = sqrt(m g l / I)
```

Define:

```text
a(theta) = sin(theta / 2)
b(omega) = omega / omega_s
r(x) = sqrt(a(theta)^2 + b(omega)^2 + eps_phi^2)
```

where `eps_phi` is a small positive regularization constant.

Define phase proxy components:

```text
c_phi(x) = a(theta) / r(x)
s_phi(x) = b(omega) / r(x)
```

At the upright target, the desired phase proxy is approximately:

```text
c_phi = 1
s_phi = 0
```

Define the phase proxy error:

```text
e_phi(x) = [c_phi(x) - 1, s_phi(x)]^T
```

This phase proxy is not a state coordinate. It should not be used to propagate the dynamics. It is only a smooth feature used to tell the cost whether the pendulum is approaching the upright energy level with the correct timing and velocity direction.

## 6. Energy gate

Phase should not dominate when the pendulum has very wrong energy. Use an energy gate:

```text
w_E(x) = exp( - e_E(x)^2 / sigma_E^2 )
```

where `sigma_E` is a positive parameter controlling how close the system must be to the upright energy before phase and local stabilization become strongly weighted.

Interpretation:

1. If `|e_E|` is large, then `w_E` is small and the cost mainly asks the controller to pump energy.
2. If `|e_E|` is small, then `w_E` is near one and phase plus local stabilization become important.

## 7. Local upright error

Define wrapped upright angle error:

```text
e_theta(x) = atan2(sin(theta - pi), cos(theta - pi))
```

Define local upright error vector:

```text
e_loc(x) = [e_theta(x), omega / omega_s]^T
```

For smoother SQP behavior, the cost may also use trigonometric penalties such as:

```text
1 + cos(theta)
sin(theta)
```

instead of relying entirely on angle wrapping.

## 8. CLBF compatible energy phase value candidate

Define the energy phase value candidate:

```text
V_ep(x) =
    q_E e_E(x)^2
  + q_phi w_E(x) e_phi(x)^T Q_phi e_phi(x)
  + q_loc w_E(x) e_loc(x)^T Q_loc e_loc(x)
```

where:

1. `q_E > 0` is the scalar energy error weight.
2. `q_phi >= 0` is the scalar phase weight.
3. `q_loc >= 0` is the scalar local upright stabilization weight.
4. `Q_phi` is a positive definite matrix for phase proxy error.
5. `Q_loc` is a positive definite matrix for local upright error.
6. `w_E(x)` is the energy gate.

This is the primary state cost and later Lyapunov candidate.

Implementation guidance:

1. Start with `q_phi = 0` and `q_loc = 0`, debug energy pumping.
2. Add `q_loc`, debug upright stabilization.
3. Add `q_phi`, debug phase timing.
4. Do not tune all weights at once.

## 9. Burst and coast stage costs

Define normalized input:

```text
s = u / u_max
```

For active burst steps, use:

```text
ell_b(x, u, u_prev) =
    V_ep(x)
  + rho_sat (1 - (u / u_max)^2)^2
  + rho_du ((u - u_prev) / u_max)^2
```

where:

1. `rho_sat >= 0` weights the preference for near saturated burst control.
2. `rho_du >= 0` weights input smoothness.
3. `u_prev` is the previously applied input.

The saturation attraction term is smooth and suitable for SQP. It encourages `u` near `+u_max` or `-u_max` without introducing binary variables.

Do not start with the hard lower bound:

```text
|u| >= eta_u u_max
```

because it is nonconvex and creates disconnected feasible regions.

For coast steps, use:

```text
ell_c(x, u_h) =
    V_ep(x)
  + rho_c (u_h / u_max)^2
```

where:

1. `u_h` is the coast command.
2. `rho_c >= 0` is the coast input penalty.

For the first version, implement two coast modes behind a configuration flag:

1. `hold_last`: `u_h = v_{N_b - 1}`
2. `zero`: `u_h = 0`

Use `hold_last` first if the current research intent is to test constant actuation during coast. Use `zero` as an ablation.

## 10. Total horizon and split ratio

Use one total prediction horizon:

```text
N
```

where `N` is the number of discrete simulation steps in the prediction window.

Use a split ratio:

```text
lambda in [0, 1]
```

where `lambda` is the burst to whole horizon ratio.

For a fixed candidate `lambda`, define:

```text
N_b = max(1, round(lambda N))
N_c = N - N_b
```

where:

1. `N_b` is the burst horizon in discrete steps.
2. `N_c` is the coast horizon in discrete steps.
3. `N = N_b + N_c`.

Important solver instruction:

Do not put the rounding operation inside SQP. Enumerate candidate ratios outside SQP.

Use a finite split candidate set:

```text
Lambda = {0.10, 0.15, 0.20, 0.25, 0.33, 0.50}
```

For the first version, a smaller set is acceptable:

```text
Lambda = {0.10, 0.20, 0.30, 0.40}
```

For each `lambda`, the optimizer sees a fixed smooth problem.

## 11. MPC problem for one split candidate

At decision epoch `k`, measured state is:

```text
x_tk
```

For one candidate split ratio `lambda`, solve:

```text
J_star(x_tk, lambda) =
min over v_0, ..., v_{N_b - 1}
[
    sum_{i = 0}^{N_b - 1} ell_b(x_i, v_i, v_{i - 1})
  + sum_{j = 0}^{N_c - 1} ell_c(x_{N_b + j}, u_h)
  + ell_f(x_N)
]
```

where:

1. `v_i` is the optimized burst input at local index `i`.
2. `x_i` is the predicted state at local index `i`.
3. `u_h` is the coast command.
4. `ell_f` is the terminal cost.

Use terminal cost:

```text
ell_f(x_N) = q_f V_ep(x_N)
```

where `q_f > 0` is the terminal cost weight.

Subject to:

```text
x_0 = x_tk
```

Burst dynamics:

```text
x_{i + 1} = f_d(x_i, v_i, dt)
for i = 0, ..., N_b - 1
```

Coast command:

```text
u_h = v_{N_b - 1}
```

or, for zero coast:

```text
u_h = 0
```

Coast dynamics:

```text
x_{N_b + j + 1} = f_d(x_{N_b + j}, u_h, dt)
for j = 0, ..., N_c - 1
```

Input constraints:

```text
|v_i| <= u_max
for i = 0, ..., N_b - 1
```

State constraints:

```text
x_i in X
for i = 0, ..., N
```

Then select the split ratio:

```text
lambda_star = argmin over lambda in Lambda of J_star(x_tk, lambda)
```

Apply:

```text
u_{t_k + i} = v_i_star
for i = 0, ..., N_b_star - 1
```

Then coast:

```text
u_{t_k + i} = u_h_star
for i = N_b_star, ..., N - 1
```

unless early wake is triggered.

## 12. SQP solvability guidance

The first implementation should use single shooting:

1. The decision variables are only the burst inputs `v_i`.
2. The predicted states are generated by forward simulation.
3. This keeps the NLP dimension low for the two state pendulum.

Use multiple shooting only if single shooting becomes unstable or if path constraints become too tight.

Keep the optimization SQP friendly:

1. Enumerate `lambda` outside the solver.
2. Keep `N_b` and `N_c` fixed within each subproblem.
3. Use smooth costs.
4. Avoid absolute values in objectives.
5. Avoid `max`, `min`, `if`, and `round` inside the continuous optimizer.
6. Use normalized variables.
7. Warm start from the previous solution.
8. Use continuation when enabling new cost terms.

Suggested continuation sequence:

1. Set `rho_sat = 0`, `q_phi = 0`, `q_loc = 0`. Debug energy cost only.
2. Enable `q_loc`.
3. Enable `rho_sat`.
4. Enable `q_phi`.
5. Add early wake.
6. Add constraints beyond input bounds.
7. Add noise and model mismatch only after deterministic success.

## 13. Natural frequency and total prediction horizon

Use the passive small angle natural frequency around the downward equilibrium to initialize the total prediction horizon.

The undamped natural frequency is:

```text
omega_n = sqrt(m g l / I)
```

where `omega_n` is in radians per second.

The small angle natural period is:

```text
T_0 = 2 pi / omega_n
```

If the pendulum is a point mass at distance `l`, then:

```text
I = m l^2
omega_n = sqrt(g / l)
T_0 = 2 pi sqrt(l / g)
```

Use the total prediction time:

```text
T_N = alpha_T T_0
```

where `alpha_T` is a horizon multiplier.

Recommended first values:

```text
alpha_T in {0.25, 0.50, 0.75, 1.00}
```

Start with:

```text
alpha_T = 0.50
```

Then:

```text
N = round(T_N / dt)
```

Example for `l = 1 m` and `dt = 0.02 s`:

```text
omega_n = sqrt(9.81) = 3.13 rad/s
T_0 = 2.01 s
T_N = 1.00 s
N = 50
```

The natural frequency is only a starting scale. Large amplitude pendulum motion does not keep the small angle natural period. Near the upright separatrix, the passive period becomes very large. Control inputs also change the realized trajectory timing, especially when feedback changes effective stiffness or damping. Therefore, the natural period should initialize the horizon, not determine it permanently.

## 14. Early wake trigger

During coast, compare the observed state against the predicted state.

Let:

```text
x_hat_{t_k + i}
```

be the observed state during execution.

Let:

```text
x_exec_star_{i | k}
```

be the predicted execution state from the selected burst plus coast rollout.

Define prediction deviation:

```text
d_i = (x_hat_{t_k + i} - x_exec_star_{i | k})^T S_x (x_hat_{t_k + i} - x_exec_star_{i | k})
```

where `S_x` is a positive definite deviation weighting matrix.

Trigger early wake if:

```text
d_i > delta_x
```

where `delta_x` is the allowed prediction deviation.

Also trigger early wake if any state constraint margin is close to violation:

```text
h_x(x_hat_{t_k + i}) > - delta_h
```

where:

1. `h_x(x) <= 0` defines state constraints.
2. `delta_h > 0` is the safety margin.

For the MPC only stage, do not use Q degradation as a trigger because there is no learned Q critic yet.

## 15. Later CLBF extension

The current value candidate is:

```text
V_ep(x)
```

Later, turn it into a control Lyapunov condition by adding:

```text
V_ep(f_d(x, u)) - V_ep(x) <= - alpha_V V_ep(x) + s_V
```

where:

1. `alpha_V` is a positive decrease rate.
2. `s_V >= 0` is a relaxation slack.

For safety, define a barrier function:

```text
h(x) >= 0
```

where `h(x)` is positive inside the safe set.

Use a discrete barrier condition:

```text
h(f_d(x, u)) >= (1 - alpha_h) h(x) - s_h
```

where:

1. `alpha_h` is the barrier rate.
2. `s_h >= 0` is a relaxation slack.

Do not add CLBF constraints in the first working version. The current implementation should only make the cost and feature design compatible with this future step.

## 16. Suggested code structure

The repository currently has planning documents. Add a source structure like this:

```text
wake_sleep_mpc/
    config/
        pendulum_default.yaml
    src/
        wake_sleep_mpc/
            __init__.py
            dynamics/
                pendulum.py
                integrators.py
            costs/
                energy_phase.py
            controllers/
                split_ratio_mpc.py
            simulation/
                rollout.py
                experiment_runner.py
            logging/
                records.py
                metrics.py
    tests/
        test_energy_consistency.py
        test_rk4_pendulum.py
        test_cost_zero_near_goal.py
        test_split_ratio_dimensions.py
        test_mpc_single_step.py
    docs/
        Plan.md
        wake_sleep_mpc_dev_spec.md
        handoff_split_ratio_mpc.md
```

Minimum module responsibilities:

1. `pendulum.py`  
   Implement continuous dynamics, input constraints, energy, and natural frequency.

2. `integrators.py`  
   Implement RK4.

3. `energy_phase.py`  
   Implement `V_ep`, `ell_b`, `ell_c`, and `ell_f`.

4. `split_ratio_mpc.py`  
   Implement split ratio enumeration and SQP calls.

5. `rollout.py`  
   Execute selected burst and coast sequence.

6. `experiment_runner.py`  
   Run deterministic episodes.

7. `records.py`  
   Save state, input, energy, selected split ratio, solver status, and early wake events.

8. `metrics.py`  
   Compute success, cost, constraint violation, control sparsity, and computation time.

## 17. Minimum tests before tuning

Implement these tests before trying to tune the controller.

### Test 1: Energy consistency

With `u = 0` and `b = 0`, energy should remain nearly constant over a short simulation.

Expected:

```text
max |E(t) - E(0)| < tolerance
```

If this fails, check the gravity sign.

### Test 2: Input bounds

The MPC result must satisfy:

```text
|u_i| <= u_max
```

for all burst inputs.

### Test 3: Cost near goal

At:

```text
theta = pi
omega = 0
```

the value `V_ep` should be near zero.

### Test 4: Cost at downward rest

At:

```text
theta = 0
omega = 0
```

the value `V_ep` should be positive.

### Test 5: Split ratio dimensions

For each `lambda`, verify:

```text
N_b >= 1
N_c >= 0
N_b + N_c = N
```

### Test 6: Deterministic replay

Given the same initial state and config, the simulator and MPC should produce reproducible trajectories.

## 18. Initial configuration recommendation

Start with this configuration:

```text
dt: 0.02
g: 9.81
m: 1.0
l: 1.0
I: 1.0
b: 0.05
u_max: 0.7 * m * g * l

alpha_T: 0.5
split_ratios: [0.10, 0.20, 0.30, 0.40]

sigma_E: 0.5
eps_phi: 1.0e-6

q_E: 1.0
q_phi: 0.0
q_loc: 0.0
q_f: 5.0

rho_sat: 0.0
rho_du: 1.0e-3
rho_c: 1.0e-3

coast_mode: hold_last
solver: SQP
shooting: single
```

After energy pumping works, change:

```text
q_loc: 1.0
rho_sat: 0.05
q_phi: 0.1
```

Then retune.

## 19. Implementation milestones

### Milestone 1: Dynamics and energy

1. Implement pendulum dynamics with corrected sign convention.
2. Implement RK4.
3. Implement energy functions.
4. Pass energy consistency tests.

### Milestone 2: Cost function

1. Implement `V_ep`.
2. Implement burst, coast, and terminal costs.
3. Pass goal and downward rest cost tests.

### Milestone 3: Fixed split MPC

1. Set one fixed split ratio.
2. Solve the burst sequence with SQP.
3. Roll out the coast segment.
4. Verify input constraints.

### Milestone 4: Split ratio enumeration

1. Add `Lambda`.
2. Solve one subproblem per split candidate.
3. Select the minimum cost candidate.
4. Log selected split ratio over time.

### Milestone 5: Closed loop wake sleep execution

1. Apply burst commands.
2. Apply coast command.
3. Re solve only at the next planned wake time.
4. Add early wake trigger.

### Milestone 6: Benchmark metrics

Record:

1. Final success or failure.
2. Time to enter goal set.
3. Time held inside goal set.
4. Total cost.
5. Fraction of active control steps.
6. Fraction of saturated control steps.
7. Number of MPC calls.
8. Mean and maximum solve time.
9. Constraint violation count.
10. Early wake count.

## 20. Deferred features

Do not implement these until the deterministic MPC only version works:

1. Learned Q critic.
2. Neural network terminal value.
3. Piece wise linear prediction.
4. Process noise.
5. Model mismatch.
6. Robust tube MPC.
7. CLBF constraints.
8. Spacecraft benchmark.
9. Multi node communication.
10. Real time parallel MPC.

## 21. Final implementation principle

The controller should be written so that the terminal cost interface can later be replaced.

Current MPC only terminal score:

```text
terminal_score = ell_f(x_N)
```

Future learned Q terminal score:

```text
terminal_score = alpha_Q Q_theta(x_{N_b}, u_h, N_c)
```

Therefore, implement the terminal scoring as an interchangeable component:

```text
TerminalScorer.evaluate(...)
```

This keeps the current deterministic benchmark compatible with the future wake sleep RL extension.
