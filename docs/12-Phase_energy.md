# Furuta Pendulum: Orbital Phase Guided Energy Burst Controller

**Development handoff | 2026-11-06**  
**Project:** Burst-Coast-MPC, `dev` branch  
**Scope:** Improved physically structured swing up and intermittent actuation controller for the Furuta pendulum. This document is an implementation specification and experiment plan. It does not concern prospective supervisors or external research positioning.

## 0. Executive summary

Develop and validate a simple, dense control baseline that (i) estimates a natural orbital phase from conservative Furuta mechanics, (ii) proposes finite torque pulses centered on nominal phases `π/2` and `3π/2`, (iii) requests a bounded fraction of the remaining mechanical energy error per pulse, (iv) rejects pulses that violate predicted rotary arm angle constraints, and (v) brakes proactively if passive motion threatens the arm limit. The controller should transition to an existing local upright stabilizer in the capture region.

**Critical design boundaries:** Phase centers are hypotheses, not proven maxima of energy injection. The conservative phase map only applies to regular orbital families, not separatrices or equilibria. Total mechanical energy alone does not uniquely determine upright capture. Endpoint checks at one half or one full nominal period do **not** establish trajectory safety; evaluate extrema throughout the prediction interval. Braking is an explicit mode, not an assumed consequence of skipping the prior pulse. No stability or safety guarantees are claimed before verification.

The initial deliverable is a working controller that uses the existing full nonlinear state model for candidate rollouts. Orbital atlas acceleration and learned enhancements are deferred until the heuristic is validated.

## 1. Existing repository context

Source: `https://github.com/VictorWuyueXia/Burst-Coast-MPC/tree/dev` (inspected 2026-11-06). Relevant modules:

- `src/rotary_pendulum/environment/dynamics.py`: `derive_model`, `state_derivative`, `rk4_step`, `energy_components`; reference continuous dynamics and RK4 integration.
- `src/rotary_pendulum/environment/jax_dynamics.py`: JAX model and physical constants; use for batched rollouts where appropriate.
- `src/rotary_pendulum/heuristic/energy.py`: existing physical-body energy encoder and signed work baseline.
- `src/rotary_pendulum/mpc/features.py`: terminal swing energy, phase chasing, local capture, and arm limit costs. Its current “phase chasing” term is a control sensitivity proxy, **not** the orbital phase derived below.
- `src/rotary_pendulum/visualization/phase.py`: existing normalized state plane phase ring diagnostics, **not** the proposed Routh phase.
- `src/rotary_pendulum/mpc/controller.py`, `src/rotary_pendulum/mpc/discrete_model.py`, `src/rotary_pendulum/configs/*.yaml`: control and model integration points.

Keep the existing implementations and artifacts functional. Implement as a separately selectable controller and reuse the existing dynamics and capture controller. Do not replace the MPC or baseline heuristics in place.

### State convention

Use the repository's convention without changes:

\[
x=(\theta,\alpha,\omega,\nu),\qquad\omega=\dot\theta,\quad\nu=\dot\alpha.
\]

- `θ`: rotary **arm** angle, subject to the existing soft limit `θ_max = 0.75π` unless a run config specifies another value.
- `α`: pendulum angle with `α=0` downward and `α=π` upright, modulo `2π`.
- `u`: commanded rotary arm torque.

**Terminology warning:** Earlier discussion sometimes called the constrained angle `α`. This handoff consistently treats the constrained arm angle as `θ`. If an experiment imposes actual *pendulum* angle constraints, those require a separate feasibility analysis because the pendulum is unactuated.

## 2. Exact plant model and energy definitions

Use parameters from `derive_model()` rather than introducing new numerical constants. Define

\[
\begin{aligned}
A(\alpha)&=J_b+J_p\sin^2\alpha,\\
c(\alpha)&=J_c\cos\alpha,\\
\Delta_M(\alpha)&=A(\alpha)J_p-c(\alpha)^2>0,\\
V_p(\alpha)&=G(1-\cos\alpha),\quad G=m_p g l_p.
\end{aligned}
\]

`J_b` is the existing `base_inertia_kg_m2`; `J_p` is `pendulum_inertia_kg_m2`; `J_c` is `coupling_inertia_kg_m2`; `G` is `gravity_torque_nm`. The arm inertia `J_a` is `arm_inertia_kg_m2`.

The Lagrangian, kinetic energy, and Hamiltonian are

\[
\begin{aligned}
L&=\tfrac12A\omega^2+c\omega\nu+\tfrac12 J_p\nu^2-V_p,\\
K&=\tfrac12A\omega^2+c\omega\nu+\tfrac12 J_p\nu^2,\\
H&=K+V_p.
\end{aligned}
\]

The mechanical Hamiltonian has the same numerical value as total energy in this natural mechanical model. Torque and damping do not change its definition, but remove conservation:

\[
\boxed{\dot H=u\omega-b_a\omega^2-b_p\nu^2.}
\]

Here `b_a = rotary_damping_nms`, `b_p = pendulum_damping_nms`.

### Energy feature representations

To compare against the previous project formulation, define explicitly:

\[
\begin{aligned}
K_a&=\tfrac12J_a\omega^2,\\
K_c&=\tfrac12(A-J_a)\omega^2+c\omega\nu,\\
K_p&=\tfrac12J_p\nu^2,\\
E_p&=K_p+V_p. 
\end{aligned}
\]

Then

\[
\boxed{\begin{aligned}
E_1&=(K_a,K_c+K_p,V_p),\\
E_2&=(K_a,K_c,E_p),\\
E_3&=(K_a,K_c,K_p,V_p).
\end{aligned}}
\]

All components sum to `H` (in `E_2`, using its third component `E_p`). **`K_c` can be negative** due to the cross term; it is not a standalone positive kinetic energy. The three `E` representations are useful for evaluation but are not globally injective state coordinates; particularly, they omit the absolute arm angle and lose signed state information. Do not assume that `E_3` plus an instantaneous energy direction uniquely determines a controlled state.

## 3. Conservative Routh reduction and orbital phase

For the ideal autonomous plant (`u=b_a=b_p=0`), the arm angle is cyclic. Its conjugate momentum is

\[
\boxed{p=p_\theta=\frac{\partial L}{\partial\omega}=A(\alpha)\omega+c(\alpha)\nu.}
\]

Under torque and damping,

\[
\boxed{\dot p=u-b_a\omega.}
\]

Eliminate the arm speed:

\[
\omega=\frac{p-c(\alpha)\nu}{A(\alpha)}.
\]

Define the Routh effective inertia and potential:

\[
\begin{aligned}
D(\alpha)&=J_p-\frac{c(\alpha)^2}{A(\alpha)},\\
V_{\mathrm{eff}}(\alpha;p)&=V_p(\alpha)+\frac{p^2}{2A(\alpha)}.
\end{aligned}
\]

Then the exact reduced energy identity is

\[
\boxed{H=\tfrac12D(\alpha)\nu^2+V_{\mathrm{eff}}(\alpha;p).}
\]

For fixed `H,p`, a selected regular orbital family has

\[
\nu=\sigma\sqrt{\frac{2[H-V_{\mathrm{eff}}(\alpha;p)]}{D(\alpha)}},\quad\sigma\in\{-1,+1\}.
\]

For a libration between adjacent turning points `α_- < α_+`, obtained from `H = V_eff(α;p)`, the period is

\[
\boxed{T(H,p,\mathcal B)=2\int_{\alpha_-}^{\alpha_+}\sqrt{\frac{D(a)}{2[H-V_{\mathrm{eff}}(a;p)]}}\,da.}
\]

Here `𝓑` is a connected orbit family/branch label, necessary when multiple orbit components have the same `H,p`. Define

\[
\tau(\alpha;H,p,\mathcal B)=\int_{\alpha_-}^{\alpha}\sqrt{\frac{D(a)}{2[H-V_{\mathrm{eff}}(a;p)]}}\,da.
\]

A continuous, uniformly increasing phase on the libration is

\[
\phi=\begin{cases}
2\pi\tau/T,&\nu>0,\\
2\pi-2\pi\tau/T,&\nu<0.
\end{cases}
\quad (\bmod\ 2\pi).
\]

The expressions agree by continuity at turning points. Therefore, **on an ideal conservative regular orbit**:

\[
\boxed{\dot H=0,\qquad\dot p=0,\qquad\dot\phi=\Omega(H,p,\mathcal B)=\frac{2\pi}{T(H,p,\mathcal B)}.}
\]

For rotations, use a one revolution integral and consistent unwrapped phase convention. The formula does not define a regular periodic phase at an equilibrium or a separatrix; near separatrices the period becomes ill conditioned. Because the absolute arm angle is cyclic, a periodic reduced trajectory need not imply periodic `θ`.

**Attribution:** This construction applies established Routh reduction and classical action angle theory to the present Furuta model. It is not a new theorem. The controlled controller and its performance remain research hypotheses.

### Controlled phase dynamics: exact local form, not globally valid

Within a regular, differentiable orbital coordinate chart, `φ=Φ(α,H,p,𝓑)` satisfies

\[
\dot\phi=\Omega(H,p,\mathcal B)+\Phi_H\dot H+\Phi_p\dot p.
\]

Substituting the exact energy and momentum balances:

\[
\boxed{\dot\phi=\Omega+(\Phi_H\omega+\Phi_p)u
+\Phi_H(-b_a\omega^2-b_p\nu^2)-\Phi_p b_a\omega.}
\]

The derivatives `Φ_H, Φ_p` are evaluated on a consistent branch at fixed `α`. A regular chart or numerical continuation is required near turning points; **do not directly differentiate a singular quadrature expression**. Near separatrices, equilibria, discontinuous regime changes, or atlas failures, use the full original ODE.

For stage one, the phase coordinate is used for **timing events and rollout diagnostics**, not as a replacement for controlled state integration.

## 4. Controller specification

### 4.1 Goals and configurable parameters

Main target: reach upright rest and transfer to the existing capture controller. The target mechanical energy is

\[
H^\star=2G.
\]

This is **necessary but not sufficient** for capture: the energy might reside in arm motion. Also monitor `p`, arm angle, arm velocity, pendulum angle and velocity.

Candidate tunables (no numerical defaults are scientifically established yet):

| Parameter | Meaning |
|---|---|
| `k_energy` | Fraction of present energy tracking error requested at each pulse; `0 < k_energy < 1` |
| `torque_max` | Physical motor torque magnitude limit from config |
| `pulse_time_max` | Maximum powered duration per opportunity |
| `phase_half_width` | Half width of each phase window around `π/2`, `3π/2` |
| `arm_angle_max` | `θ_max`; default to existing run config constraint |
| `arm_margin` | Safety margin inside physical arm limit |
| `prediction_dt` | Maximum rollout step for safety checking |
| `brake_gain` | Desired arm velocity decay gain |
| `capture_thresholds` | Full state capture region and hysteresis |
| `energy_tolerance` | Avoid chattering near energy target |
| `period_atlas_validity` | Regular orbit exclusion tolerances near boundaries/separatrices |

### 4.2 Pulse opportunity detection

Nominal event phases

\[
\boxed{\phi_c\in\{\pi/2,3\pi/2\}.}
\]

Allow actuation only in a symmetric phase window around an event center:

\[
\mathrm{wrap}_{[-\pi,\pi)}(\phi-\phi_c)\in[-\delta_\phi,+\delta_\phi].
\]

Track phase unwrapping, center crossing, and previous intervention IDs to avoid repeated firing at a single event. A planned pulse may be started before the center to straddle it, requiring a predicted start time; for the first version, event trigger at window entry is acceptable with a clear log that it is an approximation to symmetric centering. Stop at window exit or the time cap, whichever occurs first.

**Do not assume these are maximum work opportunities:** instantaneous motor energy power is `uω`, so for bounded `|u|` the maximum instantaneous absolute work rate occurs at maximum `|ω|`, which generally depends on `H,p,𝓑`. Record `|ω(H,p,φ)|` over each orbit and compare actual maxima to the two fixed centers. Retain the fixed centers as the specified experimental baseline even if suboptimal.

### 4.3 Per pulse energy goal

At each eligible opportunity request

\[
\boxed{\Delta H_g=k_{\mathrm{energy}}(H^\star-H),\quad 0<k_{\mathrm{energy}}<1.}
\]

Signed energy change is allowed; if the energy is above target, the requested change is negative. The controller chooses a torque magnitude and duration satisfying:

\[
|u|\le u_{\max},\qquad 0\le\tau_a\le\tau_{\max},
\]

and approximately `H(t+τ_a)-H(t) ≈ ΔH_g`. Exact energy balance:

\[
\Delta H=\int_t^{t+\tau_a}(u\omega-b_a\omega^2-b_p\nu^2)\,ds.
\]

For a first guess with constant torque and coast predicted arm displacement `Δθ_0(τ)`:

\[
\tilde u(\tau)=\frac{\Delta H_g}{\Delta\theta_0(\tau)},\qquad
u_{\mathrm{cand}}(\tau)=\mathrm{clip}(\tilde u(\tau),-u_{\max},u_{\max}),
\]

This ratio assumes the pulse does not materially change the arm trajectory and ignores damping, so it is only an initialization. If `|Δθ_0|` is too small, treat the opportunity as ineffective. Evaluate each candidate under the full nonlinear dynamics and select feasible torque and duration by bounded scalar search or a small enumerated grid; minimize absolute energy tracking error after the pulse plus a momentum penalty if desired. Do not force infeasible energy goals.

For example, fixed durations in a small grid up to `pulse_time_max`, with bounded analytic torque guesses and optionally `±u_max`, give a deterministic implementation. This is **not** a general high dimensional MPC trajectory optimization.

### 4.4 Candidate modes

Three modes:

1. `COAST`: `u=0`.
2. `ENERGY_PULSE`: apply `u_cand` for selected `τ_a ≤ τ_max`, then coast.
3. `BRAKE`: torque chosen to reduce arm velocity/avoid arm boundary, continuously screened for feasibility.

**Priority:** Safety and viability checks precede energy pulse execution. The brake mode may interrupt a pulse. In nominal operation, no torque is applied outside allowed phase windows except braking or local capture stabilization.

## 5. Prediction based constraint logic

### 5.1 Horizon definitions

Let `T0 = T(H,p,𝓑)` be the **current nominal conservative period**. Use

\[
T_{1/2}=0.5T_0,\qquad T_1=T_0.
\]

The period is only a prediction clock; a powered/damped trajectory will not generally retain this period. Define a maximum allowable prediction horizon so `T0→∞` near a separatrix does not cause numerical blowup; when orbit phase is invalid, switch to a configured fixed time horizon and full ODE rollouts.

The user proposed screening at `0.5T0` and `T0`. **Implement safety as continuous interval checking, not endpoint checks.** Approximate

\[
M_\theta^{(m)}(T_h)=\max_{0\le s\le T_h}|\theta_m(t+s)|
\]

on a sufficiently refined rollout, with interpolation and margin for between sample overshoots. Prefer event/root detection if possible. Require `M_θ ≤ θ_max − arm_margin` for a candidate to be considered safe over that horizon. Discrete sampling alone is not a mathematical safety proof.

### 5.2 Initial decision logic

At every controller update, in addition to evaluating at phase events:

1. Simulate `COAST` over `[0,T_half]`. If arm constraint violation is predicted, **enter braking assessment immediately**, rather than waiting for the next phase center. Compute whether a braking action actually yields a feasible trajectory. If no action is feasible, log a constraint infeasibility and invoke the runtime's explicit emergency/saturation policy; do not claim safety.
2. If the immediate half period forecast is safe, and currently inside an eligible phase window, evaluate `COAST` and `ENERGY_PULSE` over `[0,T_1]`, including the pulse itself and subsequent coast.
3. If passive trajectory is safe but pulse trajectory is unsafe, **skip the pulse**.
4. If both are safe, select an energy improving pulse if its gain exceeds a deadband/actuation cost; otherwise coast.
5. If passive is unsafe on the full horizon but a pulse is safe, a candidate action may be selected for safety even if it has poor energy performance, but do not rely exclusively on the phase window for this case. Evaluate braking and recovery as well.
6. If both are unsafe, evaluate explicit `BRAKE` and/or recovery candidates. An infeasible result is a genuine limitation of the torque authority and initial condition.

**Do not assume** the next phase opportunity naturally brakes the arm. Braking is separately commanded and tested. Repeat safety assessment at each integration/control update; candidate switching can occur earlier than the next phase event.

### 5.3 Braking candidate

Obtain arm acceleration from the existing full dynamics in affine input form:

\[
\ddot\theta=f_\theta(x)+g_\theta(x)u,\qquad
g_\theta(x)=\frac{J_p}{\Delta_M(\alpha)}>0.
\]

A simple desired braking acceleration `θ̈_des = −k_b ω` yields

\[
\boxed{u_B=\mathrm{clip}\left(\frac{-k_b\omega-f_\theta(x)}{g_\theta(x)},-u_{\max},u_{\max}\right).}
\]

This is a nominal velocity damping command, **not** a certified stopping guarantee. It may produce position drift and might not prevent boundary crossing. Evaluate full braking trajectory, and optionally compare maximum opposing torque and a higher order barrier torque interval. Braking must stay active until an explicit safe release condition is met (hysteresis prevents mode chatter).

An optional future improvement is a scalar barrier safety filter with `h_+=θ_max−θ`, `h_-=θ+θ_max` and second order constraints, subject to verified initial conditions and input feasibility. Do not claim formal forward invariance merely from discrete rollouts.

## 6. Capture and phase validity

Energy tracking alone does not guarantee upright stabilization. At the desired upright equilibrium:

\[
\alpha\equiv\pi\ (\mathrm{mod}\ 2\pi),\quad\nu=0,\quad\omega=0,\quad\theta\approx\theta_{\mathrm{ref}},\quad H=2G,\quad p=0.
\]

Transition to the existing local stabilizer when the **full physical state** lies within a verified capture region. Use a hysteretic entry/exit rule. Do not control directly by orbital phase at the separatrix or upright equilibrium; `T` is singular/ill conditioned there.

If period/phase estimates are invalid during swing up or near a separatrix, the phase controller should fall back to an established energy controller, an explicit initial excitation, or a bounded full state pulse planner. This is particularly important at exact downward rest: phase is undefined and purely energy proportional feedback may produce zero torque.

## 7. Period computation and atlas

### Stage 1: reference numerical phase (offline + diagnostic)

For each conservative state, compute `H,p`, classify orbit branch from the effective potential, find the turning points or rotational branch, and numerically integrate `T`. Handle integrable endpoint singularities by a suitable coordinate substitution or robust quadrature. Validate against full ODE cycle times. Compute phase by the normalized elapsed time integral and branch direction, with continuous unwrapping.

### Stage 2: precomputed atlas (only after baseline works)

Precompute `T(H,p,𝓑)` on a branch aware adaptive grid, and optionally an orbital decoder `α(H,p,φ,𝓑)`, `ω(H,p,φ,𝓑)`, `ν(H,p,φ,𝓑)`. Perform `O(1)` local table interpolation at runtime for grid based lookup (with approximation error); the exact general period quadrature itself is not an elementary constant cost operation. Explicitly mark invalid regions and critical boundaries. Maintain a full ODE fallback.

For controlled trajectories, update the phase consistently from the measured physical state or by integrating a validated local forced phase model; **do not simply propagate `φ += Ω(H,p)dt` across significant torque or damping and assume exactness**.

## 8. Suggested code interface and integration

Create a separate controller module under `src/rotary_pendulum/heuristic/` (exact filenames left to local code style), with isolated functions:

```python
orbital_invariants(state, model) -> (H, p)
classify_orbit(H, p, model) -> branch_or_invalid
orbital_period(H, p, branch, model, atlas=None) -> T_or_invalid
orbital_phase(state, H, p, branch, model, atlas=None) -> phi_or_invalid
candidate_energy_pulses(state, phase, config, model) -> list[Pulse]
rollout_candidate(state, action_schedule, horizon, config, model) -> Trajectory
trajectory_arm_margin(trajectory, theta_max) -> min_margin
braking_candidate(state, config, model) -> action
controller_step(state, controller_state, config, model) -> (torque, controller_state, diagnostics)
```

- Keep the pulse scheduling state: current branch, continuous phase, last fired phase event, current active pulse and stop condition, brake mode latch, local capture mode.
- Reuse `state_derivative()`/`rk4_step()` for ground truth; use JAX versions only after numerical equivalence tests.
- Ensure `theta` is **unwrapped** for arm angle constraints and net rotation calculations. The repository normalizes `alpha`, not `theta`, in RK4.
- Support logged controls and candidate rollout traces for reproducibility.
- In diagnostics distinguish `phase_geometric_existing`, `phase_orbital_new`, `phase_valid`, and `phase_center_event`.
- Keep numeric and symbolic models consistent if extending into MPC; do not import the existing `mpc/features.py` phase proxy as orbital phase.

## 9. Evaluation and ablations

### 9.1 Physics unit tests

- `H` from sum of `E_1`, `E_2`, `E_3` agrees with `energy_components()` and is finite across a state grid.
- `p=Aω+cν`; numerical differentiation agrees with `ṗ=u−b_aω`.
- Numerical differentiation of total energy agrees with `Ḣ=uω−b_aω²−b_pν²`.
- Routh reconstructed `ω` and `ν` agree with original state at regular sample points, using branch and phase.
- For ideal conservative ODE, `H,p` remain constant to numerical tolerance; `φ(t)` advances at `2π/T` on regular periodic trajectories.
- The orbit atlas rejects equilibria, separatrices, and invalid branches; no silent extrapolation.
- A pair of states with identical energies and opposite velocities yields different controlled `Ḣ` under the same nonzero torque, demonstrating observability limitations.

### 9.2 Controller comparisons

Compare at least:

A. Existing energy MPC (reference).  
B. Existing heuristic energy controller (reference).  
C. New continuous energy and momentum feedback without phase gating.  
D. Phase gated pulse controller without constraint screening.  
E. Phase gated pulse controller with candidate screening and braking.  
F. Optional: orbit dependent centers from maximizing `|ω(H,p,φ)|`, compared with fixed `π/2,3π/2`.

Run matched initial states: exact downward rest, perturbed downward, moving starts, low amplitude swing, medium amplitude swing, near upright, and states near arm limits. Include states on both sides of arm and pendulum velocity sign ambiguities. Repeat with conservative, nominal damping, and parameter perturbed models.

### 9.3 Metrics

- Swing up/capture success rate under a **full state** capture definition.
- Time to capture, steady upright error, and time spent in capture state.
- Maximum absolute rotary arm angle and number of constraint violations, including between sample checks.
- Fraction of controller updates with infeasible safety actions and number of emergency fallbacks.
- Total absolute work `∫|uω|dt` and signed work `∫uωdt` separately.
- Total powered duration `∫1_{|u|>ε}dt` and **number of actuation switch ons** (do not conflate these).
- Torque saturation fraction, pulse success in achieving `ΔH_g`, brake activations.
- Prediction error of candidate rollouts and computation time per decision.
- Phase alignment: distance of observed `|ω|` maxima from `π/2,3π/2` over representative regular orbits.

### 9.4 Pass criteria before atlas acceleration

- Conservative physics identities and orbit period tests pass within documented numeric tolerance.
- Pulse controller can reliably produce energy change on representative valid orbits under full ODE prediction.
- Angle screening reduces violations relative to the unscreened baseline, while documenting infeasible cases.
- Behavior at downward rest and near the separatrix is explicit, finite, and nonchattering.
- Ablation establishes whether fixed phase gating adds benefit over a continuous or nongated energy controller.

## 10. Development sequence

1. Add analytical invariant helpers and unit tests using existing dynamics.
2. Build offline reference orbit period and phase computation for a handful of regular conservative orbits; visualize `α,ω,ν,H,p,φ` and `|ω|` against phase. Verify proposed phase centers.
3. Implement the phase event state machine with a fixed pulse width and a bounded fractional energy target, using full ODE rollout for actual candidate evaluation.
4. Add the passive half period and full period safety screening, explicit braking mode, and infeasibility handling.
5. Reuse existing local capture control; add excitation/fallback where phase is invalid.
6. Execute matched baseline experiments and ablations; inspect failure categories.
7. Only then build adaptive `(H,p,branch)` period tables and optional orbit decoder to reduce runtime.

## 11. Explicit research assumptions and open decisions

1. **Angle constrained:** The current handoff interprets angle constraint as the rotary arm `|θ|≤θ_max`. Additional `α` constraints would change feasibility substantially.
2. **Phase centers:** `π/2` and `3π/2` are user proposed fixed centers. Their equivalence to highest energy injection must be tested; they are not a theorem.
3. **Pulse centering:** Implement symmetric phase windows and record whether actual powered pulse straddles the center; selecting `τ` and exact start time can be refined after baseline.
4. **Target energy:** `H*=2G` is upright rest total energy but does not guarantee convergence. Include momentum and full state capture criteria.
5. **Safety:** Screening by finite rollouts is a predictive heuristic, not a proof. Brake feasibility must be checked; the next intervention is not assumed to brake naturally.
6. **Period:** `T` depends on the current **conservative** orbit and branch and may become infinite near the separatrix. Use finite fallback horizons when invalid.
7. **Starting at rest:** Need explicit symmetry breaking/excitation because orbital phase may be undefined and simple feedback can stall.
8. **Control interface:** Ensure constraints can interrupt or override phase gated energy injection; otherwise hard angle protection is impossible in many states.

## 12. Literature basis (background attribution, not project scope)

- J. Á. Acosta, *Furuta's Pendulum: A Conservative Nonlinear Model for Theory and Practise*, Mathematical Problems in Engineering (2010), DOI: `10.1155/2010/742894`. Conservative Furuta mechanical model and integrability.
- M. B. Vizi, D. M. Horvath, G. Stepan, *Routh Reducibility and Controllability of Unstable Mechanical Systems*, Acta Mechanica (2022), DOI: `10.1007/s00707-022-03146-1`. Furuta Routh reduction and cyclic momentum treatment.
- V. I. Arnold, *Mathematical Methods of Classical Mechanics*, 2nd ed., Springer (1989), Chapter 10. Action angle coordinates for regular periodic integrable motion.
- K. J. Åström and K. Furuta, *Swinging up a Pendulum by Energy Control*, Automatica (2000). Classical energy driven swing up.

**Novelty boundary:** The conservative phase formula `φ̇=2π/T(H,p,branch)` is a direct application of established theory; possible project contributions reside in control synthesis, phase window selection, intervention scheduling, safety screening, and experimentally demonstrated computational or control benefits.
