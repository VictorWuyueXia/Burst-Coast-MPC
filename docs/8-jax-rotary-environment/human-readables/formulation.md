# JAX Rotary-Pendulum Environment: Physical Formulation

## One plant, two clocks

The existing [rotary-pendulum formulation](../../6.1-rotary-pendulum-MPC-formulation.md) supplies the governing mechanics. The new environment changes the integration and decision clocks, not the nominal mechanism. The physical state is

$$
x=[\theta,\alpha,\omega,\nu]^T,
$$

where $\theta$ is the unwrapped rotary-arm angle in radians, $\alpha$ is the unwrapped pendulum angle in radians measured from downward, $\omega=\dot\theta$ is arm angular velocity in radians per second, and $\nu=\dot\alpha$ is pendulum angular velocity in radians per second. The input $u=\tau$ is signed motor-shaft torque in newton-meters. All four states are observed without measurement noise in this show-of-concept environment.

Let $f(x,u)$ denote the existing nominal continuous-time derivative. The environment integrates it by a fixed fourth-order Runge–Kutta map $F_{\Delta t}$:

$$
\begin{aligned}
k_1&=f(x,u),\\
k_2&=f(x+\tfrac{\Delta t}{2}k_1,u),\\
k_3&=f(x+\tfrac{\Delta t}{2}k_2,u),\\
k_4&=f(x+\Delta t k_3,u),\\
F_{\Delta t}(x,u)&=x+\tfrac{\Delta t}{6}(k_1+2k_2+2k_3+k_4).
\end{aligned}
$$

Here $\Delta t=0.02$ seconds is the fixed physics interval; each $k_i$ is a state derivative evaluated at an RK4 stage. This map advances one 50 Hz physics step. It must remain callable directly, without episode or reward logic, so later MPC/MPPI can batch and differentiate nonlinear predictions.

A controller decision is held constant for $N_h=5$ physics steps:

$$
x_{j+1}=F_{\Delta t}^{[N_h]}(x_j,u_j),\qquad
\Delta T=N_h\Delta t=0.10\text{ seconds}.
$$

$j$ counts controller decisions, $u_j$ is the torque chosen at decision $j$, and $F_{\Delta t}^{[N_h]}$ means five sequential applications of the same physics map. Thus the decision rate is 10 Hz. The finite horizon is $N_{\max}=1000$ physics steps, or 20 seconds and 200 complete decisions. Early termination can occur inside a five-step hold.

## Goal and constraint semantics

The upright target remains the complete state $[0,\pi,0,0]^T$. Define $\beta=\operatorname{atan2}(\sin(\alpha-\pi),\cos(\alpha-\pi))$, the wrapped angular error from upright in radians. Phase 10 revised the live goal tolerances to $\epsilon_\theta=\epsilon_\beta=0.08$ radians, $\epsilon_\omega=0.15$ radians per second, and $\epsilon_\nu=0.20$ radians per second after measuring the 100 ms action resolution. These are configured benchmark choices, not learned parameters. The original phase-8 validation used `0.05` for all four limits; its plant-parity evidence remains valid, while task results must state the contract revision. A physics sample is inside the goal when

$$
g(x)=\mathbf 1\{|\theta|\leq\epsilon_\theta,\ |\beta|\leq\epsilon_\beta,\ |\omega|\leq\epsilon_\omega,\ |\nu|\leq\epsilon_\nu\}=1.
$$

$\mathbf 1\{\cdot\}$ is the indicator of the stated condition. The consecutive goal count $c$ increases by one at each inside-goal physics sample and resets to zero outside it. Success occurs at $c=5$, representing a 100 ms sampled hold. A reset starts at $c=0$ even if its initial state lies within the goal; no success is awarded without five subsequent samples.

The physical arm angle is never wrapped for constraint checks. The RL episode fails when $|\theta|\geq\theta_{\max}$ at an evaluated physics state, where $\theta_{\max}=\pi/2$ radians is the agreed ±90° cable-travel limit. This is a hard episode termination, unlike the old MPC's soft arm penalty. The 50 Hz simulator checks each physics result, including the four intermediate results inside a 10 Hz action hold. It does **not** certify that no excursion occurred between two 20 ms samples; the validation plan tests whether that resolution is adequate in the intended state range. If success and timeout coincide at step 1000, success wins; a constraint violation always takes precedence.

Torque is bounded by the current nominal rated-torque limit $u_{\max}=0.0204$ N·m, read from the existing physical configuration. The environment applies $\operatorname{clip}(u,-u_{\max},u_{\max})$ at the plant boundary, matching the current simulator's actuator semantics. This limit is not a hardware-current or thermal approval.

## Continuously distributed resets

The following are provisional show-of-concept distributions, not calibrated operating envelopes. Each coordinate is sampled independently and uniformly within its stated interval, except the moving-swing pendulum angle, whose direction has an independent fair sign. All ranges lie inside the arm limit at reset. A balanced benchmark draws each stratum with probability $1/3$; callers may also request one stratum explicitly for stratified tests.

| Stratum | $\theta$ (rad) | $\alpha$ (rad) | $\omega$ (rad/s) | $\nu$ (rad/s) | Interpretation |
|---|---:|---:|---:|---:|---|
| Downward | $[-0.20,0.20]$ | $[-0.20,0.20]$ | $[-0.5,0.5]$ | $[-0.5,0.5]$ | Perturbed low-energy starts |
| Moving swing | $[-0.50,0.50]$ | $s[0.40,2.60]$, $s\in\{-1,+1\}$ | $[-2,2]$ | $[-6,6]$ | Both swing directions and appreciable motion |
| Near upright | $[-0.25,0.25]$ | $\pi+[-0.25,0.25]$ | $[-1,1]$ | $[-1,1]$ | Capture and recovery starts |

These numbers are experiment-design bounds chosen for coverage, not measured Quanser uncertainty. The physical parameters are **not** randomized. Fixed pseudorandom seeds make validation sets reproducible, but each sampled initial state is continuous rather than selected from a deterministic catalog. Before finalizing the ranges, inspect accepted-state fractions, arm-limit failures, goal-at-reset frequency, and representative uncontrolled and controlled trajectories by stratum.

## What the environment does not decide

The environment returns physical state and terminal outcomes, not a reward. Energy-pumping incentives, actuator-on duration, switching penalties, and value definitions are deliberately deferred to [future RL work](future_rl.md). Later batched linearization is also a separate measured optimization: the nonlinear JAX map is the reference against which any local $A$ and $B$ rollout approximation must be judged.
