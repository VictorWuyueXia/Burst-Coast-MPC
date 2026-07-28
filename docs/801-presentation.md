# Burst-Coast MPC: Toward Max-Coast and Varying Temporal-Resoluiton Control
<!-- 
*Initial advisor presentation: idea, formulation, benchmark progress, and research choices*

--- -->

## 1. Long-term goal

Varying-timestep control only search for more critical times --> varying horizon, only replan at more critical times 

Navigation plus control slowly variant natural settings, where passive control is dominant:
- spacecrafts, sailboats
<!-- - natural orbital or attitude dynamics can provide useful motion without continuous actuation; -->
- actuator cycles and processor time are limited;
- long nonlinear prediction horizons are expensive;
- safety-critical controllers should be certifiable and explainable.

A representative objective is

$$
\min_{\pi}\;
J_{\mathrm{task}}
+w_a\sum_tJ_{\mathrm{action}}
+w_e\sum_tJ_{\mathrm{state}}
+w_s\sum_k t_{\mathrm{solve},k},
$$

The explainable controller challenge: a strategic layer chooses **how long to actuate and when to replan**, while constrained MPC computes the physical commands. A neural network never commands the plant directly, and we try to make it explainable like a normal function.

<!-- **Suggested motivation figure:** a spacecraft path drawn as long blue coast arcs separated by short orange thrust arcs, with "optimizer awake" windows above the thrust arcs and callouts for propellant, computation, safety, and arrival. -->

---

## 2. General formulation and fast grid computation

For a constrained nonlinear system,

$$
\dot x=f(x,u)
% \qquad x_{i+1}=F_{\Delta t}(x_i,u_i),
% \qquad x_i\in\mathcal X,
% \quad u_i\in\mathcal U.
$$

At replanning epoch $k$, choose a timing action

$$
a_k=(B_k,H_k)\in\mathcal A,
\qquad 1\le B_k\le H_k,
\qquad
u_{i\mid k}=
\begin{cases}
v_{i\mid k}, & i<B_k,\\
u_{\mathrm{coast}}, & B_k\le i<H_k.
\end{cases}
$$

$B_k$ is the "burst" action horizon, $H_k$ is the prediction or replanning horizon, and $u_{\mathrm{coast}}$ is prescribed, commonly as zero input. For each candidate,

$$
J_{\mathrm{MPC}}^*(x_k;a_k)
=\min_{v_{0:B_k-1}}
\left[
\sum_{i=0}^{B_k-1}\ell_b(x_{i\mid k},v_{i\mid k},\Delta v_{i\mid k})
+\sum_{i=B_k}^{H_k-1}\ell_c(x_{i\mid k},u_{\mathrm{coast}})
+V_f(x_{H_k\mid k})
\right]
$$

subject to the model and constraints,.

The strategic segment cost includes physical and computational consequences,

$$
c_k=w_t c_{time}
+w_u\sum_{i\in k}c_{input}
+w_s t_{\mathrm{solve},k}
+w_{\mathrm{terminal}},
\qquad r_k=-c_k,
$$

and the scheduling critic follows Bellman's optimality equation,

$$
Q^*(s_k,a_k)=\mathbb E\!\left[c_k+
\gamma Q^*(s_{k+1},a')\right],
\qquad
a_k^*=\arg\min_{a\in\mathcal A}\widehat Q(s_k,a).
$$

MPC then solves only the selected $(B_k,H_k)$ physical-control problem.

```mermaid
flowchart LR
    X["Observed state x_k"] --> G["Finite burst-horizon grid"]
    G --> T["Learned value: a grid of Q values of (B_k, H_k) pair"]
    T --> A["arg min selects B_k, H_k"]
    A --> M["Solve the selected constrained MPC"]
    M --> E["Active burst then passive coast"]
    E --> XN["Next replanning state"]
```

<!-- **Progress.** The early controller enumerated split/horizon variants outside the nonlinear program, giving independent fixed-size problems that were process-parallelized. The current inner baseline is serial so solve-time labels are meaningful; Monte Carlo episodes still run in parallel processes. Learned deployment already broadcasts the state across the complete action grid for batched tensor evaluation. -->

**Potential acceleration path:**

1. Linearize about a nominal trajectory, and move state/action rollouts and value evaluation into GPU tensors so the candidate dimension is parallel.
2. Precompute a finite state space in ODE or PDE and build DP problem.
3. Reuse suffix values with $V_j(x)=\min_a\{c(x,a)+V_{j+1}(F(x,a))\}$, replanbefore reaching horizon.

<!-- **Recommended computation figure:** two $(B,H)$ heat maps. The left shows one CPU solve and solve time per cell; the right shows one GPU tensor evaluation, the minimum cell, and its burst/coast timeline. -->

---

## 3. First benchmark: torque-limited inverted pendulum

<!-- Weak actuation makes the pendulum a useful nonlinear benchmark: it must exploit phase, energy accumulation, and coasting rather than move directly to the upright equilibrium. -->

With $x=[\theta,\omega]^T$, $\theta=0$ upright,

$$
\dot\theta=\omega,
\qquad I\dot\omega=mg\ell\sin\theta-b\omega+u,
\qquad I=m\ell^2
$$

The mechanical energy, target energy, and normalized error are

$$
E(\theta,\omega)=\frac12 I\omega^2+mg\ell(1+\cos\theta),
\qquad E^*=2mg\ell,
\qquad e_E(x)=\frac{E(x)-E^*}{E^*}.
$$

Prediction horizon $H_k$ uses a fixed ratio to natural period $T_n$, and allowed action $B_k$ a varying ratio to $H_k$

<!-- $$
H_k=\max\!\left(1,\left\lceil\frac{\bar H_kT_n}{\Delta t}\right\rceil\right),
\quad
B_k=\max\!\left(1,\operatorname{round}(\bar B_kH_k)\right),
\quad C_k=H_k-B_k.
$$ -->

The current inner MPC cost is

$$
J_{\mathrm{MPC}}
=\sum_{i=0}^{H_k-1}q_Ee_E(x_{i\mid k})^2
+\sum_{i=0}^{B_k-1}q_{\Delta u}
\left(\frac{u_{i\mid k}-u_{i-1\mid k}}{u_{\max}}\right)^2.
$$

<!-- Phase and local-upright features are logged and visualized but are not weighted in this baseline.  -->
The learned cost and reward are

$$
c_k=w_t(t_{k+1}-t_k)
+w_u\sum_{i=t_k}^{t_{k+1}-1}\left(\frac{u_i}{u_{\max}}\right)^2
+w_c\frac{t_{\mathrm{solve},k}}{\Delta t}
+P_f\mathbf 1_{\{\mathrm{terminal\ failure}\}},
\qquad r_k=-c_k.
$$

The enforced input and coast constraints are

$$
|u_{i\mid k}|\le u_{\max}<mg\ell,
\qquad u_{i\mid k}=0\ \text{for}\ B_k\le i<H_k.
$$

The reported operating and goal sets are

$$
\mathcal X=\{(\theta,\omega):|\theta|\le\theta_{\max},\ |\omega|\le\omega_{\max}\},
\qquad
\mathcal G=\{(\theta,\omega):|\operatorname{wrap}(\theta)|\le\varepsilon_\theta,
\ |\omega|\le\varepsilon_\omega\}.
$$

The goal must hold for consecutive steps.
<!-- samples. The state limits are currently monitored diagnostics, not hard MPC constraints; hard safety enforcement remains future work. -->

![Recorded frame of the Burst-Coast MPC realtime diagnostics and pendulum window](figures/advisor/realtime_window.png)

*Recorded replay of the actual real-time layout: kinetic and potential energy, phase, action, and pendulum animation. This goal-reaching run is benchmark evidence, not a safety certificate.*

![Typical replanning-level time-series artifact](figures/advisor/timeseries_result.png)

*A standard per-run artifact: sampled $(\bar B,\bar H)$ actions, segment timing, solve time, immediate cost, and backward return cost.*

To view in live:
```
burst-coast-mpc --inverted-pendulum mpc-only
```

---

## 4. Failed training, recovery plan, and next research choice

The previous traning failed:
- mismatch between optimum training goal $Q^*$ and pure monte carlo offline data $Q^\mu$
- temporal model (LSTM) made same state-action from different trajectories different meanings
- almost all state-action pairs has been seen in training data
<!-- 
The previous critic fit logged Monte Carlo returns, but its greedy closed-loop policy later collapsed toward minimum horizons and nearly zero actuation. Exploratory episodes often reached the goal, so exploration masked deterministic policy failure.

The preserved artifacts point to coupled causes:

- action semantics and the horizon range changed after offline training;
- random-continuation returns estimate $Q^\mu$, not deployed $Q^*$;
- learned physical weights changed the intended time/effort/compute objective;
- training and deployment used inconsistent compute-cost estimates;
- episode-local replay caused forgetting, while a raw grid minimum exploited unsupported actions;
- one initial state and no locked greedy validation gate allowed collapse to pass. -->

### Current recovery plan

1. Reconstruct a complete Markov state, including remaining episode time information.
2. The new neural network structure treats the benchmark as a Markovian problem
<!-- 2. Calibrate solver time under single-worker conditions and recompute the fixed immediate cost without altering raw data. -->
<!-- 3. Pretrain only an initialization on recalculated behavior-policy returns. -->
3. Train twin conservative cost critics with target networks,
   $$
   Q_C^-(s,a)=\max(Q_1^-(s,a),Q_2^-(s,a)),
   \qquad
   y=c_{\mathrm{calibrated}}+(1-d)\min_{a'}Q_C^-(s',a'),
   $$
   so unsupported actions cannot appear artificially cheap.
4. Move more towards on-policy interactive training with exploration, instead of rely on monte carlo.
<!-- 5. Add common-state action branches and policy-directed trajectories while retaining all earlier training data. -->
<!-- 6. Select only by deterministic multi-state validation, then freeze the method before opening the test set. -->

### Near-future choices

<!-- **Recommended immediate direction:** finish the trustworthy $Q$ critic, then use its value sequence to choose an **early replanning schedule by dynamic programming**. This most directly tests the max-coast hypothesis with the existing pipeline. -->

Some ideas for next step to-do:

- **Theory:** explainable neural control-Lyapunov/barrier functions for liveness, and safety.
- **Application** A simplified 2D CR3BP spacecraft navigation with this algorithm to achieve target chase or stablity
- **Adaptive discretization:** dynamic $\Delta t$ for MPC, or comparison with MPPI.
- **Reduced-order support:** explore energy-phase representations for coast progress and lower-dimensional reasoning.
- **Broader use:** the same outer RL scorer plus inner MPC as a learned optimizer for general non-convex problems.

<!-- **Supplementary figures to make next:** greedy-vs-exploratory horizon collapse; an energy-phase portrait colored by selected coast duration and safe set; and a value-along-coast curve marking the dynamic-programming early-replan point. -->

<!-- ### Questions for the advisor

- Is the primary claim actuation economy, computation economy, or a constrained trade-off?
- Should the next milestone prioritize a deployable critic or a formal CLF/CBF argument?
- Which spacecraft dynamics should follow the pendulum: attitude, relative orbital motion, or low-thrust transfer? -->
