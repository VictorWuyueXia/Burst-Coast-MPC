# Deferred RL and Predictive-Control Plan

This page is a reminder of intended later experiments, **not** a reward or network specification for the present environment work. The first learning phase is ordinary value learning and interaction with the JAX environment. Large-batch MPPI/MPC prediction, sampled local linear models, and joint learning with a controller in the loop follow only after the nonlinear simulator is validated.

## Measured execution addendum

The validated simulator has two deliberately layered interfaces. `rk4_step(x, u)` advances only the smooth nonlinear plant by one 20 ms physics interval and accepts arbitrary leading batch axes; use it for MPC/MPPI prediction, custom horizons, and differentiation. `step(env_state, u)` holds one clipped action for up to five physics intervals and owns goal dwell, arm-limit failure, timeout, and episode counters; use `jax.vmap(step)` for RL environment batches. The latter calls the former, so there is one dynamics implementation rather than two simulators.

On one NVIDIA L40S, complete 20-second, 200-decision `step` rollouts were measured with inputs and scans kept on-device. Final-state-only execution reached 1,048,576 parallel environments in 162 ms using a 199 MiB allocator peak; 8,388,608 also fit, but took 5.48 s and is not an efficient training batch. Retaining the complete six-field environment history used 1.32 GiB for 262,144 environments, 5.27 GiB for 1,048,576, and 21.09 GiB for 4,194,304. These are environment-only measurements: policy activations, optimizer state, reward/value buffers, and logging require additional VRAM. Start independent RL sessions at 65,536 or 262,144 environments per GPU, then tune against the actual learner update cost.

The eight L40S GPUs currently support eight **independent** one-GPU training sessions by launching one process with one `CUDA_VISIBLE_DEVICES` value per card. This environment does not yet create a synchronized eight-GPU learner; that later work needs explicit parameter and batch sharding plus gradient synchronization. Give each session a distinct root PRNG key and separate artifact directory. Add an on-device reward and automatic-reset wrapper before training, since a terminal lane freezes logically but remains inside the fixed scan until reset.

JAX's default allocator reserved about 34.5 GiB of each 46.1 GiB card even though the simulator's active use was much smaller. One process per GPU is therefore appropriate. Do not place multiple default JAX processes on one card; if sharing is intentional, explicitly set `XLA_CLIENT_MEM_FRACTION` or disable preallocation with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The full measurements, including GPU telemetry and machine-readable tables, are in the [validation results](../validation-results/20261005/interpretation_summary.md).

## Learning objective to formulate next

The desired behavior is to pump the pendulum toward the upright energy state while minimizing actuator-on time. The user also wants a preference for zero or near-maximum-magnitude torque, discouragement of small-to-maximum input changes, and weak time and/or energy tie-breakers. These requirements need explicit, testable definitions before training. In particular, “energy reached” is not success: success remains the complete upright state and 100 ms hold, and arm-limit failure remains terminal.

For the existing pendulum-relative swing-energy convention, a useful diagnostic is

$$
E_s(x)=\tfrac12 J_p\nu^2+G(1-\cos\alpha),\qquad E_s^\star=2G.
$$

$J_p$ is the nominal pendulum rotational inertia derived from the fixed rod mass and length; $\nu$ and $\alpha$ are state coordinates defined in [the formulation](formulation.md); $G=m_pgl$ is the gravity-energy coefficient, where $m_p$ is pendulum mass, $g$ is gravity, and $l$ is pivot-to-center distance. $E_s$ is computed from state, in joules, and $E_s^\star$ is the upright-rest energy under the downward-zero convention. It measures passive pendulum swing energy, not full-body mechanical energy and not attainment of the four-state goal. Whether the RL objective should use $E_s$, full mechanical energy, or a progress potential is **undecided**; the reward will be examined empirically in the learning phase.

At 10 Hz, one action $u_j$ is held for five 50 Hz steps. Define $Q(z_j,u_j)$ as the expected discounted or finite-horizon return after choosing **one** such action and then following the learned policy; it is not a value over burst/coast horizon choices. $V(z_j)$ is the value before choosing that action. $z_j$ is the Markov observation supplied to learning and will at least include the four physical states, the goal-hold count, and the remaining episode time. If the reward penalizes a change from the previous applied torque, include that previous torque in $z_j$ as well. Thus a cumulative actuator-on duration can be expressed as a sum of per-action costs, and a switching cost becomes Markov after adding one previous-action variable; non-Markovian training is **not inevitable**. A hard total-on-time budget would require remaining budget in $z_j$, but no such budget is currently specified.

The next phase should compare a genuinely discrete set $\{-u_{\max},0,+u_{\max}\}$ with a bounded continuous action and a loss that encourages the same extremes. It must decide whether “least on time” is a primary lexicographic criterion, a weighted cost, or a constrained objective, and whether small-to-maximum changes deserve a special asymmetric penalty. The respective indicator thresholds, coefficients, discount or finite-horizon convention, and failure/success returns are all experiment choices to be tuned **after** environment validation. Do not quietly inherit the older burst-coast MPC objective: it previously favored minimum sufficient torque, which is not the same preference.

## Later implementation sequence

| Later milestone | Work | Validation and success flag |
|---|---|
| R1. Reward and Markov state | Define success-first objective and switching/on-time accounting; compare action representations | Hand-check trajectory rankings; no hidden dependence on unobserved history; arm violations never count as success |
| R2. V/Q baseline | Train simple Flax value networks with Optax using batched JAX interactions | Held-out continuously sampled initial states; report per-stratum success, violation, completion time, on-time, switches, energy traces, and seed spread |
| R3. Fast prediction | Benchmark JAX nonlinear MPPI rollouts against sampled local linear or piecewise-affine $A,B$ models | Compare accuracy and wall time over the actual candidate batches and horizons; accept linearization only where it improves the speed–error tradeoff |
| R4. Joint controller learning | Let MPPI/MPC consume value priors while collecting policy-improvement data | Compare with R2 and controller-only baselines on the same frozen seeds and physical constraints; demonstrate improvement without increased violations |

For R3, a local linearization of the 10 Hz nonlinear decision map $F$ around a sampled reference state $\bar x$ and torque $\bar u$ has the form

$$
x^+\approx F(\bar x,\bar u)+A(x-\bar x)+B(u-\bar u),\qquad
A=\left.\frac{\partial F}{\partial x}\right|_{(\bar x,\bar u)},\quad
B=\left.\frac{\partial F}{\partial u}\right|_{(\bar x,\bar u)}.
$$

$x^+$ is the predicted next physical state, $A$ is its state Jacobian, and $B$ is its torque Jacobian, both computed from the validated JAX map. Writing $x^+=Ax+Bu$ without an affine offset is correct only in deviation coordinates or at an appropriate equilibrium. The later experiment must measure how far each sampled local model remains accurate during long-horizon prediction before using it for fast batch rollout.

The intended learning stack is JAX environment + Flax + Optax. A Gymnax-style adapter is optional only if a chosen training implementation actually needs it; the environment's own pure reset/step interface remains the core. No algorithm, architecture width, optimizer rate, reward coefficient, linearization-grid size, or eight-GPU sharding scheme is locked before its respective experiment. The eight-L40S server should first prove single-GPU workload scaling, then use multiple devices if measured batch throughput warrants it.
