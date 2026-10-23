# Understanding the current TD3 task and its use as an MPC terminal value

**The saved networks provide the ingredients for a learned MPC terminal cost, but their terminal-cost accuracy has not yet been established.** Existing experiments validate the actor's control performance. They do not independently validate the critics' predicted returns or an MPC controller using those predictions. This document explains the implemented `rotary-stationary-td3-v3` task and separates that evidence from the proposed MPC use.

## Learning guide

### What problem is the agent learning to solve?

The task is to swing up a rotary pendulum, approach upright with low angular speeds, and continue balancing while penalizing motor effort and arm displacement. It is a **continuing, discounted control task**: reaching upright does not end the task, and future rewards receive progressively smaller weights. The reward and policy have no explicit time dependence; this is what *stationary* means here. It does not mean that the pendulum is always motionless.

The current algorithm is **Twin Delayed Deep Deterministic Policy Gradient (TD3)**. An **actor** chooses a continuous action. Two **critics** estimate how much discounted reward will follow a state–action pair. **Temporal-difference (TD) learning** trains those estimates using the reward from one observed transition plus a prediction at the next state. Twin critics and delayed policy updates address value-estimation errors in the original [TD3 method](https://arxiv.org/abs/1802.09477); teacher demonstrations and imitation are additions in this repository.

| Component | Meaning in this implementation |
|---|---|
| Environment | Nonlinear rotary-pendulum simulator with known mechanical parameters |
| Physical state | Arm angle, pendulum angle, arm angular velocity and pendulum angular velocity |
| Observation | Five numerical features encoding that four-dimensional state |
| Action | One normalized motor torque, held constant for 100 ms |
| Reward | Negative weighted cost accumulated over that 100 ms interval |
| Actor output | One normalized torque command |
| Critic output | One scalar estimate of discounted future reward per critic |
| Learned models | One actor and two critics, plus slowly updated copies used during training |
| Not learned here | Dynamics, prediction horizon, sleep duration or a burst/coast policy |

### State, observation and action

At decision index t, define the simulated physical state x_t as

$$
x_t=(\theta_t,\alpha_t,\omega_t,\nu_t).
$$

Here theta is arm angle in radians, alpha is pendulum angle in radians, omega is arm angular velocity in rad/s, and nu is pendulum angular velocity in rad/s. These values come from the simulator; hardware use would require measured or estimated values. Pendulum angle zero is downward; angles near either minus pi or plus pi are upright. The stored pendulum angle is wrapped to [−pi, pi). Arm angle remains signed and is not periodically wrapped.

The feature map h produces the observation o_t supplied to both actor and critics:

$$
o_t=h(x_t)=\left(\frac{\theta_t}{\theta_{\max}},\sin\alpha_t,\cos\alpha_t,
\frac{\omega_t}{s_\omega},\frac{\nu_t}{s_\nu}\right).
$$

The configured soft arm limit theta_max is 135 degrees, or 3pi/4 radians. The fixed speed scales s_omega and s_nu are computed from the mechanical model and defined in the reference below. Sine and cosine encode pendulum orientation continuously across its angle-wrap boundary. These five features retain the physical state information needed by the nominal dynamics; they contain no elapsed time, remaining time or previous success count. Feature normalization does not clip states at the arm limit.

Let pi_phi denote the actor, with trainable weights phi. Its normalized output a_t and physical torque u_t are

$$
a_t=\pi_\phi(o_t),\qquad u_t=u_{\max}a_t,\qquad -1\leq a_t\leq1.
$$

The positive torque limit u_max is fixed separately for each trained model: 0.00459 or 0.002295 N m. For example, an action of 0.5 applies 0.002295 N m with the first model. The action is a direct torque command; there is no teacher contribution or runtime arm-constraint filter. Training adds exploration noise and clips the resulting action to this interval. Evaluation uses the deterministic actor output.

### Three different meanings of “model”

The **physical model** computes the next state under an applied torque. It is supplied by the simulator, not fitted by TD3. Each 100 ms decision contains five 20 ms observation intervals, each integrated with ten 2 ms fourth-order Runge–Kutta steps.

The **actor network** maps five features to a scalar action through fully connected hidden layers with SiLU activation and a final tanh activation. SiLU is the smooth activation that multiplies its input by the logistic sigmoid; tanh bounds the action between −1 and 1.

Each **critic network** concatenates the five features with the applied normalized action, giving six inputs, and returns one unbounded scalar through a linear output layer. The two critics have independent weights. The selected 4.59 mN m model uses three hidden layers of 256 units in each network; the 2.295 mN m model uses three layers of 512 units. Neither critic predicts the next physical state. TD3 learns from simulator transitions without differentiating through the simulator, so this is model-free learning in a simulated environment with known dynamics.

### Reward, return and value are different quantities

A **reward** describes one decision interval. Here it is nonpositive: it penalizes deviation from upright low-speed balance, pendulum-energy error, squared normalized torque and arm displacement. The arm penalty increases more rapidly beyond 108 degrees, before the 135-degree soft limit. There is no separate elapsed-time penalty in this stationary reward, even though an inherited `time_cost_per_s` field remains in the saved settings.

A **return** adds rewards over the future. Let r_t be the reward computed from the simulated transition at decision t, let gamma be the configured discount 0.995 per 100 ms, and let j count subsequent decisions. The discounted return G_t is

$$
G_t=\sum_{j=0}^{\infty}\gamma^j r_{t+j}.
$$

Higher return means lower accumulated cost. Discounting gives a characteristic weighting span of about 200 decisions, or 20 seconds; it does not terminate the task after 20 seconds. The 600-second evaluation duration is a separate control-performance test.

The **action-value function** Q^pi asks: “If we apply this action now and then follow policy pi, what return should we expect?” Its associated **state-value function** V^pi asks the same question when pi also selects the first action:

$$
Q^\pi(o,a)=\mathbb{E}[G_t\mid o_t=o,\ a_t=a,\ a_{t+j}=\pi(o_{t+j})\text{ for }j\geq1],
\qquad V^\pi(o)=Q^\pi(o,\pi(o)).
$$

The expectation denotes averaging over any transition randomness; the nominal simulator and fixed evaluation policy are deterministic from a given state. The learned critics approximate action values. They are not supervised measurements of exact future return, and neither optimality nor exact evaluation of the final deterministic actor has been established. Target-action noise, the minimum of two critics and incomplete fitting all affect their estimates.

### What happens during training?

1. **Prepare demonstrations.** The augmented teacher supplies trajectories. Whole trajectories qualify only if they avoid arm crossings and finish balanced. Separate episodes provide training and validation data. Sampling balances swing-up, capture, braking, recovery and quiet-balance transitions.
2. **Supervised initialization.** Behavior cloning fits actor actions to teacher actions. Critic pretraining fits approximate discounted teacher returns. Those returns include an estimated continuation beyond the recorded trajectory, so they are initialization targets, not exact infinite-duration labels.
3. **Collect transitions.** The current actor, with exploration noise, controls 256 parallel simulated environments. Each stored transition contains observation, applied action, reward and the actual next observation. A replay buffer stores past transitions for repeated training use.
4. **Update critics.** A batch mixes 50% demonstrations and 50% online replay. Each critic fits a one-step TD target. This is off-policy learning: stored actions need not have been chosen by the current actor.
5. **Update the actor and target networks.** Every second critic update, the actor increases the first critic's predicted value while retaining an imitation penalty on demonstration samples. Slowly moving network copies provide the next-state values used in subsequent TD targets.

Training resets an environment after 120 seconds to collect new starting conditions. This is a **sampling truncation**, not a terminal physical state: the replay record keeps the pre-reset next observation and `done=False`. Successful balancing also leaves `done=False`. The critic therefore learns continuing value instead of being told that future reward vanishes at success or at a collection boundary.

The reset groups are named downward, capture, braking and recovery. A source-level qualification matters: `targeted_resets` places the last three groups' pendulum angles near zero, which is downward under the current convention. Those names should not be interpreted as evidence that online resets actually start near upright. Teacher-data categories and actual physical states should be inspected separately when assessing value coverage.

### Why does TD learning use another prediction as its target?

For a fixed policy, remaining return decomposes into the next reward plus discounted return after that transition. TD learning uses this recursive relationship to update value estimates without waiting for a complete future trajectory. Using an estimated next-state value is called **bootstrapping**.

For one stored transition (o, a, r, o_next), let y be the computed training target, Q_bar1 and Q_bar2 the target critics, and a_next the target actor's noise-perturbed, clipped next action. Then

$$
y=r+\gamma\min_{i\in\{1,2\}}Q_{\bar\psi_i}(o_{\mathrm{next}},a_{\mathrm{next}}),
\qquad \delta_i=y-Q_{\psi_i}(o,a).
$$

Here i selects a critic, psi_i denotes its trainable weights, a bar denotes slowly updated target weights, and delta_i is the TD error. The target is held fixed during each gradient calculation. For illustration only, a reward of −0.2 and a next-state value of −10 give y = −10.15. A current prediction of −8 is then trained downward toward −10.15. This update improves consistency with the current target; it does not independently establish that −10.15 is the true return.

The critic loss is mean squared TD error. The actor loss combines negative predicted value and squared imitation error. **Reward defines desirable behavior; loss defines how network weights are fitted.** The actor's imitation term is a training regularizer, not an additional environment reward. Exact losses and update constants appear below.

### How would this become an MPC terminal cost?

MPC predicts a finite segment of motion and optimizes its actions. A terminal cost assigns a remaining cost to the state at the end of that segment. Let F be the proposed learned terminal cost, using the saved online critics and actor:

$$
F(x)=-\min_{i\in\{1,2\}}Q_{\psi_i}\bigl(h(x),\pi_\phi(h(x))\bigr).
$$

The actor is needed because each critic requires an action as well as a state. The minus sign converts reward value to cost. Taking the minimum critic value is equivalent to choosing the larger of the two cost estimates; it does not provide a certified bound. This formula defines a proposed use, not an already validated MPC integration.

Let H be a chosen positive integer prediction horizon in 100 ms decisions, k the prediction-step index, x_0 the current physical state, x_k the predicted state, a_k an optimized normalized action, and f_0.1 the simulator's 100 ms transition map. With stage cost ell equal to the negative of the implemented transition reward, the consistent discounted objective is

$$
\min_{a_0,\ldots,a_{H-1}} J,
\qquad J=\sum_{k=0}^{H-1}\gamma^k\ell(x_k,a_k)+\gamma^H F(x_H),
\qquad x_{k+1}=f_{0.1}(x_k,u_{\max}a_k).
$$

J is the total predicted cost. The terminal term approximates the cost of following the learned actor from x_H onward; the state-value conversion already includes its first action, so that action must not be charged again as a separate stage. MPC can subsequently replan, but the value being approximated remains actor continuation, not the exact value of future MPC replanning. Different stage costs, torque limits or discount conventions would require a new consistency argument or value training.

For burst/coast control, predict the coast dynamics and accumulate their costs explicitly. Evaluate this terminal value where 100 ms feedback control resumes. For an interval lasting d seconds, with d a chosen duration, the corresponding discount factor is gamma raised to d/0.1; costs within that interval must also preserve the original time weighting. The present critics do not represent repeated future sleep schedules, switching costs or energy saved by sleeping.

### Is it ready, and what evidence is still required?

**Ready for a terminal-value evaluation experiment; not yet established as an accurate MPC terminal cost.** The [existing study summary](interpretation_summary.md) reports 600-second actor tests. Nominal full-duration compliance was 510/512 episodes at 4.59 mN m and 494/512 at 2.295 mN m. Checkpoints were selected by actor compliance, final balance and torque RMS, not by critic return accuracy. Successful actor control can coexist with inaccurate value magnitudes.

| Assessment | Existing evidence | Required validation and success criterion |
|---|---|---|
| Input and objective compatibility | Five-feature continuing critics and matching actor are saved | Reproduce features, action scaling, transition reward and discount exactly for each checkpoint |
| Value accuracy | No independent critic-versus-return assessment identified in the study records | Freeze networks; compare each critic and their minimum against independent discounted rollouts from held-out and MPC-reachable states; report bias, absolute error and worst errors by state group |
| Action and state ranking | Actor tests do not measure rankings | Compare predicted preference with rollout-return preference, particularly for nearby actions and states that MPC must distinguish |
| Numerical behavior during optimization | Networks are differentiable individually; no terminal-cost optimization test reported | Check values and directional derivatives near upright, arm limits and critic-switching points; verify the optimizer does not exploit unsupported value peaks |
| Closed-loop benefit | No matched MPC terminal-cost comparison reported | Compare zero terminal cost and the learned cost with identical dynamics, costs, horizons and starts; report total cost, failures, arm crossings and solve time |

For value validation, hold the first action fixed and then follow the frozen actor when testing Q; use the actor immediately when testing V. Extend rollouts until the remaining discounted contribution is demonstrably negligible or independently bounded. Do not use the critic under test to supply its own reference continuation. Set numerical error tolerances before evaluating results, relative to the return differences that change MPC's action choice; no such acceptance thresholds are established by the current study. Parallel simulation can evaluate independent starts and first actions efficiently.

Even a small TD loss only indicates agreement with bootstrapped targets on sampled data. It does not ensure accurate values outside that data, hard arm-constraint satisfaction or closed-loop stability. At gamma = 0.995, a systematic target bias can propagate over many steps. For this proof-of-concept study, the next useful result is a measured value-accuracy and matched MPC comparison, rather than further actor training justified solely by balancing success.

## Implementation reference

This section records the exact mathematical definitions and saved settings for reproduction. Experimental arrays, logs and checkpoints remain under `artifacts/`; the sections above are the primary learning explanation.

### Feature scales and reward construction

Let I_p be pendulum hinge inertia in kg m², g_p the gravitational torque coefficient in N m, I_b the base inertia in kg m² and I_c the coupling inertia in kg m². These are computed from configured masses, lengths and gravity by `derive_model`. The derived natural angular frequency Omega_n and the feature scales are

$$
\Omega_n=\sqrt{\frac{I_b g_p}{I_b I_p-I_c^2}},\qquad
s_\omega=\theta_{\max}\Omega_n,\qquad s_\nu=2\sqrt{g_p/I_p}.
$$

Both speed scales have units rad/s. They normalize inputs; they are not velocity constraints. Define beta as signed upright angle error, E as the relative hinge-swing energy in joules, and E_star as its upright target:

$$
\beta=\operatorname{atan2}(-\sin\alpha,-\cos\alpha),\qquad
E=\tfrac12 I_p\nu^2+g_p(1-\cos\alpha),\qquad E_\star=2g_p.
$$

E is the implemented pendulum-energy shaping quantity, not the total mechanical energy of the coupled apparatus. Let b be an upright low-speed score and A the dimensionless arm penalty:

$$
b(x)=\exp\left[-\tfrac12\left((\beta/0.16)^2+(\nu/0.4)^2+(\omega/0.3)^2\right)\right],
\qquad A(\theta)=\left(\frac{\theta}{\theta_{\max}}\right)^2+
\left(\frac{\max(|\theta|-0.8\theta_{\max},0)}{0.2\theta_{\max}}\right)^2.
$$

The three denominators are tuned widths: 0.16 rad for upright error, 0.4 rad/s for pendulum speed and 0.3 rad/s for arm speed, in that order. Define c(x,a) as a positive cost rate, with configured weights w_b, w_E, w_u and w_A:

$$
c(x,a)=w_b[1-b(x)]+w_E|E/E_\star-1|+w_u a^2+w_A A(\theta),
\qquad (w_b,w_E,w_u,w_A)=(1,1,2,0.2).
$$

The weights express numerical reward units per second; they do not represent physical electrical power. All four values are tuning choices shared by the selected models. With m indexing the five 20 ms endpoints and x_t,m denoting their simulated states under the held action, the exact interval reward is

$$
r_t=-0.02\sum_{m=1}^{5}c(x_{t,m},a_t),\qquad \ell(x_t,a_t)=-r_t.
$$

Thus the cost uses intermediate states, not just the state at the decision boundary. All components vanish at the ideal upright, arm-centered, zero-speed, zero-torque equilibrium. Arm violations remain finite-cost events, not hard exclusions.

### Losses, target updates and pretraining

For a minibatch B of 512 transitions, with n indexing samples, the implemented joint critic loss is

$$
L_Q=\frac{1}{2|B|}\sum_{n\in B}\sum_{i=1}^{2}
\left[Q_{\psi_i}(o_n,a_n)-y_n\right]^2.
$$

Here |B| is batch size, and y_n is the TD target already defined. Target-action smoothing draws independent standard normal noise z, scales it by 0.02, clips it to ±0.05, then clips the noisy target actor action to ±1:

$$
\epsilon=\operatorname{clip}(0.02z,-0.05,0.05),\quad z\sim\mathcal N(0,1),\qquad
a_{\mathrm{next}}=\operatorname{clip}(\pi_{\bar\phi}(o_{\mathrm{next}})+\epsilon,-1,1).
$$

Clip means limiting a scalar to the indicated interval. This training-target noise is distinct from the online exploration standard deviation of 0.01. Both are in normalized action units. Let D be the 256 demonstration samples within B, and a_n^teach their recorded teacher actions. The actor loss is

$$
L_\pi=-\frac{1}{|B|}\sum_{n\in B}Q_{\psi_1}(o_n,\pi_\phi(o_n))+
\frac{\lambda}{|D|}\sum_{n\in D}\left[\pi_\phi(o_n)-a_n^{\mathrm{teach}}\right]^2,
\qquad \lambda=1000.
$$

Lambda is the tuned imitation weight. The first critic, after its current update, supplies the actor objective; the minimum of the critics is used in TD targets, not in this actor loss. Both losses use Adam at learning rate 0.0003, with global gradient-norm clipping at 10. Every second critic update also applies

$$
\bar\phi\leftarrow0.995\bar\phi+0.005\phi,\qquad
\bar\psi_i\leftarrow0.995\bar\psi_i+0.005\psi_i.
$$

These constants control target-network averaging. Their numerical agreement with the discount is coincidental; averaging and discounting serve different purposes.

During pretraining, actor loss is only the mean squared teacher-action error. Critics fit recorded approximate teacher returns. Let T be the number of decisions in a teacher trajectory, r_j its reconstructed stationary rewards, and G_hat_t its approximate return label:

$$
\widehat G_T=\frac{\frac1{100}\sum_{j=T-100}^{T-1}r_j}{1-\gamma},\qquad
\widehat G_t=r_t+\gamma\widehat G_{t+1}\quad(t=T-1,\ldots,0).
$$

The boundary label assumes that the mean reward of the last ten seconds continues indefinitely. It is a heuristic continuation estimate even for a qualified trajectory. Critic pretraining replaces y_n in the squared-error loss by these labels. Online TD3 subsequently uses observed rewards and target networks instead of those return labels.

| Setting | 4.59 mN m model | 2.295 mN m model |
|---|---:|---:|
| Hidden widths in actor and each critic | 256, 256, 256 | 512, 512, 512 |
| Supervised pretraining updates | 20,000 | 100,000 |
| Online transitions at selected checkpoint | 524,288 | 786,432 |
| Online transitions in completed training run | 1,048,576 | 1,048,576 |
| Replay capacity / parallel environments | 262,144 / 256 | 262,144 / 256 |
| Critic updates per parallel collection | 4 | 4 |
| Online transitions before TD updates start | 16,384 | 16,384 |

### Source and checkpoint map

All links below are relative to this document. The saved checkpoint contains online and target network parameters plus optimizer state. Evaluate the online actor and critics together using their matching settings; do not substitute older eight-feature networks or a different torque cap.

| Definition | Source |
|---|---|
| Five-feature observation | [`observe`](../../../../src/rotary_pendulum/RL/jax_residual_task.py) |
| Physical constants and transition integration | [`jax_dynamics.py`](../../../../src/rotary_pendulum/environment/jax_dynamics.py), [`derive_model`](../../../../src/rotary_pendulum/environment/dynamics.py) |
| Reward, resets and continuing replay | [`jax_stationary_task.py`](../../../../src/rotary_pendulum/RL/jax_stationary_task.py) |
| Networks, TD targets and online losses | [`jax_td3.py`](../../../../src/rotary_pendulum/RL/jax_td3.py) |
| Supervised losses and teacher return labels | [`jax_demonstrations.py`](../../../../src/rotary_pendulum/RL/jax_demonstrations.py), [`prepare.py`](../../../../scripts/experiments/stationary-balancing/prepare.py) |
| Training and checkpoint selection | [`rotary_stationary_workflow.py`](../../../../src/bringup/rotary_stationary_workflow.py), [`jax_stationary_evaluation.py`](../../../../src/rotary_pendulum/RL/jax_stationary_evaluation.py) |
| Selected checkpoint identities and hashes | [`selected_models.json`](../../../../artifacts/rotary_pendulum/stationary-balancing-20261013/machine-scannables/selected_models.json) |
| 4.59 mN m settings and weights | [`settings.json`](../../../../artifacts/rotary_pendulum/stationary-balancing-20261013/wave-2/quarter_bc1000/machine-scannables/settings.json), [`best.msgpack`](../../../../artifacts/rotary_pendulum/stationary-balancing-20261013/wave-2/quarter_bc1000/machine-scannables/best.msgpack) |
| 2.295 mN m settings and weights | [`settings.json`](../../../../artifacts/rotary_pendulum/stationary-balancing-20261013/wave-2/eighth_wide_bc1000/machine-scannables/settings.json), [`best.msgpack`](../../../../artifacts/rotary_pendulum/stationary-balancing-20261013/wave-2/eighth_wide_bc1000/machine-scannables/best.msgpack) |
