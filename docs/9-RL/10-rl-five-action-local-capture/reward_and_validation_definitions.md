# Reward and Validation Definitions

Historical version-2 reward definitions. Current code implements the [version-3 dense reward](dense_reward_proposal.md), with arm excursions nonterminal and success after 100 ms. Historical campaign results below must not be interpreted as version-3 evidence.

## Human quick reading

`Tight` and `near` name two distributions of episode starting states. They do not define success and they are not fuzzy labels assigned to a trajectory. Tight asks whether the controller can finish capture when the mechanism starts locally around upright. Near asks whether it can recover from a substantially wider upright neighborhood. The success box is a third, smaller object: the state must remain inside it for 100 ms to terminate successfully.

The completed campaign trained only from tight starts because its stage-0 tight fraction was `1.0`. Near was held out for validation. No trial passed stage 0, so none trained on near resets in a later stage.

The learner receives a terminal success or failure amount, a cost for every powered second, and a weaker cost for every elapsed second. Energy and capture terms are potential-difference shaping. They provide stepwise learning feedback, but their sum over a complete episode is a fixed offset determined by the initial state. Consequently, shaping does not change which complete trajectory has the best base return from one fixed start.

## Physical symbols

The physical state is

$$
x=[\theta,\alpha,\omega,\nu]^T.
$$

- $\theta$ is the unwrapped rotary-arm angle in radians. Zero is the centered arm position.
- $\alpha$ is the unwrapped pendulum angle in radians measured from downward. Upright is $\alpha=\pi$ modulo $2\pi$.
- $\omega=\dot\theta$ is arm angular velocity in radians per second.
- $\nu=\dot\alpha$ is pendulum angular velocity in radians per second.

The wrapped error from upright is

$$
\beta=\operatorname{atan2}(\sin(\alpha-\pi),\cos(\alpha-\pi)).
$$

Thus $\beta=0$ is upright, and $\beta$ always lies in $[-\pi,\pi]$ even when the unwrapped pendulum has completed many rotations.

## Tight, near, goal, and wave

### Tight reset and validation set

A tight state is sampled independently and uniformly from

$$
\theta\sim U(-0.08,0.08),\quad
\beta\sim U(-0.12,0.12),\quad
\omega\sim U(-0.15,0.15),\quad
\nu\sim U(-0.30,0.30).
$$

The simulator state uses $\alpha=\pi+\beta$. Episode time, goal-hold count, and terminal flags are reset to zero or false. Tight is used for:

1. stage-0 local-capture training resets;
2. 512 fixed validation starts checked every 1,048,576 collected transitions;
3. the first local-capture promotion threshold, which requires at least 90% success and zero arm violations twice consecutively.

Tight is deliberately larger than the success box in $\beta$ and $\nu$. A tight reset therefore still requires control rather than automatically counting as captured.

### Near-upright reset and validation set

A near state is sampled independently and uniformly from

$$
\theta\sim U(-0.25,0.25),\quad
\beta\sim U(-0.25,0.25),\quad
\omega\sim U(-1,1),\quad
\nu\sim U(-1,1).
$$

Near is environment reset stratum `2`. It tests wider local recovery: larger arm displacement, larger upright error, and substantially larger velocities than tight. Validation uses 1,024 fixed near starts and requires at least 80% success with at most 1% arm violations for promotion.

The completed version-2 trials used `stage_zero_tight_fraction=1.0`. Therefore all stage-0 training resets were tight. Near remained a held-out validation distribution because no trial passed stage 0.

### Success box and dwell

A 20 ms physics state is inside the goal exactly when

$$
|\theta|\le0.08\ \mathrm{rad},\quad
|\beta|\le0.08\ \mathrm{rad},\quad
|\omega|\le0.15\ \mathrm{rad/s},\quad
|\nu|\le0.20\ \mathrm{rad/s}.
$$

The goal counter increments at every consecutive inside-goal physics sample and resets to zero immediately upon leaving. Success requires five consecutive samples, hence $5\times0.02=0.10$ seconds. Reset always sets the counter to zero, even if the sampled initial state is already inside the box.

An arm violation occurs when $|\theta|\ge\pi/2$. An episode otherwise times out after 1,000 physics steps, or 20 seconds. Arm violation takes precedence over success; success takes precedence over timeout.

### Wave

A wave is experiment orchestration rather than a physical or RL variable. One campaign wave launches eight fresh trials concurrently, normally one per GPU. Each trial has its own network initialization, optimizer, replay, and random seed. Trials within a wave compare one controlled parameter family.

The current stage-0 budget per trial is 8,388,608 transitions, with validation every 1,048,576 transitions. Therefore one full wave contains:

$$
8\ \text{trials}\times8{,}388{,}608
=67{,}108{,}864\ \text{training transitions}
$$

and $8\times8=64$ scheduled validation evaluations.

## Reward functions

### Energy and capture quantities

The pendulum-relative swing energy is

$$
E_s(x)=\frac12J_p\nu^2+G(1-\cos\alpha),
$$

where $J_p=0.000133128\ \mathrm{kg\,m^2}$ is pendulum inertia and $G=0.01518588\ \mathrm{N\,m}$ is the gravity-energy coefficient. The upright-rest target is

$$
E_\star=2G=0.03037176\ \mathrm{J}.
$$

The dimensionless energy error is

$$
e_E(x)=\frac{E_s(x)-E_\star}{E_\star}.
$$

Downward rest has $e_E=-1$; upright rest has $e_E=0$. Kinetic energy can make $e_E$ positive when the pendulum moves faster than the upright-rest energy level.

The tolerance-normalized capture error is

$$
q_c(x)=\frac14\left[
\left(\frac{\theta}{0.08}\right)^2+
\left(\frac{\beta}{0.08}\right)^2+
\left(\frac{\omega}{0.15}\right)^2+
\left(\frac{\nu}{0.20}\right)^2
\right].
$$

$q_c$ is a smooth scalar used for shaping. It is not the success test: success checks all four limits separately and requires dwell.

### Reward potential

For an active augmented state $z$, the two potential components are

$$
\Phi_E(z)=-\frac{e_E(x)^2}{1+e_E(x)^2},
$$

$$
\Phi_C(z)=-w_c\frac{q_c(x)}{1+q_c(x)}.
$$

$w_c$ is `capture_weight`. It was `1.0` in Waves 1 and 2. Wave 3 tested `0.5`, `1.0`, and `2.0`; the handoff diagnostic used `2.0`. Both potentials are defined as zero after success, arm violation, or timeout. The total potential is

$$
\Phi(z)=\Phi_E(z)+\Phi_C(z).
$$

Moving toward target energy or the capture box makes $\Phi$ less negative and produces positive immediate shaping. Moving away produces negative shaping. Both fractions saturate far from their targets.

### Base reward

Decision $j$ starts at $z_j$, applies action $a_j$, and ends at $z_{j+1}$. Its actual elapsed time is

$$
\Delta t_j=0.02(n_{j+1}-n_j),
$$

where $n$ is the physics-step counter. Normally $\Delta t_j=0.10$ seconds, but an event can terminate a decision after one to four physics steps.

Let $I_s$, $I_f$, and $I_t$ indicate a newly reached success, arm failure, or timeout. The base reward is

$$
r_j^{\mathrm{base}}
=5I_s-5I_f-2I_t
-0.05\Delta t_j\mathbf1\{a_j\ne0\}
-0.005\Delta t_j.
$$

The terms have these roles:

| Term | Value | Use |
|---|---:|---|
| Success | $+5$ once | Prefer completing the capture task |
| Arm failure | $-5$ once | Reject trajectories crossing $|\theta|=\pi/2$ |
| Timeout | $-2$ once | Reject never completing within 20 seconds |
| Powered time | $-0.05$ per powered second | Prefer less actuator-on duration |
| Elapsed time | $-0.005$ per second | Weakly prefer earlier completion |

For a normal 100 ms nonterminal decision:

| Action | Base reward before any terminal amount |
|---|---:|
| Off | $-0.0005$ |
| Either pump action | $-0.0055$ |
| Either fine action | $-0.0055$ |

This is why the present reward does not make fine torque cheaper than pump torque. Pump magnitude is 22.5 times fine magnitude, but the powered-time indicator only asks whether torque is zero.

### Training reward stored in replay

The replay transition stores

$$
r_j^{\mathrm{train}}
=r_j^{\mathrm{base}}+\Phi(z_{j+1})-\Phi(z_j).
$$

The Double-DQN target, with discount $\gamma=1$, is

$$
y_j=r_j^{\mathrm{train}}
+(1-d_j)Q_{w^-}\!\left(o_{j+1},\arg\max_b Q_w(o_{j+1},b)\right),
$$

where $d_j$ is the terminal flag, $Q_w$ is the online network, $Q_{w^-}$ is the target network, and $o_{j+1}$ is the next seven-feature observation. Terminal transitions do not bootstrap.

Over a complete episode ending at terminal state $z_T$, the shaping terms telescope:

$$
\sum_{j=0}^{T-1}r_j^{\mathrm{train}}
=\sum_{j=0}^{T-1}r_j^{\mathrm{base}}+\Phi(z_T)-\Phi(z_0)
=\sum_{j=0}^{T-1}r_j^{\mathrm{base}}-\Phi(z_0),
$$

because $\Phi(z_T)=0$. For one fixed initial state, $-\Phi(z_0)$ is the same for every complete action sequence. Shaping supplies denser temporal feedback without changing the complete-return action preference.

### Where each reward is used

- The one-step exploration heuristic evaluates all five actions and chooses the largest immediate $r^{\mathrm{train}}$.
- Replay stores $r^{\mathrm{train}}$, and the Q network learns shaped finite-deadline return.
- Evaluation logs both base and training returns.
- Three-decision planning sums $r^{\mathrm{base}}$ along candidate sequences, then optionally adds zero, handcrafted potential, or converted learned Q as its terminal value.
- At one state, adding the action-independent potential back to learned Q does not change action ranking. It is required when comparing values at different predicted states.

## Interpretation of the failed campaign

The trajectory plots show repeated pump actions, almost no fine actions, pendulum speeds near $20\ \mathrm{rad/s}$, and either arm failure or 20-second timeout. This follows from three interacting facts:

1. success is sparse and requires a continuous 100 ms hold;
2. potential shaping preserves complete-return preferences and therefore cannot by itself make uphold valuable;
3. every nonzero action receives the same powered-time cost despite the 22.5-fold pump/fine magnitude difference.

These observations motivate a fixed diagnostic of fine-only local reachability and a torque-magnitude base cost before another training wave.
