# Interpretation and operating contract

This exports `wave-2/quarter_bc1000/best.msgpack`, selected after 524,288 online
transitions by 256-start validation: first maximize arm-compliant final balance,
then final upright fraction, then minimize final torque RMS. Its score was
(1, 1, −0.0000696212 N·m). The independently selected 2.295 mN·m model is a different
policy and is not included. “Best” refers to this recorded selection, not proven
optimality or minimum torque across different motor caps.

Actor and both online critics each have three 256-unit SiLU hidden layers.
The actor maps five features to a tanh action; each critic concatenates those
features with normalized applied torque and emits an unrestricted scalar.
Weights are copied without quantization. The original training snapshot supplies
the reference outputs; its source hashes and checkpoint SHA-256 are recorded.

## State, actuator and plant

Physical state consists of arm angle θ (unwrapped radians), pendulum angle α
(radians, zero down), and measured angular velocities ω and ν (rad/s).
The arm scale L is the prescribed 135° range, 2.356194490192345 rad.
Features are θ/L, sin α, cos α, ω/30.592532352353444 and ν/21.36069107055664.
Speed divisors are computed from the nominal plant and recorded numerically;
they are normalization constants, not admissibility bounds.

Applied action a is physical torque divided by the **0.00459 N·m** training cap.
The action is held constant for 100 ms. Dynamics use ten 2 ms RK4 steps per 20 ms
reward/observation sample. The physical configuration's 0.0204 N·m motor rating
is a simulator ceiling, **not this policy's permitted torque**. Rescaling this
actor to a different cap invalidates the learned value contract.

The nominal lossless model uses gravity 9.81 m/s², arm mass/length 0.095 kg/0.085 m,
pendulum mass/length 0.024 kg/0.129 m, and zero joint damping. Exact inertias and
coupling are in the manifest. Inputs assume full physical-state availability;
noise, delay, motor dynamics and hardware parameter mismatch are not validated.
There is no hard arm stop or runtime constraint filter. The ±135° arm restriction
is soft; additional cost begins at 108°. No hard speed bounds are supplied.

## What the Q values mean

Define upright error β = atan2(−sin α, −cos α), computed from measured α.
The model-derived pendulum inertia I is 0.000133128 kg·m², and gravitational
coefficient G is 0.01518588 N·m. Relative pendulum energy E and target E⋆,
upright score U, arm penalty A and reward rate ρ are computed as follows:

$$
E=\tfrac12 I\nu^2+G(1-\cos\alpha),\quad E_\star=2G,
\qquad U=\exp\!\left[-\tfrac12\left((\beta/0.16)^2+(\nu/0.4)^2+(\omega/0.3)^2\right)\right],
$$
$$
A=(\theta/L)^2+\left[\max(|\theta|-0.8L,0)/(0.2L)\right]^2,
\quad \rho=-(1-U)-|E/E_\star-1|-2a^2-0.2A.
$$

The widths (0.16 rad, 0.4 rad/s, 0.3 rad/s) and weights (1, 1, 2, 0.2) are
chosen training parameters. U rewards quiet uprightness; energy error encourages
swing-up; the last two terms penalize effort and arm travel. E is the training
energy proxy, not total coupled mechanical energy. Reward r for each action is
0.02 times the sum of ρ at its five post-step samples. Rewards are nonpositive;
higher Q means lower predicted accumulated cost, not a success probability.

With γ = 0.995 (chosen discount per 100 ms action), k the future action index,
and π the learned continuation policy, each critic approximates:

$$
Q(x,a)\approx\mathbb E\!\left[\sum_{k=0}^{\infty}\gamma^k r_k\mid x_0=x,a_0=a,\pi\right].
$$

Here x is physical state encoded by the five features; rₖ is the computed future
reward. TD3 targets use clipped target-policy noise (standard deviation 0.02,
clip ±0.05 in normalized action) and lagged twin critics, so the estimate is not
an exact deterministic-policy return. Taking the critics' minimum is not a
certified bound. Sampling resets at 120 s retain pre-reset next-state bootstrap
with `done=False`; capture does not terminate the objective. There are no clock
features. The legacy `time_cost_per_s` setting is unused by this formulation.

For cost-minimizing MPC with the same reward, a candidate terminal cost is
minus `v_actor`, discounted by γ to the number of preceding 100 ms steps.
`v_actor` evaluates both critics at the actor's action. Use `q_min` instead when
the first tail action is specified; it already includes that action's reward,
so avoid counting it twice. Simulate any proposed coast interval and its cost
explicitly before attaching the feedback tail. Different control intervals,
reward weights, torque caps or dynamics require a new justification/calibration.

## Evidence and limits

Training combines augmented heuristic demonstrations (energy swing-up, LQR local
balance and predictive braking), actor/critic pretraining and online TD3 with
imitation weight 1000. The deployed actor has none of those teacher controllers.
Targeted starts cover downward swing-up, moving capture, outward braking and
perturbed-upright recovery. These distributions are not a certified safe domain.

Independent 600 s tests had 512 starts per condition. All ended balanced;
510/512 nominal and 507/512 disturbed episodes also respected the arm bound for
the full duration and held within 5° for the final 10 s. Nominal final torque
RMS was 0.0693 mN·m. Disturbances at 120 and 300 s displaced the four physical
coordinates by signed (0.05 rad, 0.12 rad, 0.2 rad/s, 0.35 rad/s).
Some recoveries involved a fall and fresh swing-up. These tests do not establish
indefinite stability, hard arm safety, critic calibration or MPC performance.

The portable implementation passes 1,024 reference comparisons, single/batched
inference and finite-gradient checks on CPU. This validates the export, not the
Q estimates against realized discounted returns. Before trusting MPC terminal
values, compare them with long policy rollouts at the proposed terminal states
and actions, especially near arm boundaries and outside demonstration coverage.
