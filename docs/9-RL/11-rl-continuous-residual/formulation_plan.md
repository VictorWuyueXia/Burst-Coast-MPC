# Continuous analytical prior and residual TD3

## Quick reading

The motor acts on the arm, so pendulum energy error divided by time is a desired
**power**, not a torque. The controller analytically computes how motor torque
changes pendulum energy at the current state, then solves a regularized scalar
inverse. This is inexpensive and has no action search. It is only an
instantaneous approximation: holding its answer for 100 ms can change the
direction and strength of energy transfer during that interval.

An arm filter predicts three choices: the requested torque, zero torque and an
inward centering/braking torque. It prefers the request whenever its predicted
path stays inside the arm bounds. Prediction is used only for this filter.
The filter is a practical nominal-model check, not a constraint guarantee.

The residual actor starts at zero. Later TD3 can correct, cancel or reverse the
heuristic. Its critics take the actual physical torque after filtering. Reward
and strict holding criteria remain the previously accepted dense formulation.
No training is part of the present heuristic validation.

## State, units and constants

The measured/simulated state is

$$
x=(\theta,\alpha,\omega,\nu).
$$

Here arm angle θ is unwrapped in radians, pendulum angle α is measured from
downward in radians, and ω and ν are their velocities in rad/s. Upright error
is computed as β = atan2(sin(α−π), cos(α−π)). The simulator's nominal physical
parameters define pendulum inertia J = 0.000133128 kg m², coupling coefficient
B = 0.00013158 kg m², gravitational coefficient G = 0.01518588 N m and base
arm inertia J₀. The code obtains all four from `jax_dynamics.MODEL`, derived
from the checked-in physics configuration; there is no duplicate parameter fit.

The physical torque clip is 0.0204 N m. The unchanged policy limit is
U = 0.45 × 0.0204 = 0.00918 N m. Decisions last Δ = 0.1 s; the RK4 physics
interval is h = 0.02 s. Arm bound Θ = π/2 rad is a nominal filter bound and
an excursion diagnostic, not an episode termination condition.

## Analytical energy request

Pendulum energy E and the upright target E* are computed from state and physics:

$$
E=\frac12J\nu^2+G(1-\cos\alpha),\qquad E^*=2G=0.03037176\ \mathrm{J}.
$$

Define the mass-matrix entries A = J₀ + J sin²α and C = B cosα, and determinant
D = AJ−C². Let fν(x,0) denote the pendulum acceleration returned by the
analytical ODE with zero applied motor torque. The computed drift a has units
W, and torque sensitivity b has units 1/s:

$$
\dot E=a(x)+b(x)u,\qquad
a=\nu[Jf_\nu(x,0)+G\sin\alpha],\qquad
b=-\frac{JC\nu}{D}.
$$

Desired power P* = (E*−E)/τ is in W. The positive tunable energy time τ is in
seconds; it controls how quickly the instantaneous law asks to remove the
energy deficit. A positive sensitivity floor b₀ in 1/s prevents inversion of
negligible control authority. The closed-form regularized least-squares law is

$$
u_E=\operatorname{clip}\left(
\frac{b(P^*-a)}{b^2+b_0^2},-U,U\right).
$$

It minimizes `(a + b u − P*)² + b₀² u²` before clipping. At horizontal or zero
pendulum speed, b vanishes: motor torque has no first-order instantaneous
pendulum-energy authority. This does not imply zero authority over an entire
100 ms interval.

At downward near-rest, defined by E < 0.05 E*, |ν| < 0.2 rad/s and
|ω| < 0.15 rad/s, replace this request with a 0.000408 N m startup pulse.
Its sign is negative if θ > 0 and positive otherwise. These fixed thresholds
and pulse amplitude are explicit prototype choices; the pulse is memoryless,
acts toward arm center when θ ≠ 0 and breaks symmetry positively at exact rest.
The residual can subsequently cancel it. Outside that region the law above is
used, without a hidden local stabilizer or controller switch.

## Arm filter

Prediction length N is a tunable integer of 20 ms steps, initially N = 20;
H = Nh is the corresponding prediction duration, initially 0.4 s. At each
state, compute desired arm acceleration and bounded brake torque:

$$
\ddot\theta_b=-2\omega/H-\theta/H^2,\qquad
u_b=\operatorname{clip}\left(
\frac{\ddot\theta_b-f_\theta(x,0)}{J/D},-U,U\right).
$$

The zero-torque arm acceleration fθ(x,0) is computed by the same ODE. The
centering term makes this more than pure velocity damping: it requests inward
recovery near a boundary. Clipping respects the policy cap.

Predict candidates `(requested, 0, u_b)` in parallel. Hold each for the first
five physics steps, then predict the same centering/braking feedback refreshed
every five steps. Inspect **every** 20 ms sample. A candidate is feasible if
its maximum predicted |θ| is strictly below Θ. Choose the first feasible
candidate in the stated order. If none is feasible, choose the smallest
`max|θ| + 0.001|θ_final|`; the second term breaks near-ties toward arm center.
All quantities in that score are in radians. This is best effort, not a claim
that coasting or braking always prevents crossing.

Filter modes are integers: 0 accepted request, 1 coast, 2 brake, 3 no feasible
candidate/best effort. The last mode can apply any of the three candidates;
the actual torque is always logged. Existing arm reward remains the only arm
penalty. No new terminal event or reward term is introduced by filtering.

## Observation, action and learning

Let elapsed time t and deadline T be in seconds, with T selected from 20, 40
and 60. Let c be the current consecutive strict-goal physics-sample count.
The computed eight-component observation is

$$
o=\left(\theta/\Theta,\sin\alpha,\cos\alpha,
\omega/s_\omega,\nu/s_\nu,t/60,(T-t)/60,c/5\right),\qquad
s_\omega=\Theta\sqrt{\frac{J_0G}{J_0J-B^2}},\quad s_\nu=2\sqrt{G/J}.
$$

The arm scale uses the coupled small-oscillation frequency stored in the model;
the pendulum scale uses the isolated target-energy velocity. Both elapsed and
remaining time are included so T can be recovered even if future training
mixes deadlines. No last-action feature is needed: there are no switch costs,
actuator state or action-history-dependent rewards in this phase.

The actor's bounded output z = tanh(network(o)) represents a residual in units
of twice the policy cap. Exploration/smoothing noise ε is in units of U:

$$
u_{request}=\operatorname{clip}(u_E+U(2z+\epsilon),-U,U),\qquad
u_{applied}=\operatorname{ArmFilter}(x,u_{request}).
$$

At z = −u_E/(2U), the residual cancels the heuristic. A larger opposite-signed
residual reverses it; the range spans nearly the entire opposite torque cap
even when the heuristic saturates. No residual noise is used in validation.
The actor's last layer is initialized to zero, so its initial deterministic
behavior equals the heuristic plus filter exactly.

Actor and each independent critic have two hidden layers of 128 SiLU units.
The actor maps eight inputs to one residual. Each critic maps the eight inputs
and actual torque divided by U to one scalar return. There are two online
critics and corresponding target copies, plus an actor target copy.

For reward r, terminal flag d, next observation o′, smoothed and filtered
target action u′, and target critics Q′₁, Q′₂, the finite-horizon TD3 target is

$$
y=r+(1-d)\min_{i\in\{1,2\}}Q'_i(o',u'/U).
$$

Discount is exactly one. Both timeout and successful hold set d = 1; neither
bootstraps. Each critic minimizes mean squared error to y. Every second critic
update also minimizes `−mean Q₁(o,u_applied(o)/U)` for the actor, then updates
all targets with 0.995 old + 0.005 new. Target noise has standard deviation
0.2 and is clipped to ±0.5; behavior noise initially has standard deviation
0.1. Both act on the combined request before policy clipping and filtering.
Adam learning rate is 0.0003, with gradient norm clipping at 10.

The hard filter and torque clipping have zero actor gradient in overridden or
saturated regions, except where the selected candidate is the differentiable
request. No straight-through derivative is invented. This is a known learning
limitation to monitor with filter-intervention rates. Replay contains the
actual normalized torque, not the residual or rejected request.

## Reward and terminal contract

At each active post-integration physics sample use normalized energy error
e = E/E*−1, upright credit q computed below, and the five reward rates:

$$
q=\exp\left[-\tfrac12\left((\beta/0.16)^2+(\nu/0.4)^2+
(\omega/0.3)^2\right)\right],
$$

$$
r_h=h\left[-|e|-0.02|u_{applied}|/U
-1.05(1+t/T)-0.2(\theta/\Theta)^2+q\right].
$$

The upright widths are respectively in rad, rad/s and rad/s; rate coefficients
are tunable reward units per second. A decision reward sums its active physics
samples. This retains the five accepted dense terms: energy, torque, growing
time cost, arm deviation and smooth upright credit. No potential difference,
success bonus or extra arm-limit penalty is added. The time term exceeds the
maximum upright credit, preserving the preference for early completion.
Changing T also stretches this normalized time-cost ramp; compare success,
energy and arm behavior directly across deadlines rather than raw returns.

Strict success requires |θ| ≤ 0.08 rad, |β| ≤ 0.08 rad, |ω| ≤ 0.15 rad/s and
|ν| ≤ 0.20 rad/s for five consecutive physics samples (100 ms). Success ends
the episode immediately, even within a decision. Otherwise it ends at T.
Physical state freezes after termination and all further rewards are zero.

## Validation definitions

Matched resets are reused at all three deadlines. Each random stratum contains
64 episodes. All intervals below are independent uniforms:

| Label | θ (rad) | α (rad) | ω (rad/s) | ν (rad/s) |
| --- | --- | --- | --- | --- |
| downward | ±0.2 | ±0.2 | ±0.5 | ±0.5 |
| moving | ±0.5 | random sign × [0.4, 2.6] | ±2 | ±6 |
| near | ±0.25 | π ±0.25 | ±1 | ±1 |
| tight | ±0.08 | π ±0.12 | ±0.15 | ±0.30 |

Four separate probes are exact downward rest, exact upright rest and mirrored
boundary approaches `(θ,α,ω,ν) = ±(1.45,0.4,2,0)`. They do not enter random
success averages. Reports distinguish strict success from merely reaching
90% target energy or passing within 0.16 rad of upright. These latter events
can occur at reset and do not establish swing-up or capture. Full 20 ms states,
100 ms actions, reward components, filter modes and terminal outcomes are saved.
