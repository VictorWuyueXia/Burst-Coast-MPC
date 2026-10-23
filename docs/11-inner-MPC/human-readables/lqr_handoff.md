# How the existing LQR was made, and how CEM hands over

The local controller is the previously used discrete linear-quadratic regulator (LQR), not PID. It feeds back all four physical state errors and has no integral memory. CEM handles swing-up and recovery; the LQR takes over only after a measured interval of low-speed near-upright motion. Both continue within the same 40-second session. The weak angle/speed additions to the MPC objective are disabled; the original broad arm-centering term remains.

## Linearization of the simulator

The measured/simulated state contains arm angle theta, pendulum angle alpha (zero down), arm speed omega and pendulum speed nu. Angles are radians, speeds rad/s. The desired arm angle and both speeds are zero; upright is alpha = −pi, equivalent to +pi. Define the wrapped upright error beta and the local error vector e:

$$
\beta=\operatorname{atan2}(-\sin\alpha,-\cos\alpha),\qquad
 e=(\theta,\beta,\omega,\nu)^\mathsf{T}.
$$

The normalized command a is dimensionless; physical torque is u = u_max a, with the inherited motor limit u_max = 0.00459 N m. Let f be the existing nonlinear simulator advanced for one 100 ms control interval at constant torque, using five 20 ms RK4 blocks with ten 2 ms integration steps per block. JAX automatic differentiation at upright rest computes the state Jacobian A and normalized-input Jacobian B:

$$
A=\left.\frac{\partial f}{\partial e}\right|_{e=0,a=0},\qquad
B=\left.\frac{\partial f}{\partial a}\right|_{e=0,a=0},\qquad
e_{k+1}\approx A e_k+B a_k.
$$

A is a 4-by-4 matrix and B is a 4-by-1 matrix. Index k denotes a 100 ms control decision. This is a local approximation to the actual coupled rotary-pendulum dynamics, not four independently tuned proportional gains.

## The quadratic design objective

The earlier experiment chose state-weight matrix W and input weight R heuristically; they were not learned from RL. The arm scale theta_max is the existing 135° boundary converted to radians. The upright-angle scale 0.15 rad emphasizes small upright error, while the angular-speed coefficients are 0.25 in inverse squared rad/s units. R = 1 penalizes normalized control effort:

$$
W=\operatorname{diag}(\theta_{\max}^{-2},\,0.15^{-2},\,0.25,\,0.25),\qquad
R=1,\qquad
\min_{a_0,a_1,\ldots}\sum_{k=0}^{\infty}(e_k^\mathsf{T}We_k+a_k^\mathsf{T}Ra_k).
$$

For the linear unconstrained model, the old script solved the discrete algebraic Riccati equation with SciPy's `solve_discrete_are`. P is its symmetric value matrix and K is the resulting feedback row vector:

$$
P=W+A^\mathsf{T}PA-A^\mathsf{T}PB(R+B^\mathsf{T}PB)^{-1}B^\mathsf{T}PA,
$$

$$
K=(R+B^\mathsf{T}PB)^{-1}B^\mathsf{T}PA,
\qquad u=-u_{\max}\operatorname{clip}(Ke,-1,1).
$$

The reused saved gain, in the state order above, is approximately:

$$
K=(-0.07571436,\ 15.36034805,\ -0.12182625,\ 1.22115647).
$$

Mixed signs come from the coupled dynamics and coordinate convention. The old recorded magnitudes of eigenvalues of A − BK were approximately 0.189524, 0.280134, 0.653542 and 0.917225, all below one. On the current float64 simulation map, the maximum difference from those stored values is about 2.74e−6, and all remain below one. The comparison uses an explicit 1e−5 absolute tolerance and independently requires stability. This establishes local linear stability of the unsaturated feedback, not a global guarantee with torque saturation and an arm boundary.

The new experiment reuses the saved gain from the 4.59 mN m teacher settings; it does not solve a new CPU optimization. Dynamics, rollouts, CEM and local feedback run through JAX on GPU. The older teacher also used a predictive braking filter. This study reuses its local LQR gain only; it does not silently add that filter or the original heuristic swing-up controller.

## Explicit handoff and return

The chosen entry band is pendulum error at most 5°, absolute arm speed at most 0.3 rad/s, and absolute pendulum speed at most 0.4 rad/s. These are explicit heuristic capture thresholds; they are not certified boundaries of the LQR region of attraction. A counter records consecutive 100 ms intervals in which **every executed 1 ms sample** meets all three thresholds. Initial mode is CEM, even for a near-upright start.

We screen dwell durations of 0.5 and 1.0 seconds, equivalent to five and ten consecutive qualifying intervals. At the next decision, hand over to LQR after the dwell is complete. Continue LQR while absolute upright error is below 0.25 rad and absolute pendulum speed is below 2 rad/s, the old local region. If either release threshold is reached, return to CEM and require a new dwell before handing over again. This wider release region provides hysteresis. All switches occur at 100 ms decision boundaries; the executed trajectory is still recorded at 1 ms resolution.

On return to CEM, initialize from a fresh actor rollout instead of a stale pre-LQR plan. CEM otherwise uses the existing shifted-plan warm start and two constant-control seeds. When all sessions in a GPU batch are local, skip the entire CEM solve. In a mixed batch, solve the fixed CEM batch but execute its actions only for CEM-mode sessions. This preserves paired random sampling while avoiding unnecessary optimization once the whole batch is local.

The terminal neural value remains the frozen TD3 value. It is not retrained to represent the new hybrid policy, and the handoff is not inserted inside CEM's prediction horizon. Therefore this is an explicit hybrid controller experiment, not a claim of Bellman-consistent optimal control for the hybrid policy.

## Minimal implementation and validation

`hybrid_run.py` has one compiled `advance` function and one substantial nested `optimize` branch for conditional GPU CEM execution; there are no new classes or wrapper layers. Its extra carry is the mode flag and consecutive qualifying-interval count. The experiment parameter is dwell duration; zero selects the plain-CEM control comparison. Gain and thresholds are explicit in the archived design. No hidden controller substitution is introduced.

Milestone 1: saved gain passes current-model pole checks. Milestone 2: eight-start screens complete full 40-second sessions, with handoff/release records checked against their declared predicates. Milestone 3: compare the selected hybrid against fresh plain CEM on the same 32 starts for 40 seconds. Main scores are total time within ±5° and longest continuous time there, using all post-integration 1 ms samples without initial-point overcounting or gap bridging. Also report first handoff, releases, LQR time, final holding and boundary/refinement audits.

Prediction remains one second, torque effort rate remains 2, arm-centering rate remains 10, and the hard arm bound remains ±135°. The session does not terminate upon either capture or a hold-time score. Archived sources and execution records belong under `artifacts/rotary_pendulum/inner-mpc/20261022-hybrid-*`; human summaries and figures remain separate from machine-scannable data.

The comparison pairs initial states and random seeds, not identical pre-handoff trajectories. The separate compiled CEM-only and hybrid branches differ initially at floating-point roundoff; the nonlinear sampler and repeated replanning can amplify that into different actions before handoff. Interpret the results as matched-start controller comparisons, not an exact replay intervention on a shared swing-up trajectory.
