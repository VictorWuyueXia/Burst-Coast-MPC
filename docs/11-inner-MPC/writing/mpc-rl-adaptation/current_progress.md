# Current technical basis for the MPC+RL manuscript

Audit date: 2026-10-26. This note describes the current source and locally available records. It supplies the scientific basis for [the revision plan](revision_plan.md); it does not revise the manuscript. Detailed equations, source locations, numerical settings, and evidence qualifications are separated into [the evidence ledger](machine-scannables/evidence_ledger.md).

## What the present study delivers

The present controller uses nonlinear model predictive control (MPC) with a frozen actor and two critics trained by Twin Delayed Deep Deterministic Policy Gradient (TD3). A mechanics-based teacher supplies demonstrations. The actor supplies candidate continuations, and the critics supply a terminal cost for the finite prediction horizon. MPC selects the applied torque during both swing-up and subsequent upright holding.

| Component | Current status | Scientific role |
|---|---|---|
| Coupled rotary-pendulum model | Implemented; nominal simulation records available | Predict underactuated motion under bounded motor torque. |
| Energy controller with local feedback and arm braking | Implemented demonstration teacher | Supply trajectories for actor and critic initialization and retained demonstration replay. |
| Stationary direct-torque TD3 | Trained; portable selected actor and twin critics available | Learn a state-dependent policy and its discounted continuation reward. |
| Nonlinear MPC with learned terminal cost | Official baseline; complete records available | Combine constrained finite-horizon prediction with learned continuation. |
| Small upright-error holding | Observed over the saved finite sessions | Establish nominal numerical behavior on the tested starts. |
| Complete-controller stability or invariant holding region | Not established | Remains an analysis problem involving plan memory and sampled plan replacement. |
| Learned horizon or actuation/coasting schedule | Not implemented in this controller | Belongs to the broader research objective. |

The paper's central question is therefore whether this learned-continuation construction achieves swing-up followed by small upright error in the nominal nonlinear benchmark. The current records answer that numerical question; they do not establish that the learned terminal term is necessary or superior to another terminal construction.

## Mechanical model and holding objective

Let the generalized coordinates be $q=(\theta,\alpha)^{\mathsf T}$ and the physical state be $x=(\theta,\alpha,\omega,\nu)^{\mathsf T}$, where $\omega=\dot\theta$ and $\nu=\dot\alpha$. The unwrapped arm angle $\theta$ records arm travel; the pendulum angle $\alpha\in[-\pi,\pi)$ is measured from downward. Upright corresponds to $\pm\pi$, with exact $+\pi$ stored as $-\pi$. Angles are in radians and angular velocities in radians per second. The state comes directly from the simulator. Motor torque $u$, in newton metres, acts only at the arm joint.

For uniform rods, let $m_a,m_p$ be arm and pendulum masses, $r,l$ their lengths, and $g$ gravitational acceleration. These are configured physical inputs. The derived inertial and gravitational quantities are $I_a=m_ar^2/3$, $J=m_pl^2/3$, $J_0=I_a+m_pr^2$, $b=m_prl/2$, and $G=m_pgl/2$. Here $I_a,J,J_0,b$ have inertia units and $G$ has energy units. In the current lossless model, the governing equations are

$$
M(\alpha)\ddot q+
\begin{bmatrix}
2J\sin\alpha\cos\alpha\,\omega\nu-b\sin\alpha\,\nu^2\\
-J\sin\alpha\cos\alpha\,\omega^2+G\sin\alpha
\end{bmatrix}
=\begin{bmatrix}u\\0\end{bmatrix},\qquad
M(\alpha)=\begin{bmatrix}
J_0+J\sin^2\alpha&b\cos\alpha\\
b\cos\alpha&J
\end{bmatrix}.
$$

The coupled inertia matrix explains how arm torque accelerates the passive pendulum. Integration of this equation under a constant command defines the nonlinear command map $\Phi_\Delta(x,a)$, where $\Delta$ is the fixed command duration and $a=u/U$ is normalized torque for the selected usable torque cap $U>0$. The current MPC and its selected network share the same cap. This cap differs from the larger actuator rating and the historical energy-controller setting.

Define the wrapped upright error $\beta$ and the error vector $e(x)$ by

$$
\beta=\operatorname{atan2}(-\sin\alpha,-\cos\alpha),\qquad
e(x)=(\theta,\beta,\omega,\nu)^{\mathsf T}.
$$

The controller penalizes all four error coordinates. The reported holding criterion, however, concerns only $|\beta|\le\varepsilon$, with prescribed tolerance $\varepsilon=5\pi/180$ rad. Arm containment is assessed separately against the prescribed envelope $|\theta|<\theta_{\max}$, with $\theta_{\max}=3\pi/4$ rad. Remaining in the upright band does not imply that the complete state converges to rest.

## From mechanical teacher to stationary actor and critics

The teacher's energy argument remains useful, but its purpose has changed. Define total kinetic energy $K=\tfrac12\dot q^{\mathsf T}M(\alpha)\dot q$, arm-body kinetic energy $K_a=\tfrac12 I_a\omega^2$, pendulum-body kinetic energy $K_p=K-K_a$, and potential energy $V=G(1-\cos\alpha)$. Their energy representation is $z=H(x)=(K_a,K_p,V)^{\mathsf T}$. All components have joule units. The total mechanical energy $E=K+V$ and upright-rest energy $E_\star=2G$ satisfy

$$
\dot E=u\omega,\qquad
\int_0^\Delta u\omega(s)\,\mathrm ds=u[\theta(\Delta)-\theta(0)]
$$

for one constant-torque action. Work changes total energy, while coupled motion determines its distribution. The map $H$ loses phase information, so the teacher predicts from the physical state to decode a work request into torque. The current demonstration teacher augments that energy-based action with local linear feedback and predictive arm braking. The standalone learned actor has no teacher or local-feedback contribution at inference.

The student receives a five-feature physical observation

$$
o(x)=\left(\theta/\theta_{\max},\sin\alpha,\cos\alpha,
\omega/s_\omega,\nu/s_\nu\right),\qquad
u=U\pi_\phi(o(x)).
$$

The positive speed normalizers $s_\omega,s_\nu$ are derived from the mechanical parameters, rather than imposed speed limits. The actor $\pi_\phi$ has trained parameters $\phi$ and outputs a normalized torque in $[-1,1]$. Neither elapsed time nor a success counter appears in this observation. The exact normalizers are given in the ledger.

The stationary reward is the negative accumulated cost over a command interval. Its cost rate combines a smooth preference for upright low-speed motion, a pendulum-energy proxy, squared torque, and a soft arm-position penalty. Crucially, the proxy is

$$
E_h(x)=\tfrac12J\nu^2+G(1-\cos\alpha),
$$

which omits the arm motion and kinetic coupling in $E$. The two quantities must retain different symbols. The ledger specifies the complete reward without conflating this proxy with mechanical energy.

Teacher trajectories initialize the actor by action regression and initialize the critics using approximate discounted teacher returns. Subsequent TD3 training mixes new transitions with retained demonstrations and retains an imitation term in the actor loss. Success does not terminate the continuing task, and a sampling timeout does not terminate value bootstrapping. Some reset categories have intention-based names that do not match their physical pendulum orientation; the evidence ledger records this qualification.

For critic index $j\in\{1,2\}$, the learned function $Q_j(o,a)$ approximates discounted future reward after action $a$ followed by the learned policy. Actor performance does not establish critic calibration or a rigorous bound on continuation cost.

## Learned continuation inside nonlinear MPC

At a decision time, $x_0$ is the observed state. The optimization variables are the normalized commands $\mathbf a=(a_0,\ldots,a_{N-1})$, where $N$ is the fixed horizon length and $i$ indexes predicted commands. Predicted states satisfy $x_{i+1}=\Phi_\Delta(x_i,a_i)$. The implemented objective is

$$
\min_{\mathbf a}\ \sum_{i=0}^{N-1}
\left[e(x_i)^{\mathsf T}W e(x_i)+Ra_i^2\right]+F(x_N),\qquad
F(x)=-\min_{j\in\{1,2\}}Q_j(o(x),\pi_\phi(o(x))).
$$

Here $W$ is a fixed positive diagonal state-weight matrix and $R=1$ is the normalized-input weight, taken from the existing discrete linear-quadratic regulator design. The function $F$ uses the lower twin reward estimate as a larger negative-reward terminal cost. It is not a certified conservative bound. The MPC sum has neither discounting nor time-step multiplication; the terminal coefficient is one. The critics were trained with another reward and a discount, so $F$ is a transferred continuation score rather than the exact value function of this quadratic stage cost.

Commands satisfy $|a_i|\le1$, and predicted arm positions must remain strictly within the envelope at the checked integration nodes. A finer independent integration audits the selected plan before execution. These finite-grid checks do not constitute a continuous-time or uncertainty guarantee. There is no terminal equality, terminal invariant set, or local-controller handoff.

The cross-entropy method (CEM) searches command sequences from actor-based or shifted-plan seeds and constant signed seeds. It updates a sampling distribution from low-cost feasible candidates and returns an accepted incumbent. Only the first command is applied before replanning. The stored plan and random sampling affect subsequent decisions, so the executed controller cannot be identified with a memoryless exact optimizer. The actor and terminal critics remain active after the pendulum reaches upright.

## What the evidence supports

Read-only reconstruction of all saved trajectory chunks confirms the following results. The 32 starts include four groups, with eight starts per group; not all starts require a downward swing-up.

| Quantity | Reconstructed result | Interpretation |
|---|---:|---|
| Complete sessions | 32, each 40 s | Evaluation continues after reaching upright. |
| Final ten seconds entirely within $\pm5^\circ$ | 32/32 | Finite-duration nominal upright-angle holding. |
| Mean total and longest continuous upright time | Both 37.7644 s | Under this scoring, each session stays in the band after its final acquisition. |
| Final-ten-second pooled upright-error RMS | $0.4846^\circ$ | Small pendulum error in the recorded tail. |
| Final-ten-second pooled arm RMS | $12.7478^\circ$ | Appreciable arm displacement remains. |
| Final-ten-second pooled arm-speed RMS | 0.2656 rad/s | The recorded arm continues moving. |

All saved decisions pass the declared numerical checks, but the arm envelope becomes nearly active. Recorded synchronized batch solve times also do not support a real-time claim. The portable actor package preserves separate 600-second evaluation summaries; those are actor-only results under another protocol and must not be presented as 600-second MPC results.

The current local transfer excludes comparison runs and complete training histories. It therefore supports the combined construction's nominal behavior, without isolating the effect of the terminal critic or establishing learning-curve claims. The holding-analysis documents report failed candidate invariant regions for the complete controller; their generated analysis records are not included locally. A three-page paper should present the formulation and observed holding, while leaving complete-controller stability, robust constraint satisfaction, and terminal-value ablation as explicit unresolved questions.
