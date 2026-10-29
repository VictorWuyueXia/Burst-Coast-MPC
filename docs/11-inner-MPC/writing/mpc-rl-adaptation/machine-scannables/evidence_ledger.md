# Technical evidence ledger for manuscript adaptation

Audit date: 2026-10-26. Paths are relative to the repository root. Line references identify the inspected checkout; future edits can move them. This document is the detailed source record for [current_progress.md](../current_progress.md) and [revision_plan.md](../revision_plan.md), not a new experiment report. No controller, network, or experiment settings were changed.

## Evidence status vocabulary

- **SOURCE:** inspected implementation or frozen configuration; establishes what is defined or executed, not empirical performance.
- **RECORD:** locally available saved numerical record, including independently reconstructed MPC metrics.
- **SUMMARY:** preserved aggregate result without the original trajectory/training archive locally available.
- **ANALYSIS REPORT:** documented analytic/numerical study whose generated analysis outputs are absent here.
- **OPEN:** not established by current source and available evidence.

## Source map

| ID | Status | Source and inspected lines | Supported content |
|---|---|---|---|
| M1 | SOURCE | `src/rotary_pendulum/environment/dynamics.py:29–62,83–105`; `environment/jax_dynamics.py:24–50` under the same module | Uniform-link constants and coupled nonlinear dynamics. |
| M2 | SOURCE | `src/rotary_pendulum/configs/physics.yaml`; portable network `machine-scannables/manifest.json:97–109` | Primitive physical parameters and zero nominal damping. |
| T1 | SOURCE | `src/rotary_pendulum/heuristic/energy.py:14–41`; `heuristic/decoder.py:17–88` | Mechanical energy encoding, work request, phase-dependent torque candidate selection. |
| T2 | SOURCE | `scripts/experiments/sustained-balance/run.py:57–78,113–127` | Demonstration teacher uses energy decoding, saturated local linear feedback, and predictive arm braking. |
| T3 | SOURCE | `scripts/experiments/stationary-balancing/prepare.py:26–29,70–106,139–145`; `src/rotary_pendulum/RL/jax_demonstrations.py:14–53` | Trajectory qualification/splitting, demonstration sampling, approximate return labels, actor/critic pretraining. |
| L1 | SOURCE | `src/rotary_pendulum/RL/jax_stationary_task.py:35–76,79–115` | Stationary reward, continuing control, genuine pre-reset next states with continuing bootstrap. |
| L2 | SOURCE | `src/rotary_pendulum/RL/jax_residual_task.py:22–45`; `jax_residual_control.py:21–24,118–127`; `jax_td3.py:17–43` in the same directory | Five physical features in stationary mode, model-derived scales, direct bounded action without runtime teacher/filter. |
| L3 | SOURCE | `src/rotary_pendulum/RL/jax_td3.py:82–184` | Persistent demonstration batches, noisy twin-critic TD target, critic loss, actor imitation, delayed target updates. |
| L4 | SOURCE | `src/rotary_pendulum/RL/jax_stationary_task.py:17–32`; `docs/9-RL/14-reinforcement-learning-prior/20-stationary-balancing/td_training_and_mpc.md:91` | Named targeted reset groups are centered around downward pendulum orientation. |
| L5 | RECORD | `artifacts/rotary_pendulum/final-results/rl-policy/machine-scannables/{manifest.json,weights.npz,selection.json,verification.json}` | Portable online actor/twin critics, normalization, chosen checkpoint, export parity metadata. |
| L6 | SUMMARY | Same portable package, `machine-scannables/population_performance.csv` and `human-readables/interpretation_summary.md` | Preserved 600-second actor evaluations, separate from MPC. |
| C1 | SOURCE | `docs/11-inner-MPC/human-readables/mpc_rl_baseline.md:3–41`; `scripts/experiments/inner-mpc/lqr_cost_run.py:69–81,127–173` | Current official MPC+RL baseline and no local handoff. |
| C2 | SOURCE | `src/rotary_pendulum/mpc/inner_model.py:59–72,111–130,174–199` | Quadratic stage branch, frozen terminal critics, finite-grid feasibility and refinement audit. |
| C3 | SOURCE | `src/rotary_pendulum/mpc/gpu_cem.py:20–28,48–108`; runner `lqr_cost_run.py:127–173,210–213` | CEM seed lanes, incumbent/elite update, receding execution, fail-loud rejection without a fallback controller. |
| R1 | RECORD | `artifacts/rotary_pendulum/final-results/rl-mpc/machine-scannables/session-records/` | Design, starts, 40 chunk archives, source snapshots, completion marker; all 400 decisions for 32 sessions. |
| R2 | RECORD | `artifacts/rotary_pendulum/final-results/rl-mpc/machine-scannables/{episodes.csv,summary.json,baseline_report.py}; final-results/machine-scannables/file_manifest.json` | Baseline-only integrity map, reconstructed episode/aggregate metrics and exact scoring. |
| R3 | SOURCE | `docs/11-inner-MPC/machine-scannables/lqr_stage_cost_contract.md:33–47` | Saved array semantics, synchronized batch timing, refinement and record conventions. |
| A1 | ANALYSIS REPORT | `docs/11-inner-MPC/holding-region/human-readables/actual_mpc_rl_holding.md:35–47,74–102,159–192` | Actual-network optimizer-branch analysis and rejection of a candidate physical-state holding ellipsoid. |
| A2 | ANALYSIS REPORT | `docs/11-inner-MPC/holding-region/human-readables/plan_memory_switching.md:57–133` | Conditional retained-plan certificate; candidate region fails at CEM plan replacements. |
| A3 | ANALYSIS REPORT | `docs/11-inner-MPC/11.1-swing_up_conditional_proof.md:5–7`; `proof-review/human-readables/proof_assessment.md:49–69,98–102` under `docs/11-inner-MPC/` | Swing-up reasoning relies on unverified progress/continuation hypotheses; accepted shift does not certify final-audit recursive feasibility. |

## Mechanical and task notation

The physical state is $x=(\theta,\alpha,\omega,\nu)^{\mathsf T}$, with arm angle $\theta$, downward-referenced pendulum angle $\alpha$, and signed speeds $\omega=\dot\theta$, $\nu=\dot\alpha$. Angular coordinates are radians, speeds are rad/s, and dots denote continuous-time derivatives. The arm angle is unwrapped, while pendulum angle is stored in $[-\pi,\pi)$. Generalized coordinates are $q=(\theta,\alpha)^{\mathsf T}$. Torque $u$ acts only on the arm. The nominal benchmark assumes ideal joints and zero damping.

Primitive physical inputs are arm/pendulum masses $m_a,m_p$, lengths $r,l$, and gravity $g$. The derived quantities used in the model are

$$
I_a=\frac{m_ar^2}{3},\quad J=\frac{m_pl^2}{3},\quad
J_0=I_a+m_pr^2,\quad b=\frac{m_prl}{2},\quad G=\frac{m_pgl}{2}.
$$

The first four quantities have inertia units; $G$ has energy units. Define $A(\alpha)=J_0+J\sin^2\alpha$ and $C(\alpha)=b\cos\alpha$, the diagonal and coupling coefficients of the inertia matrix. The exact nominal model is

$$
M(\alpha)=\begin{bmatrix}A&C\\C&J\end{bmatrix},\qquad
M(\alpha)\ddot q=
\begin{bmatrix}
u-2J\sin\alpha\cos\alpha\,\omega\nu+b\sin\alpha\,\nu^2\\
J\sin\alpha\cos\alpha\,\omega^2-G\sin\alpha
\end{bmatrix}.
$$

This is equivalent to the existing four-state vector field, with the first two components $\omega,\nu$. The coupled inertia solve retains the passive joint's dependence on arm torque.

For the current selected actor/MPC, define usable torque cap $U=0.00459$ N m, normalized command $a=u/U$, arm envelope $\theta_{\max}=3\pi/4$ rad, and action duration $\Delta=0.1$ s. These are frozen experiment choices, not newly tunable settings. The actuator rating in `physics.yaml` is 0.0204 N m; the original energy-controller default uses 0.01836 N m. Neither number is the current MPC torque cap.

Define the wrapped upright error $\beta=\operatorname{atan2}(-\sin\alpha,-\cos\alpha)$ and state error $e(x)=(\theta,\beta,\omega,\nu)^{\mathsf T}$. The prescribed evaluation tolerance is $\varepsilon=5\pi/180$ rad. Its criterion is $|\beta|\le\varepsilon$, without velocity or arm-centering conditions.

## Teacher energy and student reward are different objects

Teacher kinetic and potential energies are $K=\tfrac12\dot q^{\mathsf T}M\dot q$ and $V=G(1-\cos\alpha)$. With $K_a=\tfrac12I_a\omega^2$, $K_p=K-K_a$, and energy representation $z=H(x)=(K_a,K_p,V)^{\mathsf T}$, mechanical energy is $E=K+V$ and upright-rest energy is $E_\star=2G$. All have joule units. The lossless identity $\dot E=u\omega$ links shaft power to mechanical-energy change. The map $H$ is many-to-one, so it cannot replace the full state for phase-dependent prediction.

The original teacher requests work and scores numerical held-torque candidates. Work agreement is not a mandatory equality of its selected action. The augmented demonstration teacher additionally uses saturated local linear feedback near upright and a predictive arm-braking filter (T2). Qualification requires no recorded arm violation and a final ten-second upright hold. Dataset sampling balances several physical/label categories, and train/validation splitting uses whole episodes (T3). This qualification is a data selection rule, not a guarantee for every teacher initial state.

Student observation is

$$
o(x)=(\theta/\theta_{\max},\sin\alpha,\cos\alpha,
\omega/s_\omega,\nu/s_\nu),\qquad
s_\omega=\theta_{\max}\sqrt{\frac{J_0G}{J_0J-b^2}},\quad
s_\nu=2\sqrt{G/J}.
$$

The positive divisors $s_\omega,s_\nu$ are derived speed scales, approximately 30.59253235 and 21.36069107 rad/s. They are not state bounds. Exported divisors retain the saved floating-point values; do not regenerate them with changed precision or clip features silently. The actor $\pi_\phi$ has learned parameters $\phi$ and outputs $a\in[-1,1]$ through a hyperbolic tangent. The current direct actor applies torque $U\pi_\phi(o)$ without teacher or filter intervention.

For the stationary reward, define the hinge-swing energy proxy $E_h$, smooth upright preference $s(x)$, and arm penalty $p_\theta$ by

$$
E_h=\tfrac12J\nu^2+G(1-\cos\alpha),\qquad
s(x)=\exp\left[-\tfrac12\left(
\frac{\beta^2}{\sigma_\beta^2}+\frac{\nu^2}{\sigma_\nu^2}
+\frac{\omega^2}{\sigma_\omega^2}\right)\right],
$$

$$
p_\theta(\theta)=\left(\frac{\theta}{\theta_{\max}}\right)^2+
\left[\frac{\max\{|\theta|-0.8\theta_{\max},0\}}{0.2\theta_{\max}}\right]^2,
$$

$$
c(x,a)=w_s[1-s(x)]+w_E|E_h/E_\star-1|+w_u a^2+w_\theta p_\theta(\theta).
$$

The widths $\sigma_\beta,\sigma_\nu,\sigma_\omega$ are designer-chosen angle/speed scales; the nonnegative weights $w_s,w_E,w_u,w_\theta$ specify cost rates. The selected export uses widths $(0.16\ {\rm rad},0.4\ {\rm rad/s},0.3\ {\rm rad/s})$ and weights $(1,1,2,0.2)$ per second. Thus $c$ has reciprocal-second cost units and its time accumulation gives dimensionless cost. The proxy $E_h$ omits arm kinetic energy and coupling; it is not $E$.

For command index $k$, the reward is numerically accumulated at five post-step states $x_{k,j}$, with observation-sample interval $\delta=0.02$ s and sample index $j=1,\ldots,5$:

$$
r_k=-\delta\sum_{j=1}^{5}c(x_{k,j},a_k),\qquad
\mathbb E\left[\sum_{k=0}^{\infty}\gamma^k r_k\right],\qquad \gamma=0.995.
$$

The expectation is over sampled starts and trajectories during learning. The sum is the continuing discounted return, not a finite capture reward. Success does not terminate transitions. Sampling resets occur at 120 s, with the real pre-reset next observation retained and terminal flag false, so bootstrap continues (L1).

## Learning construction and retained checkpoint

Let $\mathcal B$ denote the mixed replay distribution, $\mathcal D$ the demonstration distribution, and $a^T$ a teacher action. For next observation $o'$, target actor parameters $\bar\phi$, target critic parameters $\bar\psi_j$, and clipped target smoothing noise $\epsilon$, define the target action $\tilde a=\operatorname{clip}(\pi_{\bar\phi}(o')+\epsilon,-1,1)$. Bars denote slowly updated training copies, not uncertainty bounds. With transition reward $r$ and discount $\gamma$, the implemented stationary TD target is

$$
y=r+\gamma\min_{j\in\{1,2\}}Q_{\bar\psi_j}(o',\tilde a).
$$

The online critics with parameters $\psi_1,\psi_2$ minimize mean squared error to $y$. Actor parameters minimize

$$
L_\pi=-\mathbb E_{\mathcal B}Q_{\psi_1}(o,\pi_\phi(o))+
\lambda\mathbb E_{\mathcal D}\left[(\pi_\phi(o)-a^T)^2\right].
$$

The coefficient $\lambda$ is a chosen imitation weight; the selected model uses 1000, with half each batch drawn from demonstrations. Actor and target updates occur every second critic update. Pretraining uses teacher-action regression and approximate teacher-return regression. The return labels append a constant tail estimated from the final-ten-second mean reward divided by $1-\gamma$; this is a heuristic continuing-return initialization, not exact policy evaluation (T3).

The selected portable 4.59 mN m model has three 256-unit hidden layers with SiLU activations, a scalar tanh actor output, and twin scalar linear critic outputs. Its selected checkpoint follows 524,288 online transitions; the training setting's 1,048,576-transition budget is not the chosen checkpoint time. Portable weights contain online actor/critic parameters, without optimizer or target states. Weight-file integrity was checked against the local manifest. Original `.msgpack`, complete demonstration data, and full campaign histories are not present locally.

The preserved 600-second comparison summary reports all 512 nominal actor episodes ending balanced, with 510/512 satisfying full-duration arm compliance and final uninterrupted holding. The corresponding disturbed count is 507/512. These are aggregate actor results, not reconstructed 600-second trajectories or MPC results. The weaker 2.295 mN m actor is a separate study and is not the current MPC export.

**Reset qualification.** `targeted_resets` creates groups named capture, braking, and recovery near $\alpha=0$, which is downward in the implemented dynamics. Their names do not prove near-upright reset coverage. Demonstration “capture” sampling separately uses upright error. Preserve this distinction when describing training populations or held-out tests (L4).

## Actual MPC problem and execution

For predicted command index $i=0,\ldots,N-1$, $x_0$ is the current observed state, $a_i$ are decision variables, and $x_{i+1}=\Phi_\Delta(x_i,a_i)$. The map $\Phi_\Delta$ integrates the nonlinear model over a held-command interval. The horizon is $N=10$, hence one second. The baseline stage weights are fixed by the existing discrete linear-quadratic design:

$$
W=\operatorname{diag}(\theta_{\max}^{-2},0.15^{-2},0.25,0.25),\qquad R=1.
$$

The angle scale 0.15 is radians; speed coefficients have inverse squared rad/s units. These weights normalize the quadratic stage cost. With frozen online critics $Q_1,Q_2$ and frozen actor $\pi_\phi$, the terminal score and optimization objective are

$$
F(x)=-\min_{j\in\{1,2\}}Q_j(o(x),\pi_\phi(o(x))),\qquad
J(x_0,\mathbf a)=\sum_{i=0}^{N-1}[e(x_i)^{\mathsf T}We(x_i)+Ra_i^2]+F(x_N).
$$

The command vector is $\mathbf a=(a_0,\ldots,a_{N-1})$. The MPC searches for a feasible vector with low $J$. There is no discount factor, integration-step multiplier, or additional terminal coefficient in this branch. The terminal critic is always present. The setting `rl_cost=false` selects the quadratic running cost; it does not disable $F$. Since the critic's reward differs from the MPC stage cost, standard matching-value or terminal-Lyapunov arguments cannot be asserted from this formula alone.

Constraints require $|a_i|\le1$, $|\theta|<\theta_{\max}$ at all checked forecast/audit nodes, finite values, and nominal/refined integration agreement. Prediction uses 1 ms RK4, audit/execution 0.5 ms, and saved executed samples 1 ms. There is no terminal state equality/set, speed bound, input-rate bound, or CEM-to-LQR switch. Numerical disagreement limits are $(0.001,0.001,0.01,0.01)$ in state-coordinate units.

CEM uses three seed lanes: actor rollout at the first decision or shifted previous plan with actor tail thereafter, plus constant normalized commands $+0.5$ and $-0.5$. The seed rollout uses the preserved 2 ms map. Each lane draws 1,024 samples, retains 32 elites, and performs 16 generations with initial normalized spread 0.5. Clipped Gaussian samples compete with an incumbent; elite componentwise means/spreads update the distribution. The selected plan is independently audited, the first 0.1 s command executes, and the procedure repeats. No globally optimal solution is established. If no plan passes the final check, the runner errors without executing an unapproved substitute.

The controller therefore depends on state, stored plan, and sampling history, rather than only a four-state exact minimizer. A shifted candidate does not alone prove recursive feasibility because the selected candidate must pass a separate final refinement audit. The archive's plan vector stores ten actions followed by a terminal epigraph scalar; that last scalar is not an eleventh action.

## Independent saved-record reconstruction

Read-only inspection verified all 70 payload entries in the MPC transfer manifest against byte counts and SHA-256 values. Forty chunk files concatenate to 400 decisions for each of 32 sessions, with 12,800 accepted decisions and 40,000 post-integration millisecond samples per session. The four initial groups have eight starts each. Scoring computes wrapped error at every saved sample, gives no initial-point credit, and does not bridge out-of-band gaps.

| Reconstructed metric | Value |
|---|---:|
| Duration per session | 40 s |
| Final-ten-second uninterrupted $\pm5^\circ$ holding | 32/32 |
| Mean total upright time | 37.7644375 s |
| Mean longest continuous upright time | 37.7644375 s |
| Individual total upright times | 34.642–40 s |
| Final-ten-second pooled upright-error RMS | 0.4846287416° |
| Final-ten-second upright-error extrema | −1.9475785°, +1.9366865° |
| Final-ten-second pooled arm-angle RMS | 12.7478102843° |
| Final-ten-second pooled arm-speed RMS | 0.2655934452 rad/s |
| Final-ten-second pooled pendulum-speed RMS | 0.0983533355 rad/s |
| Peak absolute executed arm angle | 134.999915255° |
| Minimum stored forecast/audit arm margin | 0.0000218073° |
| Maximum absolute normalized command | 1 |
| Synchronized whole-32-state batch solve median / p95 | 3.3920203 / 3.3930660 s |

Maximum stored refinement disagreements are $(3.4034\times10^{-5},1.04073\times10^{-4},4.41711\times10^{-4},1.35109\times10^{-3})$, with radians for the first two coordinates and rad/s for the last two. They pass the declared thresholds, without establishing continuous-time containment. The almost vanishing arm margin supplies no practical uncertainty reserve. Dividing batch solve time by 32 measures throughput rather than feedback latency; the archive does not support 100 ms real-time execution.

## Claim boundaries and unavailable evidence

| Proposed statement | Permitted status |
|---|---|
| The current construction achieves swing-up and continued small upright error on the recorded starts. | Supported nominal finite-duration demonstration; not every start is downward. |
| All physical states converge to upright centered rest. | Unsupported; substantial arm motion persists. |
| The learned terminal value is calibrated, necessary, or superior to an alternative. | OPEN; actor success and combined-controller success do not establish this. |
| The full sampled/history-dependent controller has a certified invariant holding region. | OPEN; reported candidate regions fail. |
| A numerical local optimizer branch has stable eigenvalues. | ANALYSIS REPORT; not a theorem for finite CEM execution. |
| Bounds hold continuously or robustly under model/measurement error. | Unsupported by the finite-grid nominal records. |
| MPC holds for 600 s. | Unsupported; that duration belongs to a separate actor-only summary. |
| The horizon or coasting schedule is learned. | Unsupported; both command duration and horizon are fixed here. |

The actual-holding report documents 73 exits from one proposed physical-state ellipsoid. The plan-memory report documents 57 exits from a proposed five-degree queue sublevel, all at CEM replacements; it also reports 345 replacements that lower the MPC objective while increasing that certificate. These reject specific candidate sets, rather than proving instability or excluding every possible invariant region. Corresponding generated analysis outputs are absent locally, so these counts remain ANALYSIS REPORT evidence.

The local transfer explicitly excludes tuning pilots and comparison archives. The linked `20261022-lqr-cost-confirm-summary/` and `mpc-rl-local-analysis/` experiment folders are absent here. Do not present their historical comparisons as independently reconstructed evidence. Complete learning curves, terminal-cost ablations, critic return-calibration tests, robust uncertainty studies, and a complete-controller proof remain outside the verified paper basis.
