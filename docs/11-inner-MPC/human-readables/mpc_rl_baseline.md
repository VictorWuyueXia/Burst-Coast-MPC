# MPC+RL: official development baseline

Effective 2026-10-23, **MPC+RL** names the latest nonlinear CEM MPC with the exact LQR stage weights and the frozen TD3 terminal value. It is the official baseline for subsequent development. This promotion changes documentation and baseline status only; no controller behavior changes and no new experiments are run.

## Frozen settings and evidence

The authoritative numerical record is [the completed 32-start confirmation design](../../../artifacts/rotary_pendulum/final-results/rl-mpc/machine-scannables/session-records/design.json). Keep its physics, observation encoding and checkpoint unchanged when making baseline comparisons. The [machine contract](../machine-scannables/lqr_stage_cost_contract.md) specifies the execution and record details.

| Item | Official setting |
|---|---|
| Solver | GPU CEM; 1,024 samples per seed, 32 elites, 16 generations, initial normalized-input standard deviation 0.5 |
| Seeds | Three: actor/shifted previous plan with actor tail, constant +0.5, constant −0.5 |
| Horizon / execution | 10 commands × 0.1 s = 1 s prediction; execute first 0.1 s and replan |
| Integration | 1 ms prediction, 0.5 ms independent verification; 1 ms saved trajectory |
| Constraints | Normalized torque magnitude ≤1; arm magnitude <135° at checked nodes; finite values and integration audit |
| Terminal value | Frozen TD3_20261016 twin critics and actor; unit multiplier; always enabled |
| Local handoff | None: CEM and the RL terminal value remain active after capture |
| Evaluation | Full 40 s sessions; total and maximum continuous time within ±5° of upright; no performance-based early stopping |

Let the measured physical state be x=(theta, alpha, omega, nu): arm angle, pendulum angle measured from downward, arm speed and pendulum speed, in radians and radians/s. Define the wrapped upright error beta and error vector e. The arm limit theta_max=135π/180 rad and torque limit u_max=0.00459 N m are fixed physical limits in this benchmark; a=u/u_max is normalized commanded torque. The fixed designer-chosen state weights W and input weight R are exactly those of the existing LQR:

$$
\beta=\operatorname{atan2}(-\sin\alpha,-\cos\alpha),\quad
 e=(\theta,\beta,\omega,\nu)^\mathsf T,\quad
 W=\operatorname{diag}(\theta_{\max}^{-2},0.15^{-2},0.25,0.25),\quad R=1.
$$

The angle scales are in radians; speed weights have inverse squared rad/s units. For N=10 predicted commands, k indexes command intervals, e_k is computed by the nonlinear rollout, and x_N is the terminal state. The saved actor pi and reward critics Q_1,Q_2 receive their original observation encoding. The minimized cost J is

$$
J=\sum_{k=0}^{N-1}(e_k^\mathsf TWe_k+Ra_k^2)+V_{\rm RL}(x_N),\qquad
V_{\rm RL}(x)=\max_{i\in\{1,2\}}[-Q_i(x,\pi(x))].
$$

There is no time-step multiplication, no discounting, and no extra weak angle/speed term. The quadratic angle/speed terms above remain part of the LQR stage cost. The original RL training reward differs from this stage cost; promotion does not assert that the frozen critic is its exact value function.

## What actually happens after capture

[The runner](../../../scripts/experiments/inner-mpc/lqr_cost_run.py) calls CEM at every decision and has no local-mode or stabilization-dwell branch. [The objective implementation](../../../src/rotary_pendulum/mpc/inner_model.py) always adds the terminal critic cost. The archived `rl_cost=false` selects the quadratic running-cost branch; it **does not disable the RL terminal term**. The separate historical `hybrid_run.py` implements CEM→LQR and is not the official MPC+RL baseline.

The [completed evidence](../../../artifacts/rotary_pendulum/inner-mpc/20261022-lqr-cost-confirm-summary/interpretation_summary.md) gives mean total and longest hold of 37.764 s over 32 full sessions. Residual arm motion, near-boundary trajectories and solve latency remain limitations. “Capture” here means upright-band holding, not convergence of the full state to zero.

## Can Riccati handle local keeping?

Yes, for a separately defined local linear quadratic problem. Linearize the 0.1 s discrete nonlinear map at zero arm angle, upright pendulum and zero speeds/input. Its state and normalized-input Jacobians are A and B, giving e_(k+1)=Ae_k+Ba_k locally. For the same W and R, the stabilizing discrete algebraic Riccati solution P and feedback gain K satisfy

$$
P=W+A^\mathsf TPA-A^\mathsf TPB(R+B^\mathsf TPB)^{-1}B^\mathsf TPA,
\qquad K=(R+B^\mathsf TPB)^{-1}B^\mathsf TPA,\qquad a=-Ke.
$$

P is computed once from A,B,W,R under the usual stabilizability/detectability conditions; it represents the infinite-horizon quadratic continuation cost. K maps measured local error to normalized input. This is how the existing [LQR was constructed](lqr_handoff.md).

Removing the RL terminal term alone leaves a finite-horizon problem. For a linear unconstrained model its backward Riccati recursion starts at terminal matrix P_N=0, generally giving a different gain. Replacing the RL terminal term by e_NᵀPe_N instead recovers the infinite-horizon LQR feedback in the linear unconstrained case. Ordinary Riccati does not exactly solve the current nonlinear rollout with a neural terminal cost and active hard constraints.

A future Riccati keeping mode therefore needs an explicit handoff region where the local approximation is adequate and predicted inputs/states remain feasible, preferably an invariant feasible region, plus a specified return rule. Clipping the LQR input is not an exact constrained optimum. The previous local probe found feasible LQR-generated plans with lower objective than CEM in eight sampled terminal states; this motivates investigation but proves neither global optimality nor safe switching. No such mode or terminal-cost switch is introduced by this promotion.
