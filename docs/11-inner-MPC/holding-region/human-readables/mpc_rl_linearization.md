# MPC+RL local linearization and a holding-set construction

Date: 2026-10-25. Baseline: `cdce8f93f98d6c978e6a723d4aea11f4524c59ad`. The user authorized explicitly assumed neural features because the original checkpoint is unavailable. All network-dependent numerical results below belong to a hypothetical local MPC model, not to the missing network or the executed CEM controller.

## Evidence and the operating region

Both remote branches match the local refs: `dev` is the baseline above and `main` is `285d980796fee2a701f22e7f3d2af6204aa7929d`. Inspection found no latest checkpoint or MPC trajectory archive in 90 locally reachable revisions or any of 929 local Git trees, including unreachable trees. The directory `artifacts/rotary_pendulum/` is ignored. Earlier tracked rotary NPZ records belong to the heuristic controller, not MPC+RL.

The [committed experiment discussion](../../human-readables/lqr_stage_cost.md) reports:

| Reported item | What it supports |
|---|---|
| All 32 confirmation sessions hold upright after capture over 40 s; mean longest hold 37.764 s | Finite-session angle holding, without complete-state convergence bounds |
| Roughly 20° pilot arm RMS | A small centered-arm holding set is not justified by the existing prose; the original summary is needed to establish the exact window and value |
| Stationary zero-action family prefers about +2.5° arm angle, with about 0.00557 lower objective than at zero | A possible terminal arm-direction bias, under a local parabolic approximation |
| Some confirmation intervals approach 135° arm angle | Transient boundary proximity; it does not establish the eventual oscillation amplitude |
| Eight feasible LQR-generated plans have lower objective than saved CEM plans | The executed CEM plans cannot simply be treated as exact optimizers |

The report implementation computes final-ten-second arm and upright RMS, but the generated `summary.json` is missing. No reliable four-state oscillation envelope, center, peak amplitude, period, or invariant set can be recovered from prose alone. In particular, RMS is neither peak amplitude nor a region boundary. The earlier 3.785° centered-arm example must not be called empirically supported.

## Plant, neural, ideal-MPC, and CEM linearizations

Let physical error be arm angle, wrapped pendulum upright error, arm speed, and pendulum speed:

$$
e=(\theta,\beta,\omega,\nu)^{\mathsf T},\qquad
\beta=\operatorname{atan2}(-\sin\alpha,-\cos\alpha).
$$

Angles use rad and speeds rad/s; pendulum angle from downward is denoted by the final symbol inside the trigonometric functions. Linearizing the MPC prediction map over one 0.1 s command gives

$$
e^+=Ae+Ba,\qquad a=u/u_{\max},\quad u_{\max}=0.00459\ {\rm N\,m}.
$$

The state matrix and normalized-input column are the same plant derivatives used by a local LQR if the operating point and discretization match. They are not closed-loop controller Jacobians.

For a differentiable exact MPC policy, denoted by a star, the closed-loop Jacobian is instead

$$
A_{\rm cl}=g_e+g_aD_e\mu^\star,
$$

where the two flow derivatives are evaluated at the operating state and selected action. The symbol for the nonlinear one-command flow is the same one used in the holding theorem. A neural actor derivative is not the derivative of this optimizer policy.

The terminal function used by the objective is

$$
F(e)=\max_{j=1,2}C_j(e),\qquad
C_j(e)=-Q_j(o(e),\pi(o(e))),
$$

with original scaled observation, frozen actor, and frozen reward critics. Its branch gradient includes the actor derivative, critic input derivatives, and physical observation map. If one branch is uniquely active locally, its value, gradient, and Hessian give a smooth Taylor expansion. At a branch tie there is generally no unique ordinary gradient or Hessian of the maximum; retain both branches.

## A reasonable, explicit neural assumption

Write the assumed terminal Taylor model around upright rest as

$$
F(e)\approx F_0+c^{\mathsf T}e+\tfrac12e^{\mathsf T}He.
$$

The scalar offset does not affect optimization. The gradient vector and symmetric curvature matrix are missing network properties, not controller tuning parameters. Assume one active critic branch, zero cross-curvatures, and positive curvature in the four error directions. These assumptions are plausible local features to investigate, not findings about the saved weights.

Use the recorded static-arm probe to constrain two of them. Let the approximate preferred arm angle be 2.5° and the corresponding objective drop be 0.00557. For ten zero commands at stationary upright, only the arm stage term changes. With the existing state-weight matrix and command count,

$$
W=\operatorname{diag}((135\pi/180)^{-2},0.15^{-2},0.25,0.25),\qquad N=10,
$$

the parabolic fit is

$$
\theta_s=2.5\pi/180,\quad D_s=0.00557,\quad
v_s=\frac{2D_s}{\theta_s^2},\quad
c_\theta=-v_s\theta_s,\quad H_{\theta\theta}=v_s-2NW_{\theta\theta}.
$$

The preferred angle and drop are approximate documentation-derived values; the curvature with the subscript s is the curvature of the combined static objective, not the neural function alone. Complete the unmeasured directions by the explicit assumption that their terminal curvatures equal twice their stage weights. This yields

$$
c\approx(-0.255309994,0,0,0)^{\mathsf T},\qquad
H\approx\operatorname{diag}(2.248743064,88.888888889,0.5,0.5).
$$

The arm-direction coefficients are inferred only under the parabolic fit; all other entries are assumptions. No weights have been fabricated, and this quadratic model is not claimed to reproduce a real neural network over a finite region.

## Derive feedback from the MPC objective

Use the actual ten-command, undiscounted local objective, with the linear prediction model and assumed neural terminal terms:

$$
\min_{a_0,\ldots,a_9}\sum_{i=0}^{9}(e_i^{\mathsf T}We_i+a_i^2)
+c^{\mathsf T}e_{10}+\tfrac12e_{10}^{\mathsf T}He_{10},\qquad
e_{i+1}=Ae_i+Ba_i.
$$

Near an interior optimum this finite quadratic problem has an affine solution. It is computed with backward recursion whose terminal conditions come from the neural model, not from an infinite-horizon LQR value:

$$
\Pi_{10}=H/2,\qquad p_{10}=c/2.
$$

The matrix and vector are the quadratic and linear value coefficients at each prediction index. For indices from nine down to zero, define the positive input curvature and affine action coefficients by

$$
s_i=1+B^{\mathsf T}\Pi_{i+1}B,\qquad
M_i=-s_i^{-1}B^{\mathsf T}\Pi_{i+1}A,\qquad
d_i=-s_i^{-1}B^{\mathsf T}p_{i+1},
$$

$$
\Pi_i=W+A^{\mathsf T}\Pi_{i+1}A+A^{\mathsf T}\Pi_{i+1}BM_i,\qquad
p_i=(A+BM_i)^{\mathsf T}p_{i+1}.
$$

The optimized stage action is the matrix coefficient times the predicted error plus the scalar offset. Completing the square in that action proves these expressions. The executed receding-horizon policy of this local model is therefore

$$
\mu_{\rm loc}(e)=M_0e+d_0,\qquad A_M=A+BM_0.
$$

The CPU calculation uses the MPC predictor's 100 one-millisecond RK4 linearized substeps. An independent equality-constrained quadratic-program KKT solve agrees with the first-action derivative to about 1.1e-14. The result is

$$
M_0\approx(0.058626718,-15.248753480,0.114887544,-1.210727424),
\qquad d_0\approx-0.003404714.
$$

The computed eigenvalues of the resulting state matrix are approximately 0.189524, 0.280134, 0.653950, and 0.935830. They differ from those of the previous LQR controller. The affine equilibrium is

$$
\bar e=(I-A_M)^{-1}Bd_0\approx(0.058074439,0,0,0)^{\mathsf T}.
$$

Its arm offset is about +3.32742°, not the +2.5° minimum of the restricted zero-action family. The full MPC optimizes future torques, so those two centers need not agree.

## Holding set derived from this MPC model

Center the error at the computed MPC equilibrium and construct a metric from its closed-loop matrix:

$$
z=e-\bar e,\qquad P_M-A_M^{\mathsf T}P_MA_M=W,\qquad
P_M=\sum_{j=0}^{\infty}(A_M^j)^{\mathsf T}WA_M^j\succ0.
$$

The series converges when the closed-loop matrix is Schur. The floating-point eigenvalue calculation supports that condition for this illustrative model; these numbers are not an outward-rounded implementation certificate. This construction uses an ordinary Lyapunov equation for the derived MPC feedback, not the earlier LQR Riccati matrix.

For the exact affine model, the quadratic function satisfies

$$
L_M(z)=z^{\mathsf T}P_Mz,\qquad
L_M(z^+)-L_M(z)=-z^{\mathsf T}Wz<0\quad(z\ne0).
$$

Thus any sublevel ellipsoid on which the assumed local optimizer remains feasible is invariant and attracts trajectories to its center. The computed metric and norm contraction are

$$
P_M\approx\begin{bmatrix}
2.904557&-12.515994&0.838056&-1.170490\\
-12.515994&362.112416&-16.228388&27.641621\\
0.838056&-16.228388&1.203581&-1.509121\\
-1.170490&27.641621&-1.509121&2.736227
\end{bmatrix},\qquad
q_M=\|P_M^{1/2}A_MP_M^{-1/2}\|_2\approx0.981554422.
$$

The illustrative outer radius 0.78 stays below the local model's input-feasibility radius 0.784751. Arm constraints are looser, with a computed radius 3.488023. These bounds inspect every action, the current arm state, and all 1 ms predicted arm nodes of the affine plan family; they apply only to that hypothetical linear quadratic model.

For actual MPC+RL, define the physical successor residual about this MPC-derived reference, retaining controller memory:

$$
r(z,h)=g_{\Delta}(\bar e+z,\mu(\bar e+z,h))-\bar e-A_Mz.
$$

If accepted continuation and compatible memory closure hold throughout the outer domain, and the supremum of this residual in the new metric is at most the following budget, the previous set-distance proof applies:

$$
\mathcal D=\{e:(e-\bar e)^{\mathsf T}P_M(e-\bar e)\le0.78^2\},\qquad
\Omega=\{e:(e-\bar e)^{\mathsf T}P_M(e-\bar e)\le0.75^2\},
$$

$$
\sup_{e\in\mathcal D,h\in\mathcal H}\|r(e-\bar e,h)\|_{P_M}
\le(1-q_M)0.75\approx0.013834184.
$$

The permitted memory set must be specified. The proof gives contraction of the positive part of the norm minus 0.75 and therefore Lyapunov stability of the inner set. The illustrative inner coordinate bounds are approximately:

| Coordinate | Inner ellipsoid bound |
|---|---:|
| Arm angle | +3.327° center, with ±28.312° deviation |
| Pendulum upright error | ±4.797° |
| Arm speed | ±1.27610 rad/s |
| Pendulum speed | ±1.09244 rad/s |

This scale is less inconsistent with the documented pilot arm motion than the earlier tiny centered-arm example, but it is not a measured or certified oscillation envelope. The actual residual budget is unverified. The bounds are at command instants; continuous holding additionally requires the inter-command tube check.

## Why the actual CEM linearization needs memory and switching

Let the previous ten-command plan be the continuous memory vector. At decisions after initialization, its shifted first coordinate is the previous second command; the other two seed lanes start at constants ±0.5. All seed-lane initial spreads are fixed at 0.5.

Fix that previous plan, random draws, incumbent choices, elite memberships, clipping masks, and final acceptance/seed-selection branch. Every candidate first coordinate is then formed only from the preceding first-coordinate mean/spread and fixed random numbers. Refitting uses the same coordinate of fixed elite members. Induction over CEM generations gives

$$
D_e\mu_{\rm CEM}(e,h)=0
$$

inside such a branch after warm-start initialization. The actor-dependent appended tail affects a different plan coordinate; there is no smooth coordinate mixing in CEM's mean and standard-deviation updates. The physical state affects costs and therefore selection boundaries, where this branch derivative need not apply or exist. At the first decision, the actor-rollout seed depends directly on state, so the zero-derivative statement does not apply.

Consequently, neither an actor Jacobian nor an exact-optimizer sensitivity is automatically the actual CEM policy Jacobian. A smooth branch analysis must retain the four physical states and ten plan coordinates. If the selected plan map is denoted by a calligraphic letter and the previous plan by a lowercase vector, the 14-dimensional continuous block is

$$
\begin{bmatrix}\delta e^+\\\delta p^+\end{bmatrix}
=\begin{bmatrix}
g_e+g_a\mu_e&g_a\mu_p\\
\mathcal P_e&\mathcal P_p
\end{bmatrix}
\begin{bmatrix}\delta e\\\delta p\end{bmatrix}.
$$

The delta symbol denotes a first-order perturbation; subscripts denote partial derivatives on the fixed branch. Random-key/index changes give additional discrete switching. A plant-like physical-state partial derivative is not evidence that the complete memory-dependent controller is unstable. Conversely, a stable exact-MPC Jacobian alone does not certify this implemented solver.

The assumed smooth MPC model converges to a biased equilibrium and does not generate persistent oscillations. Solver randomness, selection switching, critic seams, nonlinear behavior, or combinations of them remain candidate explanations. The reported behavior does not establish any one cause. Actual records should determine the center and envelope before a final holding region is chosen.

Checkpoint decoding, saved-record formats, planned sensitivity code, validation results, and reproduction paths are in the [implementation plan](../machine-scannables/mpc_rl_linearization_contract.md).
