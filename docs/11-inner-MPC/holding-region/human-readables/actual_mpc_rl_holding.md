# Actual-network MPC+RL linearization and holding-certificate audit

**Follow-up:** the [plan-memory switching audit](plan_memory_switching.md) constructs a weighted one-step queue certificate and finds actual CEM replacements violating its proposed five-degree region. It resolves the replacement test left open below without changing the controller.

Date: 2026-10-25. Controller source: `cdce8f93f98d6c978e6a723d4aea11f4524c59ad`; authenticated network and records supplied by `1ebd1fa3f25224edce50e77567f3d25767500d4b`. No full closed-loop experiment or GPU run was performed. The original controller is unchanged.

**Result:** actual neural derivatives and local nonlinear MPC sensitivities replace the assumed model. A smooth local optimizer branch is numerically stable near an 8-degree arm offset. Its proposed quadratic holding ellipsoid is not invariant for the implemented CEM controller: 73 saved transitions leave it. A second, memory-dependent linearization describes the frequently retained warm-start branch. Neither branch alone certifies the complete switching controller.

## What the saved trajectories establish

The archive contains 32 complete 40-second sessions and 12,800 accepted decisions. Final-ten-second pooled ranges and RMS values are:

| Coordinate | Minimum | Maximum | RMS |
|---|---:|---:|---:|
| Arm angle, degrees | -45.2133 | 22.0526 | 12.7478 |
| Pendulum upright error, degrees | -1.94758 | 1.93669 | 0.484629 |
| Arm speed, rad/s | -1.19349 | 1.13724 | 0.265593 |
| Pendulum speed, rad/s | -0.553012 | 0.654992 | 0.0983533 |

The mean arm angle pooled across this window is about -7.085 degrees. Individual session mean arm angles range from -26.864 to +2.714 degrees. Comparing each session's 30–35 s and 35–40 s means gives arm changes as large as 21.629 degrees. These records show upright-angle holding with considerable arm motion, not convergence to one observed stationary full-state center. They do not identify a common periodic orbit.

An enclosing box fitted to the pooled 30–35 s states is exceeded by 291 later millisecond samples across five sessions. Even an observed box therefore needs independent verification before it can be called invariant.

## Actual neural derivatives

Define physical error in the order

$$
e=(\theta,\beta,\omega,\nu)^{\mathsf T},\qquad
\beta=\operatorname{atan2}(-\sin\alpha,-\cos\alpha).
$$

The first coordinate is arm angle from center, the second is pendulum error from upright, and the last two are their angular velocities. Angles are radians and velocities rad/s. The pendulum angle from downward is the symbol inside the sine and cosine.

The actual frozen terminal function is

$$
F(e)=\max_{j=1,2}C_j(e),\qquad C_j(e)=-Q_j(o(e),\pi(o(e))).
$$

The observation map uses the archived scaling. The frozen actor and twin critics are the authenticated arrays; differentiation includes actor substitution and the observation map. At upright rest the second negative critic is active, separated from the first by about 0.17236 cost units. The actual gradient there is

$$
\nabla F(0)\approx(-0.257239,-6.238173,0.323349,-0.459237)^{\mathsf T}.
$$

The terminal Hessian eigenvalues are approximately -0.84863, 0.55669, 2.45001 and 2183.53480. Thus the assumed positive diagonal curvature and zero non-arm gradients were incorrect. This does not imply instability of MPC: its running cost and optimizer sensitivity also enter the closed loop.

## Nonlinear MPC optimizer sensitivity

The prediction problem uses ten normalized torque commands of 0.1 s, the recorded 0.00459 N m torque cap, 1 ms RK4 prediction, the existing state weights, both terminal epigraph inequalities, and all 1,000 predicted arm nodes. Execution uses the original 0.5 ms refinement. Multiple shooting introduces explicit predicted states but is an equivalent representation of the same nominal optimization problem.

Let the decision vector contain ten commands, one terminal epigraph scalar, and ten four-dimensional predicted endpoint states. Denote it by a lowercase vector; let the active constraint vector, multipliers and Lagrangian be

$$
y=(a_0,\ldots,a_9,t,e_1,\ldots,e_{10}),\quad
c_A(y,e)=0,\quad \lambda_A,\quad
\mathcal L(y,\lambda_A;e)=J(y,e)+\lambda_A^{\mathsf T}c_A(y,e).
$$

The subscript A denotes the active constraints, including dynamics. The current measured error is the parameter. At a regular stationary branch, the implicit derivative solves

$$
\begin{bmatrix}
\mathcal L_{yy}&(c_A)_y^{\mathsf T}\\
(c_A)_y&0
\end{bmatrix}
\begin{bmatrix}D_ey^\star\\D_e\lambda_A^\star\end{bmatrix}
=-\begin{bmatrix}\mathcal L_{ye}\\(c_A)_e\end{bmatrix}.
$$

Subscripts denote partial derivatives. The first row of the decision sensitivity gives the first-action state derivative, denoted below by the feedback row. No LQR gain is substituted.

One continuation from upright reaches a numerical equilibrium of this local optimizer branch at

$$
e_M\approx(0.139588805,0,0,0)^{\mathsf T},\qquad
\theta_M\approx7.99785^\circ,
$$

with first action about 2.72e-9. Both input bounds and arm bounds are inactive there; only the second terminal epigraph inequality is active in addition to dynamics. The reduced Lagrangian Hessian has minimum eigenvalue about 0.61776, and the KKT stationarity residual is about 3.3e-13. The derivative is

$$
K_M=D_e\mu^\star(e_M)
\approx(0.044370877,-15.163537490,0.109519876,-1.202770923).
$$

Let the refined nonlinear command map be denoted by the function below. Its derivatives are evaluated at the numerical equilibrium and selected action:

$$
e^+=g(e,a),\qquad A_M=g_e+g_aK_M
\approx\begin{bmatrix}
1.00392167&-1.02346403&0.10967978&-0.09630670\\
0.00424798&0.51644994&0.01048522&0.01541146\\
0.08223261&-20.90175641&1.20297334&-1.91235037\\
0.09657271&-10.99292095&0.23836876&-0.64963923
\end{bmatrix}.
$$

Its eigenvalues are approximately 0.189524, 0.280135, 0.652396 and 0.951651. An independent directional finite difference agrees with the implicit policy derivative to 2.3e-10. Reconstructed recorded objective and execution errors are about 3.0e-11 and 3.7e-14 at the checked point.

These are local stationary-optimizer branch results. At one recorded point a local solver reached a higher-cost basin than the CEM incumbent; that result is retained and not described as a global optimum. At another point, cost fell from the recorded 5.09326 to 2.25963. A single ideal optimizer branch cannot be identified with every finite-CEM decision.

## Lyapunov construction and an explicit rejected holding region

Use the existing positive state-weight matrix and construct the metric from the actual-network MPC branch:

$$
W=\operatorname{diag}((135\pi/180)^{-2},0.15^{-2},0.25,0.25),\qquad
P-A_M^{\mathsf T}PA_M=W.
$$

For a Schur matrix the exact solution is positive definite. The numerical metric is

$$
P\approx\begin{bmatrix}
3.18844158&-14.80822925&0.97352041&-1.38415264\\
-14.80822925&379.52993258&-17.27057534&29.26584854\\
0.97352041&-17.27057534&1.26578348&-1.60630039\\
-1.38415264&29.26584854&-1.60630039&2.88769044
\end{bmatrix}.
$$

Define the centered error and quadratic function by

$$
z=e-e_M,\qquad V(e)=z^{\mathsf T}Pz.
$$

For the reference linear map, the norm contraction is about 0.983251. For the smooth nonlinear optimizer branch, local asymptotic stability follows from the usual linearization theorem once an exact regular equilibrium and Schur bounds are established. The present root, derivative and eigenvalues are numerical evidence, not interval-certified existence bounds or a numerical region of attraction.

The local nonlinear implication can be written constructively. Let the centered smooth optimizer successor and its Taylor remainder be

$$
G_M(z)=g(e_M+z,\mu^\star(e_M+z))-e_M=A_Mz+\eta(z).
$$

At an exact regular equilibrium the remainder and its first derivative vanish at zero. On a ball inside the same optimizer branch, twice continuous differentiability gives

$$
\|\eta(z)\|_P\le c_r\|z\|_P^2,\qquad
c_r=\tfrac12\sup_{\|v\|_P\le r}\|D^2G_M(v)\|_{P\leftarrow P,P}.
$$

The induced bilinear norm takes the supremum over two unit input vectors in the same metric. The scalar is a derivative bound to compute, not a tuning change. If the ball is feasible, remains within the regular active set, and

$$
q_M+c_rr<1,\qquad q_M=\|P^{1/2}A_MP^{-1/2}\|_2,
$$

then the triangle inequality gives

$$
\|G_M(z)\|_P\le(q_M+c_rr)\|z\|_P.
$$

This proves positive invariance and exponential stability of that ball for the smooth optimizer branch. A sufficiently small positive radius exists under the exact regularity and Schur hypotheses because the second derivative is locally bounded. No numerical bound on this derivative or active-set neighborhood has been established here. For finite CEM, the successor differs from this smooth branch, so this proof cannot be transferred without an additional uniformly verified solver/history residual.

To guarantee a five-degree pendulum coordinate bound with this ellipsoid shape, its level cannot exceed

$$
R_5=\frac{(5\pi/180)^2}{(P^{-1})_{22}}\approx0.611266876,\qquad
\Omega_5=\{e:V(e)\le R_5\}.
$$

The inverse-matrix entry refers to the pendulum coordinate, second in the declared order. This construction is tested against actual accepted CEM transitions, not only against the ideal optimizer.

**It fails invariance.** There are 73 exits in the final ten seconds. For zero-based episode 7, decision 375 (37.5 to 37.6 s),

$$
e_k=(0.04506904,0.01555905,-0.22232230,0.02036381)^{\mathsf T},\quad
a_k=-0.1000419974,
$$

$$
e_{k+1}=(0.01913080,0.02371423,-0.28913248,0.16506730)^{\mathsf T},
$$

$$
V(e_k)\approx0.426495444<R_5<1.270932712\approx V(e_{k+1}).
$$

The full-precision states, center, metric and plan memory are saved. Exact rational arithmetic on the archived floating-point values verifies the strict inequalities; independent refined execution reproduces the successor. Thus this is a concrete counterexample to this ellipsoid's invariance under an actual reachable controller history. Both pendulum angles remain within five degrees: leaving the ellipsoid is not the same as losing upright-angle holding.

Across the 3,200 tail transitions, 1,433 increase this quadratic function. Even the best sampled scalar inequality of the form below has residual floor at least 2.948944283:

$$
V(e_{k+1})\le\rho V(e_k)+\epsilon,\qquad
0\le\rho<1,\quad\epsilon\ge0,\qquad b=\frac{\epsilon}{1-\rho}.
$$

The symbols are a candidate contraction factor, residual bound, and ultimate level. The best sample fit occurs at zero contraction factor and reduces to the archive bound on the next value. It is not a useful Lyapunov proof. Blocks of 2, 5, 10, 20 and 50 commands do not improve the sampled floor with this same metric. That floor would allow an ellipsoid pendulum extent near 10.98 degrees, above the intended five-degree bound. Other certificate shapes are not excluded by these failures.

## The actual retained-plan branch

In 2,531 of 3,200 tail decisions, the selected first nine commands exactly equal the shifted previous commands. This is a direct role for controller memory, not an actor substitution. The retained update appends the actual actor evaluated at the forecast after the remaining nine commands, using the original 2 ms seed integration.

The actor-rest point of this smooth update is approximately

$$
e_\pi=(-0.167391443,0,0,0)^{\mathsf T},\qquad
\theta_\pi\approx-9.59082^\circ,
$$

$$
D_e\pi(e_\pi)\approx(0.13825749,-20.53834702,0.23684785,-1.62250415).
$$

For physical state plus the previous ten-command plan, the exact retained-update Jacobian has dimension 14. Using the original 2 ms tail prediction and 0.5 ms execution maps, its computed spectral radius is about 0.984331241. Matrix algebra was checked at 70-digit precision because raw predictor coordinates are badly conditioned; neural derivatives remain decoded float64 values. This is a numerical branch result, not a common certificate over random selections.

The direct actor reference has an eigenvalue near -0.98693, consistent with a slowly damped alternating component. It does not establish a unique oscillation frequency. Saved action spectral peaks vary from 0.9 to 5 Hz across sessions. The different optimizer and retained-update centers help explain why using only the ideal optimizer's center can be misleading, without identifying a complete causal mechanism for the observed motion.

## A precise memory-aware proof target

A useful candidate must constrain the whole queued trajectory. For a common nominal command map, let the selected current plan be the ten commands, and define its predicted states and a local actor certificate by

$$
\xi_0=e,\quad \xi_{i+1}=f(\xi_i,a_i),\quad i=0,\ldots,9,\qquad
L_\pi(v)=(v-e_\pi)^{\mathsf T}P_\pi(v-e_\pi).
$$

Here the matrix is positive definite and must be certified under the actual actor; it is not the previous optimizer metric. If its actor closed loop contracts by a factor strictly below one on a specified domain, define

$$
0<\rho_\pi<1,\qquad
L_\pi(f(v,\pi(v)))\le\rho_\pi L_\pi(v),\qquad
\mathcal V(e,U)=\max_{i=0,\ldots,10}L_\pi(\xi_i).
$$

The plan vector is denoted by the capital letter. If every update retains the shift and actor tail, the predicted-state list shifts left and appends one actor-controlled successor. Therefore its maximum cannot increase, and after eleven retained updates every original list entry is replaced by an actor successor, giving

$$
\mathcal V_{k+1}\le\mathcal V_k,\qquad
\mathcal V_{k+11}\le\rho_\pi\mathcal V_k.
$$

This proves invariance and block contraction of queued-state sublevel sets for that common-map retained update. It is a constructive branch theorem with a defined role for the actor and plan memory.

For the complete existing controller, a new CEM-selected plan must also be shown to preserve the queued-state sublevel set or satisfy a verified block bound. The critic affects these selections; lower critic-guided objective does not establish this maximum-certificate condition. The different implemented integration maps add another residual to bound. Neither obligation is verified by the retained-branch eigenvalues, and neither is silently added as a new acceptance rule.

**Conclusion:** the assumptions have been replaced by actual neural and trajectory evidence. The four-state optimizer-based holding ellipsoid is disproved for the saved CEM behavior. A local optimizer branch and a frequently exercised memory branch have numerically stable linearizations, but a holding-region Lyapunov proof for the complete MPC+RL controller remains open. The next certificate should cover plan memory and switching, with explicit regional and inter-command bounds, rather than reuse the rejected four-state ellipsoid.
