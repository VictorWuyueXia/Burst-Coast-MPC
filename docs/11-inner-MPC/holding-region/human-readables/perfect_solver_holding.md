# Holding under exact optimization of the existing MPC+RL objective

Date: 2026-10-25. The controller, networks, weights, horizon, constraints and integration grids are unchanged. Only the solver is idealized: it returns a global minimizer of the existing nonlinear prediction problem at every command. No CEM history, replacements or random selections enter this idealized feedback. No full experiment was run.

**Result.** The actual-network optimizer has a numerically regular, locally stable upright branch. A local existence theorem follows if this branch is the unique global optimum at its equilibrium. A concrete candidate is the ellipsoid with quadratic level **0.36**, centered at an arm offset of **7.997846 degrees**. Its pendulum extent is **3.837124 degrees** at command instants. A specified uniform nonlinear remainder bound would prove its invariance and Lyapunov stability. Local calculations support that bound but do not certify it or global optimality. Thus the numerical ellipsoid remains a candidate for the complete globally optimized MPC+RL controller.

## The controller being proved

Let the state at command index k be

$$
e_k=(\theta_k,\beta_k,\omega_k,\nu_k)^{\mathsf T}\in\mathbb R^4.
$$

The coordinates are arm angle from center, pendulum error from upright, arm speed and pendulum speed. Angles are radians and speeds rad/s. For pendulum angle from downward, the upright error is its displacement from pi, wrapped to the interval from minus pi to pi. Local coordinates are continuous around upright. Each command lasts T=0.1 s. The normalized command a lies in [-1,1], with physical torque 0.00459a N m.

Write f for the original one-command 1 ms RK4 predictor. The decision is the ten-command vector U. Its predicted states and objective are

$$
\xi_0=e,\qquad \xi_{i+1}=f(\xi_i,a_i),\quad i=0,\ldots,9,
$$

$$
J(e,U)=\sum_{i=0}^{9}\left(\xi_i^{\mathsf T}W\xi_i+a_i^2\right)+F(\xi_{10}),
\qquad
W=\operatorname{diag}((135\pi/180)^{-2},0.15^{-2},0.25,0.25),
$$

$$
F(v)=\max\{C_1(v),C_2(v)\},\qquad
C_j(v)=-Q_j(o(v),\pi(o(v))).
$$

The actor pi and critics Q1,Q2 are the authenticated checkpoint, not fitted surrogates. The observation map is

$$
o(v)=\operatorname{diag}(d)^{-1}
(v_1,-\sin v_2,-\cos v_2,v_3,v_4)^{\mathsf T},
$$

$$
d=(2.356194490192345,1,1,30.592532352353444,21.36069107055664)^{\mathsf T}.
$$

The diagonal observation divisors and state weights are archived values. The pendulum stage cost uses the original wrapped angle outside this local chart. Feasible plans satisfy the existing action bounds and all 1,000 predicted arm-node bounds of plus or minus 135 degrees. No terminal constraint is added. Denote this feasible set by U(e), and define the globally optimal plans and first-action policy by

$$
\mathcal U^\star(e)=\operatorname*{arg\,min}_{U\in\mathcal U(e)}J(e,U),
\qquad \mu^\star(e)=a_0^\star(e).
$$

The policy is a function when the relevant optimal first action is unique. Global optimality alone does not imply uniqueness or smoothness. If optimizers tie, a guarantee must cover every allowed first action or specify the solver's tie selection.

Let g(e,a) denote the one-command execution map. The simulator uses the original 0.5 ms RK4 refinement. For the continuous mechanical model, g is the exact held-input flow over T. Both choices have the same rest equilibrium; their computed closed-loop linearizations differ by only about 2.44e-10 entrywise here. The idealized closed loop is

$$
e_{k+1}=G(e_k)=g(e_k,\mu^\star(e_k)).
$$

The actor remains inside both composed terminal critics. It does not replace the optimized first action. Both critics remain in the global optimization; the local second-critic derivative is justified only while its strict terminal gap persists.

## The actual MPC+RL equilibrium and its derivative

Let e-star be an exact root of the optimized first action at upright rest. Its refined numerical estimate is

$$
e_\star\simeq(0.139588743677376,0,0,0)^{\mathsf T},
\qquad \theta_\star\simeq7.99784588^\circ,
\qquad \mu^\star(e_\star)=0.
$$

Zero torque at this state produces no mechanical motion, so g(e-star,0)=e-star for the predictor, refined execution, and exact continuous dynamics. The center is biased in arm angle, not pendulum angle. The computed first action is about 1e-17 and refined successor defect about 3.82e-17 in the metric introduced below. These residuals are numerical evidence, not an enclosure of an exact global equilibrium.

The optimization uses 51 variables y: ten actions, one terminal epigraph scalar, and forty predicted endpoint coordinates. Let c contain the forty dynamics equalities and active second-critic epigraph inequality. With multipliers lambda and Lagrangian L, differentiating the stationary equations gives

$$
\begin{bmatrix}L_{yy}&c_y^{\mathsf T}\\c_y&0\end{bmatrix}
\begin{bmatrix}D_ey^\star\\D_e\lambda^\star\end{bmatrix}
=-\begin{bmatrix}L_{ye}\\c_e\end{bmatrix}.
$$

The first action row of this solution is the feedback derivative

$$
K=D_e\mu^\star(e_\star)
\simeq(0.044370877,-15.163537490,0.109519876,-1.202770923).
$$

At the computed point, the inactive critic gap at the optimized terminal state is 0.10959667 cost units. Action and arm bounds are inactive, the reduced Lagrangian Hessian has minimum eigenvalue 0.61776324, and stationarity residual is below 3e-13. The active constraint Jacobian has full row rank: the dynamics have a triangular endpoint block with diagonal minus identity, and the epigraph constraint has a nonzero independent epigraph column. Its multiplier is one by epigraph stationarity. These facts explain why a smooth strict local optimum is expected; they do not establish global optimality.

The complete feedback linearization is

$$
A=DG(e_\star)=g_e(e_\star,0)+g_a(e_\star,0)K.
$$

Its eigenvalues are approximately 0.189524, 0.280135, 0.652396 and 0.951651. This is an MPC sensitivity using the actual actor and critics, not an LQR substitution. Define the positive matrix P using the actual feedback matrix and existing state weights,

$$
P-A^{\mathsf T}PA=W,
$$

$$
P\simeq\begin{pmatrix}
3.18844158&-14.80822925&0.97352041&-1.38415264\\
-14.80822925&379.52993267&-17.27057535&29.26584855\\
0.97352041&-17.27057535&1.26578348&-1.60630039\\
-1.38415264&29.26584855&-1.60630039&2.88769045
\end{pmatrix}.
$$

Full-precision numerical entries are saved in `center.npz`. Using the refined-map matrix gives the displayed P; the exact continuous-flow matrix gives the same reported contraction to the shown precision. For any positive P define its norm and induced matrix norm by

$$
\|z\|_P=\sqrt{z^{\mathsf T}Pz},\qquad
q=\|A\|_P=\|P^{1/2}AP^{-1/2}\|_2\simeq0.9832505522<1.
$$

The spectral radius 0.951651 is not the contraction factor for this metric.

## Local existence theorem for the full idealized objective

**Hypotheses.** At an exact upright rest point, the identified plan is the unique global minimizer. Its inactive bounds and critic gap are strict, its reduced Hessian is positive definite, and its full feedback matrix is Schur. The prediction and execution maps are smooth on the local upright chart. The numerical calculations support the local conditions; uniqueness and global optimality remain unverified.

**Theorem.** Under these hypotheses, a positive radius exists such that the complete globally optimized MPC+RL feedback has a positively invariant ellipsoid around this rest point and exponentially converges to it. Every smaller concentric ellipsoid is a Lyapunov-stable holding set relative to that neighborhood.

**Proof.** The full-rank active constraints, positive reduced Hessian and strict active-set margins give a unique smooth local stationary optimizer. It is also the global optimizer sufficiently near the center. To prove this last assertion, suppose a sequence of global optimizers at states approaching the center stays outside the local optimizer neighborhood. Actions lie in a compact ten-dimensional box. A subsequence therefore converges to a feasible plan at the center. Continuity of costs and feasibility, and the feasible smooth local branch as an upper bound, imply that its limit is a global minimizer at the center. Uniqueness puts that limit inside the local neighborhood, a contradiction. Consequently the differentiated local policy is the complete perfect-solver feedback on some neighborhood.

Let z=e-e-star and define its nonlinear remainder by

$$
\eta(z)=G(e_\star+z)-e_\star-Az.
$$

The remainder and its first derivative vanish at zero. Twice continuous differentiability gives, on a sufficiently small closed metric ball,

$$
\|\eta(z)\|_P\le c_r\|z\|_P^2,\qquad
c_r=\frac12\sup_{\|v\|_P\le r}
\|D^2G(e_\star+v)\|_{P\leftarrow P,P}<\infty.
$$

The bilinear norm is the maximum output norm over two input vectors of unit P-norm. Its value is a derivative bound, not a controller parameter. Choose a positive r small enough that the ball stays in the feasible smooth global branch and the contraction factor below is less than one. Define

$$
\Omega_r=\{e:(e-e_\star)^{\mathsf T}P(e-e_\star)\le r^2\},
\qquad \kappa=q+c_rr<1.
$$

The triangle inequality gives

$$
\|G(e)-e_\star\|_P
\le q\|e-e_\star\|_P+c_r\|e-e_\star\|_P^2
\le\kappa\|e-e_\star\|_P.
$$

Thus G maps the ellipsoid into itself, recursively preserving feasibility. Define the Lyapunov function and its decrease by

$$
V(e)=(e-e_\star)^{\mathsf T}P(e-e_\star),\qquad
V(G(e))-V(e)\le-(1-\kappa^2)V(e),\qquad
\|e_k-e_\star\|_P\le\kappa^k\|e_0-e_\star\|_P.
$$

For any inner radius s between zero and r, the function

$$
D_s(e)=\max\{\|e-e_\star\|_P-s,0\}
$$

is distance to the inner ellipsoid in the P metric and satisfies

$$
D_s(G(e))\le\kappa D_s(e).
$$

Inside the inner ellipsoid this follows from invariance; outside it follows by subtracting s from the norm bound. This proves stability and attraction of each inner holding set. Positive-definite P makes these conclusions equivalent to Euclidean stability. The radius is positive because smoothness supplies a bounded local second derivative and q is strictly below one. This completes the local theorem.

The theorem does not assume that the critic itself is a Lyapunov function. Its curvature enters the optimizer derivative, which enters A, P and the nonlinear remainder. Each controller component therefore has a defined role in the proof.

The direct actor-tail terminal-decrease proof is unavailable in its usual origin-centered form. The actual actor has a different rest point, approximately (-0.167391443,0,0,0). Denote an exact such root by v-pi, the actor composed with observation by a-pi, and the running cost by ell. At this point the actor command is zero and the mechanical state does not move, so

$$
a_\pi(v)=\pi(o(v)),\qquad
F(f(v_\pi,a_\pi(v_\pi)))-F(v_\pi)=0,
\qquad \ell(v_\pi,0)=v_\pi^{\mathsf T}Wv_\pi\simeq0.00504713>0.
$$

Consequently the terminal-decrease inequality with right side minus the running cost fails there. Learning the critics does not supply that missing inequality. The optimizer-sensitivity theorem avoids assuming it; a different terminal analysis is not excluded by this observation.

## A numerical holding region and the exact inequalities needed to prove it

The proposed nontrivial region uses r=0.6, so its quadratic level is 0.36. If the global optimizer stays on the identified branch throughout this ellipsoid and the **uniform**, not sampled, bound is

$$
\|G(e_\star+z)-e_\star-Az\|_P\le0.02\|z\|_P^2
\quad\text{for every }\|z\|_P\le0.6,
$$

then q at most 0.9833 gives

$$
\kappa\le0.9833+0.02(0.6)=0.9953<1,
\qquad V(G(e))\le0.99062209V(e).
$$

This is a completed implication, with a quantitative decrease margin. The remainder threshold for some decrease is 0.0278333; 0.02 reserves margin. These are verification thresholds, not assumed properties of the checkpoint.

The ellipsoid has command-boundary coordinate extents

| Quantity | Bound in the proposed ellipsoid |
|---|---:|
| Arm angle | -14.1331 to +30.1288 degrees |
| Pendulum error | plus or minus 3.83712 degrees |
| Arm speed | plus or minus 1.02087 rad/s |
| Pendulum speed | plus or minus 0.873492 rad/s |

Each coordinate bound follows from r times the square root of the corresponding diagonal of P inverse. They are extents of one ellipsoid, not a box whose corners are admissible. Asymptotic error relative to upright rest is zero in pendulum angle and speeds; the arm converges to its approximately eight-degree offset. This is a local holding result and contains no swing-up or capture claim.

The saved CEM sessions motivate examining upright holding, with final-window pendulum errors about plus or minus 1.95 degrees. They do not identify trajectories of the perfect solver, and this proposed ellipsoid is not asserted to contain every recorded state. Its geometry comes from the actual MPC sensitivity and the new local calculations, not an enclosing fit to those sessions.

The offline check solved 53 isolated one-horizon problems, including center refinement, three other initializations and 48 boundary probes. No new closed-loop session was generated. Probe summaries are

| P-norm radius | Probe count | Largest successor-norm ratio | Largest remainder coefficient |
|---:|---:|---:|---:|
| 0.05 | 10 | 0.983270 | 0.000920 |
| 0.15 | 14 | 0.983399 | 0.002371 |
| 0.30 | 10 | 0.983911 | 0.005298 |
| 0.60 | 14 | 0.984946 | 0.013387 |

Directions include both signs of each coordinate-extreme direction, the worst linear contraction direction, and four fixed mixed directions at radii 0.15 and 0.60. The measured coefficient is the remainder norm divided by squared state radius. All tested boundary states move inward. However, a finite set of directions cannot upper-bound the remainder on the entire ellipsoid.

At radius 0.6 one local derivative norm reaches 1.00844. Therefore a proof requiring contraction between every pair of states in this fixed metric already fails at a sampled point. The radial Lyapunov proof above is less restrictive and remains consistent with all tested points. Local attraction to a center does not require incremental contraction between arbitrary pairs.

## Holding between commands

Let Psi-t(e) be the held-action successor at time t between zero and T, using the same optimized action chosen at the command start. Let M-t be its derivative at the center. For the continuous dynamics, the state and normalized-input linearization matrices are

$$
A_c=\begin{pmatrix}0&0&1&0\\0&0&0&1\\0&CG/D&0&0\\0&IG/D&0&0\end{pmatrix},
\qquad B_c=(0,0,J_p\tau/D,C\tau/D)^{\mathsf T},
$$

$$
D=I J_p-C^2,\quad
M_t=\exp(A_ct)+\left(\int_0^t\exp(A_cu)B_c\,du\right)K.
$$

Here I=0.00040219166666666673 and J-p=0.00013312800000000002 are inertias, C=0.00013158 is the coupling inertia, G=0.01518588 is the gravity torque coefficient, and tau=0.00459 N m is the torque limit. These are archived physical values; D is the computed upright mass determinant. The integral variable is time, not a control input.

For coordinate j define h-j as the supremum, over the command interval, of the Euclidean norm of row j of M-t times P to the power minus one half. If a uniform tube remainder satisfies

$$
\|\Psi_t(e_\star+z)-e_\star-M_tz\|_P\le c_T\|z\|_P^2,
\quad 0\le t\le T,\quad \|z\|_P\le r,
$$

then Cauchy-Schwarz proves the continuous coordinate bound

$$
|\Psi_t(e)_j-e_{\star,j}|
\le h_jr+\sqrt{(P^{-1})_{jj}}\,c_Tr^2.
$$

The analytic linear-flow calculation supports h-j at most 1.03 times the square root of the corresponding P-inverse diagonal. A 0.5 ms time grid has numerical relative maximum one; the induced derivative bound of about 97.26 per second adds at most 0.02432 between grid nodes. With the additional uniform nonlinear tube target c-T at most 0.02, the proposed ellipsoid therefore gives pendulum error at most **3.99829 degrees throughout each command**. Saved local fine-grid probe remainders have maximum 0.0133873. Uniform nonlinear continuous-time coverage remains to be verified; endpoint invariance alone does not establish this continuous claim.

## What is proved and what remains

The theorem proves existence of a local holding region **conditional on the identified regular optimum being the unique global optimum at its exact rest point**. Its proof uses the full MPC+RL objective and feedback sensitivity. The explicit level-0.36 ellipsoid is supported by nonlinear local solves and has a complete quantitative Lyapunov implication. It is not yet an unconditional numerical certificate.

Two global-optimizer checks remain for the explicit ellipsoid: exclude competing lower-cost plans throughout the region, and verify the uniform nonlinear remainder bound. A continuous-time holding claim additionally requires the tube bound. At the center, an actor-seeded solve agrees with cost 3.04808366. Other starts reach costs 27.22077 and 648.18149, confirming distinct local basins. These higher values neither disprove nor establish global optimality of the low-cost branch.

A credible completion uses interval or other validated bounds on the frozen neural graph, KKT derivatives and mechanical flow, plus branch-and-bound exclusion of competing plans outside a validated optimizer neighborhood. It requires offline computation, not retraining or a full controller experiment. If a lower-cost branch is found, its equilibrium and derivative must be analyzed instead; a perfect-solver assumption cannot select the preferred local basin by declaration.

Reproduction details and verification thresholds are in the [machine contract](../machine-scannables/perfect_solver_contract.md). The previous CEM counterexamples concern different solver actions and do not falsify this idealized local theorem; they also supply no certification for it.
