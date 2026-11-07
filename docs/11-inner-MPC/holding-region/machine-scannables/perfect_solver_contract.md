# Exact-optimization holding certificate: verification contract

Date: 2026-10-25. This contract concerns the unchanged ten-command MPC+RL objective with an ideal global solver. It excludes finite-CEM replacement and random-selection behavior by the user's requested idealization. It does not replace the learned terminal cost, change constraints, add an acceptance filter, or launch controller episodes.

The [human proof](../human-readables/perfect_solver_holding.md) states the model, theorem, candidate set and evidence boundary. All numerical data belong under `artifacts/rotary_pendulum/inner-mpc/perfect-solver-holding-20261025/`.

## Planned and implemented structure

One new straight-line module, `scripts/experiments/inner-mpc/perfect_solver_holding.py`, reuses exactly two existing substantive functions: `terminal_model(weights, divisors)` decodes the authenticated NN and its derivatives; `mpc_sensitivity(terminal_model, point_records)` builds the unchanged multiple-shooting problem, nominal predictor, refined executor and KKT derivatives. No new function, method, class, dataclass, wrapper, helper, CLI parameter, configuration parameter or dependency is introduced.

The independent mathematical input is the four-dimensional measured state. Optimizer variables are ten actions, a scalar epigraph and forty endpoint coordinates; active multipliers accompany the stationary equations. Analysis objects are the frozen network/scales, the model dictionary, center/sensitivity arrays, P-norm geometry and point records. Fixed analysis values are radii 0.05, 0.15, 0.30 and 0.60, coordinate-extreme and worst-linear directions, and four mixed directions. These values select offline verification points; they are not controller settings.

The module preserves the compact continuous workflow, concise comments, explicit failures, and no fallback behavior required by AGENTS.md. Solves depend on the refined center and sensitivity, so their continuation is sequential. NN, RK4 and linear-algebra work use the existing compiled numerical backends. Each local solve prints iterations and retains the existing 60-iteration/60-second analysis limit. No full experiment, GPU operation or controller tuning occurs.

| Milestone | Validation and success flag | Current outcome |
|---|---|---|
| Authenticate objective | SHA-256 equals original manifest; reuse existing plant and NN builders | Passed |
| Refine optimizer equilibrium | First action below 1e-11; stationarity below 1e-7; dynamics residual below 1e-8 | Passed numerically |
| Differentiate feedback | Full active KKT solve; positive tangent curvature; Schur matrix | Passed numerically |
| Probe nonlinear region | Fixed isolated solves; all norm ratios below one; record remainders without promoting them to suprema | Passed as samples |
| Establish exact local theorem | Unique global optimum at exact root, regularity, smoothness and Schur property | Conditional theorem proved; global hypothesis unverified |
| Certify level 0.36 | Uniform optimizer-branch coverage and nonlinear remainder at most 0.02 | Open |
| Certify between commands | Uniform held-flow tube remainder and linear-coordinate bounds | Linear calculation supported; nonlinear uniform verification open |

## Authenticated inputs and exact optimization semantics

Network source is `artifacts/rotary_pendulum/1-TD3_20261016/machine-scannables/weights.npz`. Its binary SHA-256 is

```text
840650b06e8cec03bc8d188b9cae6237853b0fac8ebfabeff11ba26be68b284d
```

Plant settings come from `artifacts/rotary_pendulum/inner-mpc/20261022-lqr-cost-fine-confirm-16/machine-scannables/design.json` and the NN manifest's `derived_physics`. The earlier numerical branch initializes the calculation from `actual-holding-study-20261025/machine-scannables/point_equilibrium_2.npz`. Original checkpoint and trajectory payloads are read only.

Perfect solver means minimization over all feasible original action vectors, not convergence of IPOPT to one local stationary point. The local solver's success string is not a global-optimality certificate. Retain all initializations and their distinct outcomes:

| Initialization | Final objective at refined center |
|---|---:|
| Continued low-cost branch | 3.048083661748134 |
| Zero-action rollout | 27.220771793426234 |
| Actor rollout | 3.048083661748137 |
| Actions transferred from another saved local basin | 648.1814877655364 |

The high-cost outcomes show nonconvex basins; they are not candidate actions of the asserted perfect global solver. No returned plan is substituted by a fallback. The center branch is retained as the analyzed branch, explicitly without global certification.

The 51-dimensional multiple-shooting representation is equivalent to the original action-only nominal objective. Its 1,042 constraint entries are forty dynamics equalities, two terminal epigraph inequalities, and 1,000 arm-node inequalities. Original action bounds remain separate. Feasible local points have inactive action/arm limits and active second negative critic. Both critics remain in each solve.

## Derivatives needed for a regional certificate

Let Y collect primal variables and active multipliers. Let R(Y,e) be the KKT residual, consisting of Lagrangian stationarity and active constraints. A subscript denotes a partial derivative. At a regular solution,

$$
R(Y(e),e)=0,\qquad DY=-R_Y^{-1}R_e.
$$

For physical perturbation vectors v,w, the second derivative is

$$
D^2Y[v,w]=-R_Y^{-1}\left(
R_{ee}[v,w]+R_{eY}[v,DYw]+R_{Ye}[DYv,w]
+R_{YY}[DYv,DYw]\right).
$$

The first-action component of these derivatives gives the first and second derivatives of the optimized feedback. The actor, observation map and active terminal critic must be differentiated through their full composition. Third derivatives of the objective enter this KKT calculation. An indefinite terminal Hessian does not replace the full reduced Lagrangian curvature.

For G(e)=g(e,mu(e)), where g is the execution map and mu the optimized first action,

$$
D^2G[v,w]=D^2g[(v,D\mu\,v),(w,D\mu\,w)]+g_aD^2\mu[v,w].
$$

Thus a certified supremum of this induced bilinear norm at most 0.04 throughout the proposed ellipsoid would imply the remainder coefficient at most 0.02. This is sufficient, not necessary. A direct uniform radial remainder bound can succeed when a global pairwise Jacobian contraction test fails. One sampled derivative norm is 1.00844 at radius 0.60, whereas all tested centered successor ratios remain below one.

For the local existence theorem, unique global optimality is needed only at the exact center: compactness and continuity then give a neighborhood where global minimizers follow the regular branch. For a *specified numerical radius*, that neighborhood must itself be bounded or competing plans must be excluded uniformly over the proposed state set.

An explicit global exclusion certificate can use a validated optimizer neighborhood and a positive cost gap outside it. The gap is taken outside a fixed neighborhood, not over every distinct plan; the latter has zero infimum even for a strict smooth minimizer. A validated branch-and-bound calculation can compare interval lower bounds over the remaining ten-action boxes with a feasible upper bound. Constraints and the maximum of both terminal critics must remain in that calculation. All competitors, not merely the three tested starts, must be covered. If lower-cost plans are found, the perfect-solver center must be recomputed.

## Quantitative certificate target

Use the exact mathematical rest point and P metric defined in the human proof; saved arrays are their numerical approximations. The proposed region is

$$
\Omega=\{e:(e-e_\star)^{\mathsf T}P(e-e_\star)\le0.36\}.
$$

The sufficient endpoint conditions are feasibility and global-branch coverage throughout Omega, q at most 0.9833, and

$$
\sup_{0<\|z\|_P\le0.6}
\frac{\|G(e_\star+z)-e_\star-Az\|_P}{\|z\|_P^2}\le0.02.
$$

They imply the norm factor at most 0.9953 and quadratic factor at most 0.99062209. Any smaller concentric ellipsoid is positively invariant and exponentially attractive in its distance function. No empirically fitted disturbance floor is inserted. An exact zero-action mechanical rest is shared by the nominal, refined and continuous maps.

Observed boundary remainder maxima are 0.000919806, 0.002370557, 0.005298009 and 0.013387270 at the four radii. Probe counts are 10, 14, 10 and 14. The worst successor-norm ratio at radius 0.60 is 0.984945656. These are finite sample maxima, not conservative suprema. The largest tested error radius was not declared a region of attraction.

Full regional verification must also enclose root uncertainty and numerical derivative error. A floating-point approximate center cannot be substituted into a pure quadratic Taylor-remainder claim down to arbitrarily small radius unless an exact nearby root is validated. Alternatively, a separately stated affine residual and practical-stability theorem can account for centering error. The present result uses the exact-root formulation and does not silently introduce such a residual.

## Inter-command coverage

The continuous held-input linear flow is computed from the archived upright mechanics by an augmented matrix exponential. The actual MPC derivative remains the same K; no feedback is applied inside a held command. Let M(t) be this linear map. Its derivative satisfies

$$
M'(t)=\exp(A_ct)(A_c+B_cK).
$$

Let S be a matrix with S-transpose S=P, define L=S A-c S-inverse, and let lambda-max denote the largest eigenvalue of a symmetric matrix. Then

$$
\|S M'(t)S^{-1}\|_2
\le \exp\left(t\lambda_{\max}((L+L^{\mathsf T})/2)\right)
\|S(A_c+B_cK)S^{-1}\|_2.
$$

The numerical logarithmic norm is 17.03980694 per second; the maximum derivative bound on the 0.1 s interval is 97.2602472 per second. A nearest-node distance at most 0.00025 s adds relative row margin 0.0243151 to the sampled relative maximum one. A rounded factor 1.03 leaves margin for the linear-coordinate bound. These floating-point values have not been interval-enclosed.

The refined RK4 feedback matrix and continuous held-flow feedback matrix differ by at most 2.44154e-10 entrywise. Continuous-flow contraction is 0.983250552192. Both values use the same numerical K and P. This checks integration consistency at the linear level; it is not a nonlinear integration-error certificate.

The additional uniform tube condition is a P-norm remainder coefficient at most 0.02 for every state in Omega and every time in the command interval. Together with the linear factor 1.03 it implies coordinate extents given by

$$
\sqrt{(P^{-1})_{jj}}\left(1.03(0.6)+0.02(0.6)^2\right).
$$

For pendulum angle this is 3.99828312 degrees. The local *refined-grid* tube probes have maximum coefficient 0.013387271. They are not a bound over all continuous times or all initial states.

## Reproduction and outputs

Run only the fixed local analysis from the repository root:

```powershell
C:\ProgramData\anaconda3\python.exe scripts/experiments/inner-mpc/perfect_solver_holding.py
```

The dependency versions are the existing CPU analysis versions in `actual-holding-study-20261025/machine-scannables/requirements.txt`. No JAX, GPU access or full controller runtime is needed. The driver assumes the previous authenticated actual-network analysis outputs already exist.

Machine outputs are `center.npz`, `regional_probes.npz`, `point_diagnostics.json`, `summary.json`, `linear_flow.npz`, and the captured `analysis_log.txt`. `linear_flow_summary.json` records the independently evaluated continuous linear-flow check. State order is arm angle, upright error, arm speed, pendulum speed. P radius is the norm, whereas 0.36 is its square. Costs are original undiscounted objective values. All flags for global optimality and regional derivative certification remain false.

The human artifact `interpretation_summary.md` explains which conclusions are numerical, conditional or unproved. This contract is a verifier specification for the existing controller, not an added controller condition.
