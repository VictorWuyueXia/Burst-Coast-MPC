# Actual-network holding study: evidence and reproduction

Date: 2026-10-25. Controller commit `cdce8f93f98d6c978e6a723d4aea11f4524c59ad`; artifact commit `1ebd1fa3f25224edce50e77567f3d25767500d4b`. The unchanged checkpoint hash is `840650b06e8cec03bc8d188b9cae6237853b0fac8ebfabeff11ba26be68b284d`.

## Structure and scope

The Python structure stays within the planned three substantive functions: existing `terminal_model(weights, divisors)`, `holding_statistics(record_root)`, and `mpc_sensitivity(terminal_model, point_records)`. Each function has fewer than 100 lines. The first two new analysis modules and one straight-line driver are in `scripts/experiments/inner-mpc/`. The standalone artifact verifier introduces zero functions. No classes, dataclasses, wrappers, CLI options, controller settings, fallback controls or global dependency changes were added. Compact continuous logic and loud failures are retained.

Model objects are the archived network arrays/scales, physical mechanics and runtime plant settings; nominal and refined RK4 maps; objective and KKT derivatives; local stationary solutions; saved transitions; and quadratic/queued-state certificate candidates. Numerical tolerances and the small local solver budget are fixed analysis settings, not controller parameters.

## Completed workflow

| Milestone | Validation | Result |
|---|---|---|
| Authenticate inputs | Existing transfer hashes; weights hash; 400-by-32 accepted decisions and state continuity | Passed |
| Observe holding | Full final-ten-second states, per-session means/std/drift, two-window coverage | Passed as data analysis; no invariant-region inference |
| Reconstruct model | Actual nonlinear mechanics and original nominal/refined grids | Objective mismatch about 3e-11; execution mismatch about 4e-14 at checked point |
| Differentiate local optimizer | Full active-set KKT derivative, positive reduced curvature, independent directional finite difference | Numerically regular stable local branch; directional error 2.3e-10 |
| Find local rest point | Scalar continuation of first action at zero pendulum error/speeds | About +7.99785 degrees arm angle; first action 2.72e-9 |
| Test quadratic holding | Saved CEM transitions; one-step and 2/5/10/20/50-step residual tests | Proposed five-degree ellipsoid falsified by 73 recorded exits |
| Analyze actual memory branch | Exact shifted-coordinate matches; actual actor derivatives; 14-state retained-update matrix | 2531/3200 first-nine retentions; numerical spectral radius about 0.984331241 |
| Establish full theorem | Uniform accepted continuation, switched/memory regional bounds and inter-command coverage | Not established |

## Numerical formulation and safeguards against overclaiming

The local optimizer has 51 variables: ten actions, one epigraph scalar and forty endpoint coordinates. Forty nonlinear dynamics equalities reproduce single shooting. Two terminal epigraph inequalities retain the original maximum of negative critics. One thousand arm-node inequalities and the original action bounds remain. The current-state arm condition is checked in the archived inputs; local study points are interior.

The nonlinear cost uses current through ninth stage state, the undiscounted quadratic weights and the frozen terminal network. The 1 ms predictor and 0.5 ms executor are distinct maps. Error coordinates use wrapped upright error for costs and sine/cosine for dynamics/features; no state clipping is introduced.

The initial single-shooting solve was stopped after a long silent wait. The equivalent multiple-shooting formulation and iteration logging resolved the computational issue. Initial diagnostics that transplanted a plan to a different state produced unsuitable warm starts; those results and their script snapshot are preserved under `initialization_diagnostic/` and excluded from the selected equilibrium continuation. Zero actions initialize the upright rest calculation; an actual actor rollout initializes the synthetic mean-state point; original plans initialize original recorded states. These are offline solver initializations, not changes to CEM.

The NLP is nonconvex. At recorded point `final_lane2`, a local solve reached cost 28.89537, worse than the feasible CEM cost 3.28549. Its stationary sensitivity is retained as a different local basin and is not called globally optimal. At `negative_arm`, cost improved from 5.09326 to 2.25963. The equilibrium branch is the regular continuation from the low-cost upright solution, not a claim of unique global MPC feedback.

The KKT sensitivity includes all forty dynamics constraints and numerically active inequalities/bounds. The root branch has no active action or arm bounds and one active terminal critic. The reduced curvature minimum is about 0.61776. An exact regional theorem still requires validated root/regularity, branch coverage and second-derivative bounds; floating-point stationarity alone does not provide those bounds.

## Certificate test interpretation

The metric solves the ordinary Lyapunov equation for the actual-network local optimizer Jacobian with the existing positive state-weight matrix. No earlier assumed terminal curvature or LQR Riccati gain is reused.

For candidate contraction factor and residual, the sampled ultimate-level minimization is a two-variable linear program. Defining inverse remaining contraction and ultimate level transforms each sample requirement into

$$
t=\frac{1}{1-\rho}\ge1,\qquad b=\frac{\epsilon}{1-\rho}\ge0,\qquad
b\ge V_k+t(V_{k+1}-V_k).
$$

The symbols denote the quadratic value, contraction factor, residual, inverse contraction gap and ultimate level. The optimum is 2.948944283 at zero contraction factor for all tested block lengths. This is a finite archive bound, not evidence of a uniform regional inequality.

The five-degree level is 0.611266876. Episode index 7 at decision 375 rises from 0.426495444 to 1.270932712. `invariance_exit_witness.npz` includes its actual selected and previous plans. `derivative_and_exit_validation.json` verifies the strict exit by exact rational arithmetic on the archived numbers and checks refined execution independently. The exit disproves this candidate ellipsoid's invariance for a reachable stored-plan history; it does not disprove upright-angle holding or all alternative certificates.

## Retained-update Jacobian

The continuous state for this branch is physical error plus the previous ten-command plan, in the order stored in `warm_branch_linearization.npz`. Let the coarse seed command matrix pair be the two matrices with subscript s, and the refined execution pair have subscript f. Let the actor gradient at its rest point be the row with subscript pi. The retained-update derivative obeys

$$
\delta e^+=A_f\delta e+B_f\delta p_1,
$$

$$
\delta p_i^+=\delta p_{i+1},\quad i=0,\ldots,8,
$$

$$
\delta p_9^+=K_\pi\left(A_s^9\delta e+
\sum_{j=1}^{9}A_s^{9-j}B_s\delta p_j\right).
$$

The previous action index is zero-based; index zero is already consumed and does not affect this update. The delta symbol denotes first-order perturbation. The actor row is evaluated at its numerical rest arm angle -9.59082 degrees. The map retains the original coarse 2 ms tail construction and refined 0.5 ms actual execution.

High-precision matrix arithmetic verifies the reported numerical eigenvalue magnitudes without relying on ill-scaled double-precision predictor coordinates. Neural derivatives themselves remain float64 and are not interval-enclosed. The raw augmented Lyapunov matrix is badly conditioned, around 6e15, and is not used as a reliable regional certificate. CEM replacements, sampling, elite changes, acceptance changes and critic switching are outside this one-branch matrix.

## Reproduction and files

Run from the repository root, without launching a controller campaign:

```powershell
python scripts/experiments/inner-mpc/holding_statistics.py
python scripts/experiments/inner-mpc/analyze_holding.py
python artifacts/rotary_pendulum/inner-mpc/actual-holding-study-20261025/machine-scannables/verify_local_results.py
```

The first command reads saved records; the second performs only a small fixed set of local one-horizon optimizations and matrix tests; the third performs two nearby local solves, exact witness checks and retained-branch matrix analysis. No closed-loop episodes are regenerated. All long local solves print iterations and have fixed iteration/wall-time limits.

Machine records are under `artifacts/rotary_pendulum/inner-mpc/actual-holding-study-20261025/machine-scannables/`: `holding_statistics.json`, `holding_by_episode.csv`, `recorded_transitions.npz`, per-point solutions and sensitivities, `local_sensitivities.json`, `certificate_diagnostics.json`, `invariance_exit_witness.npz`, `derivative_and_exit_validation.json`, and `warm_branch_summary.json`/NPZ. The exact nonlinear-local and retained-update matrix entries are in NPZ files; the human report rounds them for reading.

`requirements.txt` records the actual CPU analysis versions, including mpmath for high-precision matrix algebra. The repository-wide environment is unchanged. The source imports only the standalone CasADi decoder and standard numerical packages; no JAX runtime or GPU is needed.

The [CasADi documentation](https://web.casadi.org/docs/) describes the accumulation and Lagrangian-derivative operations. Proof conditions and the rejected ellipsoid are explained in the [human report](../human-readables/actual_mpc_rl_holding.md).
