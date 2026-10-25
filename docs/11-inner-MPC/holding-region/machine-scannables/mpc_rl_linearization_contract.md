# CPU checkpoint analysis and MPC linearization plan

Date: 2026-10-25. Pinned controller: `cdce8f93f98d6c978e6a723d4aea11f4524c59ad`. Current results distinguish actual source facts, reported experiments, hypothetical network features, and unavailable verification.

## Minimal structure budget

Keep compact continuous logic, minimal parameters, no fallback behavior, no try/except scaffolding, and no new configuration or CLI layers. Existing controller execution remains unchanged.

| Object | Role | Status |
|---|---|---|
| `terminal_model(weights, divisors)` | Decode original actor and both online critics; return values, total state Jacobians and composed critic Hessians | Implemented in `scripts/experiments/inner-mpc/terminal_linearization.py` |
| Straight-line assumed-model calculation | Compute finite-horizon affine MPC feedback, independently check it by KKT equations, and solve its closed-loop Lyapunov equation | Implemented under artifacts; zero extra functions |
| `holding_statistics(record_root)` | Reconstruct complete saved sessions; measure per-lane steady centers, peak/RMS deviations, drift, frequencies and plan memory | Planned only; requires records |
| `mpc_sensitivity(terminal_model, point_records)` | Differentiate the actual nonlinear MPC objective/KKT system at selected operating points; separate ideal-optimizer results from finite-CEM branches | Planned only; requires checkpoint/records |

Maximum budget is these three substantive functions and one straight-line numerical calculation. No classes, dataclasses, wrappers or extra helpers. Inputs are saved weights, observation divisors, the original record root, and selected point/plan records. Derived objects are physical-state/plan arrays, neural derivatives, objective derivatives, feedback/closed-loop matrices, and certificate geometry. Fixed baseline numerical settings are not new experiment parameters.

## File formats inferred from the committed loaders

The required bundle is `artifacts/rotary_pendulum/1-TD3_20261016/machine-scannables/`.

- `manifest.json` supplies `weights_sha256`, `observation.divisors`, `networks`, and `physical_config`.
- `weights.npz` supplies arrays named `actor.i.kernel`, `actor.i.bias`, `q1.i.kernel`, `q1.i.bias`, `q2.i.kernel`, and `q2.i.bias`, with consecutive zero-based layer indices.
- Kernels have shape `(input_width, output_width)`; biases have shape `(output_width,)`. Hidden layers use SiLU; the actor ends in scalar tanh and the critics end in a scalar affine layer.
- The actor input is five-dimensional; each critic receives those five features plus the normalized actor action. Use the online actor and online critics from one matching bundle.
- In upright-error coordinates the observation before scaling is `(theta, -sin(beta), -cos(beta), omega, nu)`. The CasADi decoder retains both the observation-map and actor-substitution derivative chains.
- `training_settings.json` and run design resolve torque cap, observation conventions, reward discount and provenance. Do not use the generic physics torque cap in place of the runtime cap.

The original run is `artifacts/rotary_pendulum/inner-mpc/20261022-lqr-cost-fine-confirm-16/machine-scannables/`. Required files are `design.json`, `starts.npz`, `completed.json`, `chunk-*.npz`, and `sources/`. Prefer the actual records over report prose.

- `plans`: `(decision, lane, 11)`, with ten normalized commands and one terminal epigraph scalar.
- `stage_states`: `(decision, lane, 10, 4)`, with the measured current state first.
- `executed`: `(decision, lane, 100, 4)`, sampled every 1 ms from refined execution.
- `plan_end` is the refined endpoint; objective reconstruction needs the matching nominal endpoint instead.
- Other fields include objective components, terminal costs, accepted flags, refinement errors, arm margins, final selected spread, and feasible counts.
- The full run has 400 decisions and 32 lanes. The exact random key follows the design seed, decision folding, batch shape and lane convention.
- The runner does not preserve every CEM elite index or full generation history in its chunks. An exact finite-CEM branch reconstruction may therefore require regenerating selected solves; that is distinct from simply decoding the checkpoint.

## Milestones and success flags

| Milestone | Work | Success flag |
|---|---|---|
| Inventory | Check local revisions, all object trees, remote refs and ignored local paths | Completed: latest relevant weights and trajectories absent |
| Decode | Build CPU Dense/SiLU/tanh graph and compare independent NumPy values/finite differences | Completed on explicitly synthetic arrays only |
| Recover actual features | Check bundle hash, shapes, divisors, actor bias/Jacobian, both critic branch gradients/Hessians and gap | Blocked by missing bundle, not by CPU capability |
| Read observed region | Use complete final 10 s records per lane; inspect drift and window dependence; report peak and RMS separately | Planned; no numerical empirical envelope available |
| Derive local MPC map | Use actual neural derivatives and nonlinear rollout objective; inspect active constraints and solver stationarity | Assumed-feature example completed; actual map unavailable |
| Certify holding | Verify augmented-domain acceptance and residual/branch coverage; add inter-command bounds | Unresolved |

For observation, start with the original report's final-ten-second window, then compare its two five-second halves to detect unresolved drift. Save per-lane means and componentwise extrema; keep within-lane oscillation separate from between-lane mean differences. Use complete-state trajectories and actions to check candidate periods. A spectral peak alone does not prove a periodic orbit, and a measured cloud does not prove invariance.

## Exact MPC sensitivity to acquire

Let the nonlinear objective be the scalar function of measured error and ten-command sequence. At a smooth, unique, interior stationary optimizer with nonsingular control Hessian, differentiating its stationarity equation gives

$$
J_U(e,U^\star)=0,\qquad
D_eU^\star=-J_{UU}^{-1}J_{Ue},\qquad
D_e\mu^\star=(1,0,\ldots,0)D_eU^\star.
$$

Here subscripts denote the stated partial derivatives, and the row selects the first command. The Hessian and mixed derivative must include nonlinear rollout curvature and the composed terminal networks. If constraints are active, differentiate the full active-set KKT equations, including the epigraph and multipliers. Check regularity before assigning a unique derivative at critic ties or active-set changes.

This sensitivity describes an exact local optimizer branch. A saved CEM plan with a nonzero stationarity residual does not satisfy the equation merely because it passed the physical audit. For the finite solver, use the memory-dependent branch map described in the human report; its fixed-history first-action physical-state derivative is zero after warm-start initialization inside a fixed branch. Switching boundaries and memory updates remain part of the certificate.

## Completed numerical checks and files

CasADi 3.8.0, NumPy 2.4.6 and SciPy 1.17.1 were available on this PC. CasADi imported and differentiated SiLU successfully. The existing JAX import failed with a `jaxlib` DLL initialization error; Flax and Optax were not installed. No dependency installation or global environment change was needed. The repository's CasADi pin remains unchanged; current validation used the available local version.

Independent synthetic validation used two hidden layers, all three networks, a nonzero physical state, and 33 finite-difference points. Maximum errors were approximately 2.8e-17 in values, 2.1e-11 in Jacobians and 2.7e-9 in critic Hessians. This validates the decoder calculation, not the missing trained model. No synthetic checkpoint was saved as an actual model.

Machine outputs are under `artifacts/rotary_pendulum/inner-mpc/mpc-rl-local-analysis/machine-scannables/`: `decoder_validation.json`, `assumed_mpc_linearization.py`, and the corresponding JSON/NPZ. The assumed model records its hypotheses, fitted/assumed terminal coefficients, affine gain, biased equilibrium, MPC-derived metric, feasibility radii, independent KKT agreement, and explicitly false actual-verification flags.

Running `scripts/experiments/inner-mpc/terminal_linearization.py` currently raises `FileNotFoundError` at the required original manifest, as intended. Once the authentic bundle exists at its documented path, it verifies the checksum and writes actual actor/branch derivatives at upright. Its output explicitly states that these are terminal-network derivatives, not the executed MPC policy Jacobian.

CasADi's [official derivative documentation](https://web.casadi.org/docs/) describes the Jacobian/Hessian operations used here. No complete controller campaign, training run, or GPU computation was executed.
