# Evidence obligations for the unchanged MPC+RL controller

Audit date: 2026-10-24. Commit: `cdce8f93f98d6c978e6a723d4aea11f4524c59ad`. These are proof-work milestones within the current project phase, not new development phases. This file specifies future acquisition work; no new experiments or rigorous regional computations were performed in this review.

## Structure and scope budget

This review adds three English Markdown documents and their three Chinese companions: the interpretation summary, human assessment, and this evidence specification. It adds zero classes, functions, methods, dataclasses, wrappers, helpers, configuration parameters, or runtime behavior. Existing source and the two discussion documents remain unchanged. Maintain the repository's compact, continuous logic, minimum-parameter discipline; do not add fallback behavior or exception scaffolding to make a certificate pass.

Existing objects to inspect are `inner_model.evaluate` and `rollout` for costs and numerical transitions; `gpu_cem.initial_plans` and `sample_gpu` for candidates and selection; `lqr_cost_run.advance` for stored-plan evolution and execution; `Actor` and `Critic` for frozen inference. Any later analysis implementation needs its own explicit small structure budget before code is added.

## Required information and acquisition plan

| Milestone | Missing information | Acquire it by | Success flag |
|---|---|---|---|
| Freeze the theorem | Exact initial set; meaningful final arm/angle/speed bounds; exact ODE or discrete-map scope; fixed random sequence or quantified outcomes | State these choices explicitly using the original experiment settings; distinguish eventual entry, invariant holding, and Lyapunov stability | Every quantifier, initial memory, dynamical map, and physical target bound is specified |
| Recover provenance | Actual network arrays, architecture, observation divisors, effective torque cap, source snapshots, and initial states | Retrieve the frozen bundle and original experiment directories listed below; verify hashes, physics, seed, and batch shape | No unresolved mismatch between executable controller and theorem |
| Audit capture evidence | First simultaneous entry into the full proposed capture region; costs and accepted status before entry; all incomplete cases | Reconstruct costs from stage states/actions/critics; inspect command states and dense executed records separately | Per-session entry time or explicit nonentry; reconstruction residual and failed cases recorded |
| Define holding domain | Certificate matrix, contraction/residual bounds, invariant domain, final physical coordinate bounds, compatible inherited memory | Compute the LQR matrix as a candidate; evaluate actual MPC successors; fit candidate bounds, falsify them, then verify them over the domain | A nonempty feasible holding domain and a small invariant target satisfy the inequalities below |
| Establish progress | Net plan-cost decrease and residual before capture; relevant terminal defects and selection gaps | Reconstruct consistently using one chosen cost/map; search adverse initial states, stored plans, solver branches, and numerical discrepancies | A rigorous positive one-step net decrease; the residual-threshold condition is one sufficient route; alternatively a verified positive finite-block decrease |
| Establish continuation | Accepted initial plans and indefinitely accepted successors in the relevant holding domain; finite accepted execution until capture | Analyze candidates actually retained through final audit, not just nominal feasibility; prove a feasible accepted successor witness or cover all specified branches | No reachable failure state within theorem scope; constraints hold at the stated temporal resolution |
| Certify coverage | Regional bounds between sampled states, reachable controller histories, and nominal/execution discrepancies | Subdivide initial/reachable sets; use validated dynamics/network bounds and selection-branch reasoning, or verified reachable-set propagation | Every point and history in the declared domain is covered, with certified numerical margins |
| Compose and bound time | Inclusion of captured augmented states in the holding domain; lower/upper plan-cost constants if a numerical capture deadline is claimed | Verify the inclusion with inherited plans; compute regional or analytic cost bounds from the actual checkpoint | An end-to-end theorem with no unverified major assumptions and explicit final region |

Recommended order: provenance and scope; holding-domain construction; empirical capture/progress falsification; accepted-continuation and regional proofs; composition. Collect diagnostic capture and cost records together. Independent initial-state probes can be batched. The final uniform claims require rigorous coverage, not only more samples.

## Exact files absent locally

All paths below are relative to the repository root. The three source directories were checked and are absent from the local working tree; the checkpoint and run directories are also absent from the pinned Git tree.

- `artifacts/rotary_pendulum/1-TD3_20261016/machine-scannables/weights.npz`
- The same bundle's `manifest.json` and `training_settings.json`.
- `artifacts/rotary_pendulum/inner-mpc/20261022-lqr-cost-fine-confirm-16/machine-scannables/`: `design.json`, `starts.npz`, `completed.json`, `chunk-*.npz`, and `sources/`.
- `artifacts/rotary_pendulum/inner-mpc/20261019-gpu-closed-cem/machine-scannables/`: original design, stored starts, and preparation/source provenance.

Optional existing evidence: the screen-16 run and local probe outputs for optimizer-gap diagnosis; the teacher settings containing the LQR gain and its derivation inputs. The Riccati matrix can be recomputed from the frozen model and existing weights if it was not saved. No new RL training is required for this audit.

The checked-in `physics.yaml` sets 0.0204 N m, while official baseline documents specify 0.00459 N m. `inner_model.rollout` multiplies actions by the runtime `plant["torque_limit_nm"]`, so the generic configuration alone does not determine the baseline input cap. Resolve this from the original design and bundle. Both damping coefficients are zero in the inspected configuration.

## Minimal rigorous conditions

The physical state, error, stage cost, selected cost, capture threshold, and terminal function are defined in the [assessment](../human-readables/proof_assessment.md). Here the command index is an integer, and the command duration is 0.1 s. The controller memory contains the previous ten-command plan and enough key/index information to determine the selected random draws. Write the augmented state as

$$
z_k=(x_k,h_k),\qquad z_{k+1}\in\mathcal T(z_k).
$$

The memory is denoted by the second component. The transition relation contains the allowed actual solver outcomes; for a fixed fully specified random sequence it is single valued wherever execution is accepted. Define a certified pre-capture augmented domain and a holding augmented domain before taking suprema.

For the one-step capture route, the needed quantities are

$$
J_k\ge J_{\min},\qquad
\bar r\ge\sup_{\substack{\text{reachable pre-capture}\text{states and allowed outcomes}}}
\left[J_{k+1}-J_k+\ell(x_k,a_{0|k})\right],\qquad
\bar r<q_D.
$$

The lower constant is a uniform lower bound on relevant selected costs; the upper residual constant must be verified on the stated augmented domain. The value 0.01 appearing in the discussion is illustrative and must not enter a certified result without evidence. The value 0.0225 is the analytic threshold for the provisional box only; recompute it if that box changes.

The less conservative one-step obligation directly bounds net decrease: the stage cost minus the residual is at least one uniform positive constant outside capture. A maximum residual above the threshold does not reject this condition. Inspect both quantities before moving to finite blocks.

For finite-block progress, let a positive integer be the block length and a strictly positive scalar be its certified decrease. Require

$$
J_{k+m}-J_k\le-\delta_m,\qquad m\in\mathbb N_{>0},\quad\delta_m>0,
$$

whenever the entire block avoids capture. Accepted execution and constraints must hold at all intermediate decisions. Keep each block length fixed for the theorem and report how it was selected. A failed one-step certificate does not imply failed reachability.

For holding, use a symmetric positive definite matrix and verified bounds satisfying

$$
L(e)=e^{\mathsf T}Pe,\qquad
\sup_{\substack{z=(x,h)\text{ in holding domain}\\z^+\in\mathcal T(z)}}
\left[L(e^+)-\rho L(e)\right]\le\varepsilon,
\qquad 0<\rho<1,\quad\varepsilon\ge0.
$$

The superscript plus denotes the next augmented or error state. The contraction factor and residual are certificate constants. Verify continued acceptance, invariance of the allowed memory set, and the physical sublevel domain with radius above the residual floor. In addition, certify that every augmented first-capture state belongs to this holding domain. A proof covering all physical points of the provisional capture rectangle is sufficient but not necessary if a smaller verified capture image is available.

## Numerical and logical checks

- `stage_states` uses 1 ms nominal prediction. `plan_end` is the refined terminal state. Reconstruct the terminal value at the state matching the cost definition; do not substitute one endpoint for the other.
- The warm-start rollout uses its default 2 ms grid. Objective evaluation uses 1 ms, execution 0.5 ms, and saved trajectories 1 ms. An exact-ODE theorem requires validated error enclosures. A declared discrete-simulation theorem avoids claiming the exact ODE but still must include the predictor/execution difference in progress comparisons.
- Step-halving agreement is an error diagnostic, not automatically an enclosure of the exact trajectory. The reported approximately 0.000075 degree arm clearance makes continuous-time constraint verification particularly demanding. Reconstruct the corresponding boundary intervals before making that claim.
- CEM's final audit can discard a lane's improved incumbent without recovering its earlier candidate. Do not infer a nonpositive selection gap or recursive accepted feasibility from within-solve best-cost monotonicity.
- The original GPU start generator perturbs four centers and mirrors its second half. The official runner takes the first 32 stored starts. Confirm preparation history before equating the archive with four continuous unmirrored boxes.
- Finite observation divisors, finite network weights, positive inertia, bounded inputs, and feasible terminal arm angles support the exact-ODE lower bound. To give a numerical deadline, compute useful constants and an initial-cost upper bound. A separate proof of optimizer continuity is unnecessary if a uniform bound on every feasible initial plan is established directly.
- Search counterexamples before expensive regional verification. Classify every obligation as analytic, rigorously numerically certified, sampled evidence, or unresolved. Preserve violations. Do not add an acceptance filter, fallback controller, extra seed, cost modification, or LQR switch silently.

## Completed checks in this review

Source inspection confirms the cost indexing, always-enabled frozen terminal value, final-audit semantics, numerical grids, and memory dependence. Independent high-precision arithmetic reproduces base inertia 0.0004021916666667, pendulum inertia 0.000133128, coupling inertia 0.00013158 (all kg m squared), gravity coefficient 0.01518588 N m, and inertia determinant 0.0000000362296758 in squared inertia units. The discussion's uniform lower eigenvalue bound is approximately 0.00005419971915 kg m squared. These are arithmetic checks, not outward-rounded certificates. No GPU run, checkpoint evaluation, archive reconstruction, or end-to-end convergence test was possible from the available local inputs.
