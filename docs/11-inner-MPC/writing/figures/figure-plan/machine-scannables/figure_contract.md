# Figure source and measurement contract

## Authority and layout

The author's latest figure sequence supersedes earlier figure choices and numbering. Figures 1, 4 and 5: wide double-column page-top. Figure 2: single-column approximately half-page height. Figure 3: single-column. Figure 6: optional conventional-performance comparison.

Source manuscript: `docs/11-inner-MPC/writing/latex-project/main.tex`. Figures 1–4 production are authorized; Figures 5–6 remain plans. The Figure 1 production contract is `../../act-coast-overview/design_contract.md`. No controller experiment is authorized by this figure task.

## Figure 1: overview, policy and pipeline

Combine the event timeline and component-role diagram. Outer MPC selects both shut-off `b` and replanning `r`; conditional inner optimization selects only `a_0,...,a_(b-1)`. Predict exact-zero input through the suffix. Frozen actor supplies candidate inputs/terminal-action query and critics supply continuation cost. Teacher supplies demonstrations, not executed MPC torque. No learned timing-policy head or unmentioned LQR handoff.

Use main labels `eq:act-coast`, `eq:act-coast-cost`, `eq:timing-inner`, `eq:timing-outer`. Event notation: `tau=b*Delta`, `h=r*Delta`, current decision `t_j`, shut-off `t_j+tau`, next decision `t_j+h`. Mode and torque are separate. The single continuation evaluation is at replanning; a maximum candidate horizon is not another terminal time. Mark the complete timing-conditioned construction as formulated rather than evaluated.

Current Figure 1 composition: concept first, workflow second. Use muted blue/orange phases, black Arial text at two sizes, and identical rightward/downward workflow arrows. Show sampled timing pairs feeding parallel conditional MPC solves. The shared critic estimate is used in each terminal cost and therefore in the corresponding optimized timing score, counted once. A direct dependency to outer selection must not imply an additional terminal penalty. Keep the retained-label purpose inventory in the production contract.

## Figure 2: apparatus, hard design limits and ideal rest

Use an original editable 3D apparatus informed by the official Quanser product photographs and brochure. Preserve the model's arm-normal hinge axis and mechanism_points link proportions. Cosmetic base, motor and bearing surfaces are illustrative, not measured parameters or a hardware test.

Follow with three orthographic views: top for fixed-reference unwrapped arm travel, side for free pendulum rotation in the arm-normal plane, and front for exact upright rest. Display hard design bounds |u| <= U and |theta| < theta_max, and ideal beta = 0 with omega = nu = 0. Do not draw the earlier tolerance cone. Penalties and finite-tolerance assessment remain described separately; the artwork does not claim exact enforcement, zero-speed success in measured runs, or a certificate. See ../../rotary-pendulum-geometry/design_contract.md.

## Shared production style

Follow Figure 2's geometric illustration style and Figure 1's blue/orange palette and Arial typography for subsequent figures. Use few marks for each relation while covering the full information needed by the narrative. Avoid both decorative complexity and sparse content that leaves the scientific explanation incomplete.

## Figure 3: energy space and sensitivities

Main sources: mechanical model and energy identities in the draft; the existing energy-plane/torque-response assets are historical explanatory material. Additional diagnostic definitions are documented in `docs/9-RL/14-reinforcement-learning-prior/20-stationary-balancing/actuation-opportunities/interpretation_summary.md` and `scheduling_formulation.md`. These are diagnostic sources, not the outer timing-controller definition.

Use physical energy coordinates `(K_a,K_p,V)`, total energy `E=K_a+K_p+V`, and upright-rest target `E_star`. The requested-work plane and mechanically compatible subset are not a complete bounded-input reachable set. A finite-duration torque endpoint curve is parameterized by torque at one declared state and command duration; it is not an instantaneous rate sensitivity.

For the two-energy diagnostic use the relative swing energy from the draft, not full-body pendulum energy:

$$
E_h=\tfrac12J\nu^2+V,\qquad
s_E=\frac{\partial\dot E}{\partial u}=\omega,\qquad
s_h=\frac{\partial\dot E_h}{\partial u}=J\nu(M^{-1})_{21}.
$$

Here `J` is pendulum hinge inertia, `omega` is arm speed, `nu` is pendulum speed, `M` is the coupled inertia matrix, and row 2/column 1 of its inverse maps arm torque into pendulum acceleration. These are instantaneous power sensitivities, not endpoint-energy derivatives.

The error-weighted preferences are

$$
c_E=(E-E_\star)s_E,\qquad c_h=(E_h-E_\star)s_h.
$$

They are the torque coefficients in the derivatives of the respective half-squared energy errors. Both positive favor negative torque for the input-dependent contributions; both negative favor positive torque; opposite signs indicate competing contributions. Zero axes mean zero corresponding direct authority. Do not introduce an equal-weight summed control objective merely to color the regions.

Natural transfer contributes to the complete relative-energy rate. Thus these signs alone do not prove full task progress, suitable coast duration or arm admissibility. The map can retrace under reflected pendulum positions; it is not a global phase clock. Full-body energy `(K_p+V)` cannot silently replace `E_h` in the same sensitivity plot. A shared positive normalization can preserve signs and relative scale; define it explicitly.

Before Figure 3 is inserted, add the required diagnostic definitions to Methods. Verify any numerical source, its energy convention, torque cap, action duration and task criterion. Historical opportunity diagnostics use older teacher settings and cannot be relabeled as current MPC+RL performance. Record availability must be verified, not inferred from a Markdown link. The documented `artifacts/rotary_pendulum/actuation-opportunities/` record path is absent in this checkout; this does not establish that the original records never existed. Verify the original archive or recompute model diagnostics before reuse.

## Figure 4: one record, synchronized traces and configurations

Detailed authority: [Figure 4 production plan](../../session-record/figure_4_plan.md) and [source contract](../../session-record/machine-scannables/source_contract.md). Use vertically matched controller blocks, adjacent expanded-transient/full-remainder views with no time omitted, and the declared transient/holding error ordinates. Four common-timestamp poses link physical configurations to the traces. The current act–coast block remains empty.

Available continuously enabled data: `artifacts/rotary_pendulum/inner-mpc/mpc-rl-transfer/` and raw confirmation under `artifacts/rotary_pendulum/inner-mpc/20261022-lqr-cost-fine-confirm-16/machine-scannables/`.

- Use a complete preselected session, such as the previously fixed downward example 0. Keep the initial condition and full duration; do not select the best-looking success.
- `starts.npz` supplies initial states/groups. State order in `executed` is arm angle, pendulum angle from downward, arm speed, pendulum speed.
- Concatenated executed records have shape `(400,32,100,4)`. The record interval is 0.001 s, and the command interval is 0.1 s. If showing time zero, prepend the actual saved initial state.
- `plans` contains N normalized future commands plus one terminal epigraph scalar. The applied input is the first command per decision, held until its update. Do not plot the epigraph as an action or all future commands as executed torque.
- Current baseline mode is continuously enabled; every command update is not a new off-to-on activation. Record-derived zero torque does not imply coast.
- Reconstruct geometry at chosen timestamps from the state/model, preserving the angle convention and the moving plane. Match markers, poses, input holds and observations exactly in time. Do not imply real hardware footage.
- Define a few keyframe events before layout tuning. Distinguish common elapsed-time snapshots from controller-specific event snapshots.
- The sparse act-coast block is deliberately blank until a genuine matched record and actual mode/event logs exist. Never downsample the continuous record to manufacture a sparse-controlled session.

Existing trajectory assets can guide selection and styling; they do not contain all requested input/mode/keyframe panels. The author explicitly permits a reserved blank act-coast block; identify that absence rather than drawing a speculative trace.

## Figures 5 and 6: final matched scenario data

Mandatory controller families: proposed act-coast; bang-bang; event-triggered MPC; maximum hands-off control; mixed temporal resolution MPC. Current periodic MPC can be an additional internal reference only. Do not assume literature algorithms are implemented or assign them invented numerical values.

Before comparison, select a precise mathematical/reference implementation for each family, its input/feedback convention, horizons, triggers or timing grids as relevant, and a transparent tuning budget. Match dynamics, starts, physical constraints, goal/holding test, assessment duration and hardware. Cross-family objectives may differ; document them instead of changing a baseline's characteristic formulation. Within-method timing ablations require matched inner cost/value conventions.

At minimum the scenario definitions should be explicit and common across methods. The current benchmark groups provide a starting point: downward neighborhood, moving swing-up, near upright and outward arm motion. A pooled atlas column needs a declared equal-scenario or per-start weighting rule. Failed, interrupted or infeasible trials must remain visible, with observed exposure separated from successful fixed-duration resource comparisons.

### Activation and time metrics

- `N_on`: off-to-on mode transitions, including initial enabled activation according to the shared trial-start convention. An always-enabled method can have one activation and long on-time; do not change the definition to force a favorable ranking.
- `T_on`: integral of enabled mode over the common assessment, including enabled zero commands. Distinguish nonzero-input support duration; do not create an undeclared numerical deadband.
- `n(T)`: feedback/planning decisions. A polarity reversal is not necessarily an activation, and event-triggered computation is not necessarily actuation shut-off.
- `C_cmp`: working expansion of “total cmp time,” total online controller computation time. Sum non-overlapping inclusive controller-decision durations, or mutually exclusive subcomponents. Do not add an inclusive outer timer to its nested inner timers.
- Specify inference, event-trigger monitoring, optimization and required verification scope. Exclude offline training, simulation advancement, figure work and unrelated file I/O. Report offline cost separately if needed.
- For parallel solvers/batches, distinguish wall-clock decision latency, summed computational work, amortized throughput and assessment totals. Compare on declared matched hardware and do not convert a batch number into a fictitious per-session latency.

Figure 5 prioritizes activation count, total control time and total computation time, with eligibility and replanning diagnostics. Figure 6 can report hold-completion time, error, residual motion, effort and violations if they add a separate conclusion. Preserve the common scenario/method order and the same failure accounting.

## Production and checks

Use the existing IEEE document layout; exact dimensions and page-top placement are verified during a later rendering/integration task. Export editable vector figures, keep numerical data and provenance separate, and make all labels readable at actual size. Do not invoke the old teacher renderer or change existing image assets in this planning update.

Validate: five required figures plus optional sixth; wide/top 1/4/5; half-page single-column 2; single-column 3; outer timing roles correct; energy definitions distinct; blank sparse block honest; four requested baselines present; metric conventions stated; both LaTeX sources unchanged. No plotting scripts, runtime classes/functions, wrappers, CLI/configuration parameters, training or experiments are introduced.
