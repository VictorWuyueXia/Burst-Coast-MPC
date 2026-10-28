# Whole-paper figure plan

This plan implements the author's revised sequence: **overview → benchmark geometry and task → energy-control interpretation → matched session records and configurations → aggregate act–coast resource performance → optional conventional performance comparison**. Figures 1–5 are the planned main figures; Figure 6 is optional. This replaces the earlier three-current/two-future recommendation, including its restrictions on standalone geometry and energy figures.

Use simple labels, one connected argument, and a distinct purpose for each panel. The fixed layouts below are author requirements. Keep theory schematics symbolic and give measured plots the units and assessment definitions needed for interpretation. Figures 1–4 have been rendered under subsequent author instructions; Figures 5–6 remain plans. No controller experiment is included in this figure work.

## Figure map and layout

| Figure | Main purpose | Required layout | Material/status |
|---|---|---|---|
| 1. Act–coast overview and pipeline | Explain the policy and which component does what | Wide, double-column, page top | Schematic from the formulation; identify proposed timing-conditioned operation |
| 2. Rotary-pendulum geometry, constraints, and goal | Define the chosen benchmark and the task visually | Single column, about half a page high | Geometry/model available; symbolic constraints and goal |
| 3. Energy control, sensitivities, and regions | Explain the mechanism and local input opportunities | Single column | Current-model diagnostics recomputed; editable figure and source data available |
| 4. One-session dynamics, control, and key configurations | Explain actual behavior on a common time axis | Wide, double-column, page top | Recorded continuous session and physical keyframes rendered; matched act–coast block left blank |
| 5. Aggregate act–coast resource atlas | Compare the proposed features across scenarios and baselines | Wide, double-column, page top | Final act–coast and matched baseline evidence still required |
| 6. Conventional control metrics, optional | Check the control quality accompanying resource differences | Optional; width chosen for legibility, double-column if an atlas | Same matched comparison runs; include only if informative |

The page-top double-column requirements apply to Figures 1, 4, and 5. Figure 2 should occupy approximately half the page height, allowing space for its caption. Determine exact physical dimensions from the existing IEEE layout when rendering; do not shrink labels to force the requested arrangement.

## Figure 1. Act–coast overview and component pipeline

**Purpose:** Introduce act–coast control as the paper's central object. A reader should understand the sequence of enabled input, coast, and renewed feedback, then see which components construct that sequence. Combine the earlier policy timeline and controller-construction diagrams into this one overview.

**Main visual:** Show the act–coast concept first in panel a: a shared input/mode timeline, blue enabled operation followed by orange coast, and concise markers for planning, shut-off and replanning. An enabled zero command remains visible. Panel b follows with a rightward workflow: observed state → timing samples → parallel conditional inner MPC solves → outer selection → event execution. Each sampled pair receives its own compatible optimized input sequence. Miniature event bars and input traces illustrate these roles without descriptive paragraphs.

**Supplementary elements:** Small actor and critic network icons sit above the workflow. The actor supplies candidate inputs to the inner batch. One shared critic-based terminal cost informs both conditional input optimization and the event scores used for timing selection; draw separate downward dependencies to the batch and selector. The latter is not a second terminal penalty. Explain teacher training, continuation interpretation and repeated feedback in the caption or Methods rather than adding another branch or return loop.

**Symbols and interpretation:** Retain only phase names, input $u(t)$, mode $m(t)$, elapsed-time markers $0,\tau,h$, component-role labels, conditional score $S$ and selected quantities marked with stars. Define them in the caption. Outer MPC samples both shut-off duration $\tau$ and event duration $h$; inner optimization is conditional on each pair. The complete event includes coast and one continuation evaluation at replanning. Separate decision markers from mode changes.

**Visual discipline:** Use Morandi blue and orange as the two semantic colors, with their pale fills and neutral outlines. No additional accent color is needed. All text is black in Arial at exactly two sizes. Every workflow arrow has one simple style and points rightward or downward. Keep exact duration-grid relations and status qualifications in the caption. The retained-label inventory and purpose of each text group are recorded in `../act-coast-overview/design_contract.md`.

**Narrative link:** Lead the Introduction and Problem Formulation with this figure. Methods refers back to its component roles and its bi-level timing/input split instead of adding another overview. Keep the benchmark geometry out of this main pipeline; it receives Figure 2.

**Status and caption:** Label the full timing-conditioned construction as formulated, not evaluated. Existing learned ingredients and the continuously enabled baseline do not establish the complete scheduled controller. The figure is a schematic, not a measured saving, hardware power diagram, or certificate. Do not add a learned timing policy or local LQR handoff.

## Figure 2. Rotary-pendulum geometry, constraints, and goal

**Purpose:** Give the reader a complete visual definition of the benchmark without treating its mechanics as the contribution. The figure must identify the actuated and passive motions, the allowed operating range, and the desired orientation. It supports the benchmark formulation rather than merely illustrating a pendulum.

**Layout:** Single-column, approximately half-page block. Draw an original shaded 3D apparatus informed by online reference photographs, followed by three orthographic views: top, side and front. Make the servo base, motor shaft, arm, hinge and slender rod recognizable.

**Required content:** Show driven and passive components and positive shaft torque. Top view defines the unwrapped arm angle from a fixed horizontal reference and a sharp allowed sector. Side view shows free rod rotation in the arm-normal vertical plane, with alpha measured from downward. Front view shows exact upright rest. Do not invent a pendulum stop or add distracting velocity arrows.

**Constraints and ideal goal:** Display the hard design requirements $|u|\le U$ and $|\theta|<\theta_{\max}$, with positive prescribed bounds. Show ideal upright rest through $\beta=0$ and $\omega=\nu=0$; define error and speeds in the caption. This supersedes the previous tolerance-band drawing. Keep actual soft penalties and finite-tolerance assessment in the text; do not imply exact enforcement or a certificate.

**Data/schematic needed:** Use official apparatus photos only as visual references, without copying images or branding. Build original editable solid faces using the benchmark's mechanism_points link geometry. Cosmetic housing dimensions and the illustrative sector are not measured parameters. Use Figure 1's palette and typography. Label purposes and checks are in ../rotary-pendulum-geometry/design_contract.md.

**Narrative link:** Place in the benchmark Method subsection and use it to prepare the governing equations. The caption defines signs, constraints, and goal; the prose explains why these quantities matter.

## Figure 3. Energy control in energy space, with sensitivities and regions

Production: [design contract](../energy-control/design_contract.md) and [figure guide](../energy-control/interpretation_summary.md). Current-model zero-input diagnostics replace unavailable historical sensitivity records for this illustration.

**Purpose:** Explain how energy feedback requests a change and how the physical state affects available input directions. Show both energy geometry and the limits of a local sensitivity interpretation. This figure supports the mechanical teacher and physical reasoning; it does not introduce a second timing controller.

**Layout:** One single-column figure with a few vertically organized, readable panels. Keep the visual sequence energy-space request → sensitivity curves → sign/preference regions. Avoid fitting every historical diagnostic into the same column.

**Panel (a), energy space:** Use the manuscript's physical coordinates $(K_a,K_p,V)$: arm-body kinetic energy, full pendulum-body kinetic energy, and potential energy. Show the upright-rest target, the requested-work energy plane, its mechanically compatible region if used, and a bounded-input endpoint curve for one declared state and action duration. Explain that mechanical compatibility is different from finite-duration reachability, and a constant-input endpoint curve is not a complete reachable set. Energy axes can share normalization by the upright-rest energy $E_\star$.

**Panel (b), sensitivity curves:** Along a declared model trajectory or recorded illustrative segment, show how the two input sensitivities change. Use total mechanical energy $E$ and explicitly identified relative swing energy $E_h=\tfrac12J\nu^2+V$, where $J$ is pendulum hinge inertia and $\nu$ its angular speed. Their sensitivities are $s_E=\partial\dot E/\partial u$ and $s_h=\partial\dot E_h/\partial u$: the instantaneous change in energy rate per unit torque. Define the model-derived expressions and the chosen segment before drawing them. Do not substitute full pendulum-body energy for $E_h$ under the same label.

**Panel (c), regions:** Use the error-weighted preferences $c_E=(E-E_\star)s_E$ and $c_h=(E_h-E_\star)s_h$. Same-sign regions admit a common torque sign for reducing both input-dependent error contributions; opposite-sign regions expose a conflict. Define zero-authority axes. Include a traced sensitivity/preference curve if useful, but do not call it a unique phase coordinate or an invariant orbit.

**Data and mathematical boundaries:** Use the dynamics, energy map, a declared state/segment, and verified prediction/sensitivity calculations. Existing energy assets and written sensitivity diagnostics are reference material, not current-controller performance records. The documented sensitivity-record directory is absent at its stated local path, so verify the original archive or recompute declared diagnostics before plotting. Distinguish instantaneous sensitivity, natural energy exchange, and a finite-duration torque-response curve. Region colors do not guarantee complete energy improvement, arm admissibility, or successful coasting. Before insertion, Methods must define the relative energy and sensitivity relations needed to read this new figure.

**Narrative link:** Explain the teacher's energy request and why energy coordinates require physical-state information. The sensitivity/region panels can motivate looking ahead through an event, but must not imply that the current outer MPC selects times by a quadrant heuristic.

## Figure 4. One-session record with synchronized controller behavior and key configurations

Detailed production plan: [composition and element purposes](../session-record/figure_4_plan.md), [reading guide](../session-record/interpretation_summary.md), and [record contract](../session-record/machine-scannables/source_contract.md). The detailed plan specifies two adjacent linear time views with no omitted interval, a tolerance-based late-error ordinate, and four common-timestamp physical keyframes. It supersedes less specific layout choices in this overview.

**Purpose:** Let the reader connect system motion, control behavior, and physical configurations in one session. Compare continuously enabled control and sparse act–coast execution using the same elapsed-time interpretation. The continuous-control record can be populated now; leave the entire act–coast session block blank until its data exist.

**Layout:** A wide page-top double-column figure with two matched session blocks. Within each block, align all dynamic traces, input/mode information, decision markers, and configuration timestamps to the same time axis. Use the same axis ranges and elapsed-time span across the two controllers when comparing the same start. A stacked arrangement can preserve full-width time axes; reserve the second block for act–coast without drawing speculative curves.

**Dynamic rows:** Show upright error and arm angle, with the goal and arm bounds. Include arm and pendulum speeds in a separate aligned row if needed to show the full state and residual motion. Preserve the full transient and the holding interval rather than displaying only a successful tail. Handle angular wrapping so plotting connections do not create false upright visits.

**Controller rows:** Show applied input, a separate enabled/coast indicator, and distinct markers for replanning and shut-off. These must be recorded controller events. Zero torque during enabled operation is not coast. The continuous controller's mode stays enabled even when its torque is zero; its periodic replanning is distinct from actuator activation.

**Key configurations:** Add a small strip of reconstructed pendulum poses linked by timestamp markers to the traces. Select a few informative instants by a declared rule, such as the actual initial state, a large swing excursion, the start of a qualifying hold, and late holding. Reconstruct each pose from that record's state and model geometry, not from an artist-selected pose. These are renderings of simulated configurations, not hardware photographs. Use common timestamps for direct cross-controller pose comparisons, or clearly label controller-specific event times.

**Data needed:** One preselected complete continuous-control session, its applied commands and update times, full state samples, and geometry mapping. The later act–coast record must use a matched start/protocol and log actual mode transitions and replanning times. Preserve a blank block now; do not duplicate, extrapolate, or sparsify the current record to fabricate act–coast behavior.

**Narrative link:** Support current Results A with the populated record, then use the matching sparse record when the act–coast experiments are completed. The caption distinguishes what is present from the reserved block and defines the keyframe selection and time alignment.

## Figure 5. Aggregate final act–coast performance atlas

**Purpose:** Evaluate the method's distinguishing resource features across a declared scenario cohort. Put task eligibility alongside the resource metrics so failures cannot appear as efficient control. This is the main final comparison atlas, not a plot that can be populated from the current continuously enabled records alone.

**Layout:** A wide double-column page-top atlas. Use consistent method order/colors and scenario groupings across metric panels. A useful backbone is metric rows by scenario columns, with a compact success/constraint-status strip and an optional pooled summary under declared scenario weighting. Each row answers a different resource question, not a different plotting style.

**Required baselines:** Bang–bang control; event-triggered MPC; maximum hands-off control; mixed temporal resolution MPC. The proposed act–coast controller is compared with all four. The present periodic continuously enabled MPC may be added as an internal reference, but cannot replace any requested baseline.

**Primary metrics:** Activation count, total control time, and total controller computation time, with replanning count as a separate diagnostic. The working interpretation of the author's “total cmp time” is total online controller computation time. Before logging, define activation as off-to-on mode change, control time as accumulated enabled duration, and computation time through a non-overlapping online timing boundary. Detailed definitions appear below.

**Comparison protocol:** Use matched initial states/scenario groups, the same governing dynamics, actuator and state limits, task/holding criteria, assessment durations, and transparent tuning and hardware budgets. Select and document an actual mathematical implementation for each named baseline before comparing it. Retain each family's declared objective rather than silently changing it to imitate this controller. For timing ablations within our method, keep cost/value conventions matched.

**Aggregation and failures:** Plot scenario-level distributions or paired results with visible denominators, not only favorable means. Declare whether the pooled result weights scenarios equally or weights individual starts. Report failed/terminated runs and their exposure separately so a short failed run is not a resource saving. Do not fill missing measurements with zero or invent baseline results.

**Data needed:** Final act–coast and baseline trajectories, actual enabled modes, activation/replanning logs, applied inputs, task/constraint outcomes, and online computation records. Cross-method comparisons require matched hardware and clear parallelism/batch accounting. Existing GPU batch throughput is not single-session feedback latency.

**Narrative link:** Lead the aggregate act–coast comparison in Results. It can include internal timing ablations as supplementary comparisons, but the four requested external baseline families remain central. A formal admissibility region or certificate is not a required panel in this resource atlas unless an actual result warrants it.

## Figure 6. Conventional controller metrics — optional

**Purpose:** Show the control quality accompanying the resource differences, if Figure 5 and the text do not already resolve it. Keep these conventional metrics separate from the act–coast-specific resource claim.

**Content:** Compare the same methods and matched cohort using selected task success/hold completion, upright tracking error, residual arm motion, control effort, and constraint-violation or margin measures. Define assessment intervals and whether an error is pooled or averaged across runs. Use only a small set that adds information; do not duplicate activation, enabled-time, or computation panels.

**Data and status:** Reuse the actual matched comparison records required for Figure 5. If there is no additional substantive conclusion, omit this figure or use a compact table. No conventional-quality comparison is currently established by the baseline-only records.

## Metric definitions for the resource atlas

| Metric | Operational meaning | Required distinction |
|---|---|---|
| Activation count $N_{\mathrm{on}}$ | Count off-to-on actuator-mode transitions, including an initial activation only when the trial starts enabled | Not replanning count; not positive/negative torque sign switches |
| Total control time $T_{\mathrm{on}}$ | Accumulated enabled duration over common assessment duration $T$ | Enabled zero torque remains enabled; do not silently replace with nonzero-input support |
| Total computation time $C_{\mathrm{cmp}}$ | Sum of measured non-overlapping online controller-computation durations | Include nested outer/inner work once; do not double-count inclusive outer time and inner calls |
| Replanning count $n(T)$ | Number of new feedback/planning decisions under the declared counting convention | Event-triggered computation does not automatically imply actuator shut-off |

With mode $m(t)=1$ while enabled and zero while coasting, the total control time is

$$
T_{\mathrm{on}}=\int_0^T m(t)\,\mathrm dt.
$$

For baselines without an explicit actuator mode, establish a common, published mode/activation convention before computing these metrics. Report nonzero-input support separately if needed; do not insert an undeclared torque threshold. Specify whether verification and inference are inside the online computation boundary, exclude offline training and figure-generation work, and distinguish wall-clock latency from aggregate computational work or amortized batch throughput.

## Shared presentation and evidence rules

- Figures 1–3 explain concepts and mechanisms; they do not prove the complete controller works. Figures 4–6 use actual records and distinguish existing from future evidence.
- Preserve the requested sizes and page-top placements. Later use the existing document class's single/double-column float mechanisms and verify placement visually; this plan does not modify layout code.
- Define every symbol, curve, region, metric and normalization when introduced. Regions in energy space, sensitivity-sign regions, and state constraint sets have different meanings.
- Keep colors and labels restrained and consistent. Put numerical settings/units in measured plots or captions when needed, while keeping overview/geometry schematics symbolic.
- Caption space counts toward the existing three-page non-figure target. Account for five main captions rather than reverting to the older smaller figure set, and avoid repeated explanations or tiny labels.
- No baseline ranking, resource saving, superiority, global optimality or certificate is claimed without the corresponding experiment or proof.

## Milestones and scope

| Milestone | Validation | Success flag |
|---|---|---|
| Figure sequence | Check all six purposes and the fixed layouts | Figures 1–5 plus optional 6 match the author's requested order and sizes |
| Mathematical definitions | Check geometry, goal, energy coordinates, sensitivities and regions | No physical/relative-energy substitution or false phase/feasibility interpretation |
| Data alignment | Trace each Figure 4 curve/pose/event to one time-stamped record | Continuous block is genuine; act–coast block remains blank until data exist |
| Baseline/metric protocol | Fix baseline versions, mode semantics and computation boundary | All four baselines are represented fairly and resource counts are comparable |
| Later production | Render only from declared schematics/records and inspect at final size | Legible panels, truthful missing-data treatment, correct time axes and page-top placement |

This task updates Markdown planning artifacts only, with Chinese companions. New plotting scripts, functions, classes, wrappers, CLI options and configuration parameters: zero. Do not run old renderers, generate figures, perform training/simulation, or edit `main.tex` here. Backend selection remains deferred to a later rendering task. Keep source data and historical figures intact and follow the repository's minimum-realization discipline.

[Reading guide](interpretation_summary.md) · [Detailed source/data contract](machine-scannables/figure_contract.md)
