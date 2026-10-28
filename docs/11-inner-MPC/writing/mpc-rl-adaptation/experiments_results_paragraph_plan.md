# Paragraph plan: current performance and future experiment placeholders

The [revised whole-paper figure plan](../figures/figure-plan/outline.md) governs all panel choices. Figure 4 uses synchronized dynamics, control and physical keyframes, with the sparse act-coast block left blank. Figure 5 is the final aggregate baseline/resource atlas; Figure 6 optionally covers conventional metrics.

Organize Experiments and Results into exactly four subsections: **A. Predictive control without act–coast**, followed by three short placeholders for intended studies. Only subsection A has supported results. B–D remain headings with keywords and a one-line argument backbone; do not draft speculative result paragraphs or insert invented values.

Use [the writing guidance](../writing_guidance.md). Give subsection A three paragraphs, about 200–250 eventual prose words, with three to five sentences per paragraph. Keep the three future stubs to roughly one short keyword line each in the eventual manuscript. The whole-paper three-page target excluding figure bodies still requires later layout checking. This plan does not change `main.tex`, launch an experiment, or produce a figure.

## IV.A. Predictive control without act–coast

**Subsection purpose:** Establish the current MPC+RL controller's nominal swing-up and continued upright performance before evaluating timing adaptation. Actuation stays enabled and feedback/replanning remains periodic. Zero torque samples do not imply coasting, and these records cannot establish act–coast savings.

### A1. What the current experiment tests

**Purpose:** Give enough protocol to interpret the existing controller and population.

**Sentence sequence:** State the question: can the current learned-continuation predictive controller reach upright and sustain small angular error under the benchmark limits? Identify the nominal benchmark, continuously enabled actuator, periodic first-command replanning, and continued control after reaching upright. Define the initial-state groups, full assessment duration, and holding assessment, with exact settings confined to Results or its caption. Explain that the actor and critics remain active throughout and that no local balancing handoff is used.

**Equations:** Refer to the existing plant, constraint, target, and metric definitions. Do not repeat training losses, MPC optimization, or resource-objective derivations. The baseline has enabled fraction $T_{\mathrm{on}}/T=1$; it is a reference condition, not a measured saving.

**Figure:** No extra figure. A concise caption or small protocol table can carry the initial-state groups and numerical settings needed to read the performance figure.

**Source:** [The official baseline](../../human-readables/mpc_rl_baseline.md) and [baseline transfer guide](../../../../artifacts/rotary_pendulum/inner-mpc/mpc-rl-transfer/interpretation_summary.md). Their current successful baseline is distinct from earlier teacher, hybrid-handoff, and shorter-horizon studies.

### A2. How a representative trajectory connects swing-up and holding

**Purpose:** Illustrate the actual controlled motion, rather than only report an endpoint score.

**Sentence sequence:** Introduce a saved representative trajectory using a declared selection rule. Explain the transition into the prescribed upright band and the subsequent continued control. Use arm angle and torque to show the motion and input that accompany small pendulum error. State that residual arm motion remains even when the pendulum orientation is well regulated.

**Equations:** No new governing equation. Explain absolute upright error $|\beta|$ and arm angle $\theta$ in the figure caption; mark the existing target and arm limits. Use the complete saved trajectory so the reader sees continued regulation after the first arrival.

**Figure — current-performance panel (a):** Synchronized dynamic-state traces, applied input, enabled mode, replanning/shut-off markers and physical keyframes, with target/operating limits. Keep the sparse act-coast block blank until its matched record exists. Use the saved representative-session selection, not a newly selected best outcome. Existing [representative-session assets](../../../../artifacts/rotary_pendulum/inner-mpc/mpc-rl-transfer/human-readables/representative_sessions.png) use fixed session indices. Inspect and simplify the asset when later preparing the paper; do not infer behavior outside the saved duration.

### A3. What the population supports and what remains unresolved

**Purpose:** Quantify performance across the saved starts and state its limits.

**Sentence sequence:** Report the complete-session count and the denominator meeting the declared final holding criterion. Quantify upright error and uninterrupted residence using the saved metrics and their assessment intervals. Pair these with arm motion and constraint-margin information so small orientation error is not mistaken for full-state convergence or an invariant target set. End with the scope: finite-duration nominal performance of the combined inner controller, without isolated terminal-value benefit, resource savings, robustness, or a continuous-time certificate.

**Equations:** Reuse Section II definitions, with the exact post-entry or final assessment interval stated before reporting a holding-error statistic. If band residence is retained, use the existing definition and explain its indicator; do not duplicate the same metric under several names. Keep pooled error separate from averages of per-session errors.

**Figure support:** Keep the current all-session holding/error results in the prose. Figure 5 will aggregate final matched act-coast and baseline scenarios, not merely reproduce the current baseline hold plot. Optional Figure 6 can compare conventional quality metrics when those matched records exist. Do not use current continuously enabled results to fill missing act-coast or baseline cells.

**Evidence boundary:** Use the current portable MPC+RL records. The older teacher-study numerical paragraph and initial-state table have been replaced in `main.tex` by the current baseline Results. Do not transfer their large population count to this controller. Actor-only long runs and hybrid LQR handoff results are separate experiments.

## IV.B. Act–coast task–resource tradeoff — placeholder

**Purpose:** Test resource reduction while preserving the same task and constraints.

**Keywords:** matched starts and cost/value conventions; periodic baseline; joint outer timing; success/violations; enabled fraction; decision rate; input effort; task error.

**Backbone:** matched protocol → task satisfaction → resource/performance tradeoff.

**Figure cue:** task/resource comparison with failures visible; no proposed improvement drawn as data.

## IV.C. Contribution of joint timing optimization — placeholder

**Purpose:** Identify what adapting both shut-off and replanning adds.

**Keywords:** both times optimized; shut-off fixed; replanning fixed; same inner controller and comparison budget.

**Backbone:** timing restrictions → matched closed-loop tests → contribution of each timing choice.

**Figure cue:** compact paired comparison, or computed timing-score map if it answers the coupling question.

## IV.D. Admissibility and operating limits — placeholder

**Purpose:** Locate useful coast durations and test any derived admissibility condition.

**Keywords:** coast-duration limits; feasible timing pairs; arm constraints; fine trajectory replay; violations; conditional certificate if established.

**Backbone:** declared condition/domain → limiting cases → supported region and failures.

**Figure cue:** sampled timing-feasibility map, or analytical bound versus replay only when that bound exists.

## Figure and claim discipline

One current-performance figure with representative-motion and population panels is enough for IV.A. The three future figure cues are planning stubs, not demands to create plots now. Reuse the generic timing overview from Methods rather than repeat it in Results. Every plotted comparison must answer its subsection's question and identify its protocol, denominator, example-selection rule, and assessment interval. To isolate timing effects, match the inner cost and continuation conventions; otherwise declare the changed objective as part of the comparison.

## Writing milestones and scope

| Milestone | Validation | Success flag |
|---|---|---|
| Current controller identity | Check baseline and transfer records. | Continuously enabled MPC+RL; no coast or local-controller handoff implied. |
| Metric provenance | Trace each quantity to a saved trajectory or summary. | Denominators, intervals, pooled statistics, and failures correctly identified. |
| Figure purpose | Match each panel to A2 or A3. | Complete motion and population evidence, without redundant plots. |
| Placeholder brevity | Inspect IV.B–D. | Exactly three keyword/backbone stubs; no invented outcomes. |
| Later integration | Replace the old teacher Results and verify references. | Current inner-controller evidence and intended timing experiments remain distinct. |

[The source/evidence contract](machine-scannables/bilevel_results_plan_contract.md) contains record paths and exact available values. New runtime classes, objects, functions, methods, wrappers, helpers, CLI/configuration parameters: zero. This is a writing plan only; keep the minimum-realization discipline and add no fallback or tuning infrastructure.
