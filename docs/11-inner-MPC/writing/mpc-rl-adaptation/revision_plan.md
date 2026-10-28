# Subsection plan for the act-coast controller paper

The narrative is **intermittent actuation and feedback timing -> coupled timing/input decisions -> controller synthesis and analysis -> benchmark evaluation**. The paper concerns the act-coast controller, its construction algorithm, and provable or quantifiable properties. The rotary pendulum is the chosen benchmark; it is not the motivation or principal contribution.

This plan replaces the previous benchmark-led structure. The [opening paragraph plan](intro_formulation_paragraph_plan.md) governs three Introduction paragraphs and five Formulation paragraphs. Use the [standing writing guidance](../writing_guidance.md): simple grammar, one continuous argument, three to five sentences per paragraph, motivation before equations, and immediate definitions and intuition. Separate the proposed controller-level problem from the current fixed-horizon inner-controller implementation described in [current_progress.md](current_progress.md).

## Proposed structure and space budget

| Section or subsection | Single purpose | Approximate page budget |
|---|---|---:|
| I. Introduction | Motivate act-coast control, its coupled decisions, and the intended algorithm/analysis. | 0.30 |
| II. Problem Formulation | Define the generic plant, event policy, task, constraints, costs, and evaluation criteria. | 0.50 |
| III.A. Chosen benchmark: dynamics and energy representation | Specialize the plant and task to the rotary pendulum. | 0.35 |
| III.B. Mechanical teacher | Explain the source of demonstration trajectories. | 0.10 |
| III.C. Stationary policy and continuation learning | Explain the learned inner-control ingredients and their value interpretation. | 0.30 |
| III.D. Predictive inner control | Define conditional input prediction, constraints, terminal scoring, and numerical execution. | 0.40 |
| III.E. Bi-level outer MPC | Formulate both timing choices, conditional input optimization, and event execution. | 0.35 |
| IV. Experiments and Results | Current non-act-coast performance plus three future-study stubs. | 0.35 |
| V. Conclusion | State the delivered controller, properties, and demonstrated scope. | 0.05 |
| Captions and references | Explain figures and retain sources needed by the argument. | 0.30 |
| **Total** | Planning budget excluding figure bodies. | **3.00** |

The combined Introduction/Formulation target is approximately 550-650 prose words and two principal display groups. These allocations are not rendered page measurements. If space is insufficient, compress teacher/search/training details and repeated explanations, not symbol definitions or the core act-coast construction. An optional abstract must fit within this total. Do not shrink typography to conceal an oversized argument.

## I. Introduction

**Purpose.** Establish the act-coast research question before the plant or learning method.

**Paragraph sequence.** I1 motivates alternating bounded actuation with unforced evolution and explains why lower enabled time and fewer decisions matter only when task and constraints are preserved. I2 explains how input sets the switch-off state and coast duration sets the next decision state, so timing and actuation must be assessed together. I3 introduces outer MPC selecting both shut-off and replanning times with conditional inner input optimization, identifies algorithmic and analytical outputs, and names the pendulum only as the benchmark.

**Mathematical content.** No equations. Define coasting as zero commanded input. Introduce continuation evaluation because feedback is delayed; do not make learned terminal costs a separate competing motivation. Reserve algorithm and proof details for Methods.

**Figure and claims.** Figure 1 explains event execution; Figure 1 explains the decision structure. State only delivered contributions. Both timing decisions are selected by the formulated outer MPC; a learned timing policy is outside this plan. A benchmark-centered or energy-decoder-centered title must be reconsidered when the final algorithm and properties are established.

## II. Problem Formulation

**Purpose.** Specify the problem to solve, the allowed controller, and how success is judged.

**Paragraph sequence.** P1 defines the known nominal nonlinear plant, available state feedback, initial-state set, and state/input constraints. P2 defines event duration, enabled cutoff, planned bounded input followed by exact-zero coast, and the next feedback time. P3 defines a target neighborhood and uninterrupted hold requirement. P4 defines a task/input running cost plus enabled-duration and decision-count charges. P5 states the synthesis/analysis objective and performance measures.

**Mathematical content.** Use exactly the model, event-input law, target/hold definitions, and resource objective in the [detailed plan](intro_formulation_paragraph_plan.md). Explain each symbol and its source immediately. Distinguish enabled zero input from coasting, and continuous-time admissibility from finite-grid checking. Keep nested optimization, learning equations, pendulum coordinates, inertia coefficients, and numerical command intervals in Methods.

**Evaluation and analysis.** Define task success, hold-completion time, violations, enabled fraction, decision rate, squared input effort, and task error here; put measured values and protocols in Results. Seek a computable sufficient condition for admissible evolution until feedback resumes. Recursive feasibility, invariance, convergence, or robustness require separate stated assumptions and proofs.

**Transition.** With the generic task fixed, Methods introduces the chosen benchmark and constructs the controller ingredients for it.

## III.A. Chosen benchmark: dynamics and energy representation

**Purpose.** Introduce the rotary pendulum explicitly as the chosen benchmark and supply its prediction model.

Move geometry, coordinate conventions, angular wrapping, signed velocities, torque direction, and full physical-state observation from the old Section II to the start of this subsection. Then use the [four-paragraph dynamics plan](dynamics_teacher_paragraph_plan.md): mechanical assumptions and Lagrangian, governing dynamics, energy partition, and phase-dependent power exchange. Remove the current text's dependence on coordinates defined in Section II. Define benchmark task-error components and specialize the generic admissible sets without inserting configured values.

Retain the compact inertia-matrix relation and necessary physical definitions. Explain why the energy map cannot replace the physical state for prediction. Numerical mechanics, integration choices, and task tolerances belong in Results or reproducibility material.

## III.B. Mechanical teacher and demonstration trajectories

**Purpose.** Explain how mechanically informed reference trajectories support learning.

Retain the two short teacher paragraphs in the [focused plan](dynamics_teacher_paragraph_plan.md). Explain the work request and bounded torque realization, then the augmented teacher's role in providing swing-up and holding demonstrations. Keep the teacher subordinate to the act-coast controller construction. Omit energy-region algebra, root-search implementation, and an independent teacher-performance narrative.

## III.C. Stationary policy and continuation learning

**Purpose.** Define the current learned ingredients and exactly what their values estimate.

Retain the four learning paragraphs in the [RL/MPC plan](rl_mpc_paragraph_plan.md): physical observation and actor, continuing return, twin-critic learning, and improvement with retained imitation. Define rewards, discount, targets, and losses where introduced. Preserve the distinction between the teacher's physical energy and the student's energy proxy. Numerical training settings belong outside theoretical prose.

The existing critics estimate learned torque-policy continuation under their training reward. They are not a certified bound or an event-level value for repeated future coast intervals and decision charges. Any later scheduling value needs its own reward, elapsed-time, and discount contract. Do not silently change the existing networks' interpretation.

## III.D. Nonlinear predictive inner control

**Purpose.** Explain constrained input prediction and the role of the learned endpoint score.

Retain the three MPC paragraphs in the [focused plan](rl_mpc_paragraph_plan.md), with the current objective and execution convention explicit. Define planned inputs, predicted states, dynamics, constraints, and the terminal score before using them. Preserve the existing distinction between the undiscounted quadratic running sum and the critic's discounted training meaning; an event-level reformulation must acknowledge any changed objective.

Describe finite CEM search as selection among candidates, not a proof of global optimality. The current baseline executes its first held command and replans. A full act-coast controller must instead specify the event execution described in Section II and assess the zero-input suffix explicitly. Do not present the current first-command baseline as an implemented outer scheduler.

## III.E. Bi-level outer MPC for shut-off and replanning

**Purpose.** Formulate the recently defined timing layer without importing older horizon-learning or inner-cutoff designs.

Follow [bilevel_optimization_paragraph_plan.md](bilevel_optimization_paragraph_plan.md): E1 assigns both shut-off and replanning to outer MPC; E2 defines the input-only conditional solve and complete enabled/coast prediction score; E3 compares timing pairs jointly; E4 specifies full-event execution and the scope of admissibility claims. Use four paragraphs and two compact display groups. Retain the recent interface's cost/discount conventions as explicitly formulated, including their unresolved relation to the resource objective and current actor tail. Do not present a scheduler or certificate as implemented.

## IV. Experiments and Results

Follow [experiments_results_paragraph_plan.md](experiments_results_paragraph_plan.md). Use exactly four subsections, with evidence only in IV.A:

- **IV.A. Predictive control without act-coast:** three paragraphs for protocol/current controller, representative swing-up and continued holding, then all-session metrics and limits. Use the official continuously enabled MPC+RL records. Replace the manuscript's older teacher-study numerical paragraph when Results is drafted.
- **IV.B. Act-coast task-resource tradeoff:** keywords only; matched protocol -> task satisfaction -> resource/performance tradeoff.
- **IV.C. Contribution of joint timing optimization:** keywords only; timing restrictions -> matched tests -> contribution of adapting each time.
- **IV.D. Admissibility and operating limits:** keywords only; declared condition/domain -> limiting cases -> supported region and failures.

The last three subsections are brief placeholders, not result narratives. Pair one saved motion illustration with population evidence for IV.A. Future figures require their own measured comparison or established analytical condition. Put numerical settings and reported values in Results; keep denominators, intervals, and failures visible. No new experiment is run by this plan.

## V. Conclusion

State the controller algorithm actually delivered, the properties established under explicit conditions, and the scope of the matched benchmark evidence. Name any remaining analytical or empirical limitation without upgrading a learned score or nominal observation into a guarantee.

## Figures with defined jobs

Follow the [revised whole-paper figure plan](../figures/figure-plan/outline.md), which supersedes the earlier figure sequence.

- **Figure 1:** double-column page-top act-coast overview, combining the policy timeline and component pipeline.
- **Figure 2:** single-column, approximately half-page benchmark geometry, constraints and goal.
- **Figure 3:** single-column energy-space control, sensitivity curves and regions.
- **Figure 4:** double-column page-top one-session dynamics/control record with synchronized physical keyframes; leave the act-coast block blank until real data exist.
- **Figure 5:** double-column page-top aggregate scenario atlas, comparing activation count, total control time and total computation time with bang-bang, event-triggered MPC, maximum hands-off control and mixed temporal resolution MPC.
- **Figure 6, optional:** conventional control-quality metrics for the same matched comparisons.

The overview keeps the act-coast controller central. Dedicated benchmark and energy figures now follow the author's explicit requirements. No figure is produced by this plan.

## Draft adaptation map

| Existing material | Planned treatment |
|---|---|
| Introduction | Replace completely with I1-I3 in the detailed opening plan. |
| Problem Formulation | Replace completely with generic P1-P5. |
| Pendulum-specific Section II definitions | Relocate to III.A before governing equations; repair dependencies. |
| Existing teacher, learning, and predictive control | Retain accurate component content under III.B-III.D and its focused plans. |
| Outer timing and mathematical properties | Include only the delivered algorithm and established results in III.E. |
| Results metrics | Define generic measures in Section II; keep settings and measured outcomes in Results. |
| Title and figure callouts | Align with the controller-level contribution and actual figure availability. |

Use headings and equation labels as anchors, not obsolete source line numbers.

## Revision milestones and success flags

| Milestone | Validation procedure | Success flag |
|---|---|---|
| Paragraph argument | Read I1-I3 and P1-P5 opening ideas. | One continuous act-coast problem; no benchmark-led motivation. |
| Mathematical contract | Trace state, inputs, feedback times, target, constraints, cost, and metrics. | All symbols defined with immediate interpretation; no hidden execution or cost change. |
| Contribution evidence | Compare each algorithm/property claim with code, proof, or recorded assessment. | Proposed scheduler distinguished from implemented inner control; no unsupported certificate. |
| Figure/evaluation logic | Match panels and metrics to questions and available evidence. | No decorative plots, imagined savings, or hidden failures. |
| Later typesetting | Compile the existing project and inspect rendered pages after prose editing. | Three-page non-figure target verified, definitions relocated consistently, references resolved. |

These are writing milestones, not new development phases. This task updates existing plans and their bilingual companions. New runtime classes, objects, functions, methods, data classes, wrappers, helpers, runtime variables, CLI options, and controller/configuration parameters: zero. Preserve compact continuous logic and minimum realization. Do not change controller code, train models, generate figures, or edit `main.tex` in this planning task.
