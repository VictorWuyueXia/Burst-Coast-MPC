# Paragraph plan: act–coast control as the research problem

The Introduction plan is replaced in response to the author's application and literature comments. Its argument is **costly interventions → input duration and restart frequency → prediction and feedback timing → joint timing-conditioned input design**. The five-paragraph Problem Formulation plan below is retained unchanged. Benchmark mechanics remain in Methods.

Apply the [standing guidance](../writing_guidance.md) and the control-academic-writing skill. Use four Introduction paragraphs and five Formulation paragraphs, each with one purpose and three to five connected sentences. The Introduction has no displayed equations; Formulation retains its two principal display groups. The expanded literature requirement supersedes the old combined 550–650-word estimate; preserve the whole-paper three-page target excluding figures through a later compression and rendered-layout check, not by deleting definitions.

The bi-level subsection formulates outer MPC selection of both shut-off and replanning times. The reported current performance concerns continuously enabled MPC+RL. Certificates and act–coast savings remain unsupported until separately established.

## Scope and companion plans

This file governs the opening rewrite. The [paragraph map](../paragraph_plan.md) and [subsection revision plan](revision_plan.md) point to the updated Introduction contract. The [figure outline](../figures/figure-plan/outline.md) retains the existing figure assignments. Detailed sources and comparison boundaries are separate in the [literature ledger](machine-scannables/intro_literature_ledger.md). This task changes plans only; it does not edit the open manuscript or its bibliography.

## Introduction

Use **four physical paragraphs in three argument blocks**: I1 applications; I2a–I2b related approaches and the remaining design question; I3 our construction and its intended benefit. Splitting the literature block prevents five method families from becoming a list inside one dense paragraph. Target approximately 340–400 English prose words, with three to five sentences per paragraph and no displayed equations. The whole-paper three-page target excluding figures remains a layout constraint to check during drafting.

### I1. Applications make intervention itself a resource

**Purpose:** Explain why controlling well can require fewer interventions, before discussing a benchmark or an algorithm.

**Opening idea:** Some controlled systems incur a cost each time actuation resumes, in addition to the cost of keeping it active.

**Sentence sequence:**

1. Introduce intermittent intervention through spacecraft propulsion: a maneuver changes the trajectory, followed by motion with the selected propulsion input off.
2. Explain the distinct resource issue: some restartable propulsion systems have finite restart capability or require startup resources, so short total burn duration alone does not resolve the operating cost.
3. Use solar sailing as a second, carefully qualified example: changing sail orientation changes the radiation-pressure force; sparse reconfiguration is a design motivation when reorientation is costly.
4. Define our act–coast concept as planned bounded actuation followed by zero commanded input, with natural dynamics continuing throughout coast.
5. Close with the common design need: choose the input, shut-off time, and next feedback decision together while preserving task performance and constraints.

**Examples and their jobs:**

| Example | Necessary contribution to the argument | Boundary and writing decision |
|---|---|---|
| Propulsive spacecraft or upper-stage maneuvers | Separates burn duration from restart frequency and directly motivates a zero-input coast interval. | Say “finite restart capability” or “startup resources.” The verified sources do not establish a universal stock of consumable ignition charges. Distinguish the selected propulsion channel from other spacecraft subsystems that may remain active. |
| Solar-sail reorientation | Shows why infrequent adjustment can matter even when the environment supplies the continuing force. | Radiation pressure remains active at fixed orientation. Treat this as sparse reconfiguration motivation, not a demonstrated zero-thrust coast or a claim that sail attitude needs no control. A direct application requires an attitude/reconfiguration model, which this paper does not supply. |
| Ocean-voyaging sailboat, reserve alternative only | Gives an accessible analogy for holding a useful configuration between adjustments. | Do not add a third example to the paragraph. If replacing solar sailing, describe a held configuration under sufficiently steady conditions; do not assert that sailors adjust only when wind changes or that fixed sail trim means zero plant input. |

**Equations and intuition:** None. Explain the physical meaning of coast in words; leave its input law and resource accounting to Formulation. The examples motivate the problem and do not constitute application validation.

**Figure:** Figure 1's event timeline supports sentence 4 by separating enabled input, shut-off, coast, and renewed feedback. No spacecraft photograph, sailboat picture, or additional application panel is needed.

**Transition:** Existing optimal-control methods address input magnitude and sparsity, which makes them the first relevant comparison.

### I2a. Sparse actuation addresses duration, with a separate question about intervention frequency

**Purpose:** Recognize the foundations for bounded and sparse input design, then distinguish the resource question our controller addresses.

**Opening idea:** Bounded-input optimal control provides direct ways to shape when actuation is applied.

**Sentence sequence:**

1. Credit classical minimum-time bang–bang control for exploiting available input authority to reach a target rapidly under its stated assumptions.
2. Explain its relevant boundary: a minimum-time objective does not itself reward long zero-input intervals or penalize restarting; this concerns the chosen objective, not a universal defect of switching control.
3. Credit maximum hands-off control for minimizing the duration of nonzero input and, under its stated normality conditions, connecting that problem to an L1 formulation.
4. Explain the specific remaining distinction: input-support duration alone does not price how zero-input intervals are distributed or how many restarts occur, so our design explicitly compares a contiguous enabled segment followed by coast.

**Equations and intuition:** No formulas or optimality derivation. If “input support” is used, immediately explain it as the times when input is nonzero. Keep the normality qualification beside the L1 statement.

**Literature and claim boundary:** Use method-led numerical citations, such as “Minimum-time bang–bang control … [reference]” and “Maximum hands-off control … [reference].” The cited hands-off paper already includes self-triggered feedback and a conditional practical-stability result; never claim that this literature ignores feedback timing, lacks analysis, or has no MPC extensions. The contrast is the selected objective and event structure.

**Figure:** No new figure. Figure 1 explains our contiguous act–coast event; it does not illustrate alleged failures of the cited methods.

**Transition:** Reducing actuation is distinct from reducing prediction complexity and the frequency of feedback optimization.

### I2b. Prediction grids, feedback triggering, and learned continuation address different timing questions

**Purpose:** Connect computationally economical MPC and recent learning-based prediction to the need for joint physical shut-off and replanning decisions.

**Opening idea:** Predictive and learning-based methods provide complementary ways to reduce the burden of feedback control.

**Sentence sequence:**

1. Credit nonuniform-horizon MPC in the verified Forbes coauthored paper for resolving fast and slow dynamics over a longer prediction span without increasing the number of optimization variables; explain within the same sentence that prediction-grid spacing alone does not select a zero-input coast.
2. Credit event-triggered MPC for reducing reoptimization or communication while retaining the stated analysis; distinguish this from switching the actuator off, since the selected scheme executes planned inputs between feedback events.
3. Introduce reinforcement learning through its relevant role, continuation assessment beyond a short prediction, and credit TD-MPC2 for combining a learned model and terminal value with trajectory optimization across diverse continuous-control tasks.
4. State the scope of that comparison: TD-MPC2's published planning problem selects action sequences rather than an explicit shut-off/replanning pair, and a learned value by itself is not a certificate of constrained coast motion.
5. Close with our precise design question: compare both timing decisions using the input trajectory and complete enabled/coast motion compatible with each pair.

**Equations and intuition:** None. Define model predictive control (MPC) at its first occurrence in the final prose and reinforcement learning (RL) only if reused. Explain a terminal value as an estimate of performance after the modeled interval. Keep RL subordinate to the timing problem rather than introducing a second motivation about learning dynamics.

**Literature and claim boundary:** Use the verified name “nonuniformly spaced prediction horizon” for the Forbes reference; “dynamic time resolution MPC” is a family description, not that paper's title or a verified claim of online grid adaptation. Confine each limitation to the cited formulation. Self-triggered control is relevant because our next decision is scheduled in advance; do not claim priority for advance scheduling or for combining learning and MPC.

**Figure:** Figure 1's component pipeline supports the closing distinction by showing both timing outputs and their conditional inner solves. Do not add a literature-comparison plot without comparable data.

**Transition:** The final paragraph explains how the nested construction makes this comparison explicit.

### I3. Our contribution is a conditional input solve inside joint timing selection

**Purpose:** State the concrete mechanism and its intended benefit without announcing unproved or unmeasured properties.

**Opening idea:** We formulate act–coast control by assigning the two timing decisions to outer MPC and the bounded input trajectory to a conditional inner optimization.

**Sentence sequence:**

1. State that outer MPC selects both shut-off and replanning times, while the inner problem optimizes the enabled input for each candidate pair.
2. Explain why this matters: every timing choice is assessed using its compatible forced and unforced motion, instead of fixing the input or one timing decision before comparing the other.
3. Explain the learned continuation estimate's role in assessing the state at replanning, with task cost and declared enabled/restart charges in the finite event score; describe the intended tradeoff as control performance versus enabled duration and intervention frequency.
4. State the delivered scope: the paper formulates this construction and evaluates the continuously enabled MPC+RL ingredients; admissibility certificates and act–coast resource comparisons remain research objectives until supported by proofs and matched results.
5. Name the rotary pendulum once as the chosen nonlinear benchmark and lead directly into the generic synthesis problem in Formulation.

**Equations and intuition:** None. The “bi-level” description means that each outer timing candidate receives a conditional inner input solution. Leave duration-grid relations, cost conventions, and numerical search to Methods.

**Figure:** Figure 1 links the controller outputs to event execution and shows how inner scores support outer timing selection. Its purpose is to make the decomposition understandable, not to imply a measured saving or a globally optimal solver.

**Claim boundary:** The current frozen critic estimates torque-policy continuation, not the value of repeated future timing decisions or restart charges. The event score is a declared surrogate for the generic resource objective. Describe joint timing as a formulated mechanism and potential benefit, not as established superiority, guaranteed stability, or a completed application controller.

## Introduction citation and drafting contract

The [source ledger](machine-scannables/intro_literature_ledger.md) records verified papers, passages, application sources, citation-selection evidence, and comparison limits. It is the detailed reference for later drafting; keep that detail out of the introduction.

Use numerical citations attached to the control concept. Bibliographic entries retain author names, but prose does not use “someone et al.” Introduce each method's actual contribution before its limitation, and express the limitation as a precise mismatch of objective, decision variables, or information pattern. Do not claim that an entire field fails at a task outside the cited paper's scope.

Retain one continuous argument: **costly interventions → input duration and restart frequency → prediction and feedback timing → joint timing-conditioned input design**. Remove software details, experiment values, benchmark mechanics, broad novelty claims, and a separate paper-organization paragraph.

## Problem Formulation

### P1. The plant, information, and operating constraints

**Purpose:** Define the control problem independently of the benchmark.

**Sentence sequence:** Introduce a known nominal nonlinear system and its continuous-time state/input model. Define admissible states, admissible inputs, and the initial-state set. State that the physical state is available at scheduled decision times and the model admits unique trajectories for admissible planned inputs. Explain that state constraints apply throughout enabled operation and coast.

**Equations:** Keep $\dot x=f(x,u)$, $x(0)\in\mathcal X_0\subseteq\mathcal X$, and $u(t)\in\mathcal U$ inline. Define $x$ as physical state, $u$ as commanded input, $f$ as known dynamics, $\mathcal X$ as the prescribed operating set, $\mathcal U$ as the prescribed bounded input set containing zero, and $\mathcal X_0$ as the initial-state set under consideration. Require $x(t)\in\mathcal X$ along the trajectory. State the model regularity needed for an analytical result adjacent to that result; do not introduce an uncertainty class without analyzing it.

**Immediate intuition:** During coast the state evolves according to $f(x,0)$; it does not stop moving. These generic constraints later specialize to the benchmark's torque and arm-travel bounds.

**Figure:** None needed. The model and sets are short enough to define in prose.

**Transition:** Specify what a controller may choose and when it receives new feedback.

### P2. What an act–coast policy chooses and executes

**Purpose:** Define the policy output and execution contract before its objective.

**Sentence sequence:** Define an event as an enabled input segment followed by zero-input motion until the next feedback decision. Introduce decision time $t_j$, event duration $h_j$, enabled duration $\tau_j$, and planned input segment $v_j$. Present the input law and interpret both branches. Explain the timing restrictions and that the full event is executed before the next state-dependent plan is selected.

The first display group is

$$
t_{j+1}=t_j+h_j,\qquad
u(t)=\begin{cases}
v_j(t-t_j),&t_j\le t<t_j+\tau_j,\\
0,&t_j+\tau_j\le t<t_{j+1}.
\end{cases}
$$

Here $j=0,1,\ldots$ indexes events, $t_0=0$, and $v_j(s)\in\mathcal U$ is the selected input at elapsed enabled time $s\in[0,\tau_j)$. The controller selects $(h_j,\tau_j,v_j)$ using the state observed at $t_j$. Use $u$ consistently as the input symbol.

**Timing and intuition:** Require $0<\tau_j<h_j$ for nonempty enabled and coast intervals. Prescribed bounds $0<h_{\min}\le h_j\le h_{\max}<\infty$ prevent accumulation of decisions in finite time and bound the event duration. Define the mode indicator $m(t)=1$ while enabled and $m(t)=0$ during coast. Enabled operation permits zero input, so a zero command does not itself mean the actuator is disabled.

**Execution boundary:** Planned inputs may change within the enabled segment without new state-dependent optimization. This intended full-event execution differs from the current inner baseline's first-command replanning. Do not silently add feedback during coast or an unscheduled restart.

**Figure:** Figure 1 labels $t_j$, $t_j+\tau_j$, and $t_{j+1}$, with a varying input profile followed by exact zero. Observation markers occur at the scheduled decision times.

### P3. What task completion requires

**Purpose:** Separate required task behavior from resource reduction.

**Sentence sequence:** Introduce reaching a prescribed target neighborhood and remaining there for a prescribed duration. Define the target through a task-error map and then define the assessment and hold durations. Explain why a single visit is insufficient and why constraint satisfaction is an additional requirement. Distinguish this finite task from any separately proved invariance or convergence property.

**Equations:** Define $\mathcal G=\{x\in\mathcal X:\|e(x)\|\le\varepsilon\}$ inline. Here $e(x)$ is a prescribed task-error map with fixed component scaling, the norm is Euclidean, and $\varepsilon>0$ is the prescribed tolerance. For assessment duration $T>0$ and required hold duration $0<T_{\mathrm{hold}}\le T$, success requires an entry time $t_g\in[0,T-T_{\mathrm{hold}}]$ such that $x(t)\in\mathcal G$ throughout $[t_g,t_g+T_{\mathrm{hold}}]$, while $x(t)\in\mathcal X$ throughout the assessment.

**Immediate intuition:** The target specifies acceptable error and uninterrupted residence specifies continued regulation. Do not assume the task is feasible for every state in $\mathcal X_0$; a formal claim must identify its feasible initial-state domain.

**Benchmark placement:** Methods defines the pendulum error and the components it includes. An orientation-only criterion cannot establish small speeds or full-state rest. Introduce coordinates and wrapping in that benchmark subsection before its governing equations.

**Figure:** None required. A mechanism drawing would not resolve an additional issue in this generic paragraph.

### P4. Which costs distinguish useful coasting from merely small input

**Purpose:** Define the resource tradeoff that the controller is designed to optimize.

**Sentence sequence:** Explain why enabled duration and decision frequency must be accounted for alongside task error and input effort. Introduce the running cost and resource charges, followed by the objective. Interpret each contribution immediately, including the enabled-time charge when the command happens to be zero. Explain that the objective ranks policies subject to task and path constraints, rather than replacing those requirements with penalties.

The second display group is

$$
\begin{aligned}
T_{\mathrm{on}}&=\int_0^T m(t)\,\mathrm dt,\qquad
n(T)=\#\{j:0\le t_j<T\},\\
J_T&=\int_0^T\ell(x(t),u(t))\,\mathrm dt
+c_{\mathrm{on}}T_{\mathrm{on}}+c_{\mathrm{dec}}n(T).
\end{aligned}
$$

Here $T_{\mathrm{on}}$ is total enabled duration, $\#$ counts events, $n(T)$ counts scheduled decisions including the initial decision, and $\ell$ is a prescribed nonnegative running cost for task error and input effort. Positive design weights $c_{\mathrm{on}}$ and $c_{\mathrm{dec}}$ express the chosen costs of enabled operation and new decisions. Give their meanings without numerical settings or a claim of measured electrical energy.

**Immediate intuition:** Input effort alone does not distinguish enabled zero input from disabled coasting. The enabled-duration charge makes switch-off consequential, while the decision charge discourages needless short events. Apply all terms over the same task duration and successful task specification.

**Scope:** This is the proposed controller-level objective. Methods must derive a consistent event score and continuation convention or identify a surrogate explicitly. The old torque-policy critic is not automatically a value for this objective: reward, elapsed duration, and discount conventions must agree. Keep the nested optimization in Methods.

**Figure:** Figure 1 shades enabled segments to explain $T_{\mathrm{on}}$ and marks decisions to explain $n(T)$. Separate input and mode traces distinguish zero input from disabled operation.

### P5. What must be delivered and how it is judged

**Purpose:** End with an explicit synthesis-and-analysis problem and measurable acceptance criteria.

**Sentence sequence:** Ask for a state-dependent rule returning admissible event durations, cutoffs, and input segments that meet the task requirement. State the primary evaluation quantities: task success and hold-completion time, path-constraint violations, enabled fraction $T_{\mathrm{on}}/T$, and decision rate $n(T)/T$. Add input effort and task error to identify resource reductions obtained by degrading control. State the analytical objective of deriving computable sufficient conditions for admissible motion between feedback decisions, with recursive feasibility and target invariance requiring their own proofs. End by placing the benchmark construction in Methods and empirical comparisons in Experiments and Results.

**Equation budget:** Keep remaining definitions inline. Input effort is $\int_0^T\|u(t)\|^2\,\mathrm dt$. Task error can be summarized by $(T^{-1}\int_0^T\|e(x(t))\|^2\,\mathrm dt)^{1/2}$; holding error needs a separately declared post-entry assessment. Hold-completion time is the earliest end of an uninterrupted target residence of length $T_{\mathrm{hold}}$. Unsuccessful runs remain failures rather than receiving an apparent successful completion time.

**Comparison:** Use the same model, initial conditions, input/state limits, target, and assessment duration for act–coast control and a controller with periodic feedback and continuous enablement. Define metrics here; place settings, sampling rules, failure accounting, aggregate results, and measured comparisons in Experiments and Results.

**Certificate boundary:** Finite-grid checks and observed zero violations are numerical evidence. A learned continuation score estimates performance. Neither alone establishes continuous-time constraint satisfaction, recursive feasibility, or target invariance. State every sufficient condition with its model, domain, and assumptions, and give its derivation in Methods or a theory subsection.

**Figure:** The later Results comparison addresses this evaluation question; it supplies no evidence inside the formulation.

## Figures and their purposes

Follow the [revised whole-paper figure plan](../figures/figure-plan/outline.md), which supersedes the earlier figure sequence.

- **Figure 1:** double-column page-top act-coast overview, combining the policy timeline and component pipeline.
- **Figure 2:** single-column, approximately half-page benchmark geometry, constraints and goal.
- **Figure 3:** single-column energy-space control, sensitivity curves and regions.
- **Figure 4:** double-column page-top one-session dynamics/control record with synchronized physical keyframes; leave the act-coast block blank until real data exist.
- **Figure 5:** double-column page-top aggregate scenario atlas, comparing activation count, total control time and total computation time with bang-bang, event-triggered MPC, maximum hands-off control and mixed temporal resolution MPC.
- **Figure 6, optional:** conventional control-quality metrics for the same matched comparisons.

The overview keeps the act-coast controller central. Dedicated benchmark and energy figures now follow the author's explicit requirements. No figure is produced by this plan.

## Relocation and downstream consistency

| Material | Destination or treatment |
|---|---|
| Current Introduction and Formulation | Replace completely; relocate definitions still needed by Methods. |
| Geometry, coordinates, angular wrapping, velocities, and torque direction | Start of the benchmark dynamics subsection in Methods; remove its dependence on coordinates defined in Section II. |
| Governing mechanics, energy map, and teacher | Methods, explicitly introduced as constructions for the chosen benchmark. |
| Fixed command interval and held torques | Numerical realization of enabled inputs in Methods, rather than the general timing problem. |
| Frozen TD3 actor/critic and CEM MPC | Existing inner-control ingredients in Methods, with their actual value and execution conventions. |
| Horizon selection, cutoff search, full-event execution, and certificates | Required controller-level Methods material if delivered; not supplied by the current fixed-horizon baseline. |
| Metric definitions now in Results | Define necessary generic metrics in Formulation; put benchmark specialization and measurement protocol in Results. |
| Historical numerical Results and benchmark-specific title | Require a separate consistency revision; they do not establish act–coast savings or certification. |

## Writing milestones and validation

These are writing checks, not new numbered development stages.

| Milestone | Validation procedure | Success flag |
|---|---|---|
| Argument | Read only the nine paragraph openings. | Each advances the act–coast problem; the benchmark appears as an evaluation choice. |
| Applications | Check each example against its source and assigned purpose. | Propulsive coast and sparse sail reconfiguration remain distinct; no universal ignition-charge claim. |
| Literature | Trace all five method families to the source ledger and read each contribution before its scope limit. | Correct Forbes terminology, qualified hands-off analysis, recent RL source, and no unsupported novelty claim. |
| Formulation | Trace observation, full-event execution, target, constraints, and cost. | All symbols and parameter origins defined; enabled zero input and coast unambiguous. |
| Analysis | Map each claimed property to a proof or numerical check and its domain. | No learned score or observed result described as a certificate. |
| Contributions | Compare the outer/inner split with the code and saved architecture. | Proposed full controller distinguished from implemented inner ingredient. |
| Figures and metrics | Match each panel to a paragraph question and each metric to a claim. | No decorative plot or unsupported savings; failures remain visible. |
| Later drafting | Use three to five sentences per paragraph, repair relocated symbols, compile the existing project, inspect rendered pages. | Fluent symbolic prose, no experimental values in the opening, consistent references, verified whole-paper length. |

Artifact structure: replace this English plan and its Chinese companion; update package interpretation summaries. New runtime classes, objects, functions, methods, data classes, wrappers, helpers, controller/configuration parameters, and CLI options: zero. These mathematical objects specify the proposed paper problem, not new software settings. Preserve the minimum-realization discipline. This task does not edit `main.tex`, train models, run simulations, generate figures, or claim compilation/page verification.

## Source basis

The [current manuscript](../latex-project/main.tex) establishes the existing inner-control scope. The [outer-loop interface](../../human-readables/formulation.md#execution-and-the-eventual-outer-loop) records timing/input coupling, continuation at feedback resumption, and missing scheduler/value compatibility. The author's latest feedback determines the new narrative priority.

## Timing and Results continuation

The [timing plan](bilevel_optimization_paragraph_plan.md) assigns both times to outer MPC and inputs only to the conditional inner solve. The [Results plan](experiments_results_paragraph_plan.md) contains one supported non-act-coast subsection and three short future-experiment stubs. Older architecture notes are not a source for these decisions. The Introduction now assigns both timing decisions to outer MPC, and the bi-level subsection is written; Results now contains current performance and three placeholders; the Introduction rewrite follows this new plan.
