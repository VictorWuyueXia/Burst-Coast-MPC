# Paragraph plan: act–coast control as the research problem

Replace the complete Introduction and Problem Formulation. The previous benchmark-led plan is superseded. The argument is **motivation for intermittent actuation → coupling of torque and feedback timing → controller construction and analysis → admissible act–coast policies → task requirements → performance criteria**. Introduce the rotary pendulum as the chosen benchmark in Methods, where its coordinates, governing equations, and task specialization belong.

Apply the [standing guidance](../writing_guidance.md) and the control-academic-writing skill. Use three Introduction paragraphs and five Formulation paragraphs, each with one purpose and three to five connected sentences. Aim for 550–650 prose words and two principal display groups across these sections; keep simple model and target definitions inline. This is an editing budget supporting the whole-paper three-page target excluding figures, not a verified page count.

The plan describes the intended paper problem. The current fixed-horizon MPC+RL supplies an inner-control ingredient; the outer scheduler and its certificates remain to be developed and checked. The final contribution statements must distinguish completed constructions and proofs from research objectives.

## Scope and companion plans

This file replaces the old opening plan and governs the detailed Introduction and Formulation rewrite. The [paragraph map](../paragraph_plan.md), [subsection revision plan](revision_plan.md), and [figure outline](../figures/figure-plan/outline.md) follow the same paragraph purposes, equation allocation, and evidence boundaries. Existing Method component plans remain the references for their local constructions. Later prose drafting edits the existing `main.tex`; this task updates planning documents only.

## Introduction

### I1. Why design an act–coast controller?

**Purpose:** Introduce the controller's objective before introducing a benchmark or learning method.

**Sentence sequence:** Open with systems that alternate bounded actuation and unforced evolution. Explain that coast intervals can reduce enabled time and the frequency of new control decisions while the plant continues to evolve. Explain that these reductions are useful only while the task and operating constraints remain satisfied. End with the design question: how much to act and how long to coast before obtaining feedback and planning again?

**Equations and intuition:** No mathematics. Define coasting immediately as zero commanded input. Distinguish enabled duration from physical energy consumption, which requires an actuator power model. The eventual prose should express the closing design question as a direct statement.

**Figure:** Figure 1, a generic act–coast timeline, introduces the two intervals and the next feedback decision.

**Transition:** Delaying feedback changes the state from which control resumes, making input and timing coupled decisions.

### I2. Why actuation and timing must be designed together

**Purpose:** Explain the difficulty specific to act–coast control.

**Sentence sequence:** Explain that the enabled input determines both task progress and the state at switch-off. Explain that coast duration then determines the next observed state and whether constraints hold along the intervening motion. Explain why a small input cost or favorable endpoint alone cannot establish an admissible coast interval. End with the need to assess timing and input choices through their complete predicted motion and subsequent control requirements.

**Equations and intuition:** No equations. Motivate continuation assessment through feedback scheduling, rather than introducing reinforcement learning as a separate topic. A feasible endpoint alone does not establish a feasible future.

**Figure:** Reuse Figure 1 to explain the dependence of the next decision state on both intervals. Any contrasting coast lengths are schematic, not measured outcomes.

**Literature role:** Locate the problem relative to self-triggered control, intermittent actuation, and predictive scheduling using primary sources verified before drafting. Compare information and actuation assumptions. This plan adds no unverified citations, novelty claims, or claims that existing methods fail.

**Transition:** The coupled decision motivates conditional input optimization within a timing-selection algorithm.

### I3. What algorithm and mathematical properties the paper develops

**Purpose:** Identify controller-level contributions and give the benchmark its proper role.

**Sentence sequence:** Introduce a bi-level construction in which the outer MPC chooses both the shut-off and next feedback times, and the inner optimization chooses the bounded input trajectory conditional on that timing pair. Explain how model prediction accounts for forced and unforced motion and how a continuation estimate can assess motion beyond the next decision. State the completed paper's intended outputs: a controller-construction algorithm, analytical properties where proved, and quantified task/resource performance. End with one sentence naming the rotary pendulum as the chosen benchmark.

**Equations and intuition:** No optimization or training equations. Explain the decomposition: each timing pair is compared using its compatible input solution. Methods supplies the nested optimization, numerical solver, learning construction, and any certificate derivation.

**Figure:** Figure 1, the outer/inner decision structure, shows what each level chooses and what information returns to the outer comparison.

**Claim boundary:** This is the intended architecture, not the status of the current fixed-horizon baseline. The recent formulation uses outer MPC for both timing decisions; no learned timing policy is part of this subsection. Do not identify the current torque-policy critic with an event-level scheduling value or announce a certificate before it is established.

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
| Argument | Read only the eight paragraph openings. | Each advances the act–coast problem; the benchmark appears as an evaluation choice. |
| Formulation | Trace observation, full-event execution, target, constraints, and cost. | All symbols and parameter origins defined; enabled zero input and coast unambiguous. |
| Analysis | Map each claimed property to a proof or numerical check and its domain. | No learned score or observed result described as a certificate. |
| Contributions | Compare the outer/inner split with the code and saved architecture. | Proposed full controller distinguished from implemented inner ingredient. |
| Figures and metrics | Match each panel to a paragraph question and each metric to a claim. | No decorative plot or unsupported savings; failures remain visible. |
| Later drafting | Use three to five sentences per paragraph, repair relocated symbols, compile the existing project, inspect rendered pages. | Fluent symbolic prose, no experimental values in the opening, consistent references, verified whole-paper length. |

Artifact structure: replace this English plan and its Chinese companion; update package interpretation summaries. New runtime classes, objects, functions, methods, data classes, wrappers, helpers, controller/configuration parameters, and CLI options: zero. These mathematical objects specify the proposed paper problem, not new software settings. Preserve the minimum-realization discipline. This task does not edit `main.tex`, train models, run simulations, generate figures, or claim compilation/page verification.

## Source basis

The [current manuscript](../latex-project/main.tex) establishes the existing inner-control scope. The [outer-loop interface](../../human-readables/formulation.md#execution-and-the-eventual-outer-loop) records timing/input coupling, continuation at feedback resumption, and missing scheduler/value compatibility. The author's latest feedback determines the new narrative priority.

## Timing and Results continuation

The [timing plan](bilevel_optimization_paragraph_plan.md) assigns both times to outer MPC and inputs only to the conditional inner solve. The [Results plan](experiments_results_paragraph_plan.md) contains one supported non-act-coast subsection and three short future-experiment stubs. Older architecture notes are not a source for these decisions. The Introduction now assigns both timing decisions to outer MPC, and the bi-level subsection is written; the Results plan remains future writing.
