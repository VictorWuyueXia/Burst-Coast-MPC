# Paragraph plan: outer MPC for shut-off and replanning timing

Use the author's latest instruction and the [recent timing interface](../../human-readables/formulation.md#execution-and-the-eventual-outer-loop). The outer MPC selects **both shut-off and replanning times**; the inner optimization selects the bounded input sequence conditional on that timing pair. Ignore the older architecture that assigned cutoff selection to the inner controller or learned a horizon-selection policy. The actor and critics retain their existing inner-control roles.

This subsection follows the predictive inner-controller subsection. Plan four paragraphs, about 250–300 eventual prose words and two compact display groups. Apply [the writing guidance](../writing_guidance.md): each paragraph opens with its intuition, introduces only necessary mathematics, and immediately explains every equation. The method is recently formulated, not implemented or certified in the current controller. This document plans prose; it does not edit `main.tex`.

## E1. Why the outer MPC selects two times

**Purpose:** Define the division of decisions before presenting the nested problem.

**Sentence sequence:** Explain that an event must decide when to stop actuation and when to acquire feedback and replan. Introduce the outer timing pair, then explain that the inner problem supplies the feasible torque sequence for that pair. Relate both times to the existing event notation. End by explaining why a timing pair must be evaluated with its compatible input plan.

**Equations and immediate intuition:** Use inline $\tau=b\Delta$ and $h=r\Delta$. Here $\Delta>0$ is the existing command duration, $b$ and $r$ are integer numbers of command intervals from the current decision to shut-off and replanning, and $1\le b<r$ requires a nonempty enabled prefix and coast. The absolute event times are $t_j+\tau$ and $t_j+h$, with $t_j$ already defined in Section II. The outer MPC chooses $(b,r)$; the inner problem chooses torque only. Neither time is an RL action.

**Figure:** Figure 1 must send a timing pair from outer MPC to the inner torque solve. Return an input plan, conditional cost, and feasibility information. Remove any arrow that assigns shut-off timing to the inner controller.

**Transition:** A comparison of timing pairs requires a conditional cost that predicts their full enabled and coast motion.

## E2. How the inner solve scores a fixed timing pair

**Purpose:** Explain the cost and feasible trajectory evaluated for each outer candidate.

**Sentence sequence:** Start from an observed state and fixed shut-off/replanning pair. Explain that free inputs act only before shut-off, while prediction continues with zero input until replanning. Introduce the recent conditional objective, with continuation evaluated at the next decision. Interpret enabled-time cost, restart cost, and the trajectory constraints immediately.

The first display group is the recent interface written with the existing terminal symbol $F$,

$$
\begin{aligned}
S(b,r;x)=\min_{a_0,\ldots,a_{b-1}}\;&
\sum_{i=0}^{r-1}\gamma^i\left[\ell_{\mathrm{base}}(x_i,a_i)
+\Delta c_{\mathrm{on}}\mathbf1_{i<b}\right]\\
&+\gamma^r\left[c_{\mathrm{wake}}+F(x_r)\right],\\
a_i&=0,\qquad b\le i<r.
\end{aligned}
$$

**Definitions:** $x$ is the observed state, $x_0=x$, and $x_i$ is the predicted state after $i$ command intervals. The existing map $x_{i+1}=\Phi_\Delta(x_i,a_i)$ predicts motion with physical torque $Ua_i$, where $U$ is the torque cap and $a_i\in[-1,1]$. The declared per-command task/input cost is $\ell_{\mathrm{base}}$; $0<\gamma<1$ is the per-command discount; $c_{\mathrm{on}}>0$ is a chosen enabled-time cost rate; and $c_{\mathrm{wake}}$ is the chosen restart charge at feedback resumption. The indicator $\mathbf1_{i<b}$ is one while enabled and zero afterward. $F(x_r)$ is the frozen actor-continuation cost already defined in Methods, evaluated after the coast. Input bounds and state constraints apply throughout the prediction, including the coast, not only at the two switching states.

**Immediate intuition:** Enabled zero torque still incurs the enabled-time charge. Unforced motion still accumulates the declared stage cost. The initial activation is already incurred; do not charge it again. Since $b<r$, the next event resumes after a genuine coast. A continuously enabled reference has no restart and must not receive a spurious restart charge.

**Cost boundary:** Preserve this recent interface as a formulated score, not an implemented objective. Its discounted score and actor tail are not automatically the undiscounted Section II resource objective or the current quadratic inner baseline. Declare the stage-cost, restart-versus-decision-charge, and discount conventions explicitly before claiming equivalence or optimizing the new controller. The frozen critic does not price repeated future coasts or restart charges. Do not retrain, rescale, or reinterpret it silently.

**Figure:** Figure 1 shows the enabled prefix, zero-input suffix, and one terminal evaluation at replanning. A fixed maximum candidate horizon bounds timing search; it is not an additional terminal evaluation time.

## E3. How the outer MPC compares timing pairs

**Purpose:** Present the actual timing optimization and explain its coupling to input design.

**Sentence sequence:** Define the prescribed admissible timing-pair set. Present joint minimization of the conditional score, then explain that changing either time changes the feasible input problem and terminal state. Exclude candidates with no feasible conditional plan. Distinguish the ideal optimization from any finite or approximate implementation.

The second display group is

$$
(b^\star,r^\star)\in\operatorname*{arg\,min}_{(b,r)\in\mathcal D}S(b,r;x).
$$

Here $\mathcal D$ is the prescribed finite set of integer timing pairs satisfying $1\le b<r$ and the event-duration bounds in Section II, with a nonempty feasible conditional input set. The star denotes a selected minimizing pair, and arg min is the set of minimizers. The outer output is shut-off duration $b^\star\Delta$ and replanning duration $r^\star\Delta$; the associated input sequence comes from the inner solve.

**Immediate intuition:** Compare each pair using its own compatible input plan. Optimizing shut-off for one coast endpoint and then changing replanning time solely by a critic value does not solve this joint problem. The notation specifies the intended ideal problem; finite CEM input search does not establish globally optimal conditional costs or timing decisions.

**Figure:** Figure 1 shows repeated conditional evaluations followed by one selected timing/input plan. A numerical timing-cost map belongs in the future timing-ablation experiment only after the candidate scores have been computed.

## E4. What the selected event executes and what can be concluded

**Purpose:** Define execution and the scope of the formulated controller.

**Sentence sequence:** Explain that the selected inner inputs are executed for the enabled prefix and are followed by exact-zero input for the coast. State that the next observation and joint timing decision occur at the selected replanning time. Explain that an admissible predicted event concerns its entire trajectory. End by distinguishing numerical checks from any separately proved continuous-time or recursive property.

**Equations:** Reuse the event law in Section II; no third display is needed. Within this event, do not replace full-event execution with the current baseline's first-command replanning or add an unmentioned emergency restart. State an analytical property only with an actual sufficient condition and its model, domain, and assumptions. No invariant set or stability theorem is currently established for this outer formulation.

**Figure:** Reuse Figure 1 to show execution through the next observation. No certificate illustration is justified by the formulation alone.

## Writing checks and scope

| Milestone | Validation | Success flag |
|---|---|---|
| Decision split | Inspect every timing/input symbol and diagram arrow. | Both times belong to outer MPC; only inputs belong to the conditional inner solve. |
| Conditional objective | Trace every stage, mode charge, constraint, and terminal evaluation. | Coast is simulated and charged; the single tail starts at replanning; cost conventions are explicit. |
| Mathematical claim | Compare formal minima with the numerical method and available proof. | No global optimality or certificate inferred from a sampled plan or critic. |
| Later drafting | Read four paragraph openings and inspect equation definitions. | One continuous argument with three to five sentences per paragraph; numerical settings remain in Results. |

[The source/evidence contract](machine-scannables/bilevel_results_plan_contract.md) holds detailed provenance and cost-compatibility questions. New runtime classes, objects, functions, methods, wrappers, helpers, CLI/configuration parameters: zero. No controller implementation, training, simulation, plotting, or manuscript edit is authorized by this planning document. Preserve the repository's compact, minimum-realization workflow without fallback behavior.
