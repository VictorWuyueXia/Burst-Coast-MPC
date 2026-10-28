# Paragraph plan: reinforcement learning and MPC with learned continuation

This is a local plan for existing inner-controller ingredients within the [act–coast paper structure](revision_plan.md). It does not replace the generic [opening formulation](intro_formulation_paragraph_plan.md), implement full-event scheduling, or provide a certificate. Its paragraph and display allocations are local drafting ceilings; the revised whole-paper budget governs final compression.

Continue directly from the teacher subsection in [main.tex](../latex-project/main.tex). The argument is **teacher references → a stationary torque policy → the return that defines future performance → value and policy learning → a learned terminal score → constrained prediction → receding-horizon execution**. Follow [the standing writing guidance](../writing_guidance.md): each paragraph begins with its motivation, introduces only the mathematics needed for that step, and interprets each equation immediately.

Plan four learning paragraphs and three MPC paragraphs, placed after the existing dynamics and teacher subsections and before **Experiments and Results**. The labels C and D identify those two subsection positions, not development phases. Aim for roughly 500–650 additional prose words, with three to five sentences per paragraph and five compact display groups. The original three-page non-figure target remains a whole-paper typesetting check. This task plans the writing and leaves `main.tex` unchanged.

## C. Stationary policy and continuation-value learning

### C1. Why the student uses the physical state to choose torque

**Purpose:** Introduce the learned controller as a state-dependent extension of the teacher's reference actions.

- **Opening sentence:** Explain that the student learns to choose torque from the current physical motion, including the phase information needed for energy transfer.
- **Second sentence:** Introduce the observation inline as $o=\operatorname{normalize}(\theta,\sin\alpha,\cos\alpha,\omega,\nu)$. Reuse the previously defined arm angle $\theta$, pendulum angle $\alpha$, and signed speeds $\omega,\nu$; define `normalize` as fixed componentwise scaling of these features, with sine and cosine retaining orientation smoothly across angular wrapping.
- **Third sentence:** Define the actor, a learned policy $\pi$, through $a=\pi(o)\in[-1,1]$ and physical torque $u=Ua$, where $a$ is normalized torque and $U$ is the already defined usable torque cap.
- **Fourth sentence:** Explain that dependence on the current physical observation makes the policy stationary and permits the same rule to operate during swing-up and subsequent holding.
- **Fifth sentence:** Connect the existing teacher trajectories to actor initialization by fitting their actions, then introduce the need to judge the future motion resulting from each action.

**Equation placement:** Keep the observation and action relations inline. Avoid a display devoted to scaling constants or network architecture. The exact fixed normalization remains in [the evidence ledger](machine-scannables/evidence_ledger.md); the actor applies a direct torque rather than a correction added to the teacher.

**Reader's takeaway:** The actor proposes an admissible torque from physical state information. The next paragraph explains what makes one proposed action preferable to another.

### C2. Why the learning objective includes continued upright motion

**Purpose:** Define the return whose future value the critics learn.

- **Opening sentence:** Explain that a useful action must contribute both to reaching upright and to maintaining low-speed motion near upright afterward.
- **Second sentence:** Define the reward $r_k$ as the negative accumulated running cost over held-action interval $k$, combining fixed positively weighted penalties for departure from upright low-speed motion, pendulum-energy error, squared normalized torque, and arm displacement with an additional penalty near the arm boundary.
- **Third sentence:** Identify the reward's hinge-swing proxy inline as $E_h=\tfrac12J\nu^2+V$, using the previously defined pendulum hinge inertia $J$ and potential energy $V$, and explain that it measures pendulum swing energy while the teacher's mechanical energy $E$ also includes arm motion and coupling.
- **Fourth sentence:** Introduce the discounted action value with the first display, then explain that it adds the rewards following the queried action and subsequent policy continuation.
- **Fifth sentence:** State that motion and return continue after reaching upright, including value continuation across sampling cutoffs, so the learned preference includes holding rather than only the first arrival.

The first display defines the ideal policy action value

$$
Q^\pi(o,a)=\mathbb E_\pi\!\left[
\sum_{k=0}^{\infty}\gamma^k r_k
\;\middle|\;o_0=o,\ a_0=a
\right].
$$

Define $k$ as the command-interval index, $o_0,a_0$ as the initial observation and queried action, and $0<\gamma<1$ as the chosen discount factor per command. The expectation follows the sampled transitions with $\pi$ used after the initial action. Explain immediately that discounting reduces the weight of increasingly distant rewards, while the continuing sum gives value to remaining upright. The neural critics introduced next approximate this ideal quantity; the symbol does not assert exact policy evaluation.

**Immediate cost intuition:** The upright/low-speed term favors continued balancing, and the energy term guides large swings before the pendulum is near upright. The input and arm terms express the effort and travel preferences that accompany both motions. Weave these reasons into the reward description, rather than presenting an unexplained inventory of penalties.

**Equation selection:** Display the return rather than the full reward implementation. Describe the four cost components and their roles in prose; keep their exact Gaussian profile, soft-boundary profile, fixed weights, and numerical accumulation in the ledger. This makes the return definition self-contained as an objective for a specified reward signal without presenting the reward as the later quadratic MPC cost.

### C3. How demonstrations and new transitions train continuation estimates

**Purpose:** Explain how the learning process obtains value estimates beyond the finite teacher records.

- **Opening sentence:** Explain that teacher trajectories supply an initial estimate of useful continuation, while newly collected transitions allow that estimate to follow the evolving student policy.
- **Second sentence:** Introduce two learned critics $Q_1,Q_2$, initialized from approximate discounted teacher-return labels, and a replay distribution mixing retained demonstrations with the student's own transitions.
- **Third sentence:** Introduce a one-step temporal-difference target: the observed reward is added to an estimated value of the next state under a slightly perturbed target-policy action.
- **Fourth sentence:** Explain the second display immediately: both critics fit the same target by squared-error regression, and using the lower next-value estimate tempers optimistic bootstrapping.
- **Fifth sentence:** Identify slowly updated target copies and delayed policy updates as the Twin Delayed Deep Deterministic Policy Gradient method (TD3), then connect value estimation to policy improvement.

For a sampled transition $(o,a,r,o')$, the second display is

$$
y=r+\gamma\min_{j\in\{1,2\}}\bar Q_j(o',\widetilde a).
$$

Here $o'$ is the next physical observation, $r$ is the accumulated transition reward, $j$ indexes the twin critics, and $y$ is their regression target. Bars denote slowly updated training copies. Define the perturbed target action inline as $\widetilde a=\operatorname{clip}(\bar\pi(o')+\eta,-1,1)$, where $\bar\pi$ is the target actor, $\eta$ is bounded training noise, and `clip` enforces the normalized torque interval. Explain that bootstrapping extends the estimate beyond the observed transition, while the teacher-return labels provide an approximate starting point.

**Equation selection:** Omit the separate critic mean-square-loss display; its meaning is already stated precisely in words. Keep return-tail initialization details, target-update coefficients, sample ratios, and reset ranges in the ledger. Sampling cutoffs preserve the physical next observation and continuation target.

### C4. Why policy improvement retains teacher guidance

**Purpose:** Explain how the student improves its action choices while continuing to use the demonstrations.

- **Opening sentence:** Explain that the actor seeks actions with larger estimated future reward while retaining the demonstrated behavior as a learning reference.
- **Second sentence:** Introduce the combined actor objective with the third display.
- **Third sentence:** Interpret the negative critic term as policy improvement and the imitation term as a penalty on departure from teacher actions at demonstrated states.
- **Fourth sentence:** Explain that continued interaction supplies new transitions and that actor updates occur less frequently than critic updates.
- **Fifth sentence:** End with the interface to MPC: freeze the trained actor and the two learned critics, retaining a policy for candidate continuation and value estimates for evaluating its future motion.

The third display is

$$
L_\pi=-\mathbb E_{\mathcal B}Q_1(o,\pi(o))
+\lambda\mathbb E_{\mathcal D}
\left[\bigl(\pi(o)-a_{\rm teach}\bigr)^2\right].
$$

Define $L_\pi$ as the actor's minimized training objective, $\mathcal B$ as the mixed replay distribution, $\mathcal D$ as the retained demonstration distribution, $a_{\rm teach}$ as a normalized teacher action, and $\lambda>0$ as the fixed chosen imitation weight. The expectations are averages over their stated data sources. Keeping the imitation term is part of the current training formulation, rather than only an initialization step. Use $L_\pi$ for this loss so it stays distinct from the mechanical Lagrangian $\mathcal L$.

**Transition to D:** The learned policy describes a useful continuation, but bounded arm travel also requires evaluating the motion produced by a proposed command sequence. This motivates combining the learned continuation score with the physical prediction model.

## D. Nonlinear predictive control with learned continuation

### D1. Why a finite prediction needs an endpoint score

**Purpose:** Motivate the learned terminal cost before stating the MPC optimization.

- **Opening sentence:** Explain that a short prediction evaluates the motion inside its horizon, while its final state determines the motion that remains afterward.
- **Second sentence:** Use the frozen actor's action at that final state to query the frozen critics, and convert their reward estimates to a cost using the fourth display.
- **Third sentence:** Explain the sign change and the lower-critic choice immediately: larger predicted reward means smaller terminal cost, and the lower of the two reward estimates gives the larger negative-reward score.
- **Fourth sentence:** State that this score estimates continuation under the learned policy and its training reward, linking the learning subsection to the finite-horizon construction.

The fourth display is

$$
F(x)=-\min_{j\in\{1,2\}}Q_j\bigl(o(x),\pi(o(x))\bigr).
$$

Define $F$ as the learned terminal cost and $o(x)$ as the fixed observation map introduced in C1. The actor and critics are the frozen learned networks, rather than the barred copies used for training targets. Explain the continuation interpretation directly; the twin minimum is an estimation choice, without a certified bound on the true return.

Use the intuition that sequences with similar costs inside a short horizon can leave different future control demands. The actor supplies a reference continuation for scoring those endpoints; D3 specifies the online controller that replans the executed commands.

**Reader's takeaway:** $F$ supplies a learned preference among prediction endpoints. The next paragraph combines that preference with the immediate state and input costs.

### D2. How constrained nonlinear prediction and learned continuation form one objective

**Purpose:** State the current MPC problem and explain each ingredient's control role.

- **Opening sentence:** Explain that MPC balances the quality of the predicted motion against the continuation assigned to its endpoint.
- **Second sentence:** Define the observed state $x$, predicted states $x_i$, normalized command vector $\mathbf a=(a_0,\ldots,a_{N-1})$, fixed horizon length $N$, and held-command prediction map $\Phi_\Delta$ obtained by integrating the previously defined dynamics under torque $Ua_i$ for duration $\Delta$.
- **Third sentence:** Introduce the fifth display after defining $e_i=e(x_i)$ with $e(x)=(\theta,\beta,\omega,\nu)^{\mathsf T}$, where $\beta$ is the previously defined wrapped upright error, and fixed positive state/input weights $W,R$ inherited from the discrete linear-quadratic design.
- **Fourth sentence:** Interpret state penalties as preferences for upright motion, low speeds, and arm centering; the input term penalizes effort, while the constraints limit commanded torque and predicted arm travel at the checked nodes.
- **Fifth sentence:** State the objective distinction explicitly: the quadratic running sum is an undiscounted design, and $F$ retains the discounted training-return meaning defined in C2.

With $i=0,\ldots,N-1$, the fifth display is

$$
\begin{aligned}
\min_{\mathbf a}\quad&
\sum_{i=0}^{N-1}\left(e_i^{\mathsf T}We_i+Ra_i^2\right)+F(x_N),\\
\text{subject to}\quad&x_0=x,\qquad x_{i+1}=\Phi_\Delta(x_i,a_i),\\
&|a_i|\le1,\qquad |\theta(t)|<\theta_{\max},\quad t\in\mathcal T.
\end{aligned}
$$

Define $x_N$ as the terminal predicted state, $W$ as the positive diagonal state-weight matrix, and $R$ as the positive scalar normalized-input weight. The set $\mathcal T$ contains the prescribed forecast integration times, and $\theta(t)$ is the arm coordinate of the candidate predicted motion at those times; $\theta_{\max}$ is the existing arm envelope. Interpret the constraints as finite-node numerical checks. The terminal state is scored by $F$, with no additional terminal equality or invariant-set condition.

**Formulation fidelity:** Keep the undiscounted sum, its pre-action states $x_0$ through $x_{N-1}$, and the terminal term exactly as stated. The current objective has no time-step multiplier, discount multiplier on $F$, fitted terminal scaling, or replacement by a Riccati terminal matrix. This is a transferred learned score alongside a different running objective; terminal-value accuracy and complete-controller stability require separate analysis.

### D3. How the finite search becomes a receding-horizon torque law

**Purpose:** Explain the actual online command selection and the actor's continuing role.

- **Opening sentence:** Explain that the nonlinear prediction and learned terminal score are evaluated through a finite search over command sequences.
- **Second sentence:** Introduce the cross-entropy method (CEM) as sampling bounded command sequences, retaining a low-cost feasible incumbent, and refining the sampling distribution from low-cost feasible candidates.
- **Third sentence:** Explain that the actor supplies an initial rollout and the continuation of a shifted previous plan, alongside constant signed seeds; reuse of the previous plan gives the implementation memory.
- **Fourth sentence:** State that the selected candidate receives a finer integration check before MPC applies only its first torque, $u=Ua_0$, observes the new state, and replans.
- **Fifth sentence:** Explain that this same MPC procedure and neural terminal score continue during upright holding, completing the connection from offline learning to online control.

**Equation placement:** No additional display. The inline first-action relation completes the execution rule. Keep CEM populations, iteration counts, Gaussian-update algebra, numerical tolerances, and computation measurements outside these Methods paragraphs. The returned plan is a finite-search candidate; its feasibility checks concern the stated grids. If no candidate passes the final audit, the experiment stops rather than applying a fallback controller; retain that source-level fact in the reproducibility record.

**Transition to results:** The method is now fully specified from observed state to executed torque. **Experiments and Results** can separately assess swing-up, continued upright error, arm motion, and constraint margins under a declared protocol.

## Equation and notation budget

| Location | Display | Why it is needed |
|---|---|---|
| C1 | None; observation and torque inline | Define the learner's input and physical action without a scaling derivation. |
| C2 | Discounted action value | Define precisely what the learned continuation means. |
| C3 | Twin-critic temporal-difference target | Explain how finite observed transitions train a future-value estimate. |
| C4 | Actor objective with retained imitation | Explain improvement beyond initialization while preserving the teacher reference. |
| D1 | Frozen learned terminal cost | Connect reward estimation to endpoint scoring. |
| D2 | Constrained finite-horizon optimization | Define predicted motion, competing costs, and admissibility. |
| D3 | None; first applied torque inline | Close the loop without adding another algorithmic notation layer. |

Reuse $x,\theta,\alpha,\omega,\nu,\beta,U,\Delta,\theta_{\max},J,V,E,E_\star$ from the preceding sections. Introduce each new object only in its assigned paragraph. Network parameter vectors, extra controller modes, and families of horizon-dependent observation symbols are unnecessary here. If a name for the MPC objective is later required, use $\mathcal J$ so it stays distinct from pendulum inertia $J$; the displayed minimization currently needs no additional name.

## Source checks and writing milestones

| Milestone | Validation procedure | Success flag |
|---|---|---|
| Paragraph architecture | Read only C1–C4 and D1–D3 opening ideas. | Each paragraph advances one step from teacher references to executed predictive control. |
| Learning fidelity | Compare C1–C4 with the stationary task, observation map, demonstration pretraining, and TD3 updates. | Physical-state-only actor; continuing return; correct energy proxy; retained imitation; target copies distinct from frozen exported networks. |
| MPC fidelity | Compare D1–D3 with the objective, CEM search, and current runner. | Correct critic sign and actor substitution; undiscounted quadratic sum; checked arm nodes; plan memory; no local handoff or learned horizon. |
| Later prose drafting | Write three to five sentences per paragraph and verify every symbol and immediate interpretation. | No numerical settings, unit lists, performance results, dense implementation expansions, or unsupported guarantee. |
| Later whole-paper integration | Insert these subsections after the teacher and before Results; compile and inspect the rendered paper. | Continuous narrative, no repeated dynamics/teacher explanation, and a verified three-page non-figure budget. |

The exact reward components, normalization, training losses, fixed settings, and numerical constraints remain in [the evidence ledger](machine-scannables/evidence_ledger.md). Relevant implementations are `jax_stationary_task.py`, `jax_residual_task.py`, `jax_demonstrations.py`, and `jax_td3.py` under `src/rotary_pendulum/RL/`, together with `src/rotary_pendulum/mpc/{inner_model.py,gpu_cem.py}` and `scripts/experiments/inner-mpc/lqr_cost_run.py`. Baseline mathematical intent is documented in [mpc_rl_baseline.md](../../human-readables/mpc_rl_baseline.md).

This task adds one English paragraph plan and its Chinese companion, updates the reading-guide links, and makes the squared-error expectation explicit in the supporting ledger. New runtime classes, objects, functions, methods, data classes, wrappers, helpers, independent runtime variables, configuration/controller parameters, and CLI options: **zero**. Preserve the compact continuous-logic and minimum-realization discipline; add no fallback behavior or configuration layers. The manuscript, training code, controller, and experiment records remain unchanged.
