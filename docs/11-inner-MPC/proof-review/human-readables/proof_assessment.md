# Assessment of the proposed MPC+RL convergence proof

Reviewed on 2026-10-24. Source commit: `cdce8f93f98d6c978e6a723d4aea11f4524c59ad`. The two local discussions are `../11.1-stability_goals_and_strategies.md` and `../11.1-swing_up_conditional_proof.md`, relative to the `proof-review` directory. This assessment concerns the official fixed-timing MPC+RL baseline, with no LQR handoff or outer burst/coast scheduler.

## Judgment

The capture theorem is a valid conditional reachability result. The lower-bound lemma is also valid for the declared exact nominal ODE and finite frozen networks. The proposed holding argument is valid if its inequality and feasibility conditions are established. **These statements do not yet prove today's intended result for the implemented controller.**

The missing work is not another telescoping argument. It is demonstrating that the actual closed-loop algorithm satisfies the assumed progress, feasibility, and holding conditions on a specified domain. The discussions acknowledge this limitation, so they should be retained as a proof strategy rather than presented as a finished certificate.

## What the mathematics establishes

Let the physical state be the arm angle, pendulum angle from downward, and their respective angular velocities. Angles are in radians and velocities in rad/s. Define the upright error and error state by

$$
x=(\theta,\alpha,\omega,\nu)^{\mathsf T},\qquad
\beta=\operatorname{atan2}(-\sin\alpha,-\cos\alpha),\qquad
e=(\theta,\beta,\omega,\nu)^{\mathsf T}.
$$

Upright rest at the centered arm is the state with zero error. The discussion's provisional capture set is

$$
\mathbf D=\{x:|\theta|\le120\pi/180,\ |\beta|\le5\pi/180,
\ |\omega|\le0.3,\ |\nu|\le0.4\}.
$$

These thresholds are proposed capture specifications, not verified limits of an invariant holding region. Let the normalized commanded torque be the physical torque divided by its limit. The implemented quadratic stage cost and its outside-capture lower bound are

$$
a=u/u_{\max},\qquad
\ell(x,a)=\left(\frac{\theta}{135\pi/180}\right)^2
+\left(\frac{\beta}{0.15}\right)^2+0.25\omega^2+0.25\nu^2+a^2,
\qquad x\notin\mathbf D\Longrightarrow\ell(x,a)\ge q_D=0.0225.
$$

The threshold follows from the smallest cost at any one defining boundary. Its four candidates are approximately 0.790123, 0.338464, 0.0225, and 0.04. Independent arithmetic confirms the stated value. This is an elementary but necessary link from cost decrease to entry into the chosen physical region.

Let the decision index be the integer number of 0.1 s command intervals. Let the selected plan contain ten normalized commands, and let its cost be evaluated consistently using the exact nominal command transition. The frozen terminal function is the larger negative twin-critic output, evaluated at the frozen actor's action and the original scaled observation:

$$
U_k=(a_{0|k},\ldots,a_{9|k}),\qquad
J_k=\sum_{i=0}^{9}\ell(x_{i|k},a_{i|k})+F(x_{10|k}),\qquad
F(x)=\max_{j=1,2}\{-Q_j(o(x),\pi(o(x)))\}.
$$

Here the predicted states start at the current physical state, the index after the vertical bar identifies the decision at which the plan is selected, the observation map is the implemented scaling, the two critics are frozen scalar reward-return networks, and the actor is the frozen normalized-action network.

The capture theorem defines an accounting residual and asks for a uniform bound before first entry:

$$
r_k=J_{k+1}-J_k+\ell(x_k,a_{0|k}),\qquad
r_k\le\bar r<q_D,\qquad \delta=q_D-\bar r>0.
$$

The bound is a certificate obligation, not a measured constant. It implies that every decision outside capture decreases the plan cost by at least the positive amount represented by the last symbol. If every relevant plan cost has a common finite lower bound, denoted by the minimum subscript, then

$$
J_K\le J_0-K\delta,\qquad
K_D\le\left\lfloor\frac{J_0-J_{\min}}{\delta}\right\rfloor+1.
$$

The integer with the capture subscript is the first entry decision; the capital integer without that subscript counts consecutive steps without entry. Continued accepted execution and consistent costs, including the next cost in each comparison, are required. This contradiction argument is correct. It proves first entry only.

The lower-bound lemma has real content: a learned terminal cost need not be nonnegative, and velocities were not assumed bounded. Finite SiLU networks grow at most linearly in their inputs; the actor is bounded; feasible terminal arm angles are bounded; and bounded torque gives at-most-linear terminal speed growth over a fixed exact-ODE horizon. The first stage's quadratic speed penalty then dominates any negative linear terminal growth. The mechanical constants and determinant in the discussion agree with the configuration and governing equations. This establishes existence of a lower bound, not its useful numerical size or the progress inequality. It does not automatically transfer to arbitrary-velocity fixed-step RK4 trajectories.

## Where the controller-specific argument remains open

**The main assumption contains almost all the convergence difficulty.** The residual definition is an identity. Its required upper bound is not a consequence of that definition. Without an independent verification method, the argument amounts to assuming a lower-bounded quantity decreases until capture and then deriving capture. This is useful as a sufficient-condition template, but it does not explain why this MPC+RL controller satisfies it.

The natural missing connection uses a shifted plan. Let the exact transition over one command be the function below, and let the appended terminal feedback initially be the frozen actor. Denote a terminal physical state by the temporary variable below. Define the terminal defect and next-plan selection gap by

$$
x^+=f(x,a),\qquad z=x_{10|k},\qquad \kappa(z)=\pi(o(z)),
$$

$$
d(z)=\ell(z,\kappa(z))+F(f(z,\kappa(z)))-F(z),\qquad
s_k=J(x_{k+1},U_{k+1})-J(x_{k+1},U_k^{\rm shift}).
$$

The shifted sequence removes the executed command and appends that terminal feedback. With identical dynamics and tail construction, cancellation gives

$$
r_k=d(x_{10|k})+s_k.
$$

This identity connects RL continuation and solver selection to progress only after their combined effect is bounded on reachable states and histories. Separate small worst-case bounds are sufficient but may be conservative; a joint residual bound is also legitimate. A direct verified residual therefore need not prove either component stabilizing in isolation.

**The uniform residual threshold is stronger than necessary.** A large residual during swing-up can be harmless when the current stage cost is larger still. Before abandoning one-step progress, test the weaker direct condition

$$
\ell(x_k,a_{0|k})-r_k\ge\delta>0
$$

on the declared pre-capture domain. The same positive constant then bounds the net cost decrease. A residual exceeding 0.0225 rejects the particular uniform-residual shortcut, not necessarily one-step descent; a positive plan-cost increment rejects one-step descent at that transition. Only then consider block progress or a different certificate.

**The RL terminal value has no established terminal-decrease property for this stage cost.** Its training reward and discount differ from the current undiscounted quadratic cost. Neither successful actor episodes nor low training error supplies the missing inequality. The fact that the terminal function is the maximum of two smooth branches also does not establish positivity or decrease. Verifying its relevant defect is necessary for the shifted-plan route; proving convergence of TD3 training is not.

**CEM improvement during one solve is different from decrease between decisions.** The code retains a nominally feasible incumbent within each seed lane. A lower nominal-cost replacement can fail the finer audit, and the prior incumbent is not restored. Thus even an audited shifted seed does not automatically survive as an executable choice. Moreover, the shift uses 2 ms seed rollouts, whereas the objective uses 1 ms prediction and execution uses 0.5 ms refinement. The sign of the selection gap against a theoretical exact-map shift cannot simply be asserted. Check `gpu_cem.py:80-108` and `lqr_cost_run.py:130-173`.

**First entry is not eventual holding.** The companion theorem expressly stops at entry into the provisional box. That box permits an arm angle of 120 degrees and has no verified invariant-domain relation to the local certificate. The documented 32-session hold metric checks pendulum angle alone. It does not establish simultaneous entry into the full box or bound the complete state indefinitely.

## What the proposed holding inequality would establish

Let a symmetric positive definite matrix define a quadratic certificate. Its entries must be computed or verified for the actual MPC feedback; the existing LQR Riccati matrix is only a candidate. Suppose the implemented successor error, including allowed stored plans and random outcomes, satisfies

$$
L(e)=e^{\mathsf T}Pe,\qquad
L(e_{k+1})\le\rho L(e_k)+\varepsilon,\qquad
0<\rho<1,\quad\varepsilon\ge0.
$$

The contraction factor and nonnegative residual are bounds to certify, not controller tuning changes. For certified radii chosen above the residual floor, define

$$
b=\frac{\varepsilon}{1-\rho},\qquad
R>r_{\Omega}>b,\qquad
\mathbf D_L=\{e:L(e)\le R\},\quad
\Omega=\{e:L(e)\le r_{\Omega}\}.
$$

If the inequality and accepted continuation hold throughout the larger domain for compatible histories, both sets are invariant at command instants and every start in the larger set enters the smaller set in finite time. The strict gap above the floor matters; the limiting floor itself need not be reached in finite time. The coordinate bounds are

$$
|e_i|\le\sqrt{r_{\Omega}(P^{-1})_{ii}},\qquad i=1,2,3,4.
$$

The index enumerates arm angle, upright error, arm speed, and pendulum speed; the diagonal entries come from the inverse certificate matrix. These bounds must fit a stated, useful near-upright specification. An arbitrarily large residual that yields a large region is not a meaningful holding result. Bounds between commands are needed if holding is claimed continuously.

The capture image, including inherited controller memory, must lie in the certified holding domain. Physical state alone does not determine CEM's action. Its stored plan and random-key/decision information must be included or quantified over. Proving the inequality for fresh initialization at captured physical states is insufficient.

## Is this proving something trivial or unnecessary?

The threshold calculation and telescoping argument are elementary supporting steps. The lower-bound lemma removes a genuine obstruction from a potentially negative learned terminal function. None is wasted, but none supplies the missing controller-specific decrease.

The terminal defect and selection gap are currently diagnostic objects rather than established hypotheses of the theorem. They become substantive when bounded or used to isolate a failure. An independently verified direct residual can replace their separate analysis without weakening the capture implication.

An independent local certificate is legitimate even if it does not resemble the RL terminal value: it certifies actions selected by the full MPC+RL controller. It need not prove that every architectural component is essential. An RL-removal ablation is needed to claim an RL benefit or necessity, not to prove convergence of the frozen combined controller. LQR stability under LQR actions, critic positive definiteness by itself, TD3 training convergence, and global CEM optimality are not substitutes for the missing closed-loop conditions.

The conventional MPC argument also distinguishes terminal decrease, feasible continuation, and solver acceptance; suboptimal MPC must account for warm-start memory. See Rawlings, Mayne, and Diehl, [Model Predictive Control, second edition, fifth printing](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-5th-printing.pdf), Assumption 2.14 and Section 2.7.

## Recommended direction

Recover and inspect the actual records first. Define a physically meaningful holding region under the unchanged MPC before fixing the swing-up destination. Then falsify one-step progress into that region; if it fails, test a finite-block certificate or direct verified reachability. A violation rejects the proposed certificate, not automatically the controller. Keep all source changes, new acceptance rules, and LQR handoffs outside this audit: each would define a different controller.

The present defensible claim is that a mathematically valid certificate framework exists and nominal finite-session behavior is reported. The end-to-end convergence theorem remains open. The next deliverable should be certified bounds and their domains, or explicit counterexamples and an explanation of which proposed condition fails.
