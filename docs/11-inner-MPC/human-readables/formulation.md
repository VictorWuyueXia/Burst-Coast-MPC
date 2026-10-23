# The inner problem: optimize a torque sequence with a learned terminal cost

**Official baseline, 2026-10-23: [MPC+RL](mpc_rl_baseline.md).** The 16-generation CEM controller with exact LQR stage weights and the frozen RL terminal value is now the development baseline. RL remains enabled after capture; no local handoff is used. Earlier study descriptions below are historical.

Current experiment: [bounded weak speed penalties](weak_speed.md), CEM only, full 40-second sessions. The earlier weak angle extension has been removed.

The historical weak angle-tracking study is documented in [weak tracking](weak_tracking.md); its added angle cost is no longer part of the current implementation.

## Scope and interpretation

The eventual controller chooses when to stop acting and when to resume feedback, while an inner optimization chooses the intervening torques. Here the two timing decisions are fixed: the actuator is enabled over the entire prediction horizon, and the optimizer chooses its direct torque sequence. The heuristic supplies historical learning data or an explicitly declared initial guess; it does not restrict the admissible torque sequence to energy-phase actions.

The revised primary experiment uses squared torque effort, squared normalized arm angle along the planned trajectory, and the frozen terminal task value. The user explicitly requested the arm-centering term after the initial effort-only study. Hard arm limits remain unchanged. The original RL running cost remains a separate diagnostic.

## State, clock and frozen networks

At prediction decision k, the state x_k contains arm angle theta_k, pendulum angle alpha_k, arm speed omega_k and pendulum speed nu_k. Angles are in radians and speeds in radians per second; the initial state x_0 is measured or read from the simulator. Pendulum angle zero is downward and plus/minus pi is upright. Arm angle is not wrapped.

Let N be the positive integer number of planned decisions, Delta the fixed 0.1 s decision duration, and T the resulting prediction duration. Let a_k be the optimized dimensionless torque and u_max the checkpoint's torque cap in N m. Physical torque u_k and the nominal transition f_Delta are

$$
x_k=(\theta_k,\alpha_k,\omega_k,\nu_k),\qquad
T=N\Delta,\quad \Delta=0.1\ \mathrm{s},\qquad
u_k=u_{\max}a_k,\quad -1\le a_k\le1,\qquad
x_{k+1}=f_\Delta(x_k,u_k).
$$

The transition is computed using fifty 2 ms RK4 steps with constant torque, grouped into five 20 ms observation intervals. N is an experimental horizon choice; Delta is inherited from learning and is not retuned here. Begin with N=10, or one second, and compare 5, 20 and 40 decisions. These are proposed experiment settings, not established useful horizons.

Use the selected `rotary-stationary-td3-v3` checkpoint at 4.59 mN m first, then repeat at 2.295 mN m with its own networks. Do not substitute the 18.36 mN m default heuristic torque cap or the 20.4 mN m physical cap. The saved models have five physical features and respectively three 256-unit or three 512-unit hidden layers. The older clock-dependent and discrete-action critics have different contracts.

Define h as the checkpoint's feature map, pi as its frozen actor, and Q_1 and Q_2 as its frozen online critics. Theta_max=3pi/4 rad is the configured 135-degree arm limit. The positive normalization scales s_omega and s_nu are computed from the configured mechanics; their exact definitions appear in the [contract](../machine-scannables/implementation_contract.md).

$$
h(x)=\left(\theta/\theta_{\max},\sin\alpha,\cos\alpha,
\omega/s_\omega,\nu/s_\nu\right),\qquad
V_Q(x)=\min_{i\in\{1,2\}}Q_i\bigl(h(x),\pi(h(x))\bigr),\qquad
F_Q(x)=-V_Q(x).
$$

V_Q is an estimated reward return; larger is preferable. F_Q converts it to a cost; smaller is preferable. The actor supplies the action input required by each critic. It does not execute a simulated extra terminal interval. Consequently, do not charge that action again outside the terminal value. The minimum is a cautious twin-critic aggregation, not a guaranteed bound on true value. These critics approximate learned actor continuation after TD3 training, not exact heuristic-policy evaluation or an optimal future MPC controller. The original [TD3 paper](https://arxiv.org/abs/1802.09477) explains the twin-critic motivation.

Do not maximize the critic over a free terminal action in the first experiment. That would introduce another action optimization, increase exposure to unsupported critic peaks and change the continuation convention. Freeze actor and critic weights, but differentiate their outputs with respect to terminal state, including the actor's dependence on that state.

## Primary objective and physical constraints

Let gamma=0.995 be the saved discount per decision, and w_u=2 be the saved squared normalized-torque cost rate in numerical cost units per second. Define ell_u as the cost of one held command and J_N as the total predicted cost. The tunable nonnegative weight w_theta is the arm-centering cost rate in numerical cost units/s; compare 0.1, 1 and 10 initially. Theta_k,j is the predicted arm angle after microstep j of decision k; delta is the 2 ms integration interval. Ell_theta sums its squared fraction of the arm limit, encouraging centering before a hard boundary becomes active:

$$
\ell_u(a)=\Delta w_u a^2,\qquad
\ell_\theta(x_k,a_k)=w_\theta\delta\sum_{j=1}^{50}(\theta_{k,j}/\theta_{\max})^2,\quad\delta=0.002\ \mathrm{s},\qquad
\min_{a_0,\ldots,a_{N-1}}J_N,\qquad
J_N=\sum_{k=0}^{N-1}\gamma^k[\ell_u(a_k)+\ell_\theta(x_k,a_k)]+\gamma^N F_Q(x_N).
$$

Here w_u is retained to avoid a new effort-to-value scale choice; it is a historical reward weight, not an electrical power model. Gamma discounts costs to the current decision. No additional terminal multiplier is introduced. Changing the effort coefficient or removing discounting would define another explicitly labeled objective.

Let x_k,j denote the computed state after j of the fifty 2 ms integration steps in decision k, and theta_k,j its arm component. The admissible set is

$$
x_0=x_{\mathrm{measured}},\qquad
x_{k+1}=f_\Delta(x_k,u_{\max}a_k),\qquad
-1\le a_k\le1,\qquad
-\theta_{\max}\le\theta_{k,j}\le\theta_{\max},\quad
k=0,\ldots,N-1,\ j=1,\ldots,50.
$$

The measured state must also satisfy the arm range. Finite numbers and a well-defined dynamics evaluation are numerical feasibility requirements. Arm constraints are checked through the trajectory, including its terminal state; checking only 100 ms endpoints can miss an excursion. Grid feasibility is not a continuous-time guarantee. Independently replay accepted plans at a finer integration step to measure intersample and integration error.

There is no terminal upright equality, prescribed energy shell, phase-sign constraint, speed limit or torque-slew constraint in this primary problem. Existing physical limits remain, but feature normalization scales are not speed limits. The soft arm preference learned inside Q remains part of that frozen value; removing it would require a different learned function.

For fixed N and an always-enabled actuator, a fixed per-wake charge and an enabled-time charge have no effect on the optimizing torque sequence. Record them later when comparing durations. Enabled zero torque is allowed and does not imply sleep; imposing a minimum nonzero torque would be an extra constraint.

## Why this is not the old RL objective

The saved per-decision reward r_RL is the negative of a cost ell_RL accumulated at five 20 ms endpoints. It includes upright/low-speed error, relative swing-energy error, squared torque and a soft arm penalty. Its exact definition is in the [contract](../machine-scannables/implementation_contract.md).

For comparison, define J_consistent using that same negative reward, and ell_state as the difference between the RL cost and the revised effort-plus-centering cost:

$$
\ell_{\mathrm{RL}}=-r_{\mathrm{RL}},\qquad
\ell_{\mathrm{state}}=\ell_{\mathrm{RL}}-\ell_u-\ell_\theta,\qquad
J_{\mathrm{consistent}}=\sum_{k=0}^{N-1}\gamma^k\ell_{\mathrm{RL}}(x_k,a_k)+\gamma^N F_Q(x_N),
\qquad
J_N=J_{\mathrm{consistent}}-\sum_{k=0}^{N-1}\gamma^k\ell_{\mathrm{state}}(x_k,a_k).
$$

Thus the primary objective includes arm centering but omits upright and energy running costs inside the explicitly planned segment, while its tail still includes them. This is a well-defined terminal-guided experiment. It can tolerate undesirable transient motion or repeatedly postpone progress as the horizon moves. Even a perfect old critic would not remove this mismatch. Horizon-dependent behavior may reflect the changed objective rather than a solver defect.

Also, the learned continuation was trained with soft arm penalties, not hard viability. A terminal state may have a good predicted value but leave no feasible continuation. This experiment makes no recursive-feasibility or stability claim. A terminal feasible set or an explicit backup constraint would be a subsequent reformulation, not a hidden repair.

Use the consistent objective to diagnose these effects. Use the primary objective for the requested experiment unless its interpretation is changed explicitly. A zero-terminal-cost ablation now trades effort against arm centering, so it need not choose zero torque even when coast is feasible.

## Execution and the eventual outer loop

For the first closed-loop study, execute the first optimized 100 ms torque, observe again, shift the plan, and solve again. “Act throughout the prediction horizon” describes the optimized plan, not a requirement to execute all of it without feedback. A full-burst execution comparison can follow after the inner solver is understood.

For the later scheduler, let b be the integer stop decision, r the integer next-wake decision, and H the fixed future search-window length, all counted from now in 100 ms decisions. Require 1 <= b <= r <= H; require b < r when a nonempty coast is intended. Let c_on be a chosen cost rate while enabled and c_wake a chosen cost for restarting; neither is assigned a value here. Let ell_base be the declared stage cost used by that future experiment, including its treatment of waiting. Define the conditional score S as

$$
S(b,r;x_0)=\min_{a_0,\ldots,a_{b-1}}
\left[\sum_{k=0}^{r-1}\gamma^k\left(\ell_{\mathrm{base}}(x_k,a_k)+\Delta c_{\mathrm{on}}\mathbf1_{k<b}\right)
+\gamma^r\bigl(c_{\mathrm{wake}}+F_Q(x_r)\bigr)\right],\qquad
a_k=0\ \text{for }b\le k<r.
$$

The indicator equals one while enabled and zero while coasting. Dynamics and arm feasibility apply through r. The initial activation cost is sunk or constant because this event begins enabled. If r=b means uninterrupted actuation, no restart occurred and c_wake must be omitted. Terminal value is evaluated at the next wake, where its actor-continuation interpretation begins. H is the search-window bound in this equation, not a second terminal evaluation time.

This equation is only an interface sketch: the old critics do not price repeated future wakes or the new enable cost. A scheduler-consistent value would eventually need the appropriate reward and, where necessary, actuator-mode input. If the intended objective instead ends at fixed H after a restart inside the window, explicitly model the resumed segment and place one terminal value at H.

Stop time, coast length and burst torque are coupled through the wake state. Choosing a stop under “coast until H,” then changing the wake time solely by its Q value, is a sequential heuristic rather than the solution of this joint score. The later outer loop should compare conditional inner solves or state that approximation explicitly. No scheduler is implemented here.

## Domain boundary

The rotary task is ready for this experiment because a nominal simulator, direct-torque actor and compatible critics exist. `src/CR3BP` currently contains only a package scaffold. The same decomposition can later apply to CR3BP, but its dynamics, units, actuator model, geometric feasibility and domain-specific value must first be defined. A rotary checkpoint is not transferable evidence for a spacecraft terminal cost.
