# Wake Sleep Model Predictive Control With Learned Horizon Q Values

## 0. Working conclusion

This document specifies a first benchmark implementation of a wake sleep controller. The controller uses constrained model predictive control (MPC) to compute only a short active burst and uses a learned action value model, called a Q critic, as the terminal cost that scores the post burst wake horizon. The first version targets deterministic benchmark success. Formal liveness, recursive feasibility, and robust safety proofs are deferred to a later paper.

The design should not rely on an L1 input norm alone. L1 promotes sparse input values, but it does not force the nonzero inputs to appear in one compact burst followed by one long coast. The burst and coast pattern must be represented explicitly in the MPC constraints and input parameterization.

The first benchmark should use a torque limited inverted pendulum with exact nonlinear dynamics. A piece wise linear prediction model can be added only after the nonlinear baseline works. The coast mode should initially hold the last burst command as a constant input. Noise and model mismatch should be introduced only after deterministic success. Early re planning should be triggered when the observed trajectory deviates from the predicted sleep trajectory beyond a threshold.

## 1. Notation

Let

$$
x_t \in \mathbb{R}^{n_x}
$$

be the state at simulator step $t$, where $n_x$ is the state dimension. Let

$$
u_t \in \mathbb{R}^{n_u}
$$

be the control input, where $n_u$ is the input dimension. The discrete dynamics are

$$
x_{t+1}=f_d(x_t,u_t),
$$

where $x_{t+1}$ is the next state and $f_d$ is either an exact numerical integration map or a local prediction model.

The hard constraints are

$$
x_t \in \mathcal{X}, \qquad u_t \in \mathcal{U},
$$

or equivalently

$$
h_x(x_t) \leq 0, \qquad h_u(u_t) \leq 0.
$$

Here $h_x$ is the state constraint function and $h_u$ is the input constraint function.

For scalar pendulum torque,

$$
\mathcal{U}=\{u: |u| \leq u_{\max}\},
$$

where $u_{\max}$ is the scalar torque limit.

The intentionally underpowered condition is

$$
u_{\max} < m g l.
$$

Here $m$ is the pendulum mass, $g$ is gravitational acceleration, and $l$ is the center of mass distance from the pivot.

The controller does not run MPC at every simulator step. It runs at decision epochs

$$
t_0,t_1,t_2,\ldots
$$

where $t_k$ is decision epoch $k$.

The next planned decision epoch is

$$
t_{k+1}=t_k+N_{w,k}.
$$

Here $N_{w,k}$ is the wake horizon or sleep duration chosen at decision epoch $k$.

## 2. Two horizon variables

The implementation should separate exactly two lengths. There is no separate prediction horizon in the first formulation.

### 2.1 Burst horizon

$$
N_{b,k} \in \mathcal{N}_b
$$

Here $N_{b,k}$ is the burst horizon at decision epoch $k$, and $\mathcal{N}_b$ is the finite candidate set of allowed burst horizons. The burst horizon is the only horizon explicitly planned by the MPC optimizer.

### 2.2 Wake horizon

$$
N_{w,k} \in \mathcal{N}_w
$$

Here $N_{w,k}$ is the wake horizon at decision epoch $k$, and $\mathcal{N}_w$ is the finite candidate set of allowed wake horizons. The wake horizon is the number of simulator steps until the next planned re optimization. It is not rolled out inside the MPC problem. It is supplied to the MPC objective through the learned Q critic terminal cost.

The wake horizon candidate set is

$$
\mathcal{N}_w=\{N_w^{(1)},N_w^{(2)},\ldots,N_w^{(M)}\}.
$$

Here $N_w^{(m)}$ is candidate wake horizon $m$, and $M$ is the number of wake horizon candidates.

The basic ordering constraint is

$$
1 \leq N_{b,k} \leq N_{w,k}.
$$

Here the left inequality enforces at least one active burst step, and the right inequality leaves a nonnegative post burst coast interval before the next planned decision epoch.

The remaining coast horizon is

$$
N_{c,k}=N_{w,k}-N_{b,k}.
$$

Here $N_{c,k}$ is the number of post burst coast steps represented by the Q critic rather than explicitly optimized by MPC.

## 3. Burst coast input structure

At decision epoch $t_k$, define the predicted state sequence

$$
\bar{x}_{0|k}=x_{t_k},
$$

where $\bar{x}_{0|k}$ is the predicted state at local index $0$ for decision epoch $k$, and $x_{t_k}$ is the measured state at simulator time $t_k$.

Define the predicted burst inputs

$$
\bar{u}_{i|k}, \qquad i=0,1,\ldots,N_{b,k}-1.
$$

Here $\bar{u}_{i|k}$ is the predicted input at local burst index $i$ for decision epoch $k$.

The burst segment input parameterization is

$$
\bar{u}_{i|k}=v_{i|k}, \qquad i=0,1,\ldots,N_{b,k}-1,
$$

where $v_{i|k}$ is the MPC optimization variable for the burst input at local index $i$.

The post burst coast command is summarized by

$$
u_{\mathrm{hold},k}=v_{N_{b,k}-1|k}.
$$

Here $u_{\mathrm{hold},k}$ is the held coast input after the burst in the initial benchmark. The MPC optimizer does not roll out the coast segment; the learned Q critic scores the cost and future consequence of holding this command for the remaining coast horizon.

Two alternative coast summaries should be implemented behind a configuration flag:

$$
u_{\mathrm{hold},k}=0,
$$

and

$$
u_{\mathrm{hold},k}=u_{\text{coast}},
$$

where $u_{\text{coast}}$ is a fixed constant selected by the experiment configuration.

The predicted burst dynamics are

$$
\bar{x}_{i+1|k}=f_d(\bar{x}_{i|k},\bar{u}_{i|k}),
\qquad i=0,1,\ldots,N_{b,k}-1.
$$

Here $f_d$ is the discrete dynamics map. The MPC problem enforces constraints over the predicted burst interval:

$$
\bar{x}_{i|k} \in \mathcal{X},
\qquad
\bar{u}_{i|k} \in \mathcal{U},
\qquad
i=0,1,\ldots,N_{b,k}-1.
$$

Here $\mathcal{X}$ is the feasible state set and $\mathcal{U}$ is the feasible input set. Post burst coast feasibility is handled by the learned Q table or database, the deterministic rollout used to populate it, and the early re planning monitor. This is the part that makes the burst sleep schedule explicit while keeping MPC limited to the burst horizon. L1 penalties can still be used, but only as secondary shaping terms.

## 4. Saturation seeking burst behavior

The controller should encourage the burst inputs to use near maximum authority. This is not the same as minimizing control energy.

For scalar torque, define the normalized burst input

$$
s_{i|k}=\frac{v_{i|k}}{u_{\max}}.
$$

Here $s_{i|k}$ is the burst input normalized by the torque limit.

The input constraint becomes

$$
|s_{i|k}| \leq 1.
$$

A smooth saturation attraction term is

$$
\ell_{\text{sat}}(v_{i|k})=
\rho_{\text{sat}}\left(1-s_{i|k}^{2}\right)^2.
$$

Here $\ell_{\text{sat}}$ is the saturation attraction cost and $\rho_{\text{sat}}\geq0$ is its weight. This term has low cost near $v_{i|k}=u_{\max}$ and $v_{i|k}=-u_{\max}$. It has high cost near zero. Use it only in the burst segment:

$$
\sum_{i=0}^{N_{b,k}-1}\ell_{\text{sat}}(v_{i|k}).
$$

For a stricter formulation, impose a lower magnitude constraint during active burst:

$$
|v_{i|k}| \geq \eta_u u_{\max},
\qquad 0 < \eta_u < 1.
$$

Here $\eta_u$ is the minimum active burst magnitude ratio. This constraint is nonconvex. For the first implementation, prefer the smooth saturation attraction term. If strict on off behavior is required later, use binary sign variables or enumerate sign sequences for the scalar pendulum.

## 5. Energy like state cost

The stage cost should be based on an energy shaped error plus local state error. Energy alone is not enough because the pendulum can have the correct energy but wrong phase.

For the pendulum, use state

$$
x=\begin{bmatrix}\theta \\ \omega\end{bmatrix},
$$

where $\theta$ is the angle and $\omega$ is angular velocity. Let the upright target be

$$
x^\star=\begin{bmatrix}\pi \\ 0\end{bmatrix}.
$$

Here $x^\star$ is the desired upright equilibrium state.

Define wrapped angular error

$$
e_\theta(x)=\operatorname{atan2}\left(\sin(\theta-\pi),\cos(\theta-\pi)\right).
$$

Here $e_\theta(x)$ is the angle error wrapped to $(-\pi,\pi]$.

Define pendulum mechanical energy, with zero potential at the downward position,

$$
E(x)=\frac{1}{2} I \omega^2 + m g l (1-\cos\theta).
$$

Here $E(x)$ is the pendulum mechanical energy and $I$ is the moment of inertia.

The upright energy is

$$
E^\star=2mgl.
$$

Here $E^\star$ is the mechanical energy at the upright equilibrium.

Define energy error

$$
e_E(x)=E(x)-E^\star.
$$

Here $e_E(x)$ is the scalar energy error.

Use the state feature vector

$$
e(x)=
\begin{bmatrix}
e_E(x) \\
e_\theta(x) \\
\omega
\end{bmatrix}.
$$

Here $e(x)$ stacks the energy error, wrapped angle error, and angular velocity.

A positive definite energy like state cost is

$$
\ell_x(x)=e(x)^T Q_e e(x),
$$

where

$$
Q_e \succ 0.
$$

Here $\ell_x(x)$ is the state cost, $Q_e$ is the feature weight matrix, and $\succ0$ means positive definite.

The full burst stage cost is

$$
\ell_b(x,u)=\ell_x(x)+\rho_{\text{sat}}\left(1-\left(\frac{u}{u_{\max}}\right)^2\right)^2+\rho_{\Delta u}(u-u_{\text{prev}})^2.
$$

Here $\ell_b(x,u)$ is the burst stage cost, $\rho_{\Delta u}\geq0$ is the smoothness weight, and $u_{\text{prev}}$ is the previous applied input. The small smoothness term should not be large. It is included only to reduce solver instability and unrealistically rapid switching.

The coast stage cost is

$$
\ell_c(x,u)=\ell_x(x)+\rho_c u^2.
$$

Here $\ell_c(x,u)$ is the coast stage cost and $\rho_c\geq0$ is the coast input penalty. For the initial hold last coast mode, $u$ is not optimized during coast. The coast input is determined by the last burst input.

## 6. Wake sleep MPC problem

At decision epoch $k$, for a candidate pair $(N_b,N_w)$, solve

$$
J_k^\star(x_{t_k},N_b,N_w)
=\min_{v_{0|k},\ldots,v_{N_b-1|k}}
\left(\sum_{i=0}^{N_b-1}\ell_b(\bar{x}_{i|k},v_{i|k})+\alpha_Q Q_\theta(\bar{x}_{N_b|k},u_{\mathrm{hold},k},N_w-N_b)
\right)
$$

Here $J_k^\star$ is the optimized MPC objective value at decision epoch $k$, $N_b$ is the candidate burst horizon, $N_w$ is the candidate wake horizon, $\ell_b$ is the burst stage cost, $\alpha_Q\geq0$ is the scalar weight on the learned terminal Q cost, $Q_\theta$ is the learned Q critic with parameters $\theta$, $\bar{x}_{N_b|k}$ is the predicted terminal burst state, $u_{\mathrm{hold},k}$ is the coast command implied by the burst plan, and $N_w-N_b$ is the remaining coast horizon represented by the Q critic.

subject to

$$
\bar{x}_{0|k}=x_{t_k},
$$

$$
\bar{x}_{i+1|k}=f_d(\bar{x}_{i|k},\bar{u}_{i|k}),
$$

where the dynamics constraint is enforced for $i=0,1,\ldots,N_b-1$.

$$
\bar{u}_{i|k}=v_{i|k}, \qquad i=0,1,\ldots,N_b-1,
$$

$$
u_{\mathrm{hold},k}=v_{N_b-1|k},
$$

$$
\bar{x}_{i|k}\in\mathcal{X},
\qquad
\bar{u}_{i|k}\in\mathcal{U}.
$$

Here the state and input constraints are enforced for $i=0,1,\ldots,N_b-1$. Optional deterministic validation may roll out the selected coast after the solve, but that validation is not part of the MPC optimization horizon.

The selected schedule is

$$
(N_{b,k},N_{w,k})=\arg\min_{N_b\in\mathcal{N}_b,\,N_w\in\mathcal{N}_w,\,N_b\leq N_w}J_k^\star(x_{t_k},N_b,N_w).
$$

Here $(N_{b,k},N_{w,k})$ is the selected burst and wake horizon pair at decision epoch $k$.

The applied input from $t_k$ to $t_{k+1}-1$ is

$$
u_{t_k+i}=\bar{u}_{i|k}^\star,
\qquad i=0,1,\ldots,N_{b,k}-1,
$$

and

$$
u_{t_k+i}=u_{\mathrm{hold},k}^\star,
\qquad i=N_{b,k},\ldots,N_{w,k}-1,
$$

unless early re planning is triggered. Here the superscript $\star$ denotes the optimized MPC solution.

## 7. Learned Q critic

The learned reinforcement learning (RL) object is only a Q critic. There is no learned actor and no separate learned state value. The MPC optimizer replaces the actor by directly optimizing the burst control variables against the learned Q terminal cost.

For a post burst state $x$, coast command $u_h$, and remaining coast horizon $N_c$, parameterize

$$
Q_\theta(x,u_h,N_c)
=e(x)^T P_\theta(u_h,N_c)e(x)+\lambda_q\psi_\theta(e(x),u_h,N_c)^2+q_\theta(u_h,N_c).
$$

Here $Q_\theta(x,u_h,N_c)$ is the learned action value, $x$ is the post burst state, $u_h$ is the coast command, $N_c$ is the remaining coast horizon, $e(x)$ is the state feature vector from Section 5, $P_\theta(u_h,N_c)$ is a positive definite matrix output by the critic, $\lambda_q\geq0$ is the neural correction weight, $\psi_\theta$ is a scalar neural correction, and $q_\theta(u_h,N_c)$ is a nonnegative horizon and command dependent offset.

Use the positive definite construction

$$
P_\theta(u_h,N_c)=L_\theta(u_h,N_c)L_\theta(u_h,N_c)^T+\epsilon_P I.
$$

Here $L_\theta(u_h,N_c)$ is a lower triangular matrix output by the critic, $\epsilon_P>0$ is a numerical floor, and $I$ is the identity matrix.

Force the neural correction to vanish at the target feature:

$$
\psi_\theta(e,u_h,N_c)
=\hat{\psi}_\theta(e,u_h,N_c)-\hat{\psi}_\theta(0,u_h,N_c).
$$

Here $\hat{\psi}_\theta$ is the unconstrained neural network head and $0$ is the zero feature vector.

Enforce the nonnegative offset with

$$
q_\theta(u_h,N_c)=\operatorname{softplus}(\hat{q}_\theta(u_h,N_c)).
$$

Here $\hat{q}_\theta$ is an unconstrained scalar network head and $\operatorname{softplus}$ is the smooth positive function $\operatorname{softplus}(z)=\log(1+\exp(z))$.

Then

$$
Q_\theta(x,u_h,N_c) \geq 0.
$$

Here nonnegativity follows from $P_\theta(u_h,N_c)\succ0$, $\lambda_q\geq0$, and $q_\theta(u_h,N_c)\geq0$. The Q value is not an infinite horizon state value. It is a wake horizon conditioned terminal action value used by MPC at the predicted post burst state.

## 8. Learning target at decision epochs

The controller operates as a variable duration decision process. Define the realized total cost over one wake sleep interval as

$$
C_k=
\sum_{i=0}^{N_{w,k}-1}\gamma^i
\ell( x_{t_k+i},u_{t_k+i}).
$$

Here $C_k$ is the realized discounted total cost from decision epoch $k$ to the next planned or early decision epoch, $\gamma$ is the discount factor with $0<\gamma\leq1$, $\ell$ is the realized one step stage cost, $x_{t_k+i}$ is the realized state at simulator time $t_k+i$, and $u_{t_k+i}$ is the realized input at simulator time $t_k+i$. Use $\gamma=1$ for finite episode deterministic benchmarks unless numerical scaling requires discounting.

The Q critic target must exclude the burst cost because the MPC objective already includes the burst stage costs explicitly. Define the realized post burst coast cost as

$$
C_k^Q=
\sum_{i=N_{b,k}}^{N_{w,k}-1}
\gamma^{i-N_{b,k}}
\ell_c(x_{t_k+i},u_{\mathrm{hold},k}).
$$

Here $C_k^Q$ is the realized discounted coast cost used to train the terminal Q critic, $N_{b,k}$ is the selected burst horizon, $N_{w,k}$ is the selected wake horizon, and $\ell_c$ is the coast stage cost.

The realized critic input is

$$
z_k=(x_{t_k+N_{b,k}},u_{\mathrm{hold},k},N_{w,k}-N_{b,k}).
$$

Here $z_k$ is the post burst state action tuple scored by the Q critic, $x_{t_k+N_{b,k}}$ is the realized state immediately after the burst, $u_{\mathrm{hold},k}$ is the coast command, and $N_{w,k}-N_{b,k}$ is the realized remaining coast horizon.

The next decision state is

$$
x_{t_{k+1}}=x_{t_k+N_{w,k}}.
$$

Here $x_{t_{k+1}}$ is the state at the next planned decision epoch.

The temporal difference target for the Q critic is

$$
y_k^Q=C_k^Q+
\gamma^{N_{w,k}-N_{b,k}}
\min_{N_b'\in\mathcal{N}_b,\,N_w'\in\mathcal{N}_w,\,N_b'\leq N_w'}
J_{\theta^-}^\star(x_{t_{k+1}},N_b',N_w').
$$

Here $y_k^Q$ is the Q learning target, $\theta^-$ denotes the target critic parameters, $N_b'$ is a candidate burst horizon at the next decision epoch, $N_w'$ is a candidate wake horizon at the next decision epoch, and $J_{\theta^-}^\star$ is the MPC objective computed with the target Q critic.

The Q critic loss is

$$
\mathcal{L}_Q(\theta)=
\frac{1}{|\mathcal{B}|}
\sum_{k\in\mathcal{B}}
\left(Q_\theta(z_k)-y_k^Q\right)^2.
$$

Here $\mathcal{L}_Q(\theta)$ is the supervised temporal difference loss, $Q_\theta(z_k)$ is shorthand for $Q_\theta(x_{t_k+N_{b,k}},u_{\mathrm{hold},k},N_{w,k}-N_{b,k})$, $\mathcal{B}$ is a replay batch of decision epoch transitions, and $|\mathcal{B}|$ is the number of transitions in that batch.

## 9. Dynamic programming Q series and wake state recording

Later versions should let the critic precompute and store all wake horizon Q values for fast MPC lookup. For every stored post burst state and coast command pair, define the Q series

$$
\mathcal{Q}_\theta(x,u_h)=
\left\{Q_\theta(x,u_h,j): j\in\mathcal{J}\right\}.
$$

Here $\mathcal{Q}_\theta(x,u_h)$ is the finite Q value series, $x$ is a post burst state, $u_h$ is the coast command, $j$ is a candidate remaining coast horizon, and $\mathcal{J}$ is the allowed wake index set.

For every candidate burst action or candidate solved burst plan, simulate the resulting post burst coast trajectory under the same coast rule:

$$
\mathcal{T}(x_0,u_h)=\{x_0,x_1,\ldots,x_{N_{\max}}\}.
$$

Here $\mathcal{T}(x_0,u_h)$ is the deterministic coast rollout, $x_0$ is the post burst state, $u_h$ is the coast command, and $N_{\max}$ is the largest wake index considered by the dynamic programming table.

For each candidate wake index $j$, define the dynamic programming score

$$
S_j=
\sum_{i=0}^{j-1}\gamma^i \ell_c(x_i,u_h)
+
\gamma^j
\min_{N_b'\in\mathcal{N}_b,\,N_w'\in\mathcal{N}_w,\,N_b'\leq N_w'}
J_{\theta^-}^\star(x_j,N_b',N_w').
$$

Here $S_j$ is the predicted total score for waking after $j$ coast steps, $\ell_c$ is the coast stage cost, $x_i$ is the $i$th state in the deterministic coast rollout, and the final term is the best future MPC plus Q score from wake state $x_j$.

The best recorded wake index is

$$
j^\star=\arg\min_{j\in\mathcal{J}} S_j.
$$

Here $j^\star$ is the best remaining coast horizon according to the dynamic programming score.

Then store

$$
\mathcal{D}(x_0,u_h)=
\left(j^\star,x_{j^\star},S_{j^\star},\mathcal{Q}_\theta(x_0,u_h)\right).
$$

Here $\mathcal{D}$ is the wake state record, $x_{j^\star}$ is the best recorded wake state, $S_{j^\star}$ is the best predicted score, and $\mathcal{Q}_\theta(x_0,u_h)$ is the stored Q value series used for fast terminal lookup by MPC.

If the intent is to wake when additional coasting stops improving the Q series, use a first local minimum rule:

$$
j^\star=\min\left\{j\in\mathcal{J}: S_{j+1}-S_j \geq -\epsilon_S\right\}.
$$

Here $\epsilon_S\geq0$ is the local improvement tolerance.

If this set is empty, use

$$
j^\star=\arg\min_{j\in\mathcal{J}}S_j.
$$

This rule implements the idea that the controller should sleep until the best strategic re planning state, rather than wake at a fixed receding horizon. In the efficient implementation, the MPC terminal term

$$
\alpha_Q Q_\theta(\bar{x}_{N_b|k},u_{\mathrm{hold},k},N_w-N_b)
$$

should be served by a nearest neighbor table, interpolation database, or learned critic cache. Here the stored key is the post burst state $\bar{x}_{N_b|k}$, the coast command $u_{\mathrm{hold},k}$, and the remaining coast horizon $N_w-N_b$.

## 10. Early re planning trigger

During sleep, the simulator or real system observes

$$
\hat{x}_{t_k+i}.
$$

The predicted executed state from the last MPC solve and coast rollout is

$$
\bar{x}^{\mathrm{exec}\star}_{i|k}.
$$

Here $\bar{x}^{\mathrm{exec}\star}_{i|k}$ is the predicted state at local execution index $i$ produced by concatenating the optimized burst rollout and the configured coast rollout.

Define the weighted deviation

$$
d_i=
\left(\hat{x}_{t_k+i}-\bar{x}^{\mathrm{exec}\star}_{i|k}\right)^T
S_x
\left(\hat{x}_{t_k+i}-\bar{x}^{\mathrm{exec}\star}_{i|k}\right),
$$

where $d_i$ is the weighted state prediction error and $S_x\succ0$ is the deviation weighting matrix.

Trigger early re planning if any of the following conditions hold:

$$
d_i>\delta_x,
$$

where $\delta_x>0$ is the allowed weighted state deviation,

$$
h_x(\hat{x}_{t_k+i})> -\delta_h,
$$

where $\delta_h>0$ is the state constraint margin, or

$$
\Delta Q_i>\delta_Q.
$$

Here $\delta_Q>0$ is the allowed Q degradation margin and

$$
\Delta Q_i=
Q_\theta(\hat{x}_{t_k+i},u_{\mathrm{hold},k}^\star,N_{w,k}-i)
-
Q_\theta(\bar{x}^{\mathrm{exec}\star}_{i|k},u_{\mathrm{hold},k}^\star,N_{w,k}-i).
$$

Here $\Delta Q_i$ is the observed increase in remaining horizon Q value relative to the predicted execution state, $u_{\mathrm{hold},k}^\star$ is the optimized coast command, and $N_{w,k}-i$ is the remaining wake horizon at local execution index $i$.

The early wake time is

$$
t_{k+1}=t_k+i.
$$

Here $t_{k+1}$ is overwritten by the early re planning time when a trigger fires. This gives the first paper a simple safety mechanism without claiming full robust invariance.

## 11. Pendulum benchmark equations

Use the nonlinear pendulum model

$$
\dot{\theta}=\omega.
$$

Here $\dot{\theta}$ is the time derivative of the pendulum angle and $\omega$ is angular velocity.

The angular acceleration equation is

$$
I\dot{\omega}=mgl\sin\theta-b\omega+u.
$$

Here $I$ is the moment of inertia, $\dot{\omega}$ is angular acceleration, $m$ is mass, $g$ is gravity, $l$ is center of mass distance, $b$ is viscous damping, and $u$ is torque input.

The underpowered condition is

$$
u_{\max}<mgl.
$$

Use a fourth order Runge Kutta integration map for the simulator and the MPC prediction model at first:

$$
x_{t+1}=f_d(x_t,u_t)=\operatorname{RK4}(x_t,u_t,\Delta t).
$$

Here $\operatorname{RK4}$ is the fourth order Runge Kutta integrator and $\Delta t$ is the simulator time step.

The initial task is swing up and stabilize near the upright set

$$
\mathcal{X}_g=
\left\{x:
|e_\theta(x)|\leq \epsilon_\theta,
\quad
|\omega|\leq \epsilon_\omega
\right\}.
$$

Here $\mathcal{X}_g$ is the goal set, $\epsilon_\theta$ is the allowed wrapped angle error, and $\epsilon_\omega$ is the allowed angular velocity error.

Success requires reaching $\mathcal{X}_g$ and staying there for $T_{\text{hold}}$ simulator steps.

## 12. Piece wise linear prediction option

Only add this after the nonlinear direct model works.

Partition the state space into regions

$$
\mathcal{R}_1,\mathcal{R}_2,\ldots,\mathcal{R}_M.
$$

Here $\mathcal{R}_r$ is state space region $r$ and $M$ is the number of local model regions.

For region $r$, approximate

$$
x_{t+1}\approx A_r x_t+B_r u_t+c_r.
$$

Here $A_r$, $B_r$, and $c_r$ are the affine dynamics coefficients for region $r$.

At MPC iteration $q$, linearize around a nominal trajectory

$$
\{x_i^{(q)},u_i^{(q)}\}_{i=0}^{N_b-1}
$$

and solve the local burst problem. Here $q$ is the sequential linearization iteration and the local index runs only over the burst horizon. Update the nominal trajectory and repeat until

$$
\max_i \|x_i^{(q+1)}-x_i^{(q)}\| \leq \epsilon_{\text{lin}}.
$$

Here $\epsilon_{\text{lin}}>0$ is the convergence tolerance for the local linearization loop.

Do not combine piece wise linearization, learned Q values, and noise in the first working version. Add one complication at a time.

## 13. Training and execution loop

At each decision epoch:

1. Read the current state $x_{t_k}$.

2. Generate candidate pairs $(N_b,N_w)$.

3. For each candidate pair, solve the burst only MPC problem with the Q critic terminal cost.

4. Select the pair with the lowest objective.

5. Apply the planned input sequence during burst and coast.

6. Monitor early re planning thresholds.

7. Store the transition

$$
\left(x_{t_k},N_{b,k},N_{w,k},U_{b,k}^\star,u_{\mathrm{hold},k}^\star,C_k,C_k^Q,x_{t_{k+1}}\right)
$$

in replay memory. Here $U_{b,k}^\star$ is the optimized burst input sequence and $C_k^Q$ is the post burst coast cost used for Q critic training.

8. Update only $Q_\theta$ from replay after each episode.

9. Periodically update the target network.

10. Save the full trajectory, schedule, cost terms, constraints, and solver diagnostics.

## 14. Development plan

### Stage 1. Deterministic environment and energy diagnostics

Build the nonlinear pendulum simulator. Validate energy computation by running passive trajectories with $u=0$ and low damping. Verify that the energy trend is physically plausible.

Deliverables:

1. `PendulumEnv.step(x,u)`.

2. `energy(x)`.

3. `state_features(x)` returning $e(x)$.

4. Rollout plots for $\theta$, $\omega$, $E$, $u$.

KPIs:

1. Integration remains numerically stable for at least 30 seconds of simulated time.

2. Passive energy is approximately conserved when $b=0$.

3. Passive energy decreases when $b>0$.

Best practice:

Use small fixed time step first. Do not tune the controller until the simulator is verified.

### Stage 2. Baseline controllers

Implement comparison controllers before the proposed method.

Required baselines:

1. Energy shaping swing up controller.

2. Fixed horizon MPC with quadratic input cost.

3. Fixed horizon MPC with L1 input cost.

4. Fixed horizon MPC with explicit burst coast input blocking.

KPIs:

1. Success rate over fixed initial states.

2. Success rate over randomized initial states.

3. Mean time to reach $\mathcal{X}_g$.

4. Active input ratio.

5. Maximum continuous coast duration.

6. Constraint violation count.

Best practice:

Do not claim novelty until the explicit burst coast baseline is implemented. The proposed method must beat that baseline, not only standard MPC.

### Stage 3. Burst only wake sleep MPC with hand Q terminal cost

Implement the MPC problem with candidate enumeration over $(N_b,N_w)$ and use a hand designed Q terminal cost

$$
Q_0(x,u_h,N_c)=e(x)^T P_0 e(x)+\rho_h u_h^2+\rho_c N_c.
$$

Here $Q_0$ is the hand designed terminal Q cost, $u_h$ is the coast command, $N_c$ is the remaining coast horizon, $P_0\succ0$ is a hand tuned feature weight matrix, $\rho_h\geq0$ is the held command penalty, and $\rho_c\geq0$ is the coast length penalty or reward coefficient.

Deliverables:

1. Candidate horizon enumeration.

2. Burst coast parameterization.

3. Constraint checking over the burst horizon plus optional post solve coast validation.

4. Warm start from the previous solution.

KPIs:

1. The controller finds burst coast solutions on at least 90 percent of deterministic test episodes.

2. The burst saturation ratio is above 80 percent during active burst steps.

3. The longest coast interval is longer than the fixed horizon MPC replanning interval.

4. Constraint violations are zero in deterministic runs.

Best practice:

Keep the candidate set small at first. Example:

$$
\mathcal{N}_b=\{2,4,6\},
\qquad
\mathcal{N}_w=\{10,20,30,40\}.
$$

### Stage 4. Learned positive definite Q critic

Train $Q_\theta(x,u_h,N_c)$ from completed wake sleep rollouts.

Deliverables:

1. Replay buffer at decision epoch level.

2. Positive definite Q critic model.

3. Target network.

4. Training loss logs.

5. Q contour plots over $(\theta,\omega)$ for each $(u_h,N_c)$ slice.

KPIs:

1. TD loss decreases and does not diverge.

2. Learned Q values are smallest near the target set for stabilizing coast commands.

3. Q contours are smooth enough for MPC optimization.

4. Replacing $Q_0$ with $Q_\theta$ improves success rate or reduces time to target.

Best practice:

Normalize all features. Clip TD targets. Keep the neural network small until the controller works.

### Stage 5. Dynamic programming Q series selector

Add the $S_j$ score, Q series storage, and best wake state recording logic.

Deliverables:

1. Rollout scoring for candidate wake indices.

2. Local minimum rule.

3. Stored Q value series and best successor state for each candidate burst plan.

KPIs:

1. Chosen wake horizons vary with state.

2. The selected wake state usually has lower Q score than earlier coast states.

3. The controller avoids unnecessary short re planning intervals when the predicted coast remains useful.

Best practice:

Use the global minimum rule first. Add the local minimum rule only after score curves are smooth and interpretable.

### Stage 6. Noise and early re planning

Introduce process noise and model mismatch after deterministic success.

Use

$$
x_{t+1}=f_d(x_t,u_t)+w_t,
$$

with

$$
w_t \sim \mathcal{N}(0,\Sigma_w).
$$

Deliverables:

1. Deviation monitor.

2. Constraint margin monitor.

3. Q degradation monitor.

4. Early re planning statistics.

KPIs:

1. Early re planning reduces constraint violations compared with fixed sleep execution.

2. Success rate degrades gradually as noise increases.

3. Mean number of extra re plans remains lower than standard MPC running every step.

Best practice:

Report noise sweeps. Do not tune thresholds on the final test set.

### Stage 7. Spacecraft benchmark only after pendulum success

Start with a simplified relative motion problem before Earth to Moon rendezvous. Use a linear Hill Clohessy Wiltshire model or a two body relative dynamics model. Add higher fidelity dynamics only after the controller behavior is understood.

The spacecraft version should use zero thrust coast as the natural coast mode:

$$
u_{\mathrm{hold},k}=0.
$$

Here $u_{\mathrm{hold},k}$ is the post burst coast command used by the Q critic and execution rollout.

KPIs:

1. Total impulse or total fuel proxy.

2. Number of burn arcs.

3. Longest coast arc.

4. Final rendezvous error.

5. Constraint violations for keep out zones and line of sight constraints.

Best practice:

Do not begin with full Earth to Moon mission dynamics. The algorithmic claim should first be isolated on a controlled relative motion benchmark.

## 15. Software structure

Recommended modules:

1. `dynamics/pendulum.py`: nonlinear dynamics, RK4 map, energy calculation.

2. `dynamics/linearized.py`: optional local linear models.

3. `mpc/wake_sleep_problem.py`: optimization problem construction.

4. `mpc/horizon_enum.py`: candidate schedule enumeration.

5. `qcritic/positive_definite_q.py`: $Q_\theta$ model.

6. `qcritic/replay_buffer.py`: decision epoch replay memory.

7. `scheduler/dp_q_series.py`: $S_j$ scoring, Q series storage, and wake state recording.

8. `safety/early_replan.py`: deviation, constraint margin, and Q degradation triggers.

9. `eval/baselines.py`: baseline controllers.

10. `eval/metrics.py`: KPI computation.

11. `scripts/train_qcritic.py`: Q critic training.

12. `scripts/run_controller.py`: closed loop evaluation.

13. `scripts/plot_rollouts.py`: trajectory and schedule visualization.

## 16. Core metrics

Report these in every experiment.

### 16.1 Task success

$$
\text{success}=1
$$

Here $\text{success}$ is the binary task success indicator. It equals $1$ if the trajectory enters $\mathcal{X}_g$ and remains there for $T_{\text{hold}}$ steps.

### 16.2 Total cost

$$
J_{\text{episode}}=\sum_{t=0}^{T-1}\ell(x_t,u_t).
$$

Here $J_{\text{episode}}$ is the total realized episode cost and $T$ is the episode length in simulator steps.

### 16.3 Active input ratio

$$
r_{\text{active}}=
\frac{1}{T}
\sum_{t=0}^{T-1}
\mathbf{1}\{|u_t|>\epsilon_u\}.
$$

Here $r_{\text{active}}$ is the fraction of episode steps with input magnitude above $\epsilon_u$.

### 16.4 Saturation ratio during burst

$$
r_{\text{sat}}=
\frac{1}{N_{\text{burst}}}
\sum_{t\in\mathcal{I}_{\text{burst}}}
\mathbf{1}\{|u_t|>\eta_{\text{sat}}u_{\max}\}.
$$

Here $r_{\text{sat}}$ is the fraction of burst steps whose input magnitude exceeds the saturation threshold, $N_{\text{burst}}$ is the total number of burst steps in the episode, $\mathcal{I}_{\text{burst}}$ is the set of burst step indices, and $\eta_{\text{sat}}$ is the saturation ratio threshold. Use $\eta_{\text{sat}}=0.9$ initially.

### 16.5 Longest coast ratio

$$
r_{\text{coast}}=
\frac{\max_j L_{\text{coast},j}}{T},
$$

where $r_{\text{coast}}$ is the longest coast ratio and $L_{\text{coast},j}$ is the length of coast segment $j$.

### 16.6 Re planning count

$$
N_{\text{plan}}=\text{number of MPC solves in one episode}.
$$

Here $N_{\text{plan}}$ is the re planning count.

### 16.7 Constraint violation count

$$
N_{\text{viol}}=
\sum_{t=0}^{T}\mathbf{1}\{h_x(x_t)>0\}
+
\sum_{t=0}^{T-1}\mathbf{1}\{h_u(u_t)>0\}.
$$

Here $N_{\text{viol}}$ is the total number of state and input constraint violations over an episode.

### 16.8 Solver time

Report mean, median, and maximum solve time per decision epoch.

## 17. Initial ablation table

Run the following ablations:

| Controller | Explicit burst coast | Q terminal cost | Wake horizon source | Early re planning |
|---|---:|---:|---:|---:|
| Energy shaping | No | No | No | No |
| Fixed horizon MPC with L2 input | No | No | No | No |
| Fixed horizon MPC with L1 input | No | No | No | No |
| Fixed horizon MPC with input blocking | Yes | No | No | No |
| Wake sleep MPC with hand Q | Yes | Hand designed | Enumerated through Q | No |
| Wake sleep MPC with learned Q | Yes | Learned critic | Critic table or enumeration | No |
| Wake sleep MPC with learned Q and early re planning | Yes | Learned critic | Critic table or enumeration | Yes |

The key comparison is against fixed horizon MPC with explicit input blocking. If the proposed controller only beats L1 MPC, the claim is weak.

## 18. Risks and simple answers for the first paper

### 18.1 Energy cost ambiguity

Problem: energy matching alone does not determine phase or direction.

Initial answer: include wrapped angle error and angular velocity error in $e(x)$.

### 18.2 Nonconvex saturation attraction

Problem: saturation seeking terms can create local minima.

Initial answer: start with short candidate burst horizons and warm start from simple sign patterns.

### 18.3 Learned Q terminal cost breaks MPC guarantees

Problem: a learned Q terminal cost may not satisfy Lyapunov decrease.

Initial answer: claim deterministic empirical benchmark success only. Keep hard constraints enforced over the burst MPC horizon, validate the selected coast rollout when deterministic validation is enabled, and do not claim formal stability in the first paper.

### 18.4 Sleep increases model mismatch risk

Problem: the controller intentionally avoids frequent feedback.

Initial answer: add early re planning thresholds using state deviation, constraint margin, and Q degradation.

### 18.5 RL credit assignment over variable durations

Problem: standard one step TD targets are wrong when wake durations vary.

Initial answer: use the variable duration target with $\gamma^{N_w}$ and decision epoch replay.

### 18.6 Q scaling can dominate MPC

Problem: learned terminal Q cost may overpower the physically meaningful MPC burst cost.

Initial answer: use a scalar multiplier $\alpha_Q$, normalize cost terms, and run an ablation sweep over $\alpha_Q$.

## 19. Minimum viable success criteria

The first complete system should satisfy all of the following on the deterministic torque limited pendulum benchmark:

1. It reaches the upright target set from selected nontrivial initial states.

2. It respects input limits and configured state constraints.

3. It produces compact near saturated burst segments.

4. It produces longer coast intervals than fixed horizon MPC.

5. It uses fewer MPC solves than standard per step MPC.

6. Its learned Q critic improves either task success, convergence time, or total cost compared with the hand designed Q terminal cost.

7. Its performance is compared against explicit input blocking, not only L1 input penalties.

## 20. Recommended first implementation settings

Use these as starting values, not as final claims.

1. Time step: $\Delta t=0.02$ seconds.

2. Episode length: $T=20$ seconds.

3. Burst horizons: $\mathcal{N}_b=\{2,4,6,8\}$.

4. Wake horizons: $\mathcal{N}_w=\{10,20,30,40,60\}$.

5. Saturation threshold: $\eta_{\text{sat}}=0.9$.

6. Active input threshold: $\epsilon_u=0.05u_{\max}$.

7. Initial Q terminal cost: quadratic $Q_0(x,u_h,N_c)=e(x)^T P_0 e(x)+\rho_h u_h^2+\rho_c N_c$.

8. Learned Q critic: two hidden layers with small width, positive definite output construction.

9. Optimizer: nonlinear solver with warm start. CasADi with IPOPT is acceptable for the first prototype.

10. Training: update the Q critic after each episode from decision epoch replay.

## 21. Developer checklist

Before adding RL:

1. The simulator passes energy checks.

2. Fixed horizon MPC runs reliably.

3. Explicit burst coast MPC runs reliably.

4. Candidate wake horizon enumeration works.

5. All predicted burst states and validated coast rollout states are logged.

6. All constraints are checked after every rollout.

Before adding noise:

1. Learned Q critic is positive definite by construction.

2. Learned Q contours are plotted for representative $(u_h,N_c)$ slices.

3. The learned Q critic improves at least one KPI.

4. Early re planning logic is implemented but disabled.

Before writing the first paper:

1. Baselines are implemented.

2. Ablations are complete.

3. Random seed sweeps are complete.

4. Failure cases are documented.

5. The claim is limited to benchmarked behavior, not formal safety or liveness.
