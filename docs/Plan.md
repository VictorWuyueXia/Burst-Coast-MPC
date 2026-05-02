# Wake Sleep Model Predictive Control With Learned Horizon Conditioned Terminal Value

## Core concept and why it is nontrivial
Your controller idea can be stated as a deliberate separation of decision time scales: a short, constraint critical burst window solved by constrained MPC, and a longer strategic window handled by a learned terminal value that is evaluated at a chosen wake up time. The key operational behavior is intermittent decision making: at a wake up instant you compute a short burst that meaningfully reshapes the state, then you intentionally accept open loop or simple closed loop evolution until the next wake up time. This is structurally closer to self triggered or minimum attention control than to conventional per step receding horizon MPC. citeturn10view4turn10view3turn28view1turn32search0

From an optimal control viewpoint, the “wake up time” is a policy dependent stopping time or macro action duration. That immediately implies that the correct value function is time aware and duration aware. In finite horizon and time limited RL, the optimal value generally depends on remaining time, and omitting time creates state aliasing. citeturn24view0

Your two proposed benchmarks align with this structure:
- Torque limited or underpowered pendulum swing up is fundamentally an energy pumping problem where myopic horizon choices can fail, and successful controllers exploit resonance and phase timing. citeturn4search0turn4search8  
- Impulsive or low thrust spacecraft guidance naturally exhibits short burns and long coast arcs, and MPC formulations for rendezvous frequently emphasize thrust bounds, line of sight cones, and fuel proxies, including norms that produce bang off bang like profiles. citeturn33view0turn28view2turn5search3turn27search0

## Closest prior art and where novelty is at risk
Your proposal overlaps with several mature research threads. The novelty risk is real if the claim is simply “MPC plus RL learns terminal value and horizon.”

Value function as terminal cost in MPC is well established. Early work explicitly studied adding a global value function approximation as the final cost to shorten the MPC horizon while retaining performance. citeturn10view1 A large modern line in learning based control uses learned terminal values to reduce planning horizon or mitigate myopia. For example, entity["people","Nicklas Hansen","icml 2022 td-mpc author"] and coauthors’ TD MPC learns a terminal value used beyond the short planning horizon, learned by temporal difference learning. citeturn12view0turn11view0 entity["people","Nathan Hatch","value of planning mpc author"] and entity["people","Byron Boots","robotics rl control researcher"] discuss using approximate value functions as terminal costs for MPC and survey several RL based approaches in that space. citeturn8view2turn7search20 Learning terminal costs for constrained MPC to reduce complexity is also an established direction, including supervised or regression based terminal cost learning for constrained MPC families. citeturn8view3turn22view3

Learning the MPC horizon with RL is already directly published. entity["people","Eivind Bøhn","mpc horizon rl author"] et al. propose learning the MPC prediction horizon as a function of state using RL and explicitly note synergy with learning the MPC value function. They evaluate on tasks including inverted pendulum and show practical horizon policies learned with SAC style methods. citeturn13view3turn13view0 A related arXiv line learns variable prediction horizons for multi robot MPC using SAC. citeturn26view0 There are also non RL variable horizon MPC frameworks where the horizon length is a decision variable tied to event timing, such as step timing in bipedal walking. citeturn16view1

Choosing when to re solve MPC is core to event triggered and self triggered MPC, which is extremely close to your “sleep until wake up time” mechanism. In self triggered MPC, the next sampling instant is computed at the current instant, often jointly with the planned control sequence. citeturn10view4turn10view3turn22view4 Classical formulations explicitly optimize the waiting time to the next sample along with control, applying the plan open loop until the next trigger. citeturn10view4 In addition, minimum attention and anytime attention control formalize “attention” as inverse inter execution time and optimize it under performance constraints, which is conceptually aligned with wake sleep scheduling. citeturn28view1turn27search8

RL for event triggering MPC also exists. For example, entity["people","Fengying Dang","rl event triggered mpc author"] and coauthors train deep RL agents to decide whether to trigger MPC recomputation to trade control performance against trigger frequency. citeturn18view1turn17search6 There is also model free learning of self triggered control policies with explicit integrated cost over control performance and resource usage. citeturn16view3turn6search0

The burst coast actuation preference is strongly connected to sparse control and maximum hands off control. entity["people","Masaaki Nagahara","maximum hands off control author"], entity["people","Daniel E. Quevedo","hands off control coauthor"], and entity["people","Dragan Nešić","hands off control coauthor"] formalize maximum hands off control as minimizing support of control and show links to L1 optimal control under normality, producing bang off bang type signals. citeturn25view3turn20search12 In spacecraft MPC tutorials, L1 norm costs are explicitly shown to produce bang off bang like input trajectories in rendezvous scenarios. citeturn28view2turn27search22

Bottom line: if the headline is “RL learns horizon and terminal cost for MPC,” you will collide with existing work. The credible novelty space is in the specific coupling of three aspects that are usually treated separately: (i) a wake sleep decision schedule that is optimized for closed loop mission performance rather than only for computation savings, (ii) a horizon conditioned terminal value that is explicitly a function of the chosen wake up duration, and (iii) explicit burst coast structure and actuator saturation goals inside the MPC optimization rather than emerging indirectly. citeturn13view3turn10view4turn25view3

## How similar papers present the idea and how to position yours
Horizon learning papers typically present the horizon as an RL action and justify it as improving tradeoffs between performance and computational cost, then benchmark against fixed horizons. citeturn13view3turn26view0 A notable pattern is that the controller still solves MPC at every time step, and the “horizon” is the prediction length, not the time until the next solve. That leaves room for you if you emphasize intermittent decision epochs as the primary object, not just adaptive prediction length. citeturn13view3turn10view4

Self triggered MPC papers frame the contribution around resource constraints and provide formal guarantees. Common ingredients are: restricting the next update interval to a finite set, proving recursive feasibility, and using relaxed dynamic programming inequalities or Lyapunov like arguments for stability. citeturn10view3turn10view4turn22view4 If you want a control community venue, this is the closest narrative style to your “sleep” mechanism.

Terminal cost learning papers frame the learned value as a way to shorten horizon without losing global optimality too badly, often with stability constraints such as descent properties or scenario based certificates. citeturn22view3turn8view3turn10view2 If you claim stability, you will be compared with this literature. If you do not claim stability, you should proactively scope your claims.

Sparse actuation papers frame the desired behavior as an explicit optimal control criterion, not as an emergent phenomenon. Maximum hands off control formalizes “coast most of the time” as an L0 style objective, and then discusses tractable surrogates like L1. citeturn25view3turn25view2turn3search1 In aerospace, similar ideas are framed as minimum fuel or minimum propellant, leading to bang bang or bang off bang thrust scheduling, while discussing numerical issues around switching. citeturn5search3turn27search0turn3search7

A strong publication positioning for you is therefore:
- Treat the wake sleep schedule as the primary decision variable, not only as a byproduct of event triggering.
- Treat the RL critic as a horizon conditioned terminal cost, meaning it approximates the cost to go after sleeping for a chosen duration, not only an infinite horizon value.
- Treat burst coast as a structural constraint or explicit regularizer inside MPC, rather than hoping RL discovers it.
- Provide a clear comparison taxonomy: fixed horizon MPC, adaptive horizon MPC, event triggered MPC, self triggered MPC, and RL triggered MPC. citeturn32search0turn10view4turn18view1turn13view3

## A concrete technical formulation that matches your intent
This section gives a formulation that directly implements “plan a short burst, then sleep until wake up.”

### Intuition first
Think of the controller as operating on two clocks. The fast clock is the plant integration step. The slow clock is the controller’s decision epoch. At each decision epoch you choose two things:
1. A short high authority burst plan that respects constraints and exploits saturation when beneficial.
2. The next decision epoch, which is the wake up time.

During sleep, you do not re optimize. You either apply a predetermined coast input, or apply a low bandwidth stabilizing policy that is inexpensive to compute. This reconciles your “burst then coast” desire with the MPC constraint handling role. citeturn10view4turn10view3turn25view3

### System and constraints
Let the discrete time dynamics be
\[
x_{t+1} = f(x_t, u_t),
\]
where:
- \(x_t \in \mathbb{R}^{n}\) is the state at step \(t\).
- \(u_t \in \mathbb{R}^{m}\) is the control input.
- \(f(\cdot)\) is a known model used for prediction.

Hard constraints are
\[
x_t \in \mathcal{X}, \quad u_t \in \mathcal{U},
\]
with \(\mathcal{X}\) and \(\mathcal{U}\) compact sets representing state and input limits. This is standard constrained MPC structure. citeturn32search0turn10view2

### Decision epochs and wake up duration
Let decision epochs be indexed by \(k\), occurring at times \(t_k\). Define the chosen sleep duration as an integer
\[
\tau_k \in \{\tau_{\min}, \ldots, \tau_{\max}\}, \quad t_{k+1} = t_k + \tau_k.
\]
This discretization matches how self triggered MPC commonly treats the next sampling instant and also avoids continuous time optimal stopping complexity in a first paper. citeturn10view4turn10view3turn28view1

### Burst coast structure inside the MPC optimization
Let \(N_b\) be the burst length in steps, with \(1 \le N_b \le \tau_k\). A minimal, implementable choice is to keep \(N_b\) fixed in the first paper and let RL only learn \(\tau_k\). A stronger variant lets RL also choose \(N_b\), but that increases discrete action space. citeturn10view4turn13view3

Define a coast input policy \(\kappa_{\text{coast}}(x)\). For spacecraft, this could be \(u=0\) in a low thrust model. For a pendulum cart, it could be \(u=0\) or a small holding torque. The key is that it is cheap and predictable, and it encodes your “sleep mode.” citeturn25view3turn33view4turn4search0

At time \(t_k\), solve for the burst controls \(U_b = \{u_{0},\ldots,u_{N_b-1}\}\) over the burst, while the coast segment is fixed by \(\kappa_{\text{coast}}\). Predicted states satisfy
\[
x_{i+1} =
\begin{cases}
f(x_i, u_i), & i=0,\ldots,N_b-1,\\
f(x_i, \kappa_{\text{coast}}(x_i)), & i=N_b,\ldots,\tau_k-1,
\end{cases}
\]
with \(x_0 = x_{t_k}\), and constraints enforced for all predicted steps \(i=0,\ldots,\tau_k-1\). This matches self triggered MPC’s “apply the planned sequence until the next trigger” semantics, but with an explicit coast mode. citeturn10view4turn10view3turn22view4

### Horizon conditioned terminal value as a learned terminal cost
Define a learned terminal value function
\[
V_{\theta}(x, \tau) \approx \text{expected future cost to go after reaching state } x \text{ at a wake up time when the next decision interval is } \tau.
\]
The dependence on \(\tau\) is crucial. Time limited RL work shows that value functions generally depend on remaining time, and including time in the state representation is the standard cure for time aliasing. citeturn24view0

Now define the wake sleep MPC objective:
\[
J_{\text{WS}}(x_{t_k}, \tau_k; U_b) =
\sum_{i=0}^{N_b-1} \ell(x_i, u_i)
+
\sum_{i=N_b}^{\tau_k-1} \ell_{\text{coast}}(x_i, \kappa_{\text{coast}}(x_i))
+
V_{\theta}(x_{\tau_k}, \tau_k),
\]
where:
- \(\ell(x,u)\) is the burst stage cost.
- \(\ell_{\text{coast}}(x,u)\) is the coast stage cost, often the same \(\ell\) with \(u\) fixed, so that sleeping is not free unless the state stays good.
- \(x_{\tau_k}\) is the predicted state at the wake up time.

This is directly aligned with the classic “terminal cost extends the horizon” idea, except the terminal cost is duration conditioned rather than assumed to be a stationary infinite horizon value. citeturn12view0turn10view1turn10view2turn24view0

You can implement wake up time selection in two practical ways:
- RL chooses \(\tau_k\), MPC solves only for \(U_b\).
- MPC enumerates a small candidate set of \(\tau\) and picks the minimizer using \(V_{\theta}\) as terminal cost, which is closer to self triggered MPC with an optimization over sampling time. citeturn10view4turn13view3turn22view4

### RL training as a semi Markov decision process
To make the RL objective mathematically match variable sleep durations, model decision making at epochs \(t_k\) as a semi Markov decision process with variable duration actions. Options theory formalizes this as policies with termination conditions. citeturn25view1turn29search2

Let the per step reward be \(r_t = -\ell(x_t,u_t)\). Define the discounted return from epoch \(t_k\) under a wake sleep policy as
\[
G_k = \sum_{j=0}^{\infty} \gamma^{j} r_{t_k+j}.
\]
Split this sum into the first \(\tau_k\) steps and the remainder:
\[
G_k
=
\sum_{j=0}^{\tau_k-1} \gamma^{j} r_{t_k+j}
+
\sum_{j=\tau_k}^{\infty} \gamma^{j} r_{t_k+j}.
\]
Factor \(\gamma^{\tau_k}\) out of the remainder:
\[
G_k
=
\sum_{j=0}^{\tau_k-1} \gamma^{j} r_{t_k+j}
+
\gamma^{\tau_k}
\sum_{j=0}^{\infty} \gamma^{j} r_{t_{k+1}+j}.
\]
The last term is \(\gamma^{\tau_k} G_{k+1}\). Taking expectation gives the semi Markov Bellman equation
\[
V(x_{t_k}) = \mathbb{E}\Big[\sum_{j=0}^{\tau_k-1} \gamma^{j} r_{t_k+j} + \gamma^{\tau_k} V(x_{t_{k+1}})\Big].
\]
This is the correct learning target when \(\tau_k\) is variable and policy dependent, and it matches the known need to account for timing in value estimation. citeturn24view0turn29search2turn25view1

A one step temporal difference target for the critic can then be
\[
y_k =
\sum_{j=0}^{\tau_k-1} \gamma^{j} r_{t_k+j}
+
\gamma^{\tau_k} V_{\theta^-}(x_{t_{k+1}}, \tau_{k+1}),
\]
where \(V_{\theta^-}\) is a slowly updated target network, following standard deep RL practice. This mirrors how TD MPC uses terminal value bootstrapping, but now at decision epochs rather than at every plant step. citeturn12view0turn24view0

## Enforcing burst coast and saturation behavior in a principled way
If you want “full strength for a short time, then coast,” you should encode it explicitly. Relying on RL alone to discover that structure is fragile.

The cleanest control theoretic encoding is sparse actuation optimal control, which directly targets minimizing the temporal support of actuation. Maximum hands off control does exactly this and explains why the resulting control becomes bang off bang under amplitude bounds. citeturn25view3turn20search12turn3search1 In discrete time MPC, the practical surrogate is an L1 penalty on control magnitude,
\[
\sum_i \lambda \|u_i\|_1,
\]
combined with hard bounds \(u_i \in \mathcal{U}\). In spacecraft rendezvous MPC, L1 costs are explicitly associated with bang off bang like thrust schedules compared with quadratic costs. citeturn28view2turn5search3

If you need truly on off thrusters, mixed integer MPC is the direct encoding, but it is computationally heavy. Many aerospace papers instead use convex proxies or smoothing to avoid solver pathologies around switching and singular Jacobians. citeturn31view1turn3search7turn27search3 For a first publication, an L1 cost plus a dead zone threshold inside the actuator model is a pragmatic bridge, and the dead zone issue is explicitly studied in event based impulsive spacecraft control where tiny impulses are filtered or nullified. citeturn33view4turn21search21

There is also a structural constraint approach that matches your “burst then coast” intent even more directly than L1 regularization: impose a move blocking constraint that forces the input to be constant or zero after the burst window. Move blocking is a standard MPC input parameterization tool. citeturn31view0turn31view1 In your case, the blocking structure is not only a complexity reduction trick, it is the behavioral prior.

Finally, if you also want the burst to use saturation instead of moderate values, note that bang bang controller synthesis itself is a known topic. A recent approach explicitly targets bang bang synthesis via approximate value functions and an MPC like iterative recalibration. citeturn28view0 That connects directly to your idea of using terminal value learning, but your setting has hard constraints and an intermittent wake schedule, which that work does not focus on. citeturn28view0turn10view3

## Key failure modes and what a first paper can realistically claim
Several problems will appear quickly. You can address many of them with bounded scope claims and simple mitigations.

Stability and recursive feasibility are the first reviewers will ask about. Classic MPC gets stability by terminal ingredients and properties of the finite horizon value function. citeturn32search0turn10view2 A learned terminal cost can break the descent property required for Lyapunov style arguments. Recent work specifically targets enforcing a descent property of learned terminal costs using scenario based guarantees. citeturn22view3 For an initial paper, two practical options are defensible:
- Restrict claims to deterministic known model simulations and present empirical stability evidence only.
- Add a local terminal set and a known stabilizing terminal controller inside that set, using the learned value only outside or as an additive shaping term.

Model mismatch is more dangerous in your setting than in standard MPC because you explicitly “sleep.” During the sleep interval, you are running more open loop than a standard time triggered MPC. That increases sensitivity to disturbances and state estimation error. Robust MPC frameworks and tube MPC based learning can give safety envelopes. citeturn22view0turn10view3 If you want an easy safety story, use a safety shield: if predicted constraints will be violated before the next wake up, force an early wake and re solve. Safe RL via MPC based shielding is a recognized pattern. citeturn22view1turn18view1

Credit assignment becomes harder when durations vary. If the critic does not condition on duration, you get time aliasing. Time aware RL literature shows that omitting remaining time can cause conflicting TD updates and suboptimal policies. citeturn24view0 Your horizon conditioned \(V_{\theta}(x,\tau)\) directly addresses this, and it is also conceptually aligned with options and semi Markov RL. citeturn25view1turn29search2

Optimization non smoothness is a practical obstacle if you push bang off bang too hard. L0 objectives are discontinuous, and even L1 can create numerical issues in some regimes. citeturn25view2turn3search7 Aerospace guidance papers explicitly discuss difficulty near switching times and often use smoothing or continuation. citeturn5search3turn27search3 For an initial paper, prefer L1 or a smooth L1 L2 combination as a controllable knob, as suggested in hands off control work. citeturn25view3

Benchmarking can be made clean if you choose baseline controllers that are canonical and match each domain.
- For the torque limited pendulum, compare against energy based swing up, which is a standard robust strategy and explicitly discusses dependence on actuation to gravity ratio. citeturn4search0turn4search8 Also compare against fixed horizon MPC and adaptive horizon schemes. citeturn13view3turn32search0  
- For spacecraft, start with a rendezvous model in Hill Clohessy Wiltshire form and impulsive delta v discretization as in standard spacecraft MPC references, then evaluate bang off bang behavior by switching from quadratic to L1 costs. citeturn33view0turn28view2 For event based and impulsive behavior, compare against event based predictive or impulsive controllers that explicitly measure impulse counts and computation time. citeturn33view4turn21search21

A first paper can answer most reviewer concerns with a focused experimental protocol:
- Provide ablations: fixed horizon MPC, RL learned horizon only, learned terminal value only, and your combined wake sleep plus horizon conditioned value. citeturn13view3turn12view0turn10view3  
- Report three metrics that map directly to your claims: cumulative task cost, constraint violation rate, and actuation sparsity. The sparsity metric should be something like fraction of time steps with \(\|u_t\|_\infty\) above a dead zone, matching thruster dead zone practice. citeturn25view3turn33view4  
- Visualize the learned wake up policy as a function of state features, similar to how horizon learning papers report the distribution of chosen horizons. citeturn13view3

If you want a stronger theory component without writing a full stability proof for a learned critic, the most credible limited claim is: under exact model predictions and with a bounded discrete set of wake durations, the controller enforces constraints by construction over the planned sleep interval, and early wake up is invoked if feasibility is lost. This is aligned with how event triggered MPC work motivates triggering to reduce computation while maintaining performance. citeturn18view1turn10view3turn22view4