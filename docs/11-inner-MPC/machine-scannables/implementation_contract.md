# Inner-MPC implementation and validation contract

Status: formulation dated 2026-10-16; inner solvers implemented and smoke-tested 2026-10-19. The appended implementation record and experiment guide describe the current executable scope. The package name was explicitly requested; the milestones below are work milestones, not additional numbered project phases.

## Source-of-truth inventory

All repository links in this document are relative to its location. Source code and saved settings take precedence over historical prose. Checkpoints were selected for actor performance; critic calibration remains pending; the 2026-10-19 solver smoke results are reported separately.

| Contract element | Inspected source |
|---|---|
| Primitive physics and 2 ms base step | [physics.yaml](../../../src/rotary_pendulum/configs/physics.yaml) |
| Derived inertias, frequency, NumPy model | [dynamics.py](../../../src/rotary_pendulum/environment/dynamics.py) |
| Batched ODE and ten-step RK4 | [jax_dynamics.py](../../../src/rotary_pendulum/environment/jax_dynamics.py) |
| Episode latches and 20 ms boundary observation | [jax_environment.py](../../../src/rotary_pendulum/environment/jax_environment.py) |
| Five-feature map and explicit float32 cast | [jax_residual_task.py](../../../src/rotary_pendulum/RL/jax_residual_task.py) |
| Physical normalization constants | [jax_residual_control.py](../../../src/rotary_pendulum/RL/jax_residual_control.py) |
| Continuing reward and replay semantics | [jax_stationary_task.py](../../../src/rotary_pendulum/RL/jax_stationary_task.py) |
| Dense/SiLU networks, online and target critics | [jax_td3.py](../../../src/rotary_pendulum/RL/jax_td3.py) |
| Existing torque-bound-only CasADi NLP | [solver.py](../../../src/rotary_pendulum/mpc/solver.py), [discrete_model.py](../../../src/rotary_pendulum/mpc/discrete_model.py) |
| Model identities and recorded hashes | [selected_models.json](../../../artifacts/rotary_pendulum/stationary-balancing-20261013/machine-scannables/selected_models.json) |

Use `best.msgpack` and `settings.json` in each selected run's `machine-scannables` directory. The saved learner contains `actor`, `critics`, `target_actor`, `target_critics` and training state. Load the online actor and both online critics together; do not mix checkpoints or target/online generations.

| Run under `artifacts/rotary_pendulum/stationary-balancing-20261013/wave-2/` | Torque cap (N m) | Hidden widths | SHA-256 of `best.msgpack`, verified |
|---|---:|---|---|
| `quarter_bc1000` | 0.00459 | 256, 256, 256 | `834cc0e545e89f5aa2c3b6c0180687c85d7151a0032b2ed3470c391a7fe28f4c` |
| `eighth_wide_bc1000` | 0.002295 | 512, 512, 512 | `485f76e3244eac2fbc0399c67e3ed72ac9581658577a4c7c863aa7201ba68b9b` |

Both use contract `rotary-stationary-td3-v3`, direct unfiltered actions, discount 0.995 and squared-torque rate 2. The inherited `time_cost_per_s` field is not used by the stationary reward. The current environment has JAX/JAXlib 0.11.2, Flax 0.12.10, SciPy 1.17.1 and CasADi 3.7.2; these match the relevant project pins. No dependency changes are needed for the proposed first solver.

## Mechanics, clocks and observation

State x=(theta,alpha,omega,nu) contains arm/pendulum angles in rad and their speeds in rad/s. Torque u is in N m. I_b, I_p and I_c are respectively base, hinge and coupling inertias in kg m²; g_p is gravitational torque in N m. They are computed by `derive_model` from the configured masses, lengths and gravity. Damping coefficients d_a and d_p have units N m s and are both zero in the current physics file. Define the inertia entries A,B, determinant D and generalized forcing R_theta,R_alpha:

$$
A=I_b+I_p\sin^2\alpha,\quad B=I_c\cos\alpha,\quad D=AI_p-B^2,
\qquad R_\theta=u-d_a\omega-2I_p\sin\alpha\cos\alpha\,\omega\nu+I_c\sin\alpha\,\nu^2,
\qquad R_\alpha=-d_p\nu+I_p\sin\alpha\cos\alpha\,\omega^2-g_p\sin\alpha.
$$

The derivative and mechanical normalization scales are computed as follows. Omega_n is the linear natural frequency in rad/s; s_omega,s_nu are feature speed scales, not bounds. Theta_max=3pi/4 rad is the arm boundary inherited from the learned task.

$$
\dot x=\left(\omega,\nu,\frac{I_pR_\theta-BR_\alpha}{D},\frac{-BR_\theta+AR_\alpha}{D}\right),\qquad
\Omega_n=\sqrt{\frac{I_b g_p}{I_b I_p-I_c^2}},\quad
s_\omega=\theta_{\max}\Omega_n,\quad s_\nu=2\sqrt{g_p/I_p}.
$$

For these mechanics D is positive for every angle: its minimum is I_b I_p minus I_c squared, which is positive for the configured model. No denominator clipping is part of the formulation. Verify positivity from the loaded physics before an experiment.

Delta=0.1 s is one control decision, delta_obs=0.02 s is one reward observation, and delta_int=0.002 s is one RK4 integration step. There are five observations and fifty microsteps per decision. A horizon N contains N independent normalized controls a_k, with k=0,...,N−1 and u_k=u_max a_k. Measured x_0 is the initial condition. States x_k,j at microstep j=1,...,50 and decision-end states x_k are computed, not optimization variables in single shooting.

The observation, actor action and twin continuation costs C_i are

$$
h(x)=(\theta/\theta_{\max},\sin\alpha,\cos\alpha,\omega/s_\omega,\nu/s_\nu),\quad
a_\pi(x)=\pi(h(x)),\quad
C_i(x)=-Q_i(h(x),a_\pi(x)),\quad i\in\{1,2\}.
$$

Here pi and Q_i are the frozen online networks; weights and widths come from the matching checkpoint. Input dimensions are five for the actor and six for each critic. There is no deadline, success memory, teacher residual, or previous torque input. F_Q(x)=max(C_1(x),C_2(x)) is the selected terminal cost.

## Exact primary nonlinear program

Let w_u=2 cost units/s and gamma=0.995 be inherited reward settings. The tunable w_theta is the normalized squared-arm cost rate, and delta=0.002 s is the microstep interval; see the revised formulation. Let t be a free scalar epigraph variable in cost units, z the complete decision vector, and g_plus,g_minus the dimensionless arm-constraint residuals. The index j identifies every 2 ms endpoint. Solve

$$
\begin{aligned}
z&=(a_0,\ldots,a_{N-1},t),\\
\min_z\quad &\sum_{k=0}^{N-1}\gamma^k[\Delta w_u a_k^2+w_\theta\delta\sum_{j=1}^{50}(\theta_{k,j}/\theta_{\max})^2]+\gamma^N t,\\
\text{subject to}\quad &-1\le a_k\le1,\\
g^+_{k,j}(z)&=\theta_{k,j}/\theta_{\max}-1\le0,\quad
g^-_{k,j}(z)=-\theta_{k,j}/\theta_{\max}-1\le0,\\
C_1(x_N)-t&\le0,\qquad C_2(x_N)-t\le0,\\
x_{k+1}&=f_\Delta(x_k,u_{\max}a_k),\qquad x_0=x_{\mathrm{measured}}.
\end{aligned}
$$

For fixed controls, the smallest feasible t equals F_Q(x_N) because gamma to the power N is positive. This proves equivalence to the twin-minimum terminal return without smoothing. Dynamics are eliminated by the rollout rather than passed as equality constraints. The inequalities defining t add no physical restriction. For SLSQP's nonnegative inequality convention, negate the listed residuals. Use action bounds directly rather than applying torque clipping inside the prediction graph.

There are N+1 variables and 100N+2 nonlinear inequality residuals. Check x_0 separately. The historical SciPy path keeps every residual. The GPU SQP path takes upper/lower extrema separately within each decision, preserving the grid-feasible set but introducing active-node kinks; its exact algorithm is in [GPU contract](gpu_contract.md). Initial t is the maximum of the two evaluated continuation costs for its initial torque seed.

Freeze parameters but preserve all input derivatives. With D_x denoting total differentiation with respect to physical state, partial_a the critic's action derivative, and partial_h its feature gradient, each terminal-cost derivative includes

$$
D_x C_i=-\left[\partial_h Q_i+\partial_a Q_i\,D_h\pi\right]D_x h.
$$

All derivatives are evaluated at h(x),pi(h(x)). Differentiate through the rollout as well. No differentiation through the converged optimizer is required: the future discrete outer search compares inner results. The primary objective now also differentiates through the arm trajectory for centering; terminal-network derivatives enter the epigraph constraints.

## Reward-consistent diagnostic, not a hidden primary cost

Define beta as upright angle error in rad, E as relative hinge-swing energy in joules and E_star as its upright target. b is a unitless low-speed upright score; P_arm is the inherited dimensionless soft arm preference:

$$
\beta=\operatorname{atan2}(-\sin\alpha,-\cos\alpha),\quad
E=\tfrac12I_p\nu^2+g_p(1-\cos\alpha),\quad E_\star=2g_p,
$$
$$
b(x)=\exp\left[-\tfrac12\left((\beta/0.16)^2+(\nu/0.4)^2+(\omega/0.3)^2\right)\right],\quad
P_{\mathrm{arm}}(\theta)=\left(\frac{\theta}{\theta_{\max}}\right)^2+
\left(\frac{\max(|\theta|-0.8\theta_{\max},0)}{0.2\theta_{\max}}\right)^2.
$$

The tuned score widths are 0.16 rad, 0.4 rad/s and 0.3 rad/s respectively. Let c_RL be the positive cost rate, with four inherited rate weights (w_b,w_E,w_u,w_A)=(1,1,2,0.2), in numerical cost units/s. Let x_k,m^obs be the state at observation endpoint m=1,...,5 under the held command:

$$
c_{\mathrm{RL}}(x,a)=w_b(1-b(x))+w_E|E/E_\star-1|+w_u a^2+w_A P_{\mathrm{arm}}(\theta),\qquad
\ell_{\mathrm{RL}}(x_k,a_k)=\delta_{\mathrm{obs}}\sum_{m=1}^{5}c_{\mathrm{RL}}(x_{k,m}^{\mathrm{obs}},a_k).
$$

Use the discounted sum of ell_RL in place of the effort sum for the explicitly labeled diagnostic. Discount is applied once per 100 ms decision, not again within its five reward samples. The primary objective uses a pure squared-arm term, not the original P_arm boundary hinge or the energy/balance running terms. Both formulations use the same terminal value and hard constraints. Neither has a success bonus or a time cost. The soft-trained terminal value does not certify future hard-constraint feasibility.

## Numerical and execution contract

- Extend the existing integrator to optionally return its ten 2 ms endpoints while preserving its current terminal-state result. Currently `rk4_step` hides these endpoints and the stationary transition returns only five 20 ms states. Do not claim the required microstep constraints are already implemented.
- Predict pure physical state: do not call an episode transition that can freeze at timeout or reset. Reuse the ODE and RK4 coefficients; expose existing microsteps rather than inventing a second dynamics model.
- Preserve periodic pendulum features and signed arm coordinates. Check physical predictions and features around minus/plus pi, using periodic angle error when comparing wrapped states. Never constrain pendulum angle to a nonwrapping physical interval.
- First reproduce saved float32 inference. For optimization, promote the stored numerical weights and normalization constants without retraining and evaluate the same feature/network algebra in float64. Bypass the explicit float32 observation cast only in this declared planning path. Measure deviations against saved inference before accepting it; x64 configuration alone is insufficient.
- Use fixed-shape JAX kernels for each tested N. Compile outside the timed decision loop. Start forward-mode constraint Jacobians for the many-output/small-input map and use supplied derivatives in SLSQP. No finite-difference production derivatives or finite-difference `workers` trick is needed.
- Numerically feasible means finite output, accepted solver status and action/constraint residuals within the declared tolerance. Independently replay the proposed command sequence. The current environment flags arm equality as a crossing (`>=`), whereas the closed mathematical feasible set uses `<=`; report this convention and require strictly interior replay samples for an executable accepted plan. Do not silently enlarge the arm range.
- A finer replay checks the whole plan; actual closed-loop monitoring checks each executed segment. No finite grid proves continuous-time safety. No terminal invariant set is imposed, so future infeasibility remains a possible experimental failure.
- Execute one 100 ms command per solve in the first closed-loop study. On missing, nonfinite, nonconverged or independently rejected solutions, record a failure and stop that episode. No automatic zero torque, teacher handover or best-effort violating command is authorized by this formulation.

## Planned structure budget

The original formulation delivery added four English Markdown documents and four Chinese companions. The 2026-10-19 implementation uses the function budget below; its additional run guide has a matching Chinese version. The existing `*-CN.*` ignore pattern applies. Keep the simplest compact line-of-logic design: no new CLI/configuration framework, fallback behavior, try/with/except flow, thin helper layers or oversized functions. Future source files must be organized by task and remain roughly 40–300 lines; do not pad or remove explanatory spacing to meet length.

The following is the maximum proposed implementation structure for a later implementation turn. The optional sampling comparison is included in this budget, not permission to create a larger framework. Public kernels do substantive work; any required static argument binding uses existing library facilities, not new wrapper functions.

| Workflow | Planned functions/methods | Role and parameter contract |
|---|---|---|
| Shared physical integration | Existing `rk4_step(x,u)` and existing local `integrate` | Add one static `record_substeps` option; return endpoints only when requested; preserve all current callers |
| Physical prediction | New `rollout(x0,actions,settings)` and one local scan body `advance` | Hold each torque for 100 ms, expose microstep/observation states; parameters are measured state, normalized sequence and matched settings |
| Objective/constraints | New `evaluate(z,x0,networks,settings,output)` | Build primary/diagnostic cost or residual vector, including features and terminal networks; `output` is a static kernel selector, not a CLI option |
| Local optimizer | New `solve_inner(x0,seeds,networks,settings)` | Bind the numerical kernels, supply derivatives/bounds to SLSQP, independently accept/reject results, return full records |
| Sampling comparison | New `sample_inner(x0,seeds,networks,settings,key)` | Batched bounded CEM over the same objective and feasibility; `key` is the recorded JAX random key |
| Experiment and meaningful tests | One experiment `main`; three tests: `test_contract_parity`, `test_derivatives`, `test_feasibility_acceptance` | Load saved data, run the declared comparisons, and test actual cross-implementation/physical acceptance behavior |

Runtime kernels and solvers may occupy two task-grouped files if needed to respect the line budget; the exact split must not add functions. There are no new classes, dataclasses, custom result objects or controller-interface wrappers. Reuse existing Actor and Critic objects. Use ordinary arrays and mappings for settings, networks, seeds and records. The primary implementation has four new top-level numerical functions, one local scan body and the already-existing integrator body; the experiment/test workflow has its own bounded four-function budget.

| Independent data/object group | Origin and role |
|---|---|
| `x0`, `actions`, `t`/`z` | Measured state; optimized N-vector; optimized scalar epigraph and its packed decision vector |
| `networks`, `settings` | Frozen online weights and matching loaded physical/reward/solver contract; no newly trainable parameters |
| `seeds`, `key` | Recorded initial plans and reproducible sampling seed; generated before independent comparisons |
| `trajectory`, `values`, `jacobian` | Computed microstep states, objective/constraint evaluations and automatic derivatives |
| `result`, `records` | Library solver result and ordinary experiment records, including rejected plans |

All remaining mathematical symbols above are components, indices or derived quantities of these groups. They are not additional independent runtime configuration knobs. The optional epigraph scalar is algebraic, not a tunable parameter.

## Proposed settings and experiment milestones

Freeze physical/reward/network settings to the checkpoint. The local-solver control budget has three choices: `horizon_steps` N=10 initially, `max_iterations`=100 and `tolerance`=1e-7. These are proposed numerical settings, not measured optima. Compare N in {5,10,20,40}. Use SLSQP's `ftol` and a separately reported maximum residual against the same stated tolerance; distinguish its aggregate stopping test from per-constraint feasibility.

The sampling budget has four choices: `population`=1024, `elite_count`=64, `generations`=5 and `initial_std`=0.5 in normalized action units. Use a diagonal Gaussian per control coordinate centered on the declared initial plan, clip samples to [-1,1], and update mean/variance from feasible elites. Zero variance is reported as collapse, not repaired with an undisclosed floor. Preserve the best checked feasible sample. Fewer than the required feasible elites stops distribution fitting and records that reason; an already evaluated feasible candidate may be reported as a sampling candidate, never as a converged gradient solution. Sampling alone, gradient alone and their composition are separate labeled experiments.

Initialization uses actor rollout or the shifted previous plan, plus constant +0.5 and −0.5 action plans for multistart comparisons. The number of starts is fixed by this list, not another tuning axis. Full random keys, population draws and accepted/rejected seeds are saved. No enabled-time or wake-price parameter is tuned in this inner-only experiment.

| Milestone | Validation procedure | Success flag or explicit outcome |
|---|---|---|
| Contract lock | Check hashes, widths, five-feature inputs, torque cap, discount; compare new float32 evaluator with current inference/reward on 128 recorded states | `contract_pass`: scaled value/reward error <=1e-5; record all absolute errors and feature extrema |
| Prediction and derivatives | Compare each held-action trajectory with existing RK4 and 1 ms replay; check both critics and arm residual directional derivatives | `numerics_pass`: 2 ms reproduction <=1e-10 in float64 with periodic angle comparison; scaled directional error <=1e-4 away from branch seams; inspect seam one-sided limits |
| Precision and feasibility | Compare promoted network values to float32 originals; include plans that cross between 100 ms endpoints and nonfinite/failing solves | `acceptance_pass`: scaled value discrepancy <=1e-5; every deliberately invalid plan rejected; no independently accepted sample touches/exceeds the physical arm limit |
| Frozen-value audit | Hold the first action then follow actor for Q; use actor immediately for V; evaluate held-out and optimizer-selected terminal states | `value_audit_complete`: bias, error, twin disagreement and ranking versus independent returns reported; readiness depends on decision-scale accuracy below |
| Matched solver comparison | Same states, objective, horizons, seeds, acceptance rules and wall-clock budget; compare SLSQP, CEM and conditional hybrid | `solver_comparison_complete`: feasibility rate, actual objective, residuals, failure causes, p50/p95 latency, compilation and resource use recorded |
| Closed-loop comparison | 128 held-out starts: 32 downward, swing-up, near-upright and outward-boundary states each, constructed in physical coordinates; 120 s episodes continuing after capture | `closed_loop_complete`: primary Q-MPC, zero-terminal ablation, frozen actor and teacher assessed; consistent-cost MPC is a separate diagnostic |
| Iterative decision | Inspect failures, tune the declared solver budgets on validation starts, repeat, then evaluate locked settings on fresh starts | `method_selected` only after feasible performance/latency tradeoff is measured; otherwise report objective/value/solver limitation and next mathematical change |

Thresholds above are proposed numerical acceptance criteria, not passed results. “Scaled error” means absolute error divided by max(1,absolute reference value), componentwise for vectors. Directional checks compare automatic derivatives with central differences at normalized-control perturbations 1e-3, 1e-4 and 1e-5, using float64 and feasible interior test points. Use one-sided checks for box endpoints. Observe agreement over more than one perturbation, rather than accepting an accidental single-step match. For refinement, flag any 2 ms versus 1 ms trajectory discrepancy greater than 1e-3 rad or 1e-2 rad/s and every changed boundary decision; do not mask it by accepting a solver status.

For value audit, let L be a reference continuation length in decisions and c_max a justified bound on subsequent per-decision cost in the visited region. The remaining discounted cost is at most gamma^L c_max/(1−gamma) only if that bound is established. Without it, report finite-return convergence under extension, not a certified infinite-return target. Start at 120 s and extend to 240 s where needed; never bootstrap reference returns with the critic being tested.

Set a decision-relevant return tolerance before selecting a solver: propose 0.05 cost units. On paired terminal states whose independent return difference exceeds twice that tolerance, require at least 95% ranking agreement for a `value_ready` claim, with bootstrap confidence intervals; report magnitude error separately and require its 95th percentile <=0.05 if using absolute values to compare horizons. These are research thresholds to validate, not properties of the checkpoints. If they fail, still run a clearly labeled surrogate-Q experiment, but do not describe its horizons as accurately valued.

Verify actual angles/velocities rather than relying on reset names. Source reinspection corrects the earlier note: `targeted_resets` applies an additional minus-pi shift after modulo, placing its non-downward groups near upright. The new experiment explicitly constructs its own four physical groups. Preserve episode-disjoint validation and final-test sets. Report boundary failures and inability to find a plan in denominators, rather than dropping failed starts.

Closed-loop metrics: discounted original RL cost, effort, capture/hold time, arm crossings/peak margin, solver failures and actual command latency. Compute original RL cost from executed trajectories even when it is not the optimized primary objective. No event-level surrogate sum is substituted for realized performance. A p95 accepted-command latency below 100 ms is the proposed nominal real-time success flag; otherwise report the simulation as offline. Hardware/resource counts accompany every comparison.

Tune in at least an initial comparison and an informed second round: local iteration cap/tolerance and sampling population/generations first, with objective fixed. If changing budget improves neither acceptance nor median objective by more than 1% over two successive changes and no important failure class improves, treat ordinary budget tuning as exhausted on that validation set. Separate any horizon change because it changes the mathematical objective. Do not tune on the final held-out cohort. Explain failure before proposing a new cost, critic retraining, viability condition or multiple-shooting transcription.

## Artifact and operational contract

Future runs belong under `artifacts/rotary_pendulum/inner-mpc/<run>/`. Put named trajectory/constraint plots and a short `interpretation_summary.md` in `human-readables`. Put settings, checkpoint hashes, code revision, dtype/device metadata, seeds, numerical-parity tables, per-solve CSVs, full microstep trajectory arrays, critic audit returns and timing records in `machine-scannables`, with task subfolders. Human plots must define units, horizon, torque cap and acceptance/failure markers. No experiment JSON, CSV or checkpoint belongs in `docs/`.

Before any GPU run inspect occupancy and follow the repository's idle-device/over-four-in-use rule; use escalated terminal execution for actual GPU visibility. Parallelize independent states, populations, starts and horizon jobs; do not spawn agents merely to run them. Print progress at compilation start/end, each CEM generation or solver batch, and at least every 30 seconds during long evaluations. Stop and instrument any unexplained long silent run before restarting it. This is a formulation experiment, not a production controller; failures remain scientific results rather than triggers for undisclosed control logic.


## Implemented interface and bounded workflow, 2026-10-19

The implementation preserves the four-function numerical budget. `inner_model.py` contains `rollout` and `evaluate`; `inner_solvers.py` contains `solve_inner` and `sample_inner`. The shared RK4 function has static `record_substeps=False` and `substeps=10` parameters; 20 substeps produces the 1 ms acceptance audit. These are grid/recording choices, not extra controller tuning weights.

`rollout(x0, actions, settings, substeps=10)` accepts arbitrary leading candidate batch axes. Its normalized action array ends in N decisions, and its returned physical trajectory ends in `(N, 5*substeps, 4)`. It never clips input or stops on success. A scan across 20 ms intervals calls the same shared RK4 kernel as the current environment.

`evaluate(z, x0, networks, settings, output)` is the common compiled evaluator. Its `z` ends in N controls and the epigraph scalar. `networks` holds the online `actor` parameters and the pair of online `critics`, in existing Flax Dense-layer format. `settings` contains the matching numeric reward, observation and torque constants plus the explicit Boolean `rl_cost`. The static `output` selector has these defined meanings:

| Selector | Return and purpose |
|---|---|
| `policy` | Frozen actor normalized action at `x0`; the supplied decision vector is ignored |
| `value` | Actor action, both continuation costs and features at `x0` |
| `objective` | Effort plus arm centering plus discounted epigraph; primary objective |
| `rl_objective` | Declared running cost plus discounted epigraph; use with `rl_cost=True` for the diagnostic |
| `constraints` | Nonnegative arm and terminal-epigraph residual vector for SLSQP |
| `score` | Actual twin-critic objective, nominal physical feasibility and both critic costs; used for CEM ranking |
| `audit` | Nominal/fine trajectories, actual objective, effort, RL stage diagnostics, margins, refinement discrepancy and physical acceptance |

`solve_inner(x0, seeds, networks, settings)` receives a matrix of initial action sequences. Its outer settings mapping contains `plant` (the numeric evaluator settings), `max_iterations`, and `tolerance`. It evaluates each declared seed, differentiates with the reusable `OBJECTIVE_GRAD` and `CONSTRAINT_JAC` library transforms, and independently audits every final candidate. SciPy convergence alone is insufficient. SLSQP seed searches are sequential within a solve; independent measured-state solves run concurrently.

`sample_inner(x0, seeds, networks, settings, key)` uses the same evaluator. Its settings mapping contains `plant`, `population`, `elite_count`, `generations`, and `initial_std`. It fits one diagonal Gaussian per declared initial sequence, batching every population together. It preserves evaluated initial candidates and sampled incumbents; it never treats an unevaluated mean as the selected plan. A distribution with too few feasible elites or any zero-variance coordinate stops updating and reports its reason. Completed CEM search is not a gradient-stationarity certificate.

Both return ordinary dictionaries with `accepted`, `selected`, `plans`, `records`, `elapsed_s` and `solver`. `selected=-1` means no executable plan; callers must inspect `accepted` before indexing. Each entry in `plans` contains its full decision vector and audit dictionary. CEM additionally returns sampled populations with random keys, plus per-seed acceptance. Initial states, bounds and finite arithmetic are part of acceptance; a failed episode stops instead of handing over to another controller.

The experiment workflow retains one `main` function in `run.py`. The straight-line `prepare.py` and `report.py` scripts handle design generation and report reconstruction, avoiding a large monolithic source file without adding helper functions. Preparation has four necessary CLI arguments: output directory, study preset, executed-decision count and random seed. Execution and reporting each take the prepared directory. Parameter variations live in the recorded design, not a new runtime configuration framework.

Preparation freezes weights and source hashes, snapshots the involved code and saves paired physical starts. Execution compiles kernels and exercises one full warm-up solve outside measurement, then schedules four independent state solves with a standard thread pool. JAX batches the CEM population; merely exposing multiple GPUs does not shard one experiment. Separate runs can use idle GPUs 6 and 7 after occupancy verification. Raw data and runtime logs remain in artifacts.

The current runner supports the primary effort-plus-centering design and short receding-horizon comparisons with the portable 4.59 mN m checkpoint. The shared evaluator also implements the reward-consistent diagnostic. The prepared matrix does not yet launch the complete actor/teacher/zero-terminal task comparison, critic-return calibration, an equal-time-budget comparison, the weaker checkpoint or CR3BP. These remain explicit later experiments, not completed milestones.

See the [experiment guide](../human-readables/experiment_guide.md) for commands and the [artifact interpretation](../../../artifacts/rotary_pendulum/inner-mpc/interpretation_summary.md) for measured outcomes. All source files respect the 40–300-line budget. The existing dependency pins suffice; strict type checking uses `--no-sqlite-cache` because this machine's Python build lacks `_sqlite3`.
