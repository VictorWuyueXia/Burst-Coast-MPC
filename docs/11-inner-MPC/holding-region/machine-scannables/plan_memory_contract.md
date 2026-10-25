# Plan-memory switching verification contract

Status: **candidate region falsified; complete-controller certificate not established**.

## Scope and implementation budget

Use controller commit 1ebd1fa3f25224edce50e77567f3d25767500d4b and its archived source. Keep all controller logic unchanged. This verification uses CPU array inference, differentiation, and saved-plan integration, not controller campaigns or training.

The implementation follows the compact-code discipline: no Python classes, dataclasses, helper wrappers, fallback behavior, exception handling, CLI options, or controller parameters. One new straight-line script, audit_plan_memory.py, performs this workflow. The existing analysis function mpc_sensitivity gains one returned coarse_prediction map; no controller function changes.

The finite structure budget is:

| Object or independent variable | Role |
|---|---|
| terminal | Saved-network CasADi function; values and Jacobian, without unused Hessian outputs |
| model | Existing analysis dictionary of nominal, execution, coarse-tail, and KKT maps |
| state, actions | Symbolic independent variables, dimensions 4 and 10 |
| queue | Symbolic function returning unweighted radius, weighted radius, objective, forecasts, and node energies |
| warm | Symbolic function producing the implemented shifted plan at a supplied successor |
| terminal_defect | Symbolic function evaluating actor terminal-cost dissipation defect |
| objective_gradient | Symbolic objective derivative with respect to the ten actions |
| actor_closed_jacobian | Symbolic derivative of the nominal actor-controlled command |
| metric, center | Stored actor-derived matrix and resting reference |
| gamma, radius | Computed diagnostic factor and five-degree projection radius |
| states, plans, forecast | Saved inputs and reconstructed forecast arrays |

There are no new Python-defined functions. CasADi functions above represent the mathematical maps directly. Manifest physics and experiment weights are fixed inputs. The last-action amplitude 0.001 and derivative step 0.000001 are fixed offline probes, not controller tuning parameters. Comparison tolerances are derived from reconstruction parity and saved in the result.

Milestones and success flags:

- **Reconstruction:** authenticate the binary network hash; stage error below \(10^{-7}\) and objective error below \(10^{-5}\). Passed at \(5.162\times10^{-9}\) and \(2.962\times10^{-7}\).
- **Replacement audit:** independently reconstruct shifted warm competitors and compare \(S,J\). Completed; preservation fails in 345 lower-cost replacements.
- **Witness validation:** store both complete plans and forecasts; check strict sublevel inequalities using exact rational arithmetic on those arrays. Passed for the two stored witnesses. This does not certify integration rounding.
- **Structural diagnosis:** check active critic separation, last-action derivative by centered difference, and a cost-decreasing feasible probe. Passed; derivative error \(4.84\times10^{-9}\).
- **Regional certificate:** require uniform nonlinear, selection, feasibility, and dense-flow bounds. Not achieved; the proposed numerical region is contradicted by archived reachable transitions.

## Exact mathematical objects

Use upright coordinates \(e=(\theta,\beta,\omega,\nu)\), normalized torque \(a\), horizon \(N=10\), and command duration 0.1 s. Let \(f_h\) denote the actual rotary-pendulum RK4 command map with microstep \(h\). The nominal map is \(f=f_{0.001}\), execution map \(g=f_{0.0005}\), and warm-tail map \(c=f_{0.002}\).

The frozen actor is \(\pi\); composed twin-critic costs are \(C_j=-Q_j(o(e),\pi(e))\), and \(F=\max(C_1,C_2)\). Observation scaling, network arrays, derived mechanics, and stage matrix \(W\) come directly from the archived manifest/design.

At a post-selection augmented state \(q=(e,U)\), let

$$
\xi_0=e,\quad \xi_{i+1}=f(\xi_i,a_i),\quad
L(v)=(v-e_\pi)^{\mathsf T}P(v-e_\pi),\quad
S(q)=\max_{i=0}^{10}\gamma^{-i/2}\sqrt{L(\xi_i)}.
$$

All formal regional statements use a fixed upright coordinate chart and a compact admissible augmented domain. If wrapped cost coordinates leave that chart, their branch changes must be included explicitly; the recorded holding forecasts do not approach that ambiguity.

The ideal successor is

$$
q_I=\left(f(e,a_0),\, [a_1,\ldots,a_9,\pi(\xi_{10})]\right).
$$

The actual successor state and warm plan are

$$
e^+=g(e,a_0),\quad
\zeta_0=e^+,\quad \zeta_{i+1}=c(\zeta_i,a_{i+1}),\ i=0,\ldots,8,
$$

$$
U_H=[a_1,\ldots,a_9,\pi(\zeta_9)],\qquad q_H=(e^+,U_H).
$$

The unchanged CEM routine chooses \(U_C=\mathcal C(e^+,U,\kappa)\), with key/counter \(\kappa\); \(q^+=(e^+,U_C)\). A universal deterministic statement must cover all admitted \(\kappa\) and ensure a plan is returned.

Define \(s=\sqrt\gamma\), \(b_\pi=[S(q_I)-sS(q)]_+\), \(b_h=|S(q_H)-S(q_I)|\), and \(b_C=[S(q^+)-S(q_H)]_+\). Uniform domain bounds \(B_\pi,B_h,B_C\) imply

$$
S(q^+)\le sS(q)+B_\pi+B_h+B_C,\qquad
R_*=\frac{B_\pi+B_h+B_C}{1-s}.
$$

The theorem is meaningful only after independently verifying those constants and recursive feasibility. Naming successor residuals does not itself verify a bound. For continuous holding, additionally certify

$$
\sup_{\substack{q\in\Omega_{\bar R}\\0\le t\le0.1}}
|\beta(\phi_t(e,a_0))|\le5\pi/180,
$$

where \(\phi_t\) is the intended physical/model flow under the held command. Finite RK4 node tests do not alone establish a continuous-time flow bound.

## Data, reproduction, and limitations

Run the script from the repository root using the local Python installation:

    C:/ProgramData/anaconda3/python.exe scripts/experiments/inner-mpc/audit_plan_memory.py

Dependencies are the versions already recorded in the actual-holding-study requirements: NumPy 2.4.6, SciPy 1.17.1, CasADi 3.8.0; Fraction is from the standard library. No new package is needed.

Binary weights must match SHA-256 840650b06e8cec03bc8d188b9cae6237853b0fac8ebfabeff11ba26be68b284d. Never normalize binary NPZ data.

All generated machine files belong under artifacts/rotary_pendulum/inner-mpc/actual-holding-study-20261025/machine-scannables:

| File | Contents |
|---|---|
| plan_memory_audit.json | Definitions, tolerances, counts, probe, and witnesses |
| plan_memory_audit.npz | 101×32 selected radii/objectives; 100×32 warm/ideal radii; replacements; forecasts; terminal dissipation |
| plan_memory_replacement_witness.npz | Decision 325, episode 10; preceding, selected, warm plans and forecasts |
| plan_memory_set_exit_witness.npz | First detected sublevel exit, decision 300, episode 6 |
| plan_memory_zero_probe.npz | Resting-reference gradient and feasible last-action probe |
| plan_memory_audit_log.txt | Incremental reconstruction output |

The 100 audited transitions start at decisions 299–398 and select queues at 300–399. Their first-nine retention mask has 2,531 true entries and 669 replacements. Full reconstructed plan equality is separately checked at \(10^{-8}\). The maximum nominal actor linear squared contraction is also evaluated; it is about 0.9976693207, slightly different from the execution-map reference.

The stored maximum actor-cost dissipation defect is positive, about 3.3000. Thus an unqualified terminal inequality \(F(f(v,\pi(v)))-F(v)\le-\ell(v,\pi(v))\) is not supported by these terminal records. This does not exclude a smaller certified terminal domain.

To obtain a different valid unchanged-controller certificate, bound replacements over a candidate joint state–plan region using verified nonlinear optimization or interval subdivision. A cost-sublevel overapproximation requires proof that the warm incumbent survives final auditing. CEM preserves nominal incumbent cost within a lane, but final auditing can reject that incumbent without restoring the original seed. Recorded comparisons show no significant selected-cost excess over warm; that finite observation does not prove the regional property.

The current numerical witnesses falsify this region with large margins. They neither prove loss of physical holding nor rule out another storage function, a narrower reachable-history description, or a different invariant-set shape.
