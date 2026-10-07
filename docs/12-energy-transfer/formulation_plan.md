# Lossless energy-space planning and a phase-aware decoder

## Current decisions and assessment

The planner uses exactly three energy coordinates and seeks upright rest. It commands mechanical work per step. The decoder reads the actual angles and signed velocities to find a corresponding torque. **Arm centering is removed; the ±90° arm limit is now soft.** The implemented decoder prioritizes predicted recovery, retains nonzero zero-work roots, and permits an explicitly reported work override when needed. Zero or unresolved requests no longer imply coast. The environment remains soft/nonterminal; the decoder uses stopping-room predictions to prioritize range control.

For this first study, both damping coefficients are zero. This makes the total-energy surface invariant during zero-torque coast. **The energy point generally moves on that surface; its three components do not stay fixed.** This is the correction needed to make the proposed geometry physically consistent.

A normalized squared distance to the three-component target is a suitable first energy-space cost. It is different from squaring only total-energy error, which cannot distinguish kinetic energy from useful height. The squared component cost becomes flat near physical upright rest, so capture must be evaluated using explicit tolerances, not just a small cost.

The analytical controller is now implemented in `src/rotary_pendulum/heuristic/`, and the shared `physics.yaml` sets both dampings to zero. Prediction and simulation use those same parameters. Previous damped artifacts remain historical. See [development plan](development_plan.md) for the implemented structure and [interpretation summary](interpretation_summary.md) for GPU results. This is a tested baseline, not a trained or successful general swing-up controller.

## Physical state and three energies

Define physical state x, measured/simulated angles θ and α in radians, signed angular velocities ω and ν in rad/s, and shaft torque u in N m:

$$
x=(\theta,\alpha,\omega,\nu),\qquad \omega=\dot\theta,\quad\nu=\dot\alpha.
$$

θ is unwrapped arm angle; α is pendulum angle from downward, with upright at π. A dot denotes differentiation with respect to time in seconds. The model constants below come from configured masses, lengths and gravity, not fitting:

| Symbol | Meaning | Nominal value |
| --- | --- | ---: |
| Jₐ | Arm link inertia about its axis | 0.000228791667 kg m² |
| J | Pendulum inertia about its hinge | 0.000133128 kg m² |
| J₀ | Arm inertia plus pendulum mass × squared arm length | 0.000402191667 kg m² |
| B | Pendulum mass × arm length × pendulum center-of-mass distance | 0.00013158 kg m² |
| G | Pendulum mass × gravity × center-of-mass distance | 0.01518588 J |

Define arm kinetic energy Kₐ, full pendulum-body kinetic energy Kₚ, potential energy V relative to downward, their encoder E, and total mechanical energy H, all energy quantities in J:

$$
K_a=\tfrac12J_a\omega^2,\qquad
K_p=\tfrac12(J_0-J_a+J\sin^2\alpha)\omega^2+B\cos\alpha\,\omega\nu+\tfrac12J\nu^2,
$$
$$
V=G(1-\cos\alpha),\qquad e=E(x)=(K_a,K_p,V)^T,\qquad H=K_a+K_p+V.
$$

Kₚ includes motion carried by the moving arm and its kinetic cross term. The cross term alone can be negative, but the full body kinetic energy is nonnegative. The old reward's relative quantity ½Jν² + V is a different diagnostic; do not mix their labels.

The target energy E★ and target vector e★ are defined by the nominal upright-rest configuration:

$$
E_\star=2G=0.03037176\;\mathrm J,\qquad e_\star=(0,0,E_\star)^T.
$$

No preferred θ is included. The JAX runtime now accepts upright rest at any arm angle; crossing ±90° is recorded separately; the decoder prioritizes predicted range recovery. Reports distinguish all captures from captures inside the soft bound. The older NumPy environment retains its own legacy goal definition.

## Zero damping: what is conserved

Set arm and pendulum damping coefficients bᵣ = bₚ = 0 N m s for this study. Define P as signed power transmitted from the arm body to the pendulum body, in W, using the existing physical ODE's arm acceleration:

$$
P=(u-J_a\dot\omega)\omega,\qquad
\dot K_a=u\omega-P,\quad \dot K_p=P-G\sin\alpha\,\nu,\quad
\dot V=G\sin\alpha\,\nu.
$$

Consequently, total mechanical energy satisfies

$$
\dot H=u\omega.
$$

For one action lasting Δ = 0.1 s, define W as **actual signed shaft work**, computed from torque and physical motion. Let t be the starting time and e′ the next encoded state:

$$
W=\int_t^{t+\Delta}u(s)\omega(s)\,ds,\qquad
\mathbf1^Te'=\mathbf1^Te+W,\qquad \mathbf1=(1,1,1)^T.
$$

The integration variable s is time in seconds. If applied torque is zero, W = 0 and the total is conserved, while internal transfer P and gravity conversion can remain nonzero. Exact upright rest is stationary in the ideal model; a nearby perturbed state need not remain there. Removing damping also removes passive dissipation, so excess energy must be extracted by negative work rather than expected to decay.

## The surface and the actually reachable subset

For an energy-only request w in J, define the candidate total-energy surface S(e,w):

$$
\mathcal S(e,w)=\{z\in\mathcal E:\ \mathbf1^Tz=\mathbf1^Te+w\}.
$$

Here z is a possible next energy vector, and ℰ is the physically admissible energy region. In addition to nonnegative kinetic energies and 0 ≤ V ≤ E★, it includes a coupling constraint derived in the [lossless contract](machine-scannables/lossless_contract.md). S is generally a two-dimensional plane portion in three-dimensional energy space. It is a necessary work-balance condition, not a claim that every point on it is reachable.

The torque itself is one scalar. Let U = 0.00918 N m be the retained policy cap, within the physical 0.0204 N m limit. Let FΔ(x,u) be the physical endpoint after holding u for Δ, and 𝒲(x,u) be its shaft work:

$$
\mathcal C_x=\{E(F_\Delta(x,u)):-U\le u\le U\},\qquad
\mathcal W(x,u)=u[F_\Delta(x,u)_\theta-\theta].
$$

The subscript θ selects the endpoint arm angle. For fixed x, varying one held torque generates at most a one-dimensional endpoint curve Cₓ, which may fold or degenerate. Prescribing w intersects that curve with the work-selected surface, usually leaving isolated roots or no root. Thus the decoder finds **a torque and its resulting energy point on the realizable subset**, rather than freely choosing any point on a surface.

At a fixed physical state, even the instantaneous input-power allocation is analytical. Define the mass-matrix entries A and C, determinant Dₘ and dimensionless fraction χ:

$$
A=J_0+J\sin^2\alpha,\quad C=B\cos\alpha,\quad D_M=AJ-C^2,
\qquad \chi=J_aJ/D_M\in(0,1).
$$

Let a(x) denote the three energy derivatives at zero torque, in W, calculated from the ODE. Then

$$
\dot e=a(x)+u\omega(\chi,1-\chi,0)^T,\qquad \mathbf1^Ta(x)=0.
$$

The applied-power increment enters the two kinetic components; potential changes through motion. This is not a fixed finite-step distribution fraction because χ, ω and the drift evolve during the action. At zero arm speed the instantaneous input contribution vanishes even though a finite torque pulse can initiate motion. Use the full held-action rollout to decode work.

## Squared component cost

Adopt the dimensionless squared energy distance Φ₂ as the first terminal/endpoint score, with equal weights and no arm-position penalty:

$$
\Phi_2(e)=\frac{\|e-e_\star\|_2^2}{2E_\star^2}
=\frac{K_a^2+K_p^2+(V-E_\star)^2}{2E_\star^2}.
$$

The subscript 2 indicates Euclidean distance; the factor ½ is conventional. This gives a geometric distance to the requested point and zero cost only at the desired three-energy target. Prefer it to a total-energy-only score: every point on H = E★ would have zero total-energy error, including rapidly moving states far from upright rest.

Near upright, define β = atan2(sin(α−π), cos(α−π)), the wrapped physical angle error in rad. At zero velocities, E★−V is approximately Gβ²/2, so Φ₂ is approximately β⁴/32. Halving a small angle error reduces this cost about sixteenfold. It is quadratic in energy coordinates but fourth-order in small physical deviations; a small score is therefore not a sufficient capture criterion. Retain the earlier linear energy deficit as a later ablation if local progress stalls, rather than introducing a mixing weight immediately.

## Implemented energy policy and soft decoder

Define dimensionless policy gain γ = `work_gain`, arm-kinetic weight κ = `kinetic_weight`, and work cap wₘₐₓ = 0.04E★ J. These are explicit heuristic choices, not fitted physical constants. The policy receives only the three energies:

$$
w=\operatorname{clip}\!\left(\gamma[E_\star-V-K_p-\kappa K_a],-w_{\max},w_{\max}\right).
$$

At κ = 1 it regulates total energy. Larger κ discourages arm kinetic storage. This is a minimal work heuristic; it does not solve an optimal energy-space planning problem. Waves 1–2 weighted both kinetic terms together; wave 3 changed to the arm-only weight above. Versioned source snapshots preserve each experiment.

The physical-state decoder performs this sequence:

1. Search 33 torques over [−U,U] for work-error sign brackets, including zero requests, and refine brackets with ten bisections. The work tolerance is εW = 10⁻⁴E★ J. Keep the grid as well as bracket candidates; this sampled search does not prove all roots were found.
2. Predict each candidate's 100 ms held pulse followed by model-based, bounded velocity-canceling commands updated every 100 ms. `recovery_steps` is the total count of 20 ms prediction samples, including the first pulse. A candidate passes the sampled recovery check if the whole path stays inside π/2−0.05 rad and final arm speed is at most 0.15 rad/s. The margin is a fixed numerical choice; no center preference is added.
3. If passing candidates realize requested work within tolerance, minimize their terminal Φ₂. Otherwise score passing candidates by Φ₂ plus one half of `work_weight` times squared work mismatch normalized by wₘₐₓ. This dimensionless weight trades energy progress against work tracking. If none passes recovery, minimize peak excess, terminal excess, terminal outward speed and then Φ₂ in that order; record recovery unresolved. Ties prefer smaller torque magnitude then positive torque. Numerical failure is explicit, not zero torque.
4. Apply only the first pulse, re-encode actual motion and report achieved work and mismatch. Selected values are `work_gain=0.04`, `kinetic_weight=1`, `work_weight=1`, `recovery_steps=20` (0.4 s). The prediction suffix is hypothetical feedback, not an executed sleep commitment.

Define δW = W−w as actual minus requested work, in J. The realized energy plane during a recovery override follows

$$
\mathbf1^Te'=\mathbf1^Te+w+\delta W.
$$

Thus the three measured energies remain truthful even when the original requested plane is infeasible or rejected for recovery. There is no hidden energy debt. Exact equations and mode definitions are in the [recovery contract](machine-scannables/decoder_recovery_contract.md).

Zero torque alone cannot guarantee the arm bound: x = (1.55,0,1,0) coasts to θ = 1.65 rad in 100 ms. Conversely, a nonzero pulse can realize zero net work by braking and reversing within the interval. The new decoder preserves that choice. A finite sampled recovery test still cannot guarantee future containment; the environment neither clips nor terminates crossings.

The old forced-coast decoder admitted downward pendulum/steady arm-rotation non-target motions, derived in the [historical implementation contract](machine-scannables/implementation_contract.md). The revision removed excursions from all 192 downward confirmation starts, but replaced rotation with bounded oscillation rather than successful swing-up. This distinction matters: useful energy transfer and phase matching must be measured separately from containment.

## What the planner needs, and what can be learned

This interpretation does not require an exact autonomous three-state ODE. The high-level representation and target geometry are energy-only, while current physical feasibility is evaluated by the phase-aware decoder. The analytical Lagrangian model already provides the power allocation, finite-step torque curve and work balance. Start with those calculations; a neural “distribution” model is unnecessary for this known nominal physics.

For a future horizon, let N be the number of decision intervals and b the number of active-work intervals before the coast suffix. A minimal objective is terminal Φ₂ at step N, with explicitly zero torque during the sleep suffix b ≤ k < N; k is the step index. A zero-work command alone no longer specifies sleep because the decoder may actively reverse or override it. Use a fixed finite deadline to assess time to target, and log total absolute work rather than adding effort weights in this first formulation. Initial studies can replan every step with b = N; a later burst/coast demonstration must execute the planned sleep interval.

A phase-independent energy transition table or NN remains an approximation if later used for multi-step planning. Work-realization status must be part of that transition, since the same e,w can be resolved in one phase and unresolved in another. Train only after fixing the decoder and observing actual transitions. No history or hidden recurrent state is added to the three-state planner. Compare alternative upper policies using the same decoder so its physical-state reasoning is credited correctly.

## Task criteria and validation milestones

Remove arm centering from this study's success definition. For physical verification, require |β| ≤ 0.08 rad, |ω| ≤ 0.15 rad/s and |ν| ≤ 0.20 rad/s for five consecutive 20 ms samples. Arm angle does not gate JAX capture; in-bound capture is reported separately. These are the retained angle/speed/dwell tolerances with arm centering removed in the JAX runtime. The energy score remains purely three-dimensional. Report arm excursions separately and distinguish first capture from sustained unforced hold.

| Stage | Validation | Success flag / current status |
| --- | --- | --- |
| 0. Lossless geometry | Explicitly zero both dampings; refine coast integration; check changing components at constant total and input-power allocation | **PASS** in the new CPU diagnostic |
| 1. Cost and rejection assumptions | Check local cost scaling and exhibit whether zero torque guarantees the arm limit | **PASS**: fourth-order local cost; zero-coast guarantee disproved |
| 2. Work decoder | Check zero-work reversal, predictive braking, work overrides, torque bounds and actual-state encoding | **Passed behavioral checks**; complete-action and full closed-loop refinement reported |
| 3. Closed-loop study | Recovery pilots, then 64 held-out starts per stratum × three seeds paired with the prior decoder; use only GPU 0–3 after allocation correction | **Range control and local capture improved; downward swing-up failed.** Remaining excursions and integration sensitivity are reported in the summary |
| 4. Burst/coast and optional NN | Execute true sleep, perturb near upright, test any later surrogate on held-out sequences including rejection | Report robust capture/sleep duration and model error; **deferred** |

The original stage-3 performance screen was not met; validating the implementation is distinct from solving swing-up. Tune on separate resets; examine physical and energy trajectories after each iteration. If cost, horizon or work-range tuning stops helping, diagnose phase reachability or decoder rejection, rather than increasing network size blindly. The earlier damped controller's success rates are historical context, not a matched lossless baseline.

## Numerical evidence and artifact interpretation

![Lossless energy geometry and local cost](human-readables/lossless_energy_geometry.png)

Left: three components change while their sum is constant. Middle: the blue coast trajectory stays on the gray constant-total plane; axes are divided by initial total energy H₀. The gray triangle is only a conservation relaxation, not the entire physically reachable set. Right: at zero speeds, squared energy distance falls fourth-order with small upright-angle error, while the linear height deficit falls second-order.

For the probe (θ,α,ω,ν) = (0,0.7,1,2), the energy split changes from [0.114396,0.581857,3.571078] mJ to [1.271114,1.395219,1.600998] mJ after 100 ms of zero torque; the total stays 4.267331 mJ. Over one second, maximum total drift is 7.81 × 10⁻⁷ J at 20 ms RK4, 2.03 × 10⁻¹¹ J at 2 ms, and 1.31 × 10⁻¹² J at 1 ms. RK4 is not exactly energy-preserving, even with zero physical damping. The 257-torque finite-step surface check has maximum residual 2.34 × 10⁻¹² J.

These are mechanics and cost probes, not controller tuning or capture results. The new figure was visually inspected. CSV/JSON records and scripts are in `machine-scannables`; the explanatory figure is in `human-readables`. The earlier damped checks remain labeled as historical in the general analytical contract.

## Structure budget and coding discipline

The current [development plan](development_plan.md) specifies every public/nested function, parameter and milestone. The module has four task-oriented files, one campaign script and four behavioral tests; no added classes, wrappers, neural network or dependency. Keep substantive code files between 40 and 300 lines. Independent episodes, torque roots and trial settings are batched; time integration and bisection remain sequential. Numerical records and plots are separated.

The original `validate_lossless.main()` and reused `energy_ledger` remain mechanics diagnostics. They copy physical parameters explicitly with zero damping. The controller now uses the shared zero-damping configuration directly. Unresolved recovery uses an explicit best-recovery action and reported actual work; numerical failure is rejected rather than disguised as successful realization.

## Sources and reproduction

Reproduce the lossless study with `MPLCONFIGDIR=/tmp/bcmpc-mpl PYTHONPATH=src .venv/bin/python docs/12-energy-transfer/machine-scannables/validate_lossless.py`. It uses the existing environment and no GPU or added dependency. The [lossless mathematical contract](machine-scannables/lossless_contract.md) specifies the exact surface, allocation, momentum and decoder rules; the [general analytical contract](machine-scannables/analytical_contract.md) retains the original Lagrangian derivation and damped evidence.

The local derivations are the basis of this proposal. For primary-source context, [Cazzolato and Prime](https://digital.library.adelaide.edu.au/items/2dee399a-84e2-4990-a1e1-c191c44d3b92) discuss Furuta dynamics and modeling approximations; [Tedrake's pendulum notes](https://underactuated.mit.edu/pend.html) explain conservative energy contours and why energy shaping and stabilization are distinct. These sources do not validate this work decoder or its work policy or soft decoder.
