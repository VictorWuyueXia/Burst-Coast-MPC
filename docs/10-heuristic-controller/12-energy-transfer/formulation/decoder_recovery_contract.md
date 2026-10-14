# Implemented recovery-decoder contract

The user authorized all three recovery changes, including explicit work overrides. This contract describes the executed `predictive_recovery_v1` decoder. The [development plan](../development_plan.md) specifies its structure budget; the [results](../../../artifacts/rotary_pendulum/experiment-results/physical-energy-transfer-controller/interpretations/interpretation_summary.md) distinguish implementation checks from unresolved swing-up and containment failures.

## Dynamics and energy accounting

Define measured/simulated state x = (θ,α,ω,ν), with arm angle θ and pendulum angle α from downward in rad, and signed speeds ω,ν in rad/s. Torque u is in N m. The nominal inertias Jₐ, J₀, J, B in kg m² are respectively arm-link, base, pendulum-hinge and coupling inertias derived by the existing model. Define mass entries A,C and determinant D, then arm drift a₀ and torque authority gᵤ:

$$
A=J_0+J\sin^2\alpha,\quad C=B\cos\alpha,\quad D=AJ-C^2,
\qquad a_0(x)=f_\omega(x,0),\quad g_u(x)=J/D>0,
$$
$$
\dot\omega=a_0(x)+g_u(x)u.
$$

Here f is the existing four-state ODE; its ω component has units rad/s². gᵤ has units of angular acceleration per torque. Thus maximum torque opposite arm speed minimizes its instantaneous outward acceleration, but pendulum drift can still dominate and changes during a held pulse. Do not use a fixed inertia to certify stopping distance.

The encoder returns arm kinetic Kₐ, full pendulum-body kinetic Kₚ and pendulum potential V, all in J, with total H and upright target E★ = 0.03037176 J. Generalized arm momentum pθ and pendulum-body transfer rate P obey

$$
p_\theta=A\omega+C\nu,\quad \dot p_\theta=u,\qquad
\dot H=u\omega,\quad P=\frac{d(K_p+V)}{dt}=(u-J_a\dot\omega)\omega.
$$

pθ is in kg m²/s and P is in W. These are analytical lossless-model identities. Reducing Kₐ does not imply all removed arm energy enters the pendulum: some can leave through negative motor work. Measure all three components and actual work.

## Candidate and recovery prediction

Retain U = 0.00918 N m torque cap, 33 grid samples and ten parallel bisection refinements per sign bracket. Set Δ = 0.1 s for the real held action and δt = 0.02 s for prediction sampling. Include zero torque and all resolved roots even when commanded work w = 0. Combine these with the bounded grid (32 bracket candidates plus 33 grid candidates); invalid roots are masked, never interpreted as coast.

A candidate starts with one real held pulse u. For a first explicit recovery continuation, compute a bounded velocity-canceling command at each subsequent 100 ms update:

$$
u_b(x)=\operatorname{clip}\left(\frac{-\omega/\Delta-a_0(x)}{g_u(x)},-U,U\right).
$$

u_b is a model-based backup command, not passive damping. Its acceleration target would stop the arm in one Δ under frozen dynamics; the full nonlinear held-action rollout decides what actually happens. There is no position-centering term. It can miss feasible recovery maneuvers; compare it with saturated opposite torque on the validation probes rather than claiming it is optimal.

Let T = `recovery_steps` × δt include the first held pulse and this feedback continuation. Require `recovery_steps` to be a multiple of five and greater than five. Pilot horizons covered 0.2–0.8 s; selected T = 0.4 s, or 20 samples. The computed small-angle period is about 0.484 s. Use inner bound Θg = π/2−0.05 rad; the 0.05 rad margin is a declared numerical design choice. A conservative sampled recovery test requires every predicted sample inside Θg and final arm speed magnitude no greater than the existing 0.15 rad/s goal tolerance.

Call the candidate set passing this test U_rec(x). This is a finite-horizon, backup-specific test, not the exact viability kernel. It can reject viable states, and a terminal low arm speed does not prove invariance under future pendulum forcing. Fine-step replay must check between coarse samples; a hard all-time guarantee would require a justified invariant terminal set or a stronger barrier construction.

## Selection and honest work mismatch

Let W(x,u) = u[θ(Δ)−θ(0)] be actual held-interval work, w the upper request, εW = 10⁻⁴E★ the existing work tolerance, and e_T(u) = (Kₐ,Kₚ,V) the predicted terminal energy. The dimensionless energy score is

$$
\Phi(e)=\frac{K_a^2+K_p^2+(V-E_\star)^2}{2E_\star^2}.
$$

First retain candidates in U_rec with |W−w| ≤ εW, and minimize Φ over them. Thus accurate work tracking is preserved whenever this recovery test finds a compatible root. For an exact tie use smaller torque magnitude then the existing positive-first rule. There is no privileged zero-request branch.

If no work-compatible candidate passes recovery, choose from U_rec by

$$
\underset{u\in\mathcal U_{\rm rec}(x)}{\operatorname{argmin}}\left[
\Phi(e_T(u))+\frac{\lambda_W}{2}\left(\frac{W(x,u)-w}{w_s}\right)^2\right],
\qquad w_s=0.04E_\star.
$$

Here wₛ is the existing work-cap scale in J and λW = `work_weight` is one dimensionless decoder tuning parameter, selected as one. Scaling mismatch by wₛ prevents its small numerical unit from making it irrelevant. Report δW = W−w; the physical transition has H′ = H+W, not H+w. Do not force later repayment with new hidden state.

If U_rec is empty, use a declared best-recovery ordering across finite candidates: minimize maximum predicted bound excess, then terminal bound excess, then terminal outward speed, then Φ. Excess is measured beyond Θg. Terminal outward speed is max(sign(θ_T)ω_T,0), where θ_T and ω_T are the predicted terminal arm angle and speed. Record `recovery_unresolved`; this ordering is not proof that the chosen action is optimal or that range recovery is physically impossible. Nonfinite dynamics are numerical failures and must not count as successful constraint handling.

The phase selection is implicit in full-state prediction of energy transfer and potential height. It does not impose a fixed phase offset or guarantee phase locking. Validation must show useful reversals, bounded motion and better swing-up separately; parking the arm is not swing-up success.

## Modes and numerical records

| Mode | Meaning | Applied command |
| --- | --- | --- |
| 0 | At least one passing recovery candidate matches requested work within εW | Lowest energy score among these candidates |
| 1 | Passing recovery candidates exist, none matches work | Lowest energy-plus-work-mismatch score |
| 2 | No candidate passes the recovery test | Explicit lexicographic best-recovery action |
| 3 | No finite valid candidate | NaN torque; evaluation rejects numerical failure |

`root_count` counts finite bracket candidates within work tolerance, not mathematically distinct roots. `recoverable_count` includes grid and bracket candidates, including possible duplicates. Finite checks include the score, peak, predicted work, request and work weight. The torque grid is cast to the physical-state dtype to preserve JAX scan carry consistency.

`requested_work_j` is the upper request. `predicted_work_j` uses the full first held interval. `predicted_peak_arm_rad` includes the initial state and all prediction samples. `predicted_terminal_arm_speed` is the signed speed after the controlled suffix. These are predicted diagnostics, not realized guarantees.

Evaluation records `start_x`, actual `work_j`, and `work_mismatch_j` (actual minus requested on active decisions, zero after termination). Actual work is torque times the executed arm displacement; an early terminal action can be shorter than 100 ms. Root-error maxima therefore use mode 0 and complete actions only. All actual energy components are re-encoded. The proposed/requested energy surface must use actual work after any override.

Episode records include `recovery_unresolved_fraction` (mode 2 share of active decisions), `work_override_fraction` (mode 1 share), `numerical_reject_fraction` (mode 3 share), and `total_absolute_work_mismatch_j` for all active decisions. Mode 2 can also mismatch work, so mode 1 frequency alone is not the total intervention frequency. `arm_reversals` counts speed sign changes ignoring magnitudes at most 0.15 rad/s; `maximum_potential_fraction` is maximum potential divided by E★; `zero_work_nonzero_torque_count` counts the zero-request/nonzero-command event. The reporting implementation specifies its numerical event thresholds. Any sampled arm excursion, endpoint in-bound capture and capture without earlier excursion are distinct metrics.

## Assessment artifacts and reproduction

`decoder_first_crossings.csv` and `decoder_recovery_audit.json` retain the previous decoder's first-crossing diagnosis and local zero-work/saturated-braking probes. They explain the revision, not its closed-loop success. The current `recovery_comparison.csv`, `recovery_confirmation_episodes.csv`, `recovery_pilots.csv`, `recovery_integration_audits.json` and `recovery_fine_summary.json` contain the new evaluation records. The readable report defines the strata and outcomes.

The existing `audit_decoder_recovery.main()` retains its 1 ms NumPy local probes and additionally runs two independent closed loops: 20 ms and 2 ms NumPy plants, the same JAX decoder, 100 ms control updates and 20 ms capture checks. Initial states are the first 16 from each stratum plus four deterministic probes in archived confirmation seed 20261008. Each plant replans from its own evolving state. The predictor stays at 20 ms, so this tests closed-loop sensitivity to plant integration rather than refining the predictor too. It prints progress each simulated second and saves full numerical traces under `artifacts/rotary_pendulum/experiment-results/physical-energy-transfer-controller/raw-runs/recovery_fine_audit/machine-scannables/`.

The local saturated-braking probes freeze at their first nonpositive speed only to measure stopping position; that freeze is diagnostic bookkeeping. Closed-loop replay freezes only at the shared capture condition, retaining the current environment semantics.

Run from repository root with the existing virtual environment and archived initial states/traces:

```bash
CUDA_VISIBLE_DEVICES=3 JAX_ENABLE_X64=true PYTHONPATH=src .venv/bin/python scripts/experiments/physical-energy-transfer-controller/audit_decoder_recovery.py
```

Use physical GPUs 0–3 only; allocation is checked before JAX initialization. Current confirmation campaigns use `recovery_steps`, `work_weight` and unchanged upper-policy parameters; old `coast_steps`/`arm_weight` campaigns require their source snapshots. Complete campaign commands and numerical limitations are in the interpretation summary.
