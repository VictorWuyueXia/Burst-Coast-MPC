# Committed controller records: figure recommendations

Reviewed commit 0e12c24, “rl and controller artifacts,” against the current five-figure paper plan. All 192 files in final-results/machine-scannables/file_manifest.json matched their listed sizes and SHA-256 values. This review reads existing records and exported figures; it does not run controllers, redraw figures, or change the manuscript.

## Recommendation

The largest scientific improvement is to Figure 3. Its current constructed zero-input example can be complemented or replaced by the newly available recorded sensitivity trajectories and finite-action comparisons. Figure 4 already uses exactly the same MPC trajectories as the relocated bundle: all 40 chunk hashes match. Actor-only and energy-heuristic records add useful supporting evidence, but their different evaluation protocols prevent a direct controller ranking.

## What the new bundles support

| Bundle | Verified content and useful role | Boundary |
|---|---|---|
| Energy controller | Two validation seeds, population metrics, fine-plant check, representative trajectories and physical keyframes. Use a short reference trajectory to explain the mechanical teacher's work/potential-energy behavior. | Historical heuristic, not the augmented demonstration teacher. Its short upright-entry criterion, larger torque allowance and wider arm envelope differ from MPC/RL. Records end after reaching its criterion and do not show sustained holding. |
| RL policy | Actor/twin-critic package, all episode metrics for nominal and disturbed long-duration tests, and three selected recorded sessions per condition. Supports sustained policy operation and specified disturbance response. | Successful actor behavior does not calibrate the critics or prove that the learned terminal cost improves MPC. Complete training histories and corresponding learning curves are not supplied. |
| RL+MPC | All 32 complete sessions, executed inputs/states, initial conditions, objective diagnostics and source snapshots. | These are the same continuously enabled sessions already used in Figure 4, not new act–coast trials. Batch timings are not single-session latency. |
| Energy sensitivity | Three recorded heuristic trajectories, three explicit torque-free releases, full energy-rate decompositions, and all 160 complete decisions of the selected slow session with counterfactual torque sweeps. | Deterministic mechanisms, not a validated quadrant controller or a matched cross-controller performance comparison. |

## Read the sensitivity plane correctly

Use the manuscript notation. Let E be total mechanical energy, V potential energy, J the pendulum hinge inertia, nu the signed pendulum speed, omega the arm speed, and M the joint inertia matrix. The archive calls relative swing energy H; in the paper it is E_h. At a fixed physical state and shaft torque u,

$$
E_h=\tfrac12J\nu^2+V,\qquad
s_E=\frac{\partial\dot E}{\partial u}=\omega,\qquad
s_h=\frac{\partial\dot E_h}{\partial u}=J\nu(M^{-1})_{21}.
$$

The matrix entry maps arm torque to pendulum acceleration. These sensitivities describe instantaneous changes in power per unit torque, not finite-action energy gains. In the undamped model,

$$
\dot E=s_Eu,\qquad \dot E_h=d_h+s_hu,
$$

where d_h is the model-computed swing-energy rate at zero torque. Thus natural exchange remains even when the actuator is disabled.

For positive torque, the raw sensitivity quadrants mean:

| Signs (s_E,s_h) | Direct input effect | Interpretation |
|---|---|---|
| (+,+) | Increase total and relative swing energy rates. | Common addition direction; negative torque reverses both contributions. |
| (-,-) | Decrease both rates. | Negative torque supplies the common addition direction. |
| (-,+) | Remove total energy while increasing the swing-energy input contribution. | Opposed energy directions; potentially useful when swinging needs support while total energy is excessive. |
| (+,-) | Add total energy while decreasing the swing-energy input contribution. | Opposed directions; negative torque reverses them. |

A quadrant does not specify whether either energy should rise. With upright-rest target E_star, define error-weighted preferences

$$
c_E=(E-E_\star)s_E,\qquad c_h=(E_h-E_\star)s_h.
$$

Each coefficient multiplies torque in the derivative of the corresponding half-squared energy error. Same positive signs favor negative torque for both input contributions; same negative signs favor positive torque; opposite signs require a task tradeoff. The raw plane answers what torque can do locally; the preference plane adds what the current energy errors request. Neither incorporates the full future motion or arm constraints.

The term “redistribution” needs care. E_h is hinge-relative energy, not full pendulum-body energy E_p. For arm-only inertia I_a and inertia entries A=M11, C=M12, J=M22, the same model gives

$$
s_p=\frac{\partial\dot E_p}{\partial u}
=\omega\left(1-\frac{I_aJ}{AJ-C^2}\right).
$$

The archive establishes that the bracket is positive throughout this model, so total and full-body sensitivities share their sign. The rich opposing-sign plane concerns relative swing energy. Near s_E=0 with appreciable s_h, call the feature **relative-swing authority with small instantaneous motor power**. Do not claim literal transfer from arm-body kinetic energy into pendulum-body energy without checking that physical energy partition and its rates.

Zero-input operation also has no universal quadrant. It requires favorable predicted natural evolution, acceptable target error and arm travel over a declared duration, and a reason to prefer inactivity over a small correction. A near-zero sensitivity does not establish any of these conditions. The coast records show repeating reduced motion with drifting arm position; the sensitivity map retraces and its polar angle is not a reliable phase clock.

## How to enhance Figure 3

Retain the energy-space geometry as a compact introductory panel. Replace the current sensitivity-versus-time panel with the requested raw (s_E,s_h) plane as the main diagnostic panel. Use the fixed slow-downward trajectory, a quiet complete background trace, direction arrows used sparingly, and a few recorded state markers. Use common positive axis normalization, preserve the blue/orange typography and palette, and avoid copying the archive's large grids, rainbow energy coloring or phase-angle panels into the single-column figure.

Keep a smaller (c_E,c_h) preference panel using exactly the same marked states. Distinguish raw response quadrants from error-dependent preferences; do not label raw quadrants as a ready-made pumping/redistribution/coast policy. A useful three-example story uses the archived pumping example C, differential-authority example D, and nearly adequate coast example F. Identify the original sample IDs in source metadata. The full six-state study belongs in supplementary material.

The archive's A and E examples supply the strongest justification for finite-horizon prediction. At A, the finite-duration preferred torque reverses the instantaneous recommendation because the state changes during the held action. At E, natural evolution raises relative swing energy sharply, so initial energy deficits alone give misleading advice. Use these examples in a compact supplementary panel of actual versus counterfactual endpoint energies or one selected torque-response curve, with drift and input contributions explicitly separated.

These comparisons use an explicit two-energy loss with a chosen relative weight, omit some controller costs and constraints, and do not execute the alternative as a closed-loop controller. Across all 160 decisions, recorded versus best-tested equal-weight torque signs differ at 50 decisions; instantaneous versus finite-action recommendations differ at 20. These counts were independently reproduced from comparison.npz. They demonstrate objective/duration sensitivity, not a performance improvement or a heuristic error rate. Example F illustrates why an activity penalty matters: its best-tested correction has a small absolute diagnostic benefit, not proof that continued actuation is worthwhile.

If even this reduced composition is too dense at 8/9-point text, keep two mechanism examples in the main figure and move the remainder to the supplement; do not shrink labels or present the full diagnostic atlas in one column. Historical heuristic diagnostics need an explicit caption distinction from current MPC+RL results. The recently committed data resolve the old local-availability gap, but not the different-protocol boundary.

## Other paper figures

| Figure | Recommended action |
|---|---|
| 1: Overview | Preserve the timing-conditioned outer/inner MPC architecture. The sensitivity maps motivate looking ahead but do not supply a learned timing policy or replace outer MPC with quadrant switching. No redesign is needed from these records alone. |
| 2: Geometry | Retain the benchmark geometry. Keep configured controller caps and assessment protocols in Results/source metadata, rather than treating the different historical limits as the same evaluation. |
| 3: Energy interpretation | Prioritize the recorded raw sensitivity plane, paired preferences, and a small number of mechanism examples. Keep physical energy geometry and relative swing energy distinct. |
| 4: Session execution | Keep the present continuous session and keyframes. The relocated records are byte-identical. Repair active extraction paths before regeneration; the local extract_session.py still names the deleted source directories. Do not put unrestricted coast releases into the blank act–coast block: these are separate tests without the proposed scheduler. |
| 5: Resource atlas | Keep the placeholder. None of these bundles supplies evaluated joint act–coast timing or all four required matched baselines. Energy heuristic, actor and MPC evaluations cannot be ranked by their existing success rates or durations. |
| Optional 6 / supplement | Prefer a compact actor-only nominal/disturbed performance panel with full population denominators and selected recovery context. Keep it separate from the MPC cohort; use a fair common protocol before making method rankings. |

A critic/actor ablation remains more directly relevant to the MPC+RL argument than adding many actor trajectories. The supplied actor results cannot replace terminal-value calibration or a matched MPC ablation. The original energy controller is useful as a mechanism/reference source, not evidence that the present demonstration teacher or MPC+RL inherits its population success rate.

## Recommended next actions and validation

1. Repair only active rendering input paths to the canonical final-results bundles, then verify the migrated record hashes. Success: Figure 4 can regenerate the same data and keyframes without simulator imports or a new run.
2. Freeze Figure 3's single recorded session and sample IDs, energy convention, common normalization, and the distinct roles of raw and preference planes. Success: every point and action annotation maps to an exact source state and the claim agrees with the full energy-rate decomposition.
3. Redraw Figure 3 in the current editable style, retaining the old asset until review. Success: raw sensitivity mapping is visible at column size, all retained text is readable, finite-duration counterfactuals are identified, and no quadrant is labeled as an implemented timing rule.
4. Put the complete action/phase diagnostics and any long-horizon actor evidence in supporting material. Keep Figure 5 empty until the required matched controller experiments exist.

This is an advisory review only. No figures, captions, controller code, renderer paths, or evaluation protocols were modified.
