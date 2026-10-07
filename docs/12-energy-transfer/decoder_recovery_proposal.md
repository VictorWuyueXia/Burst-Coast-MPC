# Decoder revision: brake early and choose the return phase

> **Status: implemented and evaluated.** The user authorized all three changes, including work overrides. The text below preserves the pre-implementation rationale and proposed acceptance criteria; references to “current” or “future” there describe that earlier assessment. The [current contract](machine-scannables/decoder_recovery_contract.md) and [results](interpretation_summary.md) supersede those implementation-status statements. All runs after the GPU allocation correction used physical GPUs 0–3 only. Range control improved; general downward swing-up and guaranteed containment did not pass.

## Recommendation and current evidence

The next change should be in the decoder: **prioritize a recoverable arm trajectory, retain nonzero zero-work solutions, and relax work tracking when range recovery requires it.** Keep the upper policy's three energy inputs and work-request interface. The environment can keep the ±90° limit soft/nonterminal while the decoder gives staying inside it first priority.

The current implementation does not reject a torque simply because it crosses the arm limit. It ranks work-compatible roots with a soft arm penalty, and coasts for zero or unresolved work requests. Reviewing the saved confirmation traces shows all 192 downward starts first crossed the limit in mode 0, while applying an accepted torque. For example, one arm was already at −72.54° and moving at −4.92 rad/s; the decoder applied −1.93 mN m because the policy requested +1.02 mJ. Its single resolved root accelerated the outward motion. Replacing only the unresolved-request coast branch would miss this failure.

This is a proposal and a bounded diagnostic study, not an implemented recovery controller. The decoder's right to override requested work is the remaining interface choice; the recommendation below assumes that override is allowed and always reports actual work. The existing controller and its earlier results remain unchanged.

## Three changes, in order

1. **Remove “zero work implies zero torque.”** Solve the work equation even when the request is zero. A pulse may first brake and then accelerate in the opposite direction, with zero net work over the whole interval. Coast remains a candidate when its predicted path is acceptable.
2. **Check stopping room before enforcing work.** Predict each candidate's actual held action and a bounded braking continuation. Reject candidates that leave no modeled room to stop/turn before the limit. This applies to successful work roots as well as unresolved requests. Do not wait for the measured angle to reach 90°.
3. **Choose a useful return motion among acceptable candidates.** Use the existing full physical model and three-component energy target to choose the phase and torque. Preserve exact work when an acceptable root exists. Otherwise select an acceptable torque with a recorded work mismatch. If no acceptable trajectory is found, choose the least predicted excursion and report recovery failure; zero torque has no privileged status.

Do not command a fixed sinusoidal arm motion or force the arm back to its center. Arm reversals should occur when stopping room and predicted pendulum-energy progress require them. Inside the range, phase-aware prediction remains free to use either direction. A successful swing-up must eventually reduce the oscillation into upright rest; sustained arm oscillation is not the terminal goal.

## Why zero-work reversal matters

Define physical state x = (θ,α,ω,ν): arm and downward-referenced pendulum angles θ,α in rad and signed angular speeds ω,ν in rad/s. Torque u is in N m; Δ = 0.1 s is the current held-action interval. Let FΔ(x,u) be the endpoint computed with the existing lossless ODE and W the actual mechanical shaft work in J:

$$
W(x,u)=u\,[F_\Delta(x,u)_\theta-\theta].
$$

The subscript θ selects arm angle. A nonzero torque can have W = 0 if the arm returns to its starting angle after reversing. Instantaneous power can be negative during braking and positive after reversal, with zero integral. This is active motor control, not passive damping.

A fine-step local probe starts at x = (1.4,0,1,0). A held torque near −0.005693 N m returns the arm almost to its initial angle in 100 ms while changing its velocity to about −0.91 rad/s. The peak arm angle is about 81.60°, and net work is within 0.1 μJ of zero. Pure coast ends at 1.5 rad with unchanged +1 rad/s arm speed. The pendulum also moves: this is a coupled-system result, not an isolated-arm calculation.

Some zero-work reversals are therefore available at the existing torque cap. They are not available for every state. At high outward speed, the required reversal torque or stopping distance may exceed the available limits.

## Braking must depend on speed and pendulum phase

Let Θ = π/2 rad be the range limit, s = +1 or −1 denote one of its sides, and vₛ = max(sω,0) be speed toward that side. With an assumed constant available deceleration a_b > 0 in rad/s², remaining angle dₛ and approximate stopping angle d_stop are

$$
d_s=\Theta-s\theta,\qquad d_{\mathrm{stop}}\approx\frac{v_s^2}{2a_b}.
$$

Both distances are angular distances in rad. The approximation explains why angle-only switching is too late. It is not the acceptance test: the Furuta arm's drift and braking authority change with pendulum angle and speed. Use the coupled ODE to predict the entire held action and braking continuation, including the control-update delay.

At the existing 0.00918 N m cap, fine-step probes with the pendulum initially downward give:

| Initial arm angle and speed | Maximum opposite torque: first stopping angle |
| --- | ---: |
| 80.21°, +1 rad/s | about 81.07° |
| 80.21°, +3 rad/s | about 88.17° |
| 88.81°, +3 rad/s | about 96.76° |

The last probe crosses even with immediate maximum opposite torque. This demonstrates that this braking law is too late there; it does not prove impossibility for every more elaborate torque waveform. A finite rollout with no excursion is likewise evidence of short-term recovery, not a proof of indefinite constraint satisfaction.

## Work tracking must yield when it conflicts with range recovery

Define e = (Kₐ,Kₚ,V) as the encoder's arm kinetic, full pendulum-body kinetic and pendulum potential energies, in J. Total physical energy is H = Kₐ + Kₚ + V. Let w be requested work, W applied work, δW = W−w the reported mismatch, and H′ the next total energy. With zero physical damping,

$$
H'=H+W=H+w+\delta W.
$$

If avoiding the boundary requires negative work, a positive or zero request cannot always be fulfilled in that same 100 ms interval. A decoder cannot guarantee both promises for every state. Re-encoding actual motion keeps the three-energy representation truthful; it does not require delivering an infeasible request.

Do not hide the mismatch in an unreported energy account. Log it, and let the unchanged upper policy see the lower measured energy next step. Avoid adding a work-debt integrator in the first revision: the current policy already responds to the observed energy deficit, and forced repayment can immediately recreate the boundary problem.

## Phase selection and minimal implementation plan

The first revision should reuse predicted squared energy distance and the full-state decoder. Evaluate reversal/coast choices over a meaningful part of the swing, with controlled recovery in the prediction instead of an unconditional coast suffix. The model's derived small-angle period is about 0.484 s, so 0.4–0.6 s is a reasonable pilot lookahead range, not an assumed fixed swing period. Nonlinear swing timing varies with energy.

Use this selection order: predicted range/recoverability first; exact work roots when available; energy-target progress to choose their physical branch. When none of those roots passes recovery checks, search the existing bounded torque grid and trade measured work mismatch against energy progress. This gives the arm an opportunity to reverse without adding a fourth planner state. Preventing excursions alone will not prove phase matching or successful swing-up.

Follow compact task-oriented files of 40–300 substantive lines: no new classes, wrappers, NN, CLI or hidden fallback rules. Independent candidates, phases and initial states remain batched. Time integration and root refinement retain their required sequential order.

| Planned structure | Role and budget |
| --- | --- |
| Current assessment: `machine-scannables/audit_decoder_recovery.py`, one `main()` | Reproduce saved first-crossing classification, zero-work reversal and early/late braking probes; write CSV/JSON, no controller mutation |
| Future `decode` revision, existing nested `predict`, `advance`, `bisect` | Keep the work roots; add explicit zero-work roots, controlled recovery prediction and candidate selection in the existing file; no new public function |
| Existing `evaluate`, `write_artifacts`, campaign `main` | Record applied/requested work, intervention reason, stopping/reversal events and excursion; keep current rollout/plot responsibilities |
| Existing behavioral-test file | Extend decoder tests with zero-work reversal, early braking, work mismatch, mirrored states and failed recovery |

Independent controller parameters: retain the policy's `work_gain` and `kinetic_weight`; replace the decoder's soft `arm_weight`/coast length with at most `work_weight` (dimensionless work-mismatch score weight) and `recovery_steps` (20 ms prediction samples). Reuse the current torque cap, grid and work tolerance. A fixed 0.05 rad inner margin is an initial numerical guard for validation, not a learned parameter or physical guarantee. Working objects are candidate states/torques, root masks, predicted paths/turning points, work errors, selection flags and existing trace dictionaries. No new controller memory is planned.

| Stage | Validation and success flag |
| --- | --- |
| 1. Decode zero work | Demonstrate finite nonzero roots, accurate net work, reversal and unchanged exact-upright rest; pass if fine-step replay confirms all accepted samples within declared work tolerance |
| 2. Prevent avoidable crossings | Sweep mirrored outward/inward states and pendulum phases at matched torque caps; compare coast, roots and braking continuation; pass if all predefined fine-step-recoverable probes stay within range, and unresolved cases are explicitly reported |
| 3. Preserve useful phase motion | Run paired prior pilot resets on the authorized GPUs 0–3, inspect each tuning wave; measure direction reversals, pendulum potential gain, work mismatch and capture separately; pass only if gains are not merely boundary parking or forced oscillation |
| 4. Confirm | Repeat three independent seeds × 64 starts per stratum against current decoder and zero torque; require zero excursions on the declared recovery test set and improved random-start capture; replay representative closed loops with smaller integration steps before a stronger claim |

The proposed guard follows the broad idea of checking and modifying a requested action using a model-predicted recovery trajectory. [Wabersich and Zeilinger's predictive safety filter](https://arxiv.org/abs/1812.05506) provides primary-source context. Its formal guarantees require assumptions and terminal conditions that this small prototype does not yet establish. Their [predictive control barrier-function paper](https://arxiv.org/abs/2105.10241) also separates recovery outside a filter's feasible set from ordinary constraint satisfaction. The numerical probes and decoder-specific recommendation here come from this repository's model, not a claim that those papers validated our implementation.
