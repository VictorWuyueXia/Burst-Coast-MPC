# Recorded heuristic decisions on the opportunity map

**The map exposes useful opportunities, but the recorded heuristic follows a different objective.** At six sampled states, its actual action has lower cost under its own archived objective than the alternative preferred by the two-energy diagnostic. This is evidence for a formulation comparison, not proof that the heuristic made six mistakes.

[Annotated sensitivity map](../../../../../../artifacts/rotary_pendulum/actuation-opportunities/decision-comparison/human-readables/decision_map.png) · [Six action comparisons](../../../../../../artifacts/rotary_pendulum/actuation-opportunities/decision-comparison/human-readables/sampled_decisions.png) · [Physical context](../../../../../../artifacts/rotary_pendulum/actuation-opportunities/decision-comparison/human-readables/decision_context.png)

The left map colors every recorded decision by the heuristic's torque; the right colors the same states by the best tested alternative. Gray curves follow the full physical record. A–F refer to the same states across plots and tables. Blue colors indicate negative torque, orange positive torque, and gray near zero. The physical-context plot supplies angles, energy evolution, and the actual-versus-alternative torque timeline. PDF and SVG versions are also provided.

## What “best” means here

Use the slow downward capture, archive lane 325, previously selected as the longest downward capture. It lasts 16.02 s under the historical 15°/100 ms capture criterion. Compare all 160 complete 100 ms decisions; omit the final 20 ms partial action from action scoring. No alternative controller is run: every comparison starts from the original recorded state.

Define E and H as total mechanical energy and relative hinge-swing energy, respectively. Their common upright-at-rest target E_star is 0.03037176 J, computed from the model. Define tau as a constant candidate torque, t as the recorded decision-start time, and lambda as a chosen relative-swing task weight. The dimensionless terminal loss L is

$$
L_\lambda(\tau)=\frac12\left[\left(\frac{E(t+0.1;\tau)}{E^\star}-1\right)^2
+\lambda\left(\frac{H(t+0.1;\tau)}{E^\star}-1\right)^2\right].
$$

The semicolon means the endpoint comes from a new 100 ms simulation under that candidate torque. Main figures use lambda = 1, an explicit equal-weight diagnostic choice. Lambda = 0.25 and 4 test sensitivity to this choice. We test 257 uniformly spaced torques from −18.36 to +18.36 mN m, plus the exact recorded torque. “Best tested” means the lowest loss in this set; it does not mean a globally optimal swing-up action. Arm constraints, effort penalties, and future capture rewards are absent from this diagnostic.

## Six sampled decisions

Torque columns are in mN m; losses are dimensionless. Selection intentionally illustrates mechanisms: largest disagreement with the instantaneous sign rule, largest gap among conflicting tasks, an early pumping state, low total-energy authority relative to swing authority, a late state, and the last recorded coast. These six are not a random sample.

| Point | Time (s) | Recorded torque | Best tested torque | Recorded loss → best loss |
|---|---:|---:|---:|---:|
| A | 0.00 | +5.737 | -18.360 | 0.975431 → 0.716104 |
| B | 0.10 | -9.180 | +18.360 | 0.929812 → 0.704698 |
| C | 0.80 | -8.033 | -18.360 | 0.590029 → 0.480557 |
| D | 2.00 | -12.623 | -18.360 | 0.237231 → 0.183366 |
| E | 15.00 | +0.005 | -7.028 | 0.0394264 → 0.0291628 |
| F | 15.90 | +0.000 | -1.291 | 1.84094e-05 → 1.56452e-07 |


- **A — finite action duration reverses the instantaneous advice.** Initially both sensitivities and both energy deficits favor positive torque. Nevertheless, negative saturation produces more energy after 100 ms because the arm reverses during the action. Total-energy gains are 6.125 mJ for the alternative and 0.491 mJ for the recorded action. A local power derivative cannot predict a whole 100 ms interval by itself.
- **B — task priority matters.** Total-energy sensitivity is strongly positive while swing sensitivity is slightly negative and almost zero. Equal weights favor positive saturation, but increasing the swing weight to 4 reverses the recommendation to negative saturation. Calling either sign universally best would conceal a priority choice.
- **C — both methods pump, with different strengths.** The actual torque is −8.03 mN m; the diagnostic favors −18.36 mN m. Despite the favorable instantaneous swing sensitivity, relative swing energy drops 0.398 mJ under the actual action, whereas it rises 0.855 mJ under the stronger action. Natural exchange and changing state matter.
- **D — strong swing authority with relatively little initial mechanical-work authority.** Total sensitivity is −0.431 rad/s and swing sensitivity −6.910 rad/s. Both objectives favor negative torque; the diagnostic favors stronger torque. Initial low work authority does not imply low work over the full interval: the stronger action adds 2.679 mJ total energy.
- **E — counteract natural swing-energy overshoot.** At 15 s both energies are initially below target, but near-coasting raises relative swing energy by 11.006 mJ during the next interval. The diagnostic favors negative torque, removes 4.005 mJ total energy, and limits the swing-energy rise to 8.023 mJ. Its initial sensitivity-only advice has the opposite torque sign. This illustrates why the map needs natural evolution or a short prediction.
- **F — coasting is already nearly adequate.** The recorded controller applies zero torque. A small negative torque reduces the diagnostic loss further, but the absolute improvement is only 0.0000183. Without an effort penalty the diagnostic still prefers actuation; this does not establish a practical advantage over coast.

## Why the original heuristic differs

The archived controller predicts elevation after 100 ms and penalizes deviation from a small requested mechanical work. It does not directly minimize the two energy errors above. Its predictions use 20 ms RK4 steps, while the archived plant uses 2 ms substeps. We reproduced its saved work and elevation cost terms to within 10⁻¹⁰; the recorded plant endpoints reproduce within 4.98 × 10⁻¹⁴. The archive's arm penalty is zero at every selected decision, and all tested historical predictions remain inside its ±180° range. This is not an arm-boundary effect.

For **all six samples**, the original objective prefers the recorded action to the two-energy recommendation. For example, at A its original cost is 0.993 for the recorded torque versus 1.460 for the alternative. At C the corresponding costs are 0.607 versus 0.784. The work penalty explains why aggressive pumping can improve our new energy loss while worsening the heuristic's objective.

Across all 160 decisions, the actual and equal-weight recommended torques have opposite signs at **50 decisions (31.25%)**. The instantaneous equal-weight sensitivity rule and the 100 ms recommendation have opposite signs at **20 decisions (12.5%)**. Changing the swing weight from 0.25 to 4 reverses the recommended sign at **23 decisions (14.4%)**. These are trajectory-specific diagnostics, not performance rates across starts.

## Assessment

The opportunity map is useful for identifying shared torque preferences, competition, and differing task authority. It needs energy errors to say what is desirable, and natural dynamics or short prediction to judge a finite action. These plots directly show where the existing heuristic differs from that interpretation.

A reasonable next experiment is a matched closed-loop comparison of the archived objective and this explicit two-energy objective, with the same torque limit and decision interval. It must measure capture time, failures, energy/effort and arm behavior before we can claim the alternative is better. No controller modification or such performance claim is made here.

The 1 ms versus 0.5 ms prediction difference is below 3.37 × 10⁻⁸ in the state coordinates. Refining the torque grid from 129 to 257 points changes the best equal-weight loss by at most 1.11 × 10⁻⁵. Exact arrays, sample selection rules and provenance are in the [methods and machine records](../../../../../../artifacts/rotary_pendulum/actuation-opportunities/decision-comparison/machine-scannables/methods.md).
