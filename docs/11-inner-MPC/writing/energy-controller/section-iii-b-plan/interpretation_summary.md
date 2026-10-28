# Reading the Section III.B plan

Read [dynamics-outline.md](dynamics-outline.md) for the current seven-paragraph plan. It replaces [outline.md](outline.md), which records the earlier six-paragraph structure. The manuscript has not been changed during this planning step.

The revised argument begins with the governing dynamics from III.A. It then develops the requested energy plane, the physically possible energies on that plane, the predicted endpoint curve obtained by varying torque, and the intersection condition for delivering the requested work. The final score considers energy distribution and arm excursion before applying an input.

The central distinctions are explicit: requested work is not commanded power; the torque-response curve lies in three-dimensional energy space and generally does not lie in the requested plane; and the Jacobian explains the energy-rate transformation while the implemented predictor integrates the physical dynamics and evaluates the energy map. Physical state existence, finite-duration reachability, and the arm-position penalty remain separate.

The actuator note uses official Quanser documentation. The current simulation bounds torque directly and contains no electrical loss model. Device-specific voltage and current limits would require an actuator-model extension before making electrical-efficiency claims. The exact Quanser model remains unspecified, so no numerical hardware rating is adopted.

[machine-scannables/dynamics_planning_audit.json](../../../../artifacts/rotary_pendulum/reference-data/10-heuristic-controller/15-controller-paper/section-iii-b-plan/machine-scannables/dynamics_planning_audit.json) records source evidence, model distinctions, the skill passes, and file-preservation checks. No compilation or controller run is needed for this plan. The Chinese companion files are ignored by the repository's existing rule.
