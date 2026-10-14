# Heuristic results: trajectory review and viability

## Quick reading

Review the saved sessions without changing the controller or starting training.
A heuristic prior does not need to solve the complete task. Judge its ability
to bring the pendulum into useful states, its arm behavior, and ultimately
whether residual learning improves faster than a matched direct-policy baseline.
The existing zero-capture result alone is not a sufficient reason to reject it.

[Results, plots and proposed viability screens](../../artifacts/rotary_pendulum/continuous-residual/review/human-readables/interpretation_summary.md)
are saved with the artifacts. This review supports a bounded residual/direct
learning comparison as a next experiment; it does not establish that the
heuristic accelerates learning or execute training.

## Analysis structure and validation plan

Add one reproducible artifact script, `review_sessions.py`, under
`artifacts/rotary_pendulum/continuous-residual/review/machine-scannables/`.
Budget: one `main` function, no classes, dataclasses, helpers or runtime changes;
40–300 lines, compact task-oriented logic, no try/with/except or extra CLI.
Vectorize independent states and lanes. Use CPU only for this saved-data review.

Inputs: existing NPZ trajectories, case-selection JSON and episode CSVs.
Independent analysis choices: 20 s coverage deadline, 10–20 s diagnostic
window, selected median-return downward/near cases, and the explicitly proposed
loose approach-region thresholds below. No learner parameters are changed.

Outputs: complete-session and first-ten-second PNGs, a 10–12 s energy-transfer
zoom, per-episode diagnostic CSV, source/lane selection JSON, and an interpretation
summary. Human figures/reports and machine data/scripts stay in separate folders.

Milestones: (1) verify selected lanes and deadlines match stored selections;
(2) recompute instantaneous versus interval power for all downward lanes, not
only one example; (3) plot 20 ms physical states and 100 ms torques, inspect the
rendered figures; (4) state observed facts separately from proposed viability
targets and causal hypotheses. Pass means numerical records and plots agree,
no hidden simulation/training is performed, and percentages identify denominators.

## Diagnostic definitions

E is pendulum kinetic-plus-potential energy in joules, computed from logged
pendulum angle and speed using nominal inertia and gravity. E* = 0.03037176 J
is the upright-rest target. For each held 100 ms decision, initial physical
state is x, applied motor torque is u, and end state is x′. Instantaneous power
is the exact energy derivative from the nominal ODE at (x,u). Interval power is

$$
P_{interval}=[E(x')-E(x)]/\Delta t.
$$

The measured active duration Δt is 0.1 s for nonterminal decisions. Both powers
are in watts. A sign-mismatch decision has initial power above +0.000001 W
and interval power below −0.000001 W. The threshold is a numerical tolerance;
rates are over all active decisions in the specified window, not conditional
on positive initial power. Saturation means at least 99.9% of the policy cap;
arm intervention means logged filter mode is nonzero.

For an explicitly proposed **approach-region diagnostic**, use pendulum angular
distance from upright ≤0.35 rad, pendulum speed ≤2 rad/s, arm speed ≤2 rad/s,
and arm position magnitude ≤1.2 rad, all simultaneously. These are analyst-chosen
screening limits, not a verified capture basin or new success condition. Count
visits within 20 s; report whether the initial state was already inside. The
existing strict hold condition remains unchanged. Energy access means at least
one logged physical sample lies between 90% and 110% of target within 20 s.
