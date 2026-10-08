# Portable animation records: single-action controller

These eight complete episodes illustrate behavior and failure mechanisms; they are not a statistical sample or a training dataset. All eventually captured. The final two also crossed the arm soft limit, which must remain visible.

| Case | Seed / original lane | Capture time (seconds) | Peak absolute arm angle (degrees) | CSV |
| --- | --- | ---: | ---: | --- |
| Typical downward swing-up | 20261022 / 118 | 11.12 | 78.34 | [Log](records/downward_median_capture_time.csv) |
| Slowest downward capture in seed 20261022 | 20261022 / 177 | 16.92 | 59.33 | [Log](records/downward_slowest_capture.csv) |
| Typical moving start without an arm crossing | 20261022 / 635 | 9.12 | 97.46 | [Log](records/moving_median_capture_time_without_arm_crossing.csv) |
| Typical near-upright capture | 20261022 / 1403 | 0.10 | 7.85 | [Log](records/near_upright_median_capture_time.csv) |
| Typical tight-upright capture | 20261022 / 1792 | 0.10 | 2.77 | [Log](records/tight_upright_median_capture_time.csv) |
| Exact downward rest | 20261022 / 2048 | 10.06 | 50.21 | [Log](records/exact_downward_rest.csv) |
| Crossing despite bounded tested torques | 20261022 / 959 | 11.24 | 184.58 | [Log](records/arm_crossing_despite_bounded_tested_torques.csv) |
| Crossing without a bounded tested torque | 20261023 / 676 | 13.02 | 189.03 | [Log](records/arm_crossing_without_bounded_tested_torque.csv) |

“Typical” means the upper median capture time among successful, noncrossing episodes in that group, sorted by duration then lane. The slow downward case is the last in that ordering. Exact downward rest is selected by its all-zero initial state. Each boundary example has the largest arm peak among audited events of its specified type and seed. The [manifest](records/manifest.json) and records/provenance/ retain the precise rule, episode metrics, settings, numerical audits and checksums.

The arm peaks shown here include the initial pose; the original episodes.csv peak_arm_rad includes only active post-integration samples. Both definitions are retained explicitly. Near-upright and tight-upright examples last only 0.1 seconds: they illustrate immediate capture, not sustained balancing.

## Make animations on another machine

CSV playback needs no simulator rerun, GPU or original full archive. Every row includes time_s (seconds), four state fields (radians and radians per second), interval torque (newton metres), decision index, four physical energy columns (joules), and origin/pivot/center/tip x/y/z coordinates (metres). CSV files contain no terminal padding.

1. Play by time_s, one original sample every 0.02 seconds (50 frames per second). Draw the arm from origin to pivot and the pendulum from pivot to tip, with equal 3D axis scales.
2. The first row is the initial state; torque 0 and decision index −1 mean no preceding interval. On later rows, preceding_interval_torque_nm is the torque applied over the integration interval ending at that row, not a newly selected next action. One decision normally spans five rows; the final action can end early.
3. Display pendulum position in 0–360 degrees, 0 downward and 180 upright. Keep arm angle and velocities signed. Do not linearly interpolate across the pendulum-angle wrap; use the saved coordinates and native frames.
4. Display time, torque, pendulum position, arm angle and whether the arm crosses ±180 degrees. First-crossing records and bounded-tested-torque counts are in the manifest. A count of zero refers to the finite tested action set, not a proof about every continuous torque.
5. End each clip at capture. If a multi-panel animation holds an ended clip on its final frame, label it “recording ended”; it is not continuing simulated balance. Convert all energy components from joules to millijoules consistently if desired, without separate display weights.

NPZ files retain 200 original decision slots plus initial_x. physics_x has shape (200, 5, 4); physics_active has shape (200, 5). Only active samples represent execution; later entries are terminal padding. State order is arm angle, pendulum position, arm speed, pendulum speed. energy_j and predicted_energy_j are ordered arm kinetic, full pendulum kinetic, potential. Use NPZ for detailed diagnostics and the already-filtered CSV for animation.

## Interpretation and next step

The median downward case demonstrates a useful starting controller; the slow case shows room to reduce capture time; the two crossings preserve its weaknesses. Those final cases are soft-limit violations, not capture failures. See the [RL assessment and plan](../../../../../docs/14-reinforcement-learning-prior/assessment_and_next_step_plan.md).

This directory is a user-requested exception to artifact exclusion: only portable data, provenance and this English interpretation are force-added to the index. The Chinese companion still follows the -CN ignore rule. Full original runs remain ignored. Transferring the bundle through Git requires committing and pushing these staged files later.
