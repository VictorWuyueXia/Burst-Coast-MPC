# Recovery replay schema, version 1

The CSV coordinate/torque conventions and NPZ array axes match the [historical replay schema](../original-analytical-controller-replay/experiment_formulation_and_analysis.md). Angles remain unwrapped, velocities are signed, positions are metres, and torque applies over the interval ending at each CSV row. Initial torque zero and decision index −1 are sentinels. CSV includes only the initial pose and active physics samples; NPZ preserves all 200 original decision slots including inactive terminal padding.

The source is `recovery_confirm_20261008/recovery`. `manifest.json` records exact selected lanes, episode metrics, durations, source-trace SHA-256, geometry and data hashes. `selection_rule` specifies outcome-stratified upper medians. `source_figure` and `figure_sha256` are null: these cases were selected from logs and are not claimed to be the cases in an existing figure. `plot_selection.json` names the new selection, not the original campaign's representative plot selection.

| Case | Lane | CSV poses including initial | Duration (s) |
| --- | ---: | ---: | ---: |
| downward | 62 | 1001 | 20.00 |
| moving | 77 | 352 | 7.02 |
| near | 148 | 99 | 1.96 |
| tight | 254 | 56 | 1.10 |

The actual JAX physics interval is 0.02 s, control interval 0.1 s. Successful episodes may stop partway through the final control interval; only the applied substeps are exported. The NumPy YAML's 0.002 s setting is not the trace sample interval. Geometry and both zero damping coefficients are preserved from the archived configuration.

Unlike historical traces, these NPZ files additionally contain `start_x` (200,4), `work_mismatch_j` (200,), `recoverable_count` (200,), and `predicted_terminal_arm_speed` (200,). They represent decision-start state, actual minus requested work in joules (zero for inactive decisions), count of passing recovery candidates, and predicted signed terminal arm speed in rad/s.

Current `mode` meanings are 0: matched work plus recovery; 1: work override plus recovery; 2: recovery unresolved with explicit best-recovery action; 3: numerical failure. These replace the older decoder's mode meanings. See the [recovery contract](../formulation/decoder_recovery_contract.md) for details. The upper policy receives exactly three energies and uses work gain 0.04 and kinetic weight 1; decoder work weight is 1, recovery horizon 20 physics samples including the first held pulse.

The replay's final frozen pose is not a new observation or a controller hold experiment. Use `physics_active` when independently rendering the NPZ; otherwise terminal padding would falsely appear to demonstrate sustained balance. The CSV already performs that filtering.
