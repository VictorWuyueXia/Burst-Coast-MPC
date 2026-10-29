# Selected-result data contract

## Collection and identity

`file_manifest.json` lists paths relative to `final-results/`, byte counts, and SHA-256 hashes. It excludes itself and ignored Chinese companions. `packaging_provenance.json` records original paths, renamed assets, and representative lane order. Source snapshots retain their original bytes; they are evidence of execution, not the live entry points.

`package_results.py` collects existing source-workstation archives without running controllers. It does not regenerate the editorial reading guides or final file manifest. Final packaging also updates active repository path references and validates the extracted payload. Downstream numerical use does not require this collection script or its historical input paths.

## Common physical state

State arrays have four last-axis coordinates: arm angle, pendulum angle from downward, arm angular velocity, and pendulum angular velocity. Angles are radians; velocities are radians/second. Upright is ±pi radians. Read each bundle's convention before wrapping pendulum positions. Arm angle remains signed and must not be wrapped for arm-range metrics.

Torque fields ending in `_nm` are newton metres. Energy fields ending in `_j` are joules; `_w` denotes watts, `_s` denotes seconds, and `_rad_s` denotes radians/second. Diagnostic loss and normalized torque fields are dimensionless. Exact controller parameters are configured values; state and torque traces are recorded simulation quantities; energy and sensitivity quantities are computed from those states and the stored model.

## RL policy

`nominal/episodes.csv` and `disturbed/episodes.csv` contain all 512 starts per condition. `population_performance.csv` contains only the selected actor, with aggregate and per-stratum rows. Its original `actor_quarter_*` run identifiers remain source identifiers; they do not request another model.

Each representative NPZ has 6,000 decisions and three selected sessions. Session axis is one, except `labels`, whose only axis is the selected session. `physics_x` has shape `(6000,3,5,4)`; `start_x` is `(6000,3,4)`. `plot_selection.json` orders the original lane IDs. Five physics samples per decision have 20-millisecond spacing over a 600-second record. The underlying plant used 2-millisecond integration.

Use `inference.py` and `weights.npz` for actor and twin-critic reuse. `operating_contract.md` defines reward and value semantics. Reference vectors validate export parity, not critic calibration. The package manifest's historical checkpoint path is provenance; inference reads its local weights.

## RL+MPC

`session-records/design.json` specifies the final controller and `starts.npz` its 32 initial states and groups. Forty `chunk-*.npz` files each contain ten sequential decisions; concatenate along axis zero. Complete `executed` data have shape `(400,32,100,4)`. The record interval is 0.001 seconds and the commanded action interval is 0.1 seconds.

Plan records contain ten normalized torques followed by a terminal epigraph scalar. Scale actual normalized commands by the stored usable torque limit; never interpret the epigraph scalar as torque. Exact objective components, feasibility flags, selected seeds, numerical acceptance diagnostics, and source snapshots remain included.

Population metrics concern ±5-degree pendulum upright error. They do not imply arm centering or convergence of both velocities. The selected runner reads the final plant dictionary and initial states directly, preserving the recorded numerical design.

## Energy controller

Three directories contain the two final seeds and the fine-plant check. Each has 2,052 episodes: four groups of 512 random starts plus four deterministic probes. Summary rows and per-episode records retain that distinction.

The four selected sessions retain 200 decision slots. `physics_x` has shape `(200,4,5,4)` and `physics_active` has shape `(200,4,5)`. Ignore frozen post-capture padding. `plot_selection.json` maps group labels to original lane numbers; its insertion order is the NPZ session order. `energy_j` stores arm kinetic, full pendulum-body kinetic, and potential energy.

The animation manifest has its own checksums relative to `animation/`. Its CSV records have already removed inactive padding and contain physical geometry for playback. The animation sample selection is based on duration and boundary events, whereas the validation figures use their recorded minimum-energy-distance selection.

## Energy/torque sensitivity

The study retains three controlled sessions and three explicitly torque-free continuations. Controlled CSV and NPZ pairs have identical columns. Their descriptive filenames identify typical downward, slow downward, and typical moving-start sessions. Coast releases identify low, intermediate, and near-upright total-energy levels.

`methods.md` defines total mechanical energy, relative hinge-swing energy, their instantaneous power derivatives with respect to torque, error-weighted task preferences, and the natural swing-energy drift. The stored full-body sensitivity is a different diagnostic. The two energy definitions are not interchangeable.

`action-comparison/` contains the 160 complete decisions of the slow downward session, candidate-action arrays, all-decision tables, six illustrative samples, and objective-weight checks. These are counterfactual one-action predictions from recorded states, not a new closed-loop controller evaluation. The quadrants describe instantaneous input preferences; finite-duration decisions also depend on drift and changing state.

The map retraces under reflected pendulum positions and does not define a unique global phase. Coast can preserve reduced periodic motion while arm position drifts. The detailed study's existing figures and limits are retained without introducing a new coast policy or gain-sensitivity campaign.
