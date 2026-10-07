# Historical confirmation replay schema, version 1

Source: `confirm_20261008/energy_soft`, selected lanes in `plot_selection.json`. `manifest.json` identifies the original figure and trace archive by SHA-256, defines the frame and control intervals, records episode outcomes and hashes the exported data/config files. Paths in archived provenance identify their original context; playback has no dependency on those paths. Historical decoder mode meanings are in the [original contract](../implementation_contract.md), not the recovery decoder contract.

## CSV: one file per case

Each CSV is UTF-8, comma separated, with one header row and float values serialized to 17 significant digits. There are 1,001 rows per case, starting at time zero. Each later row is an active 20 ms physics endpoint. No inactive padding is exported.

| Columns | Meaning and units |
| --- | --- |
| `time_s` | Elapsed simulation time, seconds, from the reset |
| `theta_rad` | Unwrapped arm angle about positive vertical, radians; zero points along positive x |
| `alpha_rad` | Pendulum angle from downward, radians; upright at π modulo 2π |
| `omega_rad_s`, `nu_rad_s` | Signed arm and pendulum angular velocities, radians/second |
| `interval_torque_nm` | Applied arm torque, N m, over the interval ending at this row; zero at initial row is a sentinel |
| `decision_index` | Zero-based original 100 ms decision index for that interval; −1 at initial row |
| `origin_x_m`, `origin_y_m`, `origin_z_m` | Stationary motor-axis origin, metres |
| `pivot_x_m`, `pivot_y_m`, `pivot_z_m` | Pendulum hinge at arm endpoint, metres |
| `center_x_m`, `center_y_m`, `center_z_m` | Pendulum center of mass, metres |
| `tip_x_m`, `tip_y_m`, `tip_z_m` | Pendulum free endpoint, metres |

Geometry uses a right-handed frame with positive z upward, arm length r = 0.085 m and pendulum length l = 0.129 m from archived physics. θ and α are the recorded angles above. Define radial vector R, positive tangential vector T and vertical vector Z; O, P, C and Q denote origin, hinge, center and tip, in metres:

$$
R=(\cos\theta,\sin\theta,0),\quad
T=(-\sin\theta,\cos\theta,0),\quad Z=(0,0,1),
$$
$$
O=(0,0,0),\quad P=rR,\quad
C=P+\tfrac{l}{2}(\sin\alpha\,T-\cos\alpha\,Z),\quad
Q=P+l(\sin\alpha\,T-\cos\alpha\,Z).
$$

These algebraic coordinates are computed with the repository's `mechanism_points`; they add no simulated dynamics. They also remove any ambiguity in the sign of pendulum motion for another renderer. Connect O to P for the arm and P to Q for the pendulum. Do not wrap angle increments before interpolation; rapid arm rotation is real in these logs.

## NPZ: original selected-lane trace

Load with `numpy.load(path, allow_pickle=False)`. `initial_x` has shape (4,) and state order `[theta, alpha, omega, nu]`. Every other key is copied exactly from the archived trace after selecting its lane; time has not been resampled. There are 200 decision slots.

- `physics_x`: (200,5,4), states after each 20 ms substep; `physics_active`: (200,5), whether that substep actually executed.
- `x`: (200,4), state after each decision; `time_s`, `elapsed_s`: (200,), absolute endpoint time and actually executed duration in seconds.
- `energy_j`: (200,3), arm kinetic, full pendulum-body kinetic and pendulum potential energy in joules.
- `torque_nm`: (200,), actual held command; `requested_work_j`, `predicted_work_j`, `work_j`, `energy_balance_j`: (200,), upper request, full-action predicted work, executed work and numerical balance residual in joules.
- `mode`, `root_count`, `predicted_peak_arm_rad`: (200,), historical decoder mode, number of resolved candidates and predicted peak absolute arm angle in radians.
- `goal_count`, `success`, `timeout`, `arm_violation`: (200,), capture dwell counter and cumulative episode flags.

For any future use of the raw arrays, respect `physics_active`; do not animate repeated frozen terminal entries. The current four cases run the full 20 seconds. The CSV alone suffices for 3D playback; NPZ retains original numerical diagnostics for auditing or rebuilding figure panels.

## Physics and dependencies

Both physical damping coefficients are zero. The archived YAML contains `simulation.timestep-s: 0.002` for the NumPy configuration; **this experiment's actual JAX sampling is 0.02 s**, recorded in the manifest and trace timestamps. Control holds are 0.1 s. Do not infer replay timing from that YAML field or reintegrate using it.

`campaign.json` preserves both archived trials; this bundle extracts only `energy_soft` (work gain 0.04, kinetic weight 1, arm penalty 0.1, no coast suffix). Provenance lists the historical software/source hashes. Replay needs only the versions in `requirements.txt`, not the original CUDA environment. Manifest hashes cover data and archived metadata; schema and requirements are versioned by Git.
