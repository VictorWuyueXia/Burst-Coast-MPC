# Official MPC+RL baseline transfer

This package contains only the latest successful official parameter setting: 16-generation CEM MPC with LQR stage weights and a frozen RL terminal value. It includes 32 complete 40-second sessions. RL remains active throughout; no local LQR handoff occurs.

## Contents

- `../../1-TD3_20261016/`: portable actor/twin-critic package, weights, inference code, normalization, reference vectors and dependency pins.
- `../20261022-lqr-cost-fine-confirm-16/`: official confirmation records, initial states, selected plans, decomposed costs, critic values, executed trajectories, source snapshots and completion marker.
- `human-readables/`: baseline-only hold-time and representative-session figures in PNG/PDF/SVG formats.
- `machine-scannables/`: baseline-only session metrics, aggregate results, report regeneration script and checksum manifest.
- `../20261019-gpu-closed-cem/machine-scannables/{design.json,starts.npz}`: configuration and initial-state inputs required by the unchanged runner. These are input dependencies, not comparison experiment records.

Tuning pilots, other controller runs, mixed comparison summaries and comparison figures are excluded from this transfer. They remain in the original machine's local workspace. Existing wider historical links in repository documentation may refer to those unbundled records.

## Interpretation

All 32 sessions hold within ±5° of upright throughout their final ten seconds. Mean total upright time and maximum continuous upright time are both 37.7644375 seconds. The hold-time plot shows all session indices; bars and markers overlap because each session remains upright after capture. Representative plots use fixed session indices 0 and 2, matching the prior examples rather than selecting the best outcomes.

The final-ten-second arm RMS is approximately 12.75°, while upright-error RMS is approximately 0.485°. Upright holding is not full-state convergence. Some trajectories approach the arm limit very closely; these are nominal simulation results, with no real-time or robustness guarantee. See `docs/11-inner-MPC/human-readables/mpc_rl_baseline.md` for the cost and fixed parameters.

## Numerical analysis and integrity

Use NumPy with `allow_pickle=False`. Concatenate `chunk-*.npz` in filename order along axis zero. The 40 chunks cover 400 decisions for 32 sessions. State order is arm angle, pendulum angle measured from downward, arm speed and pendulum speed, in radians and radians/s. A selected plan contains ten normalized commands followed by a terminal epigraph scalar; the last scalar is not an action.

The saved executed states reproduce the recorded sessions directly. Closed-loop recomputation depends on device, precision, JAX version and random sampling; exact cross-device agreement is not guaranteed. The source snapshots preserve the original algorithm. Read the repository's `lqr_stage_cost_contract.md` for record semantics.

`machine-scannables/manifest.json` lists all included payload paths, byte counts and SHA-256 values except itself and this guide. The official run also retains its original hashes and source snapshots. The manifest identifies the source-code revision before artifact packaging.

## Reproduction on another machine

Use Python 3.12.13 and the repository's pinned dependencies. The standalone RL package has a smaller `requirements.txt`; the simulation uses the repository's `jax-cuda13` extra on a compatible GPU host. Follow existing GPU occupancy rules before launching simulations.

To regenerate these baseline-only metrics and figures from saved records, run from the repository root:

```bash
MPLCONFIGDIR=/tmp/mpc-rl-matplotlib python artifacts/rotary_pendulum/inner-mpc/mpc-rl-transfer/machine-scannables/baseline_report.py
```

This performs data analysis only and needs NumPy and Matplotlib, not GPU simulation or an external plotting skill. It rewrites the baseline report outputs.

The original runner is `scripts/experiments/inner-mpc/lqr_cost_run.py confirm 16`. It refuses to overwrite its fixed artifact directory. For a new experiment, use a separate working copy and change only the output destination to a fresh directory, preserving the archived inputs and parameters. See the GPU experiment guide for its launch environment. No new simulations were run while preparing this package. Chinese companions and caches remain locally ignored under project policy.
