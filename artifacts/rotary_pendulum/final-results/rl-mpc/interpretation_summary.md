# Nonlinear MPC with a learned terminal value

This is the selected MPC+RL controller: 16-generation cross-entropy optimization, the discrete LQR state/input weights as running cost, and a frozen twin-critic terminal value. The actor and critics remain active throughout control. No local LQR handoff or scheduled coast is used.

Start with [population holding performance](human-readables/hold_metrics.png) and [representative sessions](human-readables/representative_sessions.png). PDF and SVG versions accompany both figures. Representative session indices are fixed at 0 and 2, rather than selected for the best outcome.

All 32 saved sessions last 40 seconds and remain within ±5 degrees of upright for their final ten seconds. Mean total upright time and mean longest continuous upright time are both 37.7644375 seconds. Final-ten-second upright-error RMS is approximately 0.485 degrees; arm RMS is approximately 12.75 degrees. Upright holding is not complete-state convergence.

The usable torque is ±0.00459 newton metres, the arm envelope is ±135 degrees, the command interval is 100 milliseconds, and the prediction horizon is one second. Four initial-state groups each contribute eight sessions. Some trajectories approach the arm envelope closely. These are nominal simulation results, without a real-time or robustness claim.

[Episode metrics](machine-scannables/episodes.csv) and [aggregate metrics](machine-scannables/summary.json) provide collective performance. [Session records](machine-scannables/session-records/) contain the exact design, initial states, 40 trajectory chunks, completion marker, and original source snapshots. Their historical strings and source hashes are provenance, not instructions to locate missing development archives.

Load NPZ files with NumPy and `allow_pickle=False`. Concatenate `chunk-*.npz` in filename order along axis zero. The chunks cover 400 decisions across 32 sessions. State order is arm angle, downward-referenced pendulum angle, arm speed, pendulum speed; angles are radians and speeds radians/second. `executed` retains 100 one-millisecond samples per decision. The plan's last component is a terminal epigraph variable, not an applied command.

Regenerate baseline-only metrics and figures from the repository root:

```bash
MPLCONFIGDIR=/tmp/mpc-rl-matplotlib .venv/bin/python artifacts/rotary_pendulum/final-results/rl-mpc/machine-scannables/baseline_report.py
```

This is CPU data analysis, not a new simulation. The selected model is in [rl-policy](../rl-policy/README.md). The live experiment runner now reads this final design and these initial states directly; it no longer depends on an earlier multi-configuration screening file. New experiment outputs go to the ignored `artifacts/rotary_pendulum/runs/rl-mpc/` tree.
