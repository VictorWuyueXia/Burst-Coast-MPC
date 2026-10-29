"""Regenerate official MPC+RL figures and metrics from saved confirmation records."""
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

output = Path(__file__).resolve().parents[1]
source = output / "machine-scannables/session-records"
archives = [np.load(p, allow_pickle=False) for p in sorted(source.glob("chunk-*.npz"))]
executed = np.concatenate([z["executed"] for z in archives])
assert executed.shape == (400, 32, 100, 4)
assert all(z["accepted"].all() for z in archives)
for archive in archives:
    archive.close()
states = executed.transpose(1, 0, 2, 3).reshape(32, 40000, 4)
error = np.rad2deg(np.arctan2(-np.sin(states[..., 1]), -np.cos(states[..., 1])))
time = np.arange(1, 40001) * 0.001
rows = []
for episode in range(32):
    held = np.abs(error[episode]) <= 5
    gaps = np.r_[-1, np.flatnonzero(~held), 40000]
    rows.append(dict(episode=episode, total_hold_s=float(held.sum() * 0.001),
                     max_hold_s=float(np.diff(gaps).max() * 0.001 - 0.001),
                     final_10s_hold=bool(held[-10000:].all())))
f = (output / "machine-scannables/episodes.csv").open("w", newline="")
writer = csv.DictWriter(f, fieldnames=list(rows[0]))
writer.writeheader()
writer.writerows(rows)
f.close()
summary = dict(controller="MPC+RL", count=32, duration_s=40,
               mean_total_hold_s=float(np.mean([r["total_hold_s"] for r in rows])),
               mean_max_hold_s=float(np.mean([r["max_hold_s"] for r in rows])),
               final_10s_hold=sum(r["final_10s_hold"] for r in rows),
               final_10s_arm_rms_deg=float(np.sqrt(np.mean(np.rad2deg(states[:, -10000:, 0])**2))),
               final_10s_upright_rms_deg=float(np.sqrt(np.mean(error[:, -10000:]**2))))
(output / "machine-scannables/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
ax.bar(np.arange(32), [r["total_hold_s"] for r in rows], label="Total upright time")
ax.plot(np.arange(32), [r["max_hold_s"] for r in rows], "k.", label="Longest continuous upright time")
ax.set(xlabel="Session index (0–31)", ylabel="Time within ±5° of upright (s)",
       ylim=(0, 42), title="MPC+RL · official 16-generation CEM · 40 s sessions")
ax.legend(loc="lower right")
for extension in ("png", "pdf", "svg"):
    fig.savefig(output / f"human-readables/hold_metrics.{extension}", dpi=180)
plt.close(fig)
fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, layout="constrained")
for column, episode in enumerate((0, 2)):
    axes[0, column].plot(time, error[episode], linewidth=0.8)
    axes[0, column].axhspan(-5, 5, alpha=0.2, color="green", label="Upright tolerance ±5°")
    axes[0, column].set(title=f"MPC+RL · session {episode}", ylabel="Wrapped upright error (deg)")
    axes[0, column].legend(fontsize=8)
    axes[1, column].plot(time, np.rad2deg(states[episode, :, 0]), linewidth=0.8)
    axes[1, column].axhline(135, color="red", linestyle="--", label="Arm boundary ±135°")
    axes[1, column].axhline(-135, color="red", linestyle="--")
    axes[1, column].set(xlabel="Elapsed time (s)", ylabel="Arm angle (deg)", ylim=(-140, 140))
    axes[1, column].legend(fontsize=8)
for extension in ("png", "pdf", "svg"):
    fig.savefig(output / f"human-readables/representative_sessions.{extension}", dpi=180)
plt.close(fig)
print(json.dumps(summary), flush=True)
