"""Plot complete archived failure trajectories and their conditional MPC plans."""

import json
from pathlib import Path

import matplotlib
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("artifacts/rotary_pendulum/inner-mpc/20261020-failure-analysis")
machine, human = root / "machine-scannables", root / "human-readables"
data = np.load(machine / "diagnostics.npz", allow_pickle=False)
summary = json.loads((machine / "summary.json").read_text())
time, states = data["time_s"], data["states"]
colors = ["#228833", "#4477AA", "#EE7733", "#AA3377", "#009988"]
titles = [
    "A  Upright not reached",
    "B  Capture not sustained",
    "C  Initial balance lost",
    "D  Late arrival",
]
plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "legend.frameon": False,
        "axes.titleweight": "bold",
    }
)
# All five rows establish a different part of the explanation: outcome, momentum,
# energy transfer, available arm room, and the actual applied control.
fig, axes = plt.subplots(5, 4, figsize=(16, 12), sharex=True, sharey="row")
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.89, hspace=0.3, wspace=0.19)
fig.suptitle("Four different reasons for missing the five-second score", fontsize=19, y=0.98)
fig.text(
    0.5,
    0.94,
    "All examples: best aggregate CEM, centering rate 10. Full recorded sessions; no early stops.",
    ha="center",
)
for col, case in enumerate(summary["representatives"]):
    k, episode = case["controller"], case["episode"]
    color = colors[col + 1]
    axes[0, col].set_title(f"{titles[col]}\nEpisode {episode}", fontsize=12, pad=12)
    values = [
        data["error_deg"][k, episode],
        states[k, episode, :, 3],
        data["energy_ratio"][k, episode],
        np.rad2deg(states[k, episode, :, 0]),
    ]
    for row, value in enumerate(values):
        axes[row, col].plot(time, value, color=color, lw=1.4)
    axes[4, col].stairs(
        data["torque_nm"][k, episode] * 1000, np.arange(51) / 10, color=color, lw=1.3
    )
    axes[0, col].axhspan(0, 5, color="#BBBBBB", alpha=0.5)
    axes[0, col].set_ylim(-5, 185)
    axes[0, col].set_yticks([0, 90, 180])
    axes[1, col].axhline(0, color="0.6", lw=0.7)
    axes[2, col].axhline(1, color="0.4", ls="--", lw=1)
    axes[3, col].set_ylim(-140, 140)
    axes[3, col].set_yticks([-135, 0, 135])
    for bound in [-135, 135]:
        axes[3, col].axhline(bound, color="0.4", ls="--", lw=0.8)
    for bound in [-4.59, 4.59]:
        axes[4, col].axhline(bound, color="0.4", ls="--", lw=0.8)
    for ax in axes[:, col]:
        ax.axvspan(4, 5, color="#CCBB44", alpha=0.12, zorder=0)
        ax.set_xlim(0, 5)
        ax.grid(alpha=0.16)
    axes[-1, col].set_xlabel("Time (s)")
for ax, label in zip(
    axes[:, 0],
    [
        "Distance to upright (°)",
        "Pendulum speed (rad/s)",
        "Relative pendulum energy",
        "Arm angle (°)",
        "Torque (mN m)",
    ],
    strict=True,
):
    ax.set_ylabel(label)
fig.text(
    0.5,
    0.015,
    "Gray band: within 5° of upright. Yellow interval: scored final second. "
    "Dashed lines: energy reference or actuator/arm bounds.",
    ha="center",
    fontsize=10,
)
require_matplotlib_panel_alignment(fig, json_out=machine / "representatives.alignment.json")
fig.savefig(human / "representative_failure_modes.png", dpi=300)
fig.savefig(human / "representative_failure_modes.pdf")
fig.savefig(human / "representative_failure_modes.svg")
plt.close(fig)
print("Rendered all four complete failure examples", flush=True)

# A conditional plan may end near upright while subsequent replanning changes its future.
fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex="row", sharey="row")
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.81, hspace=0.5, wspace=0.19)
fig.suptitle("Planned arrival is not a guarantee of executed capture", fontsize=18, y=0.98)
for col, case in enumerate(summary["representatives"]):
    k, episode = case["controller"], case["episode"]
    terminal = data["plan_end"][k, episode, :41]
    terminal_error = np.rad2deg(
        np.abs(np.arctan2(-np.sin(terminal[:, 1]), -np.cos(terminal[:, 1])))
    )
    axes[0, col].plot(
        np.arange(41) / 10 + 1,
        terminal_error,
        "o--",
        ms=3,
        lw=1,
        color="#4477AA",
        label="Planned 1 s endpoint",
    )
    axes[0, col].plot(
        time[1000:],
        data["error_deg"][k, episode, 1000:],
        color="#EE7733",
        lw=1.2,
        label="Executed after replanning",
    )
    axes[0, col].set_title(f"{titles[col]}\nEpisode {episode}", fontsize=12)
    axes[0, col].set_xlabel("Endpoint time (s)")
    axes[0, col].set_ylim(-5, 185)
    axes[0, col].axhspan(0, 5, color="0.8", alpha=0.4)
    axes[1, col].plot(time, data["unwrapped_turns"][k, episode], color=colors[col + 1], lw=1.4)
    axes[1, col].set_xlabel("Time (s)")
    axes[1, col].set_xlim(0, 5)
    for upright in np.arange(-3.5, 4, 1):
        axes[1, col].axhline(upright, color="0.7", ls="--", lw=0.6)
    axes[1, col].set_ylim(-0.6, 3.2)
    for ax in axes[:, col]:
        ax.grid(alpha=0.15)
axes[0, 0].set_ylabel("Distance to upright (°)")
axes[1, 0].set_ylabel("Unwrapped pendulum turns")
fig.legend(
    *axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2
)
fig.text(
    0.5,
    0.025,
    "Top: each planned endpoint assumes its full plan is followed. "
    "Bottom: dashed half-integer turns are upright; integer turns are downward.",
    ha="center",
    fontsize=10,
)
require_matplotlib_panel_alignment(fig, json_out=machine / "plans.alignment.json")
fig.savefig(human / "planned_and_executed_progress.png", dpi=300)
fig.savefig(human / "planned_and_executed_progress.pdf")
fig.savefig(human / "planned_and_executed_progress.svg")
plt.close(fig)
print("Rendered conditional plans and continuous swing angle", flush=True)

fig, ax = plt.subplots(figsize=(11, 6))
fig.subplots_adjust(left=0.25, right=0.97, top=0.78, bottom=0.12)
counts = np.array(summary["counts_by_controller"])
labels = [
    "Final-second hold",
    "Upright not reached",
    "Capture not sustained",
    "Initial balance lost",
    "Late arrival",
]
left = np.zeros(9)
for category, label in enumerate(labels):
    ax.barh(np.arange(9), counts[:, category], left=left, label=label, color=colors[category])
    for row in range(9):
        if counts[row, category]:
            ax.text(
                left[row] + counts[row, category] / 2,
                row,
                str(counts[row, category]),
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )
    left += counts[:, category]
ax.set_yticks(np.arange(9), [c["label"] for c in summary["controllers"]])
ax.invert_yaxis()
ax.set_xlim(0, 32)
ax.set_xlabel("Sessions (same 32 initial states per controller)")
fig.suptitle("Failure behavior across all nine controller settings", fontsize=17, y=0.97)
fig.legend(
    *ax.get_legend_handles_labels(),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.91),
    ncol=3,
    fontsize=10,
)
fig.text(
    0.5,
    0.02,
    "288 complete five-second sessions; categories are retrospective descriptions, "
    "never stopping rules. w = arm-centering rate.",
    ha="center",
    fontsize=9,
)
fig.savefig(human / "failure_counts_by_controller.png", dpi=300)
fig.savefig(human / "failure_counts_by_controller.pdf")
fig.savefig(human / "failure_counts_by_controller.svg")
plt.close(fig)
data.close()
(machine / "plot_failure_modes.py").write_bytes(Path(__file__).read_bytes())
print(f"All figures saved to {human}", flush=True)
