"""Plot one archived trajectory, sampled opportunities, and explicit action comparisons."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/rotary_pendulum/actuation-opportunities"
MACHINE = BASE / "decision-comparison/machine-scannables"
HUMAN = BASE / "decision-comparison/human-readables"
HUMAN.mkdir(parents=True, exist_ok=True)
(MACHINE / "qa").mkdir(exist_ok=True)
d = dict(np.load(MACHINE / "comparison.npz"))
trace = dict(np.load(BASE / "machine-scannables/controlled_1.npz"))
summary = json.loads((MACHINE / "summary.json").read_text())
samples = summary["selection"]
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
colors = ["#176B98", "#C06A27", "#368456"]
figures = []

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.subplots_adjust(left=0.08, right=0.86, bottom=0.16, top=0.78, wspace=0.3)
fig.suptitle(
    "One recorded swing-up: what the heuristic did and what the energy objective prefers",
    y=0.98,
    fontsize=13,
)
fig.text(
    0.46,
    0.92,
    "Gray curve: complete physical record. Colored dots: 100 ms decision starts.",
    ha="center",
)
fig.text(
    0.46, 0.87, "A–F identify the same sampled states in the action-comparison plots.", ha="center"
)
cmap = LinearSegmentedColormap.from_list("torque", ["#2166ac", "#d8d8d8", "#b35806"])
norm = Normalize(-18.36, 18.36)
offsets = [(-85, 80), (28, -28), (-48, 12), (-32, -32), (28, 12), (-48, 22)]
for column, key in enumerate(("actual", "recommended")):
    ax = axes[column]
    ax.plot(trace["s_total_rad_s"], trace["s_swing_rad_s"], color="#bbbbbb", lw=0.7, alpha=0.6)
    points = ax.scatter(d["s_E"], d["s_H"], c=1000 * d[key], cmap=cmap, norm=norm, s=22, zorder=3)
    for item, offset in zip(samples, offsets, strict=True):
        i = item["index"]
        ax.scatter(d["s_E"][i], d["s_H"][i], facecolors="none", edgecolors="black", s=75, zorder=4)
        ax.annotate(
            item["label"],
            (d["s_E"][i], d["s_H"][i]),
            xytext=offset,
            textcoords="offset points",
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.6},
        )
    ax.set(
        xlabel="Total-energy sensitivity [rad/s]",
        ylabel="Swing-energy sensitivity [rad/s]",
        xlim=(-13, 13),
        ylim=(-14, 14),
    )
    ax.axhline(0, color="#aaaaaa", lw=0.5)
    ax.axvline(0, color="#aaaaaa", lw=0.5)
    ax.set_title(
        "Recorded heuristic torque" if column == 0 else "Best tested torque: two-energy loss",
        pad=15,
    )
    ax.grid(False)
color_axis = fig.add_axes([0.90, 0.22, 0.015, 0.5])
fig.colorbar(points, cax=color_axis, label="Torque [mN m]")
figures.append(("decision_map", fig, [color_axis]))

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.subplots_adjust(left=0.075, right=0.98, bottom=0.09, top=0.77, hspace=0.65, wspace=0.3)
fig.suptitle(
    "Sampled decisions: actual torque versus the best tested 100 ms alternative",
    y=0.98,
    fontsize=14,
)
fig.text(
    0.5,
    0.93,
    "Blue curve: equal-weight terminal energy loss. Lower is better.",
    ha="center",
    color=colors[0],
)
fig.text(
    0.5,
    0.885,
    "Orange: recorded torque. Green: best tested torque. Gray dashed: coast.",
    ha="center",
)
for ax, item in zip(axes.flat, samples, strict=True):
    i = item["index"]
    ax.plot(1000 * d["torques"][i, :257], d["scores"][i, :257, 1], color=colors[0], lw=1.8)
    ax.axvline(0, color="#888888", ls="--", lw=0.8)
    ax.axvline(item["actual_mnm"], color=colors[1], lw=1.4)
    ax.axvline(item["best_mnm"], color=colors[2], lw=1.4)
    ax.scatter(item["actual_mnm"], item["score_actual"], color=colors[1], s=35, zorder=5)
    ax.scatter(item["best_mnm"], item["score_best"], color=colors[2], marker="D", s=25, zorder=5)
    ax.set_title(
        f"{item['label']} | {item['time_s']:.2f} s\n"
        f"Actual {item['actual_mnm']:+.2f}; best {item['best_mnm']:+.2f} mN m",
        pad=12,
    )
    ax.set(xlabel="Constant torque [mN m]", ylabel="Terminal energy loss", xlim=(-20, 20))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(alpha=0.15)
figures.append(("sampled_decisions", fig, []))

fig, axes = plt.subplots(3, 1, figsize=(13, 10))
fig.subplots_adjust(left=0.09, right=0.97, bottom=0.07, top=0.86, hspace=0.55)
fig.suptitle(
    "Physical context and torque disagreement along the original record", y=0.98, fontsize=14
)
fig.text(
    0.5,
    0.935,
    "Counterfactual actions are evaluated from recorded states; no new controller is run.",
    ha="center",
)
t = trace["time_s"]
alpha = np.rad2deg(trace["alpha_rad"]).copy()
alpha[np.r_[False, abs(np.diff(alpha)) > 180]] = np.nan
axes[0].plot(t, alpha, color=colors[0], label="Pendulum")
axes[0].plot(t, np.rad2deg(trace["theta_rad"]), color=colors[1], label="Arm")
axes[0].set_ylabel("Angle [deg; 0 down]")
axes[1].plot(t, trace["total_j"] / d["target_j"], color=colors[0], label="Total energy")
axes[1].plot(t, trace["swing_j"] / d["target_j"], color=colors[1], label="Swing energy")
axes[1].axhline(1, color="#888888", ls="--", lw=0.8)
axes[1].set_ylabel("Energy / upright target")
axes[2].step(d["time_s"], 1000 * d["actual"], where="post", color=colors[1], label="Recorded")
axes[2].scatter(d["time_s"], 1000 * d["recommended"], color=colors[2], s=12, label="Best tested")
axes[2].set_ylabel("Torque [mN m]")
for ax in axes:
    ax.set_xlim(0, 16.02)
    ax.set_xlabel("Time [s]")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2, frameon=False)
    for item in samples:
        ax.axvline(item["time_s"], color="#aaaaaa", lw=0.5, ls=":")
figures.append(("decision_context", fig, []))

for name, fig, excluded in figures:
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, exclude_axes=excluded, json_out=MACHINE / "qa" / f"{name}.alignment.json", strict=True
    )
    fig.savefig(HUMAN / f"{name}.png", dpi=300)
    fig.savefig(HUMAN / f"{name}.pdf")
    fig.savefig(HUMAN / f"{name}.svg")
    plt.close(fig)
    print(f"Exported {name}.", flush=True)
