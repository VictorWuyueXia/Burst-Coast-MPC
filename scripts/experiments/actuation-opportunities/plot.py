"""Render physical trajectories and opportunity maps from the saved analysis arrays."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "artifacts/rotary_pendulum/actuation-opportunities"
MACHINE, HUMAN = OUTPUT / "machine-scannables", OUTPUT / "human-readables"
provenance = json.loads((MACHINE / "provenance.json").read_text())
metrics = json.loads((MACHINE / "metrics.json").read_text())
target = 2 * provenance["model"]["gravity_torque_nm"]
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.labelsize": 10,
        "lines.linewidth": 1.1,
    }
)
colors = ["#176B98", "#C06A27", "#686868"]

for kind in ("controlled", "coast"):
    records = [dict(np.load(MACHINE / f"{kind}_{index}.npz")) for index in range(3)]
    if kind == "controlled":
        titles = [
            f"{entry['label']}\nLane {entry['lane']} | {float(entry['duration_s']):.2f} s"
            for entry in provenance["selection"]
        ]
        heading = "Recorded heuristic control: repeated swings during a changing energy envelope"
        rows = 5
    else:
        titles = [
            f"Release at {entry['source_time_s']:.2f} s\n"
            f"Total energy {entry['energy_fraction']:.3f} × upright target"
            for entry in provenance["releases"]
        ]
        heading = "Torque switched off: pendulum cycles repeat while the arm drifts"
        rows = 4
    fig, axes = plt.subplots(rows, 3, figsize=(15, 2.45 * rows), squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.055, top=0.84, hspace=0.52, wspace=0.33)
    fig.suptitle(heading, y=0.985, fontsize=15)
    fig.text(
        0.5,
        0.949,
        "Blue: pendulum angle / arm speed / total energy / total sensitivity",
        ha="center",
        color=colors[0],
    )
    fig.text(
        0.5,
        0.925,
        "Orange: arm angle / pendulum speed / relative swing energy / swing sensitivity",
        ha="center",
        color=colors[1],
    )
    for column, record in enumerate(records):
        t = record["time_s"]
        angle = np.rad2deg(record["alpha_rad"]).copy()
        angle[np.r_[False, abs(np.diff(angle)) > 180]] = np.nan
        axes[0, column].plot(t, angle, color=colors[0])
        if kind == "controlled":
            axes[0, column].plot(t, np.rad2deg(record["theta_rad"]), color=colors[1])
        axes[0, column].set_ylim(-200, 200)
        axes[0, column].set_yticks([-180, 0, 180])
        axes[1, column].plot(t, record["omega_rad_s"], color=colors[0])
        axes[1, column].plot(t, record["nu_rad_s"], color=colors[1])
        if kind == "controlled":
            axes[2, column].step(t, 1000 * record["torque_nm"], where="pre", color=colors[2])
        else:
            axes[2, column].plot(t, np.rad2deg(record["theta_rad"]), color=colors[1])
        axes[3, column].plot(t, record["total_j"] / target, color=colors[0])
        axes[3, column].plot(t, record["swing_j"] / target, color=colors[1])
        axes[3, column].axhline(1, ls="--", color=colors[2], lw=0.7)
        if kind == "controlled":
            axes[4, column].plot(t, record["s_total_rad_s"], color=colors[0])
            axes[4, column].plot(t, record["s_swing_rad_s"], color=colors[1])
        axes[0, column].set_title(titles[column], pad=12)
        for row in range(rows):
            axes[row, column].set_xlim(t[0], t[-1])
            axes[row, column].set_xlabel(
                "Time [s]" if kind == "controlled" else "Time since release [s]"
            )
    labels = [
        "Angle [deg]\n0 down, ±180 up",
        "Angular speed [rad/s]",
        "Torque [mN m]" if kind == "controlled" else "Arm angle [deg]",
        "Energy /\nupright target",
        "Sensitivity [rad/s]",
    ]
    for row in range(rows):
        axes[row, 0].set_ylabel(labels[row])
    for index, ax in enumerate(axes.flat):
        ax.grid(alpha=0.15)
        ax.annotate(
            chr(97 + index),
            (0, 1),
            xycoords="axes fraction",
            xytext=(-12, 8),
            textcoords="offset points",
            fontweight="bold",
        )
    name = f"{kind}_states"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=MACHINE / "qa" / f"{name}.alignment.json", strict=True
    )
    fig.savefig(HUMAN / f"{name}.png", dpi=300)
    fig.savefig(HUMAN / f"{name}.pdf")
    fig.savefig(HUMAN / f"{name}.svg")
    plt.close(fig)
    print(f"Exported {name}.", flush=True)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.subplots_adjust(left=0.085, right=0.90, bottom=0.07, top=0.86, hspace=0.65, wspace=0.37)
    fig.suptitle(
        "Repeating swings do not make the opportunity-map angle a phase", y=0.985, fontsize=15
    )
    fig.text(0.5, 0.95, "Arrows indicate time direction. Dots mark the initial state.", ha="center")
    fig.text(
        0.5,
        0.925,
        "Top: energy-colored sensitivity map. Middle: task conflict map."
        if kind == "controlled"
        else "Color: swing energy. Bottom: blue = map angle φ; orange = signed-state angle ψ.",
        ha="center",
    )
    normalizer = Normalize(
        0,
        max(np.max(r["total_j" if kind == "controlled" else "swing_j"] / target) for r in records),
    )
    for column, record in enumerate(records):
        t = record["time_s"]
        s = np.column_stack((record["s_total_rad_s"], record["s_swing_rad_s"]))
        energy = record["total_j" if kind == "controlled" else "swing_j"] / target
        segments = np.stack((s[:-1], s[1:]), axis=1)
        collection = LineCollection(
            segments, cmap="viridis", norm=normalizer, linewidths=1.25, rasterized=True
        )
        collection.set_array((energy[:-1] + energy[1:]) / 2)
        axes[0, column].add_collection(collection)
        axes[0, column].autoscale()
        axes[0, column].scatter(*s[0], s=20, color="black", zorder=4)
        extent = max(abs(s).max() * 1.12, 0.1)
        axes[0, column].set_xlim(-extent, extent)
        axes[0, column].set_ylim(-extent, extent)
        axes[0, column].set_xlabel("Total sensitivity [rad/s]")
        axes[0, column].set_ylabel("Swing sensitivity [rad/s]")
        arrow_indices = np.linspace(2, len(t) - 3, 7, dtype=int)
        if kind == "coast":
            period = metrics[f"coast_{column}"]["period_mean_s"]
            arrow_indices = np.linspace(2, int(period / 0.001) - 3, 5, dtype=int)
        for index in arrow_indices:
            axes[0, column].annotate(
                "",
                xy=s[index + 2],
                xytext=s[index - 2],
                arrowprops=dict(arrowstyle="->", color="#303030", lw=0.7),
            )
        if kind == "controlled":
            points = np.column_stack((record["c_total"], record["c_swing"])) / target
            axes[1, column].plot(points[:, 0], points[:, 1], color=colors[0], alpha=0.8)
            axes[1, column].scatter(*points[0], s=20, color="black")
            axes[1, column].set_xlabel("Total error × sensitivity / target [rad/s]")
            axes[1, column].set_ylabel("Swing error × sensitivity\n/ target [rad/s]")
        else:
            axes[1, column].plot(
                np.rad2deg(record["alpha_rad"]), record["nu_rad_s"], color=colors[0]
            )
            axes[1, column].scatter(
                np.rad2deg(record["alpha_rad"][0]), record["nu_rad_s"][0], s=20, color="black"
            )
            axes[1, column].set_xlabel("Pendulum angle [deg; 0 down]")
            axes[1, column].set_ylabel("Pendulum speed [rad/s]")
        phi = np.rad2deg(record["phi_rad"]).copy()
        radius = np.linalg.norm(s, axis=1)
        phi[radius < 0.01 * np.sqrt(np.mean(radius**2))] = np.nan
        phi[np.r_[False, abs(np.diff(phi)) > 180]] = np.nan
        axes[2, column].plot(t, phi, color=colors[0])
        if kind == "coast":
            psi = np.rad2deg(np.angle(np.exp(1j * record["psi_rad"])))
            psi[np.r_[False, abs(np.diff(psi)) > 180]] = np.nan
            axes[2, column].plot(t, psi, color=colors[1], lw=0.9)
        axes[2, column].set_ylim(-190, 190)
        axes[2, column].set_yticks([-180, 0, 180])
        axes[2, column].set_xlim(0, t[-1] if kind == "controlled" else 3)
        axes[2, column].set_xlabel(
            "Time [s]" if kind == "controlled" else "Time since release [s; detail]"
        )
        axes[2, column].set_ylabel(
            "Map angle φ [deg]" if kind == "controlled" else "Angles φ, ψ [deg]"
        )
        axes[0, column].set_title(titles[column], pad=12)
    color_axis = fig.add_axes([0.93, 0.66, 0.012, 0.18])
    fig.colorbar(
        collection,
        cax=color_axis,
        label="Total energy / target" if kind == "controlled" else "Swing energy / target",
    )
    for index, ax in enumerate(axes.flat):
        if index < 6:
            ax.axhline(0, color="#888888", lw=0.5, ls=":")
            ax.axvline(0, color="#888888", lw=0.5, ls=":")
        ax.grid(alpha=0.13)
        ax.annotate(
            chr(97 + index),
            (0, 1),
            xycoords="axes fraction",
            xytext=(-12, 8),
            textcoords="offset points",
            fontweight="bold",
        )
    name = f"{kind}_maps"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=MACHINE / "qa" / f"{name}.alignment.json",
        exclude_axes=[color_axis],
        strict=True,
    )
    fig.savefig(HUMAN / f"{name}.png", dpi=300)
    fig.savefig(HUMAN / f"{name}.pdf", dpi=300)
    fig.savefig(HUMAN / f"{name}.svg", dpi=300)
    plt.close(fig)
    print(f"Exported {name}.", flush=True)

# Input-dependent task descent and natural exchange have distinct physical meanings.
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.subplots_adjust(left=0.09, right=0.975, bottom=0.09, top=0.76, hspace=0.55, wspace=0.35)
fig.suptitle(
    "Task competition and natural energy exchange during recorded control", y=0.98, fontsize=15
)
fig.text(
    0.5,
    0.935,
    "Top: blue = total-energy task; orange = relative-swing task. Opposite signs compete.",
    ha="center",
)
fig.text(
    0.5,
    0.9,
    "Bottom: gray = natural exchange; orange = torque contribution; blue = full swing-energy rate.",
    ha="center",
)
for column in range(3):
    record = dict(np.load(MACHINE / f"controlled_{column}.npz"))
    t = record["time_s"]
    axes[0, column].plot(t, record["c_total"] / target, color=colors[0])
    axes[0, column].plot(t, record["c_swing"] / target, color=colors[1])
    axes[1, column].plot(t, 1000 * record["swing_drift_w"], color=colors[2], alpha=0.8)
    axes[1, column].plot(t, 1000 * record["swing_input_w"], color=colors[1], alpha=0.9)
    axes[1, column].plot(t, 1000 * record["swing_rate_w"], color=colors[0], lw=0.8)
    axes[0, column].set_title(provenance["selection"][column]["label"], pad=15)
    for row in range(2):
        axes[row, column].set_xlim(0, t[-1])
        axes[row, column].set_xlabel("Time [s]")
        axes[row, column].axhline(0, color="#888888", lw=0.5)
        axes[row, column].grid(alpha=0.15)
axes[0, 0].set_ylabel("Error × sensitivity /\ntarget [rad/s]")
axes[1, 0].set_ylabel("Relative swing-energy rate [mW]")
for index, ax in enumerate(axes.flat):
    ax.annotate(
        chr(97 + index),
        (0, 1),
        xycoords="axes fraction",
        xytext=(-12, 8),
        textcoords="offset points",
        fontweight="bold",
    )
fig.canvas.draw()
require_matplotlib_panel_alignment(
    fig, json_out=MACHINE / "qa/task_rates.alignment.json", strict=True
)
fig.savefig(HUMAN / "task_rates.png", dpi=300)
fig.savefig(HUMAN / "task_rates.pdf")
fig.savefig(HUMAN / "task_rates.svg")
plt.close(fig)
print("Exported task_rates.", flush=True)
