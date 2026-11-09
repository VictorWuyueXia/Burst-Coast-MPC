"""Compare paired nominal runs and plot the same three starts at two time scales."""

import csv
import io
from pathlib import Path

import matplotlib as mpl
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

machine = Path(__file__).resolve().parent
comparison = machine.parents[2] / "comparison/machine-scannables"
human = machine.parent / "human-readables"
human.mkdir(exist_ok=True)
assert np.array_equal(np.load(comparison / "starts.npy"), np.load(machine / "starts.npy"))
dt, state_dt = 0.02, 0.0005
active_modes = ("launch", "pulse", "brake", "aperiodic", "local")
sources = {"before": comparison, "centered": machine}
metadata = {
    name: list(csv.DictReader((folder / "episodes.csv").read_text(encoding="utf-8").splitlines()))
    for name, folder in sources.items()
}

# Count starts of uninterrupted actuation using control modes for both controllers.
rows = []
for name, folder in sources.items():
    for episode, result in enumerate(metadata[name]):
        record = np.load(folder / f"episode-{episode:03d}.npz", allow_pickle=False)
        torque = record["torque_nm"]
        modes = record["mode"][: len(torque)]
        on = np.isin(modes, active_modes)
        upright_feedback = modes == "local"
        final_state = record["states"][-1].copy()
        final_state[1] = np.arctan2(-np.sin(final_state[1]), -np.cos(final_state[1]))
        at_end = np.all(np.abs(final_state) < (0.08, np.deg2rad(5), 0.15, 0.20))
        switch_ons = on & ~np.r_[False, on][:-1]
        exact_zero_on = (torque == 0) & on
        if name == "centered":
            np.testing.assert_array_equal(on, record["actuation_on"])
            assert int(result["actuation_switch_ons"]) == switch_ons.sum()
            assert np.isclose(float(result["actuation_window_s"]), dt * on.sum())
        record.close()
        rows.append(dict(
            controller=name, episode=episode, status=result["status"],
            capture_s=float(result["capture_s"]), meets_hold_limits_at_end=int(at_end),
            peak_arm_deg=float(result["peak_arm_deg"]),
            arm_crossings=int(result["arm_crossings"]), upright_s=float(result["upright_s"]),
            switch_ons=int(switch_ons.sum()), actuation_s=float(on.sum() * dt),
            zero_torque_inside_s=float(exact_zero_on.sum() * dt),
            upright_feedback_s=float(upright_feedback.sum() * dt),
            switch_ons_first_10s=int(switch_ons[:500].sum()),
            actuation_s_first_10s=float(on[:500].sum() * dt),
            zero_torque_inside_s_first_10s=float(exact_zero_on[:500].sum() * dt),
            upright_feedback_s_first_10s=float(upright_feedback[:500].sum() * dt),
        ))

summary = []
for name in sources:
    subset = [r for r in rows if r["controller"] == name]
    capture = np.array([r["capture_s"] for r in subset])
    summary.append(dict(
        controller=name, runs=len(subset), upright_holds=int(np.isfinite(capture).sum()),
        median_hold_s=float(np.median(capture[np.isfinite(capture)])),
        completed=sum(r["status"] == "completed" for r in subset),
        meets_hold_limits_at_end=sum(r["meets_hold_limits_at_end"] for r in subset),
        arm_crossing_runs=sum(r["arm_crossings"] > 0 for r in subset),
        max_abs_arm_deg=max(r["peak_arm_deg"] for r in subset),
        mean_upright_s=float(np.mean([r["upright_s"] for r in subset])),
        **{f"total_{metric}": sum(r[metric] for r in subset) for metric in (
            "switch_ons", "actuation_s", "zero_torque_inside_s", "upright_feedback_s",
            "switch_ons_first_10s", "actuation_s_first_10s", "zero_torque_inside_s_first_10s",
            "upright_feedback_s_first_10s",
        )},
    ))

# Include a previously missed hold, a previously successful hold, and a regression if present.
regressions = [
    i for i in range(32)
    if np.isfinite(rows[i]["capture_s"]) and not np.isfinite(rows[32 + i]["capture_s"])
]
episodes = (0, 1, regressions[0] if regressions else 15)
for filename, records in (("paired-actuation.csv", rows), ("comparison-summary.csv", summary),
                          ("figure-cases.csv", [{"column": label, "episode": i}
                                               for label, i in zip("ABC", episodes, strict=True)])):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    (machine / filename).write_text(buffer.getvalue(), encoding="utf-8", newline="")

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9,
    "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.right": False, "axes.spines.top": False,
})
styles = {"before": ("#85808b", "--", "Before centering"),
          "centered": ("#126481", "-", "With centering")}
for limit in (40, 10):
    # Overlay the physical states and mark active time in two strips in each torque panel.
    fig, axes = plt.subplots(4, 3, figsize=(12.0, 7.3), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.10, top=0.77, wspace=0.17, hspace=0.27)
    fig.suptitle(f"Effect of arm centering during the first {limit} seconds", x=0.11,
                 y=0.975, ha="left", fontsize=15)
    handles = [Line2D([], [], color=color, linestyle=ls, label=label)
               for color, ls, label in styles.values()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.105, 0.944), frameon=False, ncol=2)
    for column, episode in enumerate(episodes):
        pendulum, arm, energy, motor = axes[:, column]
        pendulum.axhspan(0, 5, color="#e0eae2")
        for bound in (-135, 135):
            arm.axhline(bound, color="#74787c", linestyle=":", linewidth=0.9)
        energy.axhline(30.37176, color="#74787c", linestyle=":", linewidth=0.9)
        labels = []
        for index, (name, folder) in enumerate(sources.items()):
            record = np.load(folder / f"episode-{episode:03d}.npz", allow_pickle=False)
            torque = record["torque_nm"][: round(limit / dt)]
            states = record["states"][: round(limit / state_dt) + 1]
            time = np.arange(len(states)) * state_dt
            h = 1000 * record["mechanical_energy_j"][: len(states)]
            on = np.isin(record["mode"][: len(torque)], active_modes)
            record.close()
            error = abs(np.rad2deg(np.arctan2(-np.sin(states[:, 1]), -np.cos(states[:, 1]))))
            color, linestyle, _ = styles[name]
            for axis, series in zip((pendulum, arm, energy), (error, np.rad2deg(states[:, 0]), h), strict=True):
                axis.plot(time, series, color=color, linestyle=linestyle, linewidth=1.0)
            motor.stairs(1000 * torque, np.arange(len(torque) + 1) * dt, baseline=None,
                         color=color, linestyle=linestyle, linewidth=0.85)
            intervals = np.flatnonzero(np.diff(np.r_[False, on, False])).reshape(-1, 2)
            motor.broken_barh([(a * dt, (b - a) * dt) for a, b in intervals],
                              (-6.7 if index == 0 else -7.5, 0.52), facecolors=color, linewidth=0)
            labels.append(f"{'Before' if index == 0 else 'Centered'} · {len(intervals)} switch-ons · {dt * on.sum():.2f} s on")
        pendulum.set_title(f"{'ABC'[column]}  Starting state {episode}\n" + "\n".join(labels),
                           loc="left", fontsize=9.3, pad=7)
        motor.set_xlabel("Simulation time (s)")
        motor.set_xlim(0, limit)
        for axis in axes[:, column]:
            axis.grid(axis="y", color="#e8eaed", linewidth=0.55)
            axis.set_axisbelow(True)
    for axis, ylabel in zip(axes[:, 0], ("Distance from\nupright (°)", "Arm angle (°)",
                                        "Mechanical\nenergy (mJ)", "Torque (mN·m)"), strict=True):
        axis.set_ylabel(ylabel)
    axes[0, 0].set(ylim=(0, 184), yticks=(0, 90, 180))
    axes[1, 0].set(ylim=(-147, 147), yticks=(-135, 0, 135))
    axes[2, 0].set(ylim=(-0.6, 35), yticks=(0, 15, 30))
    axes[3, 0].set(ylim=(-7.65, 5.2), yticks=(-4.59, 0, 4.59))
    fig.align_ylabels(axes[:, 0])
    stem = f"paired-first-{limit}s"
    require_matplotlib_panel_alignment(fig, json_out=machine / f"{stem}-alignment.json", strict=True)
    fig.savefig(human / f"{stem}.png", dpi=300)
    fig.savefig(human / f"{stem}.pdf")
    fig.savefig(human / f"{stem}.svg")
    plt.close(fig)
