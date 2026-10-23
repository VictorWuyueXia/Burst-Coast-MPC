"""Analyze completed paired tracking studies; argument: screen or confirm."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt

stage = sys.argv[1]
root = Path("artifacts/rotary_pendulum/inner-mpc")
output = root / f"20261020-tracking-{stage}-summary"
machine, human = output / "machine-scannables", output / "human-readables"
machine.mkdir(parents=True, exist_ok=True)
human.mkdir(exist_ok=True)
runs = sorted(root.glob(f"20261020-tracking-{stage}-*/machine-scannables/design.json"))
summary, episodes, histories = [], [], []
for path in runs:
    folder = path.parent
    design = json.loads(path.read_text())
    assert (folder / "completed.json").exists(), f"Incomplete study: {folder}"
    for source, digest in design["source_sha256"].items():
        assert (
            hashlib.sha256((folder / "sources" / Path(source).name).read_bytes()).hexdigest()
            == digest
        )
    archive = np.load(folder / "starts.npz", allow_pickle=False)
    initial, groups = archive["states"].copy(), archive["groups"].copy()
    archive.close()
    chunks = sorted(folder.glob("chunk-*.npz"))
    pieces = []
    for chunk in chunks:
        archive = np.load(chunk, allow_pickle=False)
        pieces.append({name: archive[name].copy() for name in archive.files})
        archive.close()
    data = {name: np.concatenate([piece[name] for piece in pieces]) for name in pieces[0]}
    assert data["accepted"].shape == (design["decisions"], design["count"])
    assert data["accepted"].all()
    x = data["executed"].transpose(1, 0, 2, 3).reshape(design["count"], -1, 4)
    x = np.concatenate((initial[:, None], x), axis=1)
    assert np.isfinite(x).all()
    assert (np.abs(x[..., 0]) < design["plant"]["observation_scales"][0]).all()
    assert (np.abs(data["plans"][..., :-1]) <= 1).all()
    time = np.arange(x.shape[1]) * 0.001
    assert np.isclose(time[-1], design["duration_s"])
    error = np.rad2deg(np.abs(np.arctan2(-np.sin(x[..., 1]), -np.cos(x[..., 1]))))
    near = error < 5
    arm = np.rad2deg(x[..., 0])
    name = f"{design['solver'].upper()}, tracking={design['rate']:g}"
    local = []
    for lane in range(design["count"]):
        edges = np.diff(np.pad(near[lane].astype(int), (1, 1)))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        durations = (ends - starts) * 0.001
        first = np.flatnonzero(near[lane])
        first1 = starts[durations >= 1]
        first10 = starts[durations >= 10]
        row = dict(
            controller=name,
            run=folder.parent.name,
            solver=design["solver"],
            rate=design["rate"],
            episode=lane,
            group=int(groups[lane]),
            duration_s=float(time[-1]),
            first_arrival_s=float(time[first[0]]) if len(first) else None,
            first_1s_hold_completed_s=float(first1[0] * 0.001 + 1) if len(first1) else None,
            first_10s_hold_completed_s=float(first10[0] * 0.001 + 10) if len(first10) else None,
            final_1s_hold=bool(near[lane, -1000:].all()),
            final_10s_hold=bool(near[lane, -10000:].all()),
            ever_10s_hold=bool(len(first10)),
            upright_fraction_after_10s=float(near[lane, 10000:].mean()),
            escaped_after_first_10s_hold=bool(
                np.any((ends >= first10[0] + 10000) & (ends < len(time)))
            )
            if len(first10)
            else None,
            longest_dwell_s=float(durations.max()) if len(durations) else 0.0,
            final_dwell_s=float(np.cumprod(near[lane, ::-1]).sum() * 0.001),
            exits_after_first_1s_hold=int(np.sum(ends < len(time)) - np.sum(ends <= first1[0]))
            if len(first1)
            else None,
            final_10s_upright_rms_deg=float(np.sqrt(np.mean(error[lane, -10000:] ** 2))),
            final_10s_arm_rms_deg=float(np.sqrt(np.mean(arm[lane, -10000:] ** 2))),
            peak_arm_deg=float(np.max(np.abs(arm[lane]))),
            final_10s_torque_rms_mnm=float(
                np.sqrt(
                    np.mean(
                        (data["plans"][-100:, lane, 0] * design["plant"]["torque_limit_nm"] * 1000)
                        ** 2
                    )
                )
            ),
        )
        local.append(row)
    episodes.extend(local)
    summary.append(
        dict(
            controller=name,
            run=folder.parent.name,
            count=design["count"],
            duration_s=float(time[-1]),
            final_1s_hold=sum(r["final_1s_hold"] for r in local),
            final_10s_hold=sum(r["final_10s_hold"] for r in local),
            ever_10s_hold=sum(r["ever_10s_hold"] for r in local),
            reached=sum(r["first_arrival_s"] is not None for r in local),
            mean_upright_fraction_after_10s=float(
                np.mean([r["upright_fraction_after_10s"] for r in local])
            ),
            median_longest_dwell_s=float(np.median([r["longest_dwell_s"] for r in local])),
            escaped_after_first_10s_hold=sum(
                r["escaped_after_first_10s_hold"] is True for r in local
            ),
            mean_final_10s_upright_rms_deg=float(
                np.mean([r["final_10s_upright_rms_deg"] for r in local])
            ),
            mean_final_10s_arm_rms_deg=float(np.mean([r["final_10s_arm_rms_deg"] for r in local])),
            median_batch_s=float(np.median(data["solve_s"])),
            p95_batch_s=float(np.quantile(data["solve_s"], 0.95)),
            max_peak_arm_deg=max(r["peak_arm_deg"] for r in local),
            max_refinement_error=data["refinement_error"].max(axis=(0, 1)).tolist(),
            total_accepted=int(data["accepted"].sum()),
        )
    )
    np.savez_compressed(
        machine / f"{folder.parent.name}.npz",
        states=x,
        time_s=time,
        torque_nm=data["plans"][..., 0].T * design["plant"]["torque_limit_nm"],
        error_deg=error,
        plan_end=data["plan_end"],
        groups=groups,
    )
    histories.append(dict(name=name, design=design, time=time, error=error, arm=arm, rows=local))
    print(json.dumps(summary[-1]), flush=True)
stream = (machine / "episodes.csv").open("w", newline="")
writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
writer.writeheader()
writer.writerows(episodes)
stream.close()
(machine / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)
colors = ["#4477AA", "#EE7733", "#228833", "#AA3377", "#009988", "#CCBB44"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.16, top=0.72, wspace=0.28)
for k, history in enumerate(histories):
    time = history["time"]
    # Curves show completion of full ten-second holds, never instantaneous arrivals.
    completed = np.array(
        [
            r["first_10s_hold_completed_s"]
            for r in history["rows"]
            if r["first_10s_hold_completed_s"] is not None
        ]
    )
    progress = (completed[:, None] <= time[None, ::100]).sum(axis=0)
    axes[0].plot(time[::100], progress, color=colors[k], label=history["name"], lw=1.8)
    axes[1].plot(
        time[::100],
        (history["error"][:, ::100] < 5).sum(axis=0),
        color=colors[k],
        lw=1.3,
        alpha=0.85,
    )
for ax in axes:
    ax.set_xlabel("Elapsed simulation time (s)")
    ax.set_ylim(-0.5, histories[0]["design"]["count"] + 0.5)
    ax.grid(alpha=0.18)
axes[0].set_title("A  Ever completed a 10 s upright hold")
axes[1].set_title("B  Currently within 5° of upright")
axes[0].set_ylabel("Number of sessions")
axes[1].set_ylabel("Number of sessions")
fig.suptitle(f"Long-duration weak-tracking comparison: {stage}", fontsize=17, y=0.98)
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.91),
    ncol=2,
    fontsize=10,
)
fig.text(
    0.5,
    0.025,
    "A past hold does not guarantee continued balance. "
    "All sessions run for the full common duration.",
    ha="center",
    fontsize=9,
)
require_matplotlib_panel_alignment(fig, json_out=machine / "progress.alignment.json")
fig.savefig(human / "hold_progress.png", dpi=300)
fig.savefig(human / "hold_progress.pdf")
fig.savefig(human / "hold_progress.svg")
plt.close(fig)

fig, axes = plt.subplots(
    2, len(histories), figsize=(4 * len(histories), 6), sharex=True, sharey=True
)
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.13, top=0.82, wspace=0.14, hspace=0.3)
for col, history in enumerate(histories):
    for row, lane in enumerate([0, 2]):
        axes[row, col].plot(history["time"], history["error"][lane], color=colors[col], lw=0.8)
        axes[row, col].axhspan(0, 5, color="0.7", alpha=0.4)
        axes[row, col].set_ylim(-4, 185)
        axes[row, col].set_yticks([0, 90, 180])
        axes[row, col].grid(alpha=0.15)
    axes[0, col].set_title(history["name"], fontsize=11)
    axes[1, col].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Downward start, episode 0\nUpright error (°)")
axes[1, 0].set_ylabel("Near-upright start, episode 2\nUpright error (°)")
fig.suptitle("Matched swing progress: identical initial states", fontsize=17, y=0.97)
fig.text(
    0.5,
    0.9,
    "Fixed episodes 0 and 2 in every setting; "
    "full traces, no smoothing or favorable-case selection.",
    ha="center",
)
fig.text(
    0.5,
    0.025,
    "Gray band = upright ±5°. Large excursions after arrival indicate loss of capture.",
    ha="center",
    fontsize=10,
)
require_matplotlib_panel_alignment(fig, json_out=machine / "sessions.alignment.json")
fig.savefig(human / "matched_sessions.png", dpi=300)
fig.savefig(human / "matched_sessions.pdf")
fig.savefig(human / "matched_sessions.svg")
plt.close(fig)
(machine / "tracking_report.py").write_bytes(Path(__file__).read_bytes())
print(f"Complete: {output}", flush=True)
