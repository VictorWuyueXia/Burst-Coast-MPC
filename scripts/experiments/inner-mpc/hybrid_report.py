"""Audit full CEM–LQR sessions and compare hold times; argument: screen or confirm."""

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
output = root / f"20261022-hybrid-{stage}-summary"
machine, human = output / "machine-scannables", output / "human-readables"
machine.mkdir(parents=True, exist_ok=True)
human.mkdir(exist_ok=True)
paths = sorted(root.glob(f"20261022-hybrid-{stage}-*/machine-scannables/design.json"))
paths.sort(key=lambda p: json.loads(p.read_text())["dwell_s"])
summary, episodes, histories = [], [], []
for path in paths:
    design = json.loads(path.read_text())
    folder = path.parent
    assert (folder / "completed.json").exists(), f"Incomplete: {folder}"
    for source, digest in design["source_sha256"].items():
        assert (
            hashlib.sha256((folder / "sources" / Path(source).name).read_bytes()).hexdigest()
            == digest
        )
    archive = np.load(folder / "starts.npz", allow_pickle=False)
    initial, groups = archive["states"].copy(), archive["groups"].copy()
    archive.close()
    pieces = []
    for chunk in sorted(folder.glob("chunk-*.npz")):
        archive = np.load(chunk, allow_pickle=False)
        pieces.append({k: archive[k].copy() for k in archive.files})
        archive.close()
    data = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    count, dwell = design["count"], design["dwell_s"]
    assert design["duration_s"] == 40 and data["accepted"].shape == (400, count)
    assert data["accepted"].all() and np.isfinite(data["executed"]).all()
    assert (np.abs(data["executed"][..., 0]) < design["plant"]["observation_scales"][0]).all()
    assert (np.abs(data["action"]) <= 1).all()
    assert (data["refinement_error"] <= [0.001, 0.001, 0.01, 0.01]).all()
    # Independently reconstruct the complete switching state machine and local feedback.
    quiet, local, states = np.zeros(count, int), np.zeros(count, bool), initial
    for decision in range(400):
        beta = np.arctan2(-np.sin(states[:, 1]), -np.cos(states[:, 1]))
        region = (np.abs(beta) < 0.25) & (np.abs(states[:, 3]) < 2)
        expected = (
            (local | (quiet >= round(dwell * 10))) & region if dwell else np.zeros(count, bool)
        )
        np.testing.assert_array_equal(data["local"][decision], expected)
        np.testing.assert_array_equal(data["handoff"][decision], expected & ~local)
        np.testing.assert_array_equal(data["release"][decision], ~expected & local)
        np.testing.assert_array_equal(
            data["solver_called"][decision], np.full(count, not expected.all())
        )
        error = states.copy()
        error[:, 1] = beta
        np.testing.assert_allclose(
            data["action"][decision, expected],
            np.clip(-error @ design["lqr_gain"], -1, 1)[expected],
            atol=1e-12,
        )
        current = data["executed"][decision]
        beta = np.abs(np.arctan2(-np.sin(current[..., 1]), -np.cos(current[..., 1])))
        settled = (
            (beta <= np.deg2rad(5))
            & (np.abs(current[..., 2]) <= 0.3)
            & (np.abs(current[..., 3]) <= 0.4)
        ).all(axis=1)
        quiet, local, states = np.where(settled, quiet + 1, 0), expected, current[:, -1]
        np.testing.assert_array_equal(data["quiet"][decision], quiet)
    x = data["executed"].transpose(1, 0, 2, 3).reshape(count, 40000, 4)
    error = np.rad2deg(np.abs(np.arctan2(-np.sin(x[..., 1]), -np.cos(x[..., 1]))))
    rows = []
    for lane in range(count):
        mask = error[lane] <= 5
        gaps = np.r_[-1, np.flatnonzero(~mask), len(mask)]
        handoffs = np.flatnonzero(data["handoff"][:, lane])
        row = dict(
            dwell_s=dwell,
            episode=lane,
            group=int(groups[lane]),
            total_hold_s=float(mask.sum() * 0.001),
            max_hold_s=float((np.diff(gaps) - 1).max() * 0.001),
            first_handoff_s=float(handoffs[0] * 0.1) if len(handoffs) else None,
            handoffs=len(handoffs),
            releases=int(data["release"][:, lane].sum()),
            post_handoff_peak_error_deg=(
                float(error[lane, handoffs[0] * 100 :].max()) if len(handoffs) else None
            ),
            lqr_time_s=float(data["local"][:, lane].sum() * 0.1),
            final_10s_hold=bool(mask[-10000:].all()),
            final_error_deg=float(error[lane, -1]),
            final_arm_deg=float(np.rad2deg(x[lane, -1, 0])),
        )
        assert 0 <= row["max_hold_s"] <= row["total_hold_s"] <= 40
        rows.append(row)
    episodes.extend(rows)
    summary.append(
        dict(
            dwell_s=dwell,
            count=count,
            duration_s=40,
            mean_total_hold_s=float(np.mean([r["total_hold_s"] for r in rows])),
            mean_max_hold_s=float(np.mean([r["max_hold_s"] for r in rows])),
            median_max_hold_s=float(np.median([r["max_hold_s"] for r in rows])),
            handed_over=sum(r["handoffs"] > 0 for r in rows),
            releases=sum(r["releases"] for r in rows),
            final_10s_hold=sum(r["final_10s_hold"] for r in rows),
            latest_first_handoff_s=max(
                [r["first_handoff_s"] for r in rows if r["handoffs"]], default=None
            ),
            mean_lqr_time_s=float(np.mean([r["lqr_time_s"] for r in rows])),
            cem_batches=int(data["solver_called"][:, 0].sum()),
            median_batch_s=float(np.median(data["solve_s"])),
            all_local_median_batch_s=(
                float(np.median(data["solve_s"][~data["solver_called"][:, 0]]))
                if (~data["solver_called"][:, 0]).any()
                else None
            ),
            accepted=int(data["accepted"].sum()),
            peak_arm_deg=float(np.rad2deg(np.abs(x[..., 0]).max())),
            max_refinement_error=data["refinement_error"].max(axis=(0, 1)).tolist(),
        )
    )
    histories.append(dict(dwell=dwell, x=x, error=error, rows=rows, initial=initial))
    np.savez_compressed(
        machine / f"dwell-{dwell:g}.npz",
        states=x,
        initial=initial,
        time_s=np.arange(1, 40001) * 0.001,
        local=data["local"].T,
        action=data["action"].T,
        error_deg=error,
        groups=groups,
    )
    print(json.dumps(summary[-1]), flush=True)
for h in histories[1:]:
    np.testing.assert_array_equal(h["initial"], histories[0]["initial"])
f = (machine / "episodes.csv").open("w", newline="")
w = csv.DictWriter(f, fieldnames=list(episodes[0]))
w.writeheader()
w.writerows(episodes)
f.close()
(machine / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
paired = []
if histories[0]["dwell"] == 0:
    for h in histories[1:]:
        for metric in ("total_hold_s", "max_hold_s"):
            delta = np.array([r[metric] for r in h["rows"]]) - np.array(
                [r[metric] for r in histories[0]["rows"]]
            )
            paired.append(
                dict(
                    dwell_s=h["dwell"],
                    metric=metric,
                    mean_delta_s=float(delta.mean()),
                    improved=int((delta > 0).sum()),
                    worsened=int((delta < 0).sum()),
                )
            )
(machine / "paired.json").write_text(json.dumps(paired, indent=2) + "\n")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
colors = {0: "#555555", 0.5: "#0072B2", 1.0: "#D55E00"}
labels = ["CEM" if h["dwell"] == 0 else f"CEM → LQR\n{h['dwell']:g} s dwell" for h in histories]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
for ax, metric, title in zip(
    axes,
    ("total_hold_s", "max_hold_s"),
    ("Total upright time", "Longest continuous hold"),
    strict=True,
):
    values = np.array([[r[metric] for r in h["rows"]] for h in histories])
    ax.plot(np.arange(len(histories)), values, color=".8", linewidth=0.6, alpha=0.6)
    for i, h in enumerate(histories):
        ax.scatter(np.full(len(h["rows"]), i), values[i], color=colors[h["dwell"]], s=16)
        ax.plot([i - 0.15, i + 0.15], [values[i].mean()] * 2, color="black", linewidth=3)
    ax.set(
        xticks=np.arange(len(histories)),
        xticklabels=labels,
        ylim=(0, 41),
        ylabel="Time (s)",
        title=title,
    )
fig.suptitle(
    f"{len(histories[0]['rows'])} paired starts · full 40 s · ±5° upright band\n"
    "Dots: sessions; lines: matched starts; black bars: means",
    fontsize=12,
)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=machine / "hold_metrics.alignment.json")
fig.savefig(human / "hold_metrics.png", dpi=300)
fig.savefig(human / "hold_metrics.pdf")
fig.savefig(human / "hold_metrics.svg")
plt.close(fig)
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout="constrained")
for row, lane in enumerate([0, 2]):
    axes[row, 0].axhspan(0, 5, color=".9")
    for h in histories:
        label = "CEM" if h["dwell"] == 0 else f"LQR after {h['dwell']:g} s dwell"
        axes[row, 0].plot(
            np.arange(1, 40001) * 0.001,
            h["error"][lane],
            color=colors[h["dwell"]],
            linewidth=0.8,
            label=label,
        )
        axes[row, 1].plot(
            np.arange(1, 40001) * 0.001,
            np.rad2deg(h["x"][lane, :, 0]),
            color=colors[h["dwell"]],
            linewidth=0.8,
        )
        first = h["rows"][lane]["first_handoff_s"]
        if first is not None:
            axes[row, 0].axvline(first, color=colors[h["dwell"]], linestyle=":", linewidth=1)
    axes[row, 0].set_yscale("symlog", linthresh=10)
    axes[row, 0].set_yticks([0, 5, 10, 30, 90, 180], labels=["0", "5", "10", "30", "90", "180"])
    axes[row, 0].set(ylabel="Upright error (°)", ylim=(0, 180), title=f"Episode {lane}: pendulum")
    axes[row, 1].set(
        ylabel="Arm angle (°)", ylim=(-135, 135), title=f"Episode {lane}: arm centering"
    )
for ax in axes[-1]:
    ax.set(xlabel="Time (s)", xlim=(0, 40))
axes[-1, 0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=9, frameon=False)
fig.suptitle(
    "Full sessions · dotted lines: first LQR handoff\n"
    "Pendulum scale: linear to 10°, logarithmic above",
    fontsize=12,
)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=machine / "representative_sessions.alignment.json")
fig.savefig(human / "representative_sessions.png", dpi=300)
fig.savefig(human / "representative_sessions.pdf")
fig.savefig(human / "representative_sessions.svg")
plt.close(fig)
(machine / "hybrid_report.py").write_bytes(Path(__file__).read_bytes())
print(f"Report: {output}", flush=True)
