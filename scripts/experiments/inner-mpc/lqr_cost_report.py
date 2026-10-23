"""Audit LQR-cost CEM and compare complete sessions; argument: screen or confirm."""

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
output = root / f"20261022-lqr-cost-{stage}-summary"
machine, human = output / "machine-scannables", output / "human-readables"
machine.mkdir(parents=True, exist_ok=True)
human.mkdir(exist_ok=True)
paths = sorted(root.glob(f"20261022-lqr-cost-fine-{stage}-*/machine-scannables/design.json"))
if stage == "confirm":
    paths = [
        root / f"20261022-hybrid-confirm-{d}/machine-scannables/design.json" for d in (0, 0.5)
    ] + paths
histories, rows, summaries = [], [], []
for path in paths:
    design = json.loads(path.read_text())
    folder = path.parent
    assert (folder / "completed.json").exists(), f"Incomplete: {folder}"
    for source, digest in design["source_sha256"].items():
        assert (
            hashlib.sha256((folder / "sources" / Path(source).name).read_bytes()).hexdigest()
            == digest
        )
    start = np.load(folder / "starts.npz")
    initial, groups = start["states"].copy(), start["groups"].copy()
    start.close()
    chunks = []
    for p in sorted(folder.glob("chunk-*.npz")):
        z = np.load(p)
        chunks.append({k: z[k].copy() for k in z.files})
        z.close()
    data = {k: np.concatenate([p[k] for p in chunks]) for k in chunks[0]}
    count = design["count"]
    assert design["duration_s"] == 40 and data["accepted"].shape == (400, count)
    assert data["accepted"].all() and np.isfinite(data["executed"]).all()
    assert (np.abs(data["executed"][..., 0]) < design["plant"]["observation_scales"][0]).all()
    assert (data["refinement_error"] <= [0.001, 0.001, 0.01, 0.01]).all()
    if "lqr_state_weights" in design["plant"]:
        label = f"LQR cost + RL, {design['generations']} gen"
        assert design["plant"]["discount"] == 1 and not design["plant"]["rl_cost"]
        actions = data["plans"][..., :-1]
        assert (np.abs(actions) <= 1).all() and (data["arm_margin_rad"] > 0).all()
        error = data["stage_states"].copy()
        error[..., 1] = np.arctan2(-np.sin(error[..., 1]), -np.cos(error[..., 1]))
        parts = np.sum(error**2 * np.array(design["plant"]["lqr_state_weights"]), axis=-2)
        for i, name in enumerate(("centering", "upright_tracking", "speed_arm", "speed_pendulum")):
            np.testing.assert_allclose(data[name], parts[..., i], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(data["effort"], np.sum(actions**2, axis=-1), atol=1e-10)
        np.testing.assert_allclose(
            data["objective"],
            parts.sum(axis=-1) + data["effort"] + data["terminal_costs"].max(axis=-1),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_array_equal(data["stage_states"][0, :, 0], initial)
        np.testing.assert_array_equal(data["stage_states"][1:, :, 0], data["executed"][:-1, :, -1])
        np.savez_compressed(
            machine / f"cost-components-{design['generations']}.npz",
            objective=data["objective"],
            state_components=parts,
            effort=data["effort"],
            terminal=data["terminal_costs"].max(axis=-1),
        )
    else:
        label = "Original CEM" if design["dwell_s"] == 0 else "Original CEM → LQR"
        assert (np.abs(data["action"]) <= 1).all()
    x = data["executed"].transpose(1, 0, 2, 3).reshape(count, 40000, 4)
    error = np.rad2deg(np.abs(np.arctan2(-np.sin(x[..., 1]), -np.cos(x[..., 1]))))
    current = []
    for lane in range(count):
        held = error[lane] <= 5
        gaps = np.r_[-1, np.flatnonzero(~held), len(held)]
        row = dict(
            controller=label,
            episode=lane,
            group=int(groups[lane]),
            total_hold_s=float(held.sum() * 0.001),
            max_hold_s=float((np.diff(gaps) - 1).max() * 0.001),
            final_hold_s=float((len(held) - gaps[-2] - 1) * 0.001),
            final_10s_hold=bool(held[-10000:].all()),
            first_upright_s=float(np.flatnonzero(held)[0] * 0.001 + 0.001) if held.any() else None,
            final_error_deg=float(error[lane, -1]),
            final_arm_deg=float(np.rad2deg(x[lane, -1, 0])),
        )
        assert 0 <= row["max_hold_s"] <= row["total_hold_s"] <= 40
        current.append(row)
    rows.extend(current)
    summary = dict(
        controller=label,
        count=count,
        duration_s=40,
        mean_total_hold_s=float(np.mean([r["total_hold_s"] for r in current])),
        mean_max_hold_s=float(np.mean([r["max_hold_s"] for r in current])),
        final_10s_hold=sum(r["final_10s_hold"] for r in current),
        latest_final_hold_start_s=float(40 - min(r["final_hold_s"] for r in current)),
        ever_upright=sum(r["first_upright_s"] is not None for r in current),
        peak_arm_deg=float(np.rad2deg(np.abs(x[..., 0]).max())),
        final_10s_arm_rms_deg=float(np.sqrt(np.mean(np.rad2deg(x[:, -10000:, 0]) ** 2))),
        final_10s_upright_rms_deg=float(np.sqrt(np.mean(error[:, -10000:] ** 2))),
        median_batch_s=float(np.median(data["solve_s"])),
        accepted=int(data["accepted"].sum()),
        max_refinement_error=data["refinement_error"].max(axis=(0, 1)).tolist(),
    )
    summaries.append(summary)
    histories.append(dict(label=label, rows=current, x=x, error=error, initial=initial))
    np.savez_compressed(
        machine / f"trajectory-{len(histories) - 1}.npz",
        states=x,
        error_deg=error,
        initial=initial,
        groups=groups,
        time_s=np.arange(1, 40001) * 0.001,
    )
    print(json.dumps(summary), flush=True)
for h in histories[1:]:
    np.testing.assert_array_equal(h["initial"], histories[0]["initial"])
f = (machine / "episodes.csv").open("w", newline="")
w = csv.DictWriter(f, fieldnames=list(rows[0]))
w.writeheader()
w.writerows(rows)
f.close()
(machine / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
paired = []
for h in histories[1:]:
    for metric in ("total_hold_s", "max_hold_s"):
        delta = np.array([r[metric] for r in h["rows"]]) - np.array(
            [r[metric] for r in histories[0]["rows"]]
        )
        paired.append(
            dict(
                controller=h["label"],
                reference=histories[0]["label"],
                metric=metric,
                mean_delta_s=float(delta.mean()),
                improved=int((delta > 0).sum()),
                worsened=int((delta < 0).sum()),
                tied=int((delta == 0).sum()),
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
colors = ["#555555", "#0072B2", "#D55E00", "#009E73"]
labels = [h["label"].replace(" + RL, ", " + RL\n").replace("Original ", "") for h in histories]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
for ax, metric, title in zip(
    axes,
    ("total_hold_s", "max_hold_s"),
    ("Total upright time", "Longest continuous hold"),
    strict=True,
):
    values = np.array([[r[metric] for r in h["rows"]] for h in histories])
    ax.plot(np.arange(len(histories)), values, color=".8", linewidth=0.6, alpha=0.6)
    for i, h in enumerate(histories):
        ax.scatter(np.full(len(h["rows"]), i), values[i], color=colors[i], s=16)
        ax.plot([i - 0.15, i + 0.15], [values[i].mean()] * 2, color="black", linewidth=3)
    ax.set(
        xticks=np.arange(len(histories)),
        xticklabels=labels,
        ylim=(0, 41),
        ylabel="Time (s)",
        title=title,
    )
fig.suptitle(
    f"{len(histories[0]['rows'])} matched starts · full 40 s · ±5° upright band\n"
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
for row, lane in enumerate((0, 2)):
    axes[row, 0].axhspan(0, 5, color=".9")
    for i, h in enumerate(histories):
        axes[row, 0].plot(
            np.arange(1, 40001) * 0.001,
            h["error"][lane],
            color=colors[i],
            linewidth=0.8,
            label=h["label"],
        )
        axes[row, 1].plot(
            np.arange(1, 40001) * 0.001,
            np.rad2deg(h["x"][lane, :, 0]),
            color=colors[i],
            linewidth=0.8,
        )
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
    "Complete sessions · gray band: ±5° upright\nPendulum scale: linear to 10°, logarithmic above",
    fontsize=12,
)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=machine / "representative_sessions.alignment.json")
fig.savefig(human / "representative_sessions.png", dpi=300)
fig.savefig(human / "representative_sessions.pdf")
fig.savefig(human / "representative_sessions.svg")
plt.close(fig)
(machine / "lqr_cost_report.py").write_bytes(Path(__file__).read_bytes())
print(f"Report: {output}", flush=True)
