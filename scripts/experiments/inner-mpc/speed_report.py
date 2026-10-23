"""Paired 40-second CEM speed-cost analysis; argument: screen or confirm."""

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
output = root / f"20261020-speed-{stage}-summary"
machine, human = output / "machine-scannables", output / "human-readables"
machine.mkdir(parents=True, exist_ok=True)
human.mkdir(exist_ok=True)
paths = sorted(root.glob(f"20261020-speed-{stage}-cem-*/machine-scannables/design.json"))
paths.sort(key=lambda p: json.loads(p.read_text())["rate"])
summary, episodes, histories = [], [], []
for path in paths:
    folder = path.parent
    design = json.loads(path.read_text())
    assert (folder / "completed.json").exists(), f"Incomplete: {folder}"
    assert design["duration_s"] == 40 and design["solver"] == "cem"
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
        pieces.append({name: archive[name].copy() for name in archive.files})
        archive.close()
    data = {name: np.concatenate([p[name] for p in pieces]) for name in pieces[0]}
    assert data["accepted"].shape == (400, design["count"]) and data["accepted"].all()
    x = data["executed"].transpose(1, 0, 2, 3).reshape(design["count"], 40000, 4)
    assert np.isfinite(x).all()
    assert (np.abs(x[..., 0]) < design["plant"]["observation_scales"][0]).all()
    assert (np.abs(data["plans"][..., :-1]) <= 1).all()
    assert (data["refinement_error"] <= [0.001, 0.001, 0.01, 0.01]).all()
    error = np.rad2deg(np.abs(np.arctan2(-np.sin(x[..., 1]), -np.cos(x[..., 1]))))
    local = []
    for tolerance in (2, 5, 10):
        for lane in range(design["count"]):
            inside = error[lane] <= tolerance
            edges = np.diff(np.pad(inside.astype(int), (1, 1)))
            lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
            row = dict(
                rate=design["rate"],
                episode=lane,
                group=int(groups[lane]),
                tolerance_deg=tolerance,
                total_hold_s=float(inside.sum() * 0.001),
                max_hold_s=float(lengths.max() * 0.001) if len(lengths) else 0.0,
            )
            assert 0 <= row["max_hold_s"] <= row["total_hold_s"] <= 40
            local.append(row)
    episodes.extend(local)
    main = [r for r in local if r["tolerance_deg"] == 5]
    summary.append(
        dict(
            rate=design["rate"],
            count=design["count"],
            duration_s=40,
            mean_total_hold_s=float(np.mean([r["total_hold_s"] for r in main])),
            median_total_hold_s=float(np.median([r["total_hold_s"] for r in main])),
            mean_max_hold_s=float(np.mean([r["max_hold_s"] for r in main])),
            median_max_hold_s=float(np.median([r["max_hold_s"] for r in main])),
            median_batch_s=float(np.median(data["solve_s"])),
            accepted=int(data["accepted"].sum()),
            peak_arm_deg=float(np.rad2deg(np.abs(x[..., 0]).max())),
            max_refinement_error=data["refinement_error"].max(axis=(0, 1)).tolist(),
            mean_speed_cost=float((data["speed_arm"] + data["speed_pendulum"]).mean()),
            mean_effort_cost=float(data["effort"].mean()),
            mean_centering_cost=float(data["centering"].mean()),
            mean_terminal_cost=float(
                design["plant"]["discount"] ** 10 * data["terminal_costs"].max(axis=-1).mean()
            ),
            median_inband_arm_speed_rad_s=float(np.median(np.abs(x[..., 2][error <= 5]))),
            median_inband_pendulum_speed_rad_s=float(np.median(np.abs(x[..., 3][error <= 5]))),
            sessions_with_10s_hold=sum(r["max_hold_s"] >= 10 for r in main),
        )
    )
    np.savez_compressed(
        machine / f"rate-{design['rate']:g}.npz",
        states=x,
        initial=initial,
        time_s=np.arange(1, 40001) * 0.001,
        error_deg=error,
        groups=groups,
        torque_nm=data["plans"][..., 0].T * design["plant"]["torque_limit_nm"],
    )
    histories.append(
        dict(
            rate=design["rate"],
            error=error,
            states=x,
            initial=initial,
            rows=main,
            all_rows=local,
            seed=design["seed"],
        )
    )
    print(json.dumps(summary[-1]), flush=True)
assert histories[0]["rate"] == 0
for history in histories[1:]:
    np.testing.assert_array_equal(history["initial"], histories[0]["initial"])
    assert history["seed"] == histories[0]["seed"]
stream = (machine / "episodes.csv").open("w", newline="")
writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
writer.writeheader()
writer.writerows(episodes)
stream.close()
paired = []
for history in histories[1:]:
    for tolerance in (2, 5, 10):
        for metric in ("total_hold_s", "max_hold_s"):
            baseline = np.array(
                [r[metric] for r in histories[0]["all_rows"] if r["tolerance_deg"] == tolerance]
            )
            values = np.array(
                [r[metric] for r in history["all_rows"] if r["tolerance_deg"] == tolerance]
            )
            delta = values - baseline
            paired.append(
                dict(
                    rate=history["rate"],
                    tolerance_deg=tolerance,
                    metric=metric,
                    mean_delta_s=float(delta.mean()),
                    median_delta_s=float(np.median(delta)),
                    improved=int((delta > 0).sum()),
                    worsened=int((delta < 0).sum()),
                    tied=int((delta == 0).sum()),
                )
            )
(machine / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(machine / "paired.json").write_text(json.dumps(paired, indent=2) + "\n")
(machine / "metric_contract.json").write_text(
    json.dumps(
        dict(
            duration_s=40,
            sample_dt_s=0.001,
            samples_per_session=40000,
            initial_sample_included=False,
            primary_tolerance_deg=5,
            sensitivity_tolerance_deg=[2, 10],
            group_codes={0: "downward", 1: "swing_up", 2: "near_upright", 3: "outward_arm"},
            total_hold_s=(
                "0.001 times count of post-integration samples with "
                "absolute wrapped upright error <= tolerance"
            ),
            max_hold_s=(
                "0.001 times longest consecutive run of accepted upright samples; no gap bridging"
            ),
            states=(
                "arm angle rad, pendulum angle rad zero down, arm speed rad/s, pendulum speed rad/s"
            ),
            comparison=(
                "paired starts and common CEM random seed within round; "
                "descriptive, not independent replicates"
            ),
        ),
        indent=2,
    )
    + "\n"
)
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
colors = {0: "#555555", 0.01: "#0072B2", 0.03: "#009E73", 0.1: "#D55E00"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
for ax, metric, title in zip(
    axes,
    ("total_hold_s", "max_hold_s"),
    ("Total upright time", "Longest continuous hold"),
    strict=True,
):
    values = np.array([[r[metric] for r in h["rows"]] for h in histories])
    ax.plot(np.arange(len(histories)), values, color="0.78", alpha=0.55, linewidth=0.6)
    for index, h in enumerate(histories):
        ax.scatter(
            np.full(len(h["rows"]), index), values[index], color=colors[h["rate"]], s=14, alpha=0.7
        )
        ax.plot(
            [index - 0.15, index + 0.15], [values[index].mean()] * 2, color="black", linewidth=3
        )
    ax.set(
        xticks=np.arange(len(histories)),
        xticklabels=[f"{h['rate']:g}" for h in histories],
        xlabel="Speed-cost strength (units/s)",
        ylabel="Time (s)",
        ylim=(0, 41),
        title=title,
    )
fig.suptitle(
    f"CEM · {len(histories[0]['rows'])} paired starts · 40 s · ±5° upright band\n"
    "Dots: sessions; lines: matched starts; black bars: means",
    fontsize=12,
)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=machine / "hold_metrics.alignment.json")
fig.savefig(human / "hold_metrics.png", dpi=300)
fig.savefig(human / "hold_metrics.pdf")
fig.savefig(human / "hold_metrics.svg")
plt.close(fig)
# Show both effect extremes alongside fixed downward and near-upright examples.
delta = np.array([r["total_hold_s"] for r in histories[-1]["rows"]]) - np.array(
    [r["total_hold_s"] for r in histories[0]["rows"]]
)
lanes = [0, 2, int(delta.argmax()), int(delta.argmin())]
titles = [
    "Fixed downward start",
    "Fixed near-upright start",
    "Largest gain at highest strength",
    "Largest loss at highest strength",
]
fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout="constrained")
for ax, lane, title in zip(axes, lanes, titles, strict=True):
    ax.axhspan(0, 5, color="0.88")
    for h in histories:
        ax.plot(
            np.arange(1, 40001) * 0.001,
            h["error"][lane],
            color=colors[h["rate"]],
            linewidth=0.8,
            alpha=0.85,
            label=f"Strength {h['rate']:g}",
        )
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([0, 5, 10, 30, 90, 180], labels=["0", "5", "10", "30", "90", "180"])
    ax.set(ylabel="Upright error (°)", ylim=(0, 180), title=f"{title} · episode {lane}")
axes[-1].legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=len(histories), fontsize=10, frameon=False
)
axes[-1].set(xlabel="Time (s)", xlim=(0, 40))
fig.suptitle(
    "Full sessions · shaded ±5° band · angle scale linear to 10°, logarithmic above", fontsize=12
)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=machine / "representative_sessions.alignment.json")
fig.savefig(human / "representative_sessions.png", dpi=300)
fig.savefig(human / "representative_sessions.pdf")
fig.savefig(human / "representative_sessions.svg")
plt.close(fig)
(machine / "representative_selection.json").write_text(
    json.dumps(dict(episodes=lanes, reasons=titles), indent=2)
)
(machine / "speed_report.py").write_bytes(Path(__file__).read_bytes())
print(f"Report data and figures: {output}", flush=True)
