"""Recompute sampled goals and collect complete campaign tables and readable figures."""

import csv
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rotary_pendulum.heuristic.artifacts import write_artifacts

root = Path(__file__).resolve().parents[3] / (
    "artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/raw-runs"
)
output = root.parent
machine, human = output / "records", output / "figures"
human.mkdir(exist_ok=True)
rows, episodes, audits, checks = [], [], [], []
selections = {}
boundary_events = []
for stage in (
    "screening",
    "arm_penalty",
    "refinement",
    "low_gain",
    "final_refinement",
    "confirmation",
    "validation_20261022",
    "validation_20261023",
    "fine_plant",
):
    campaign = json.loads((root / stage / "machine-scannables/campaign.json").read_text())
    initial = np.load(root / stage / "machine-scannables/initial_states.npz")
    labels = initial["labels"]
    records = machine / "run-records" / stage
    records.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / stage / "machine-scannables/provenance.json", records / "provenance.json")
    for trial in campaign["trials"]:
        name = trial["name"]
        source = root / stage / name / "machine-scannables"
        print("Audit", stage, name, flush=True)
        data = np.load(source / "trajectories.npz")
        archive = data
        data = {key: archive[key] for key in archive.files if key != "allow_pickle"}
        archive.close()
        for key in ("start_x", "x", "physics_x"):
            data[key][..., 1] = (data[key][..., 1] + np.pi) % (2 * np.pi) - np.pi
        data["pendulum_angle_deg"] = np.rad2deg(data["physics_x"][..., 1])
        x, active = data["physics_x"], data["physics_active"]
        inside = active & (np.abs(data["pendulum_angle_deg"]) >= 165)
        inside = inside.transpose(0, 2, 1).reshape(-1, len(labels))
        ticks = np.arange(1, len(inside) + 1)[:, None]
        streak = ticks - np.maximum.accumulate(np.where(inside, 0, ticks), axis=0)
        np.testing.assert_array_equal((streak >= 5).any(axis=0), data["success"][-1])
        np.testing.assert_array_equal(
            ((abs(x[..., 0]) >= np.pi) & active).any(axis=(0, 2)), data["arm_violation"][-1]
        )
        assert np.all((x[..., 1] >= -np.pi) & (x[..., 1] < np.pi))
        complete = np.isclose(data["elapsed_s"], 0.1)
        checks.append(
            {
                "stage": stage,
                "case": name,
                "goal_recomputed": True,
                "arm_flags_recomputed": True,
                "angle_range_pass": True,
                "max_100ms_predicted_actual_work_difference_j": float(
                    abs(data["predicted_work_j"][complete] - data["work_j"][complete]).max()
                ),
                "max_100ms_predicted_actual_energy_difference_j": float(
                    abs(data["predicted_energy_j"][complete] - data["energy_j"][complete]).max()
                ),
            }
        )
        audit = json.loads((source / "integration_audit.json").read_text())
        audits.append({"stage": stage, "case": name, **audit})
        handle = (source / "episodes.csv").open()
        case_rows = list(csv.DictReader(handle))
        handle.close()
        episodes.extend(
            {"stage": stage, "case": name, "seed": campaign["seed"], **row} for row in case_rows
        )
        for label in dict.fromkeys(labels):
            selected = labels == label
            success, crossed = data["success"][-1, selected], data["arm_violation"][-1, selected]
            peak = np.where(active, abs(x[..., 2]), 0).max(axis=(0, 2))[selected]
            final = abs(data["x"][-1, selected, 2])
            rows.append(
                {
                    "stage": stage,
                    "case": name,
                    "seed": campaign["seed"],
                    "stratum": str(label),
                    "count": int(selected.sum()),
                    "captures": int(success.sum()),
                    "crossings": int(crossed.sum()),
                    "clean_captures": int((success & ~crossed).sum()),
                    "median_duration_s": float(np.median(data["time_s"][-1, selected])),
                    "median_peak_arm_speed_rad_s": float(np.median(peak)),
                    "median_final_arm_speed_rad_s": float(np.median(final)),
                    "p95_final_arm_speed_rad_s": float(np.quantile(final, 0.95)),
                    "maximum_arm_angle_deg": float(
                        np.rad2deg(abs(x[..., 0][:, selected])[active[:, selected]]).max()
                    ),
                    "median_final_total_energy_j": float(
                        np.median(data["energy_j"][-1, selected].sum(-1))
                    ),
                    **{key: trial[key] for key in ("work_gain", "work_weight", "arm_limit_weight")},
                }
            )
        if (
            stage in ("confirmation", "validation_20261022", "validation_20261023")
            and name == "gain_0.004_work_0.02"
        ):
            for lane in np.flatnonzero(data["arm_violation"][-1]):
                crossed = (abs(x[:, lane, :, 0]) >= np.pi) & active[:, lane]
                decision, sample = np.argwhere(crossed)[0]
                before = data["start_x"][decision, lane]
                boundary_events.append(
                    {
                        "seed": campaign["seed"],
                        "lane": int(lane),
                        "time_s": float(decision * 0.1 + (sample + 1) * 0.02),
                        "start_arm_deg": float(np.rad2deg(before[0])),
                        "start_arm_speed_rad_s": float(before[2]),
                        "bounded_torque_options": int(data["bounded_count"][decision, lane]),
                        "previous_predicted_peak_deg": float(
                            np.rad2deg(data["predicted_peak_arm_rad"][decision - 1, lane])
                        ),
                        "peak_arm_deg": float(np.rad2deg(abs(x[:, lane, :, 0])).max()),
                    }
                )
        target = records / name
        target.mkdir(exist_ok=True)
        for filename in (
            "summary.csv",
            "episodes.csv",
            "integration_audit.json",
            "settings.json",
            "plot_selection.json",
        ):
            shutil.copy2(source / filename, target / filename)
        if (
            stage in ("validation_20261022", "validation_20261023", "fine_plant")
            and name == "gain_0.004_work_0.02"
        ):
            chosen = json.loads((source / "plot_selection.json").read_text())
            crossings = np.flatnonzero(data["arm_violation"][-1]).tolist()
            selections[stage] = {"plot_lanes": chosen, "crossing_lanes": crossings}
            lanes = sorted(set(chosen.values()) | set(crossings))
            np.savez_compressed(
                target / "representatives.npz",
                initial_x=np.column_stack(
                    (
                        initial["x"][lanes, 0],
                        (initial["x"][lanes, 1] + np.pi) % (2 * np.pi) - np.pi,
                        initial["x"][lanes, 2:],
                    )
                ),
                labels=labels[lanes],
                lanes=np.array(lanes),
                **{key: data[key][:, lanes] for key in data if key != "allow_pickle"},
            )
            temporary = Path(tempfile.mkdtemp(prefix="single-action-plots-"))
            write_artifacts(
                temporary / "session",
                {key: data[key] for key in data if key != "allow_pickle"},
                labels.tolist(),
                {**campaign, **trial},
            )
            shutil.copy2(
                temporary / "session/human-readables/representative_sessions.png",
                human
                / (
                    "two_millisecond_plant_validation_trajectories.png"
                    if stage == "fine_plant"
                    else f"validation_seed_{stage.removeprefix('validation_')}_trajectories.png"
                ),
            )
            shutil.rmtree(temporary)
    initial.close()

for filename, data in (("summary.csv", rows), ("episodes.csv", episodes)):
    handle = (machine / filename).open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(data[0]))
    writer.writeheader()
    writer.writerows(data)
    handle.close()
(machine / "boundary_events.json").write_text(json.dumps(boundary_events, indent=2) + "\n")
(machine / "integration_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
(machine / "representation_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
(machine / "plot_selection.json").write_text(json.dumps(selections, indent=2) + "\n")
selected = [
    r
    for r in rows
    if r["stage"].startswith("validation_")
    and r["case"] == "gain_0.004_work_0.02"
    and r["stratum"] != "probe"
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
labels = ("downward", "moving", "near", "tight")
x = np.arange(4)
captures = [sum(r["captures"] for r in selected if r["stratum"] == label) for label in labels]
crossings = [sum(r["crossings"] for r in selected if r["stratum"] == label) for label in labels]
axes[0].bar(x, captures, color="#288569", label="Captured")
axes[0].bar(x, crossings, color="#c35541", label="Had an arm-limit crossing")
axes[0].set(
    xticks=x,
    xticklabels=["Downward", "Moving", "Near upright", "Tight upright"],
    ylabel="Episodes / 1,024 per group",
    ylim=(0, 1150),
)
for i, (capture, crossing) in enumerate(zip(captures, crossings, strict=True)):
    axes[0].text(
        i, capture + 18, f"{capture} captured\n{crossing} crossed", ha="center", fontsize=9
    )
axes[0].legend(loc="lower right")
peak, final = [], []
for seed in (20261022, 20261023):
    p = root / f"validation_{seed}" / "gain_0.004_work_0.02/machine-scannables/trajectories.npz"
    d = np.load(p)
    peak.extend(
        np.where(d["physics_active"][:, :512], abs(d["physics_x"][:, :512, :, 2]), 0).max(
            axis=(0, 2)
        )
    )
    final.extend(abs(d["x"][-1, :512, 2]))
    d.close()
axes[1].boxplot([peak, final], tick_labels=["Peak during swing-up", "At capture"], showfliers=True)
axes[1].set(ylabel="Absolute arm speed [rad/s]", title="1,024 downward starts; no speed cap")
fig.suptitle("Selected controller: independent validation seeds 20261022–20261023")
fig.savefig(human / "validation.png", dpi=150)
plt.close(fig)
print(
    "Recorded",
    len(episodes),
    "episode evaluations;",
    len(checks),
    "case runs;",
    "down median peak/final",
    np.median(peak),
    np.median(final),
    flush=True,
)
