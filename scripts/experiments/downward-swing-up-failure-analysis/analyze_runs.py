"""Classify failures from physics samples and compare paired Monte Carlo interventions."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    rows, audits, selections = [], [], []
    for campaign_path in arguments.runs:
        campaign = json.loads((campaign_path / "campaign.json").read_text())
        for case in campaign["cases"]:
            for seed in campaign["seeds"]:
                name = f"{case['name']}_{seed}"
                root = campaign_path / "records" / name
                print(f"Analyzing {name}", flush=True)
                initial = np.load(campaign_path / f"initial_{seed}.npz")
                data = dict(np.load(root / "trajectories.npz"))
                provenance = json.loads((root / "provenance.json").read_text())
                audit = json.loads((root / "integration_audit.json").read_text())
                audits.append({"case": case["name"], "seed": seed, **audit})
                retained = output / "provenance" / name
                retained.mkdir(parents=True, exist_ok=True)
                for filename in ("provenance.json", "integration_audit.json", "summary.json"):
                    shutil.copy2(root / filename, retained / filename)
                shutil.copy2(campaign_path / f"{name}.log", retained / "run.log")
                shutil.copy2(
                    campaign_path / "sources" / case["name"] / "experiment.patch",
                    retained / "experiment.patch",
                )
                count = len(initial["x"])
                physical = data["physics_x"].transpose(0, 2, 1, 3).reshape(1000, count, 4)
                sample_active = data["physics_active"].transpose(0, 2, 1).reshape(1000, count)
                active = data["elapsed_s"] > 0
                denominator = active.sum(axis=0)
                beta = np.arctan2(
                    np.sin(physical[..., 1] - np.pi), np.cos(physical[..., 1] - np.pi)
                )
                goal = (abs(beta) <= provenance["goal_angle_rad"]) & sample_active
                potential = (1 - np.cos(physical[..., 1])) / 2
                peak = np.where(sample_active, potential, 0).max(axis=0)
                run_length, longest = np.zeros(count, int), np.zeros(count, int)
                for inside in goal:
                    run_length = np.where(inside, run_length + 1, 0)
                    longest = np.maximum(longest, run_length)
                success = data["success"][-1]
                np.testing.assert_array_equal(longest >= 5, success)
                failure = np.where(
                    success,
                    "captured",
                    np.where(
                        goal.any(axis=0),
                        "brief_visit",
                        np.where(peak < 0.5, "low_lift", "partial_lift"),
                    ),
                )
                target = 0.03037176
                energy = data["energy_j"] / target
                positive = np.maximum(data["work_j"], 0).sum(axis=0)
                negative = -np.minimum(data["work_j"], 0).sum(axis=0)
                total = positive + negative
                recurrence = np.full(count, np.inf)
                lag = np.zeros(count, int)
                # Descriptive tail recurrence, not a proof of a stable limit cycle.
                for offset in range(5, 31):
                    difference = (data["x"][-50:] - data["x"][-50 - offset : -offset]) / [
                        np.pi,
                        np.pi,
                        5,
                        10,
                    ]
                    distance = np.sqrt(np.mean(difference**2, axis=(0, 2)))
                    better = distance < recurrence
                    recurrence = np.minimum(recurrence, distance)
                    lag = np.where(better, offset, lag)
                masks = {
                    "exact_fraction": data["mode"] == 0,
                    "override_fraction": data["mode"] == 1,
                    "unresolved_fraction": data["mode"] == 2,
                    "no_root_fraction": data["root_count"] == 0,
                    "no_bounded_fraction": data["bounded_count"] == 0,
                    "speed_reject_fraction": (data["bounded_count"] > 0)
                    & (data["recoverable_count"] == 0),
                    "saturated_fraction": abs(data["torque_nm"])
                    >= 0.999 * provenance["torque_limit_nm"],
                    "negative_despite_request_fraction": (data["requested_work_j"] > 0)
                    & (data["work_j"] < 0),
                    "exact_energy_gap_fraction": (data["mode"] == 0)
                    & (data["energy_preference_gap"] > 0.01),
                    "unreachable_request_fraction": (
                        data["requested_work_j"] > data["grid_work_max_j"]
                    )
                    | (data["requested_work_j"] < data["grid_work_min_j"]),
                }
                metrics = {
                    "success": success,
                    "excursion": data["arm_violation"][-1],
                    "duration_s": data["time_s"][-1],
                    "peak_potential": peak,
                    "closest_upright_deg": np.rad2deg(
                        np.where(sample_active, abs(beta), np.inf).min(axis=0)
                    ),
                    "longest_goal_samples": longest,
                    "goal_visits_samples": goal.sum(axis=0),
                    "peak_arm_deg": np.rad2deg(
                        np.where(sample_active, abs(physical[..., 0]), 0).max(axis=0)
                    ),
                    "tail_total_energy": energy[-50:].sum(axis=-1).mean(axis=0),
                    "tail_arm_energy": energy[-50:, :, 0].mean(axis=0),
                    "tail_potential": energy[-50:, :, 2].mean(axis=0),
                    "tail_recurrence_rms": recurrence,
                    "tail_recurrence_lag_s": lag * 0.1,
                    "positive_work_j": positive,
                    "negative_work_j": negative,
                    "cancelled_work_fraction": np.divide(
                        2 * np.minimum(positive, negative),
                        total,
                        out=np.zeros(count),
                        where=total > 0,
                    ),
                    "max_cumulative_energy_error_j": abs(
                        np.cumsum(data["energy_balance_j"], axis=0)
                    ).max(axis=0),
                    **{
                        key: (mask & active).sum(axis=0) / denominator
                        for key, mask in masks.items()
                    },
                }
                for lane, label in enumerate(initial["labels"]):
                    rows.append(
                        {
                            "case": case["name"],
                            "stage": campaign_path.name,
                            "seed": seed,
                            "lane": lane,
                            "stratum": str(label),
                            "failure": str(failure[lane]),
                            **{key: value[lane].item() for key, value in metrics.items()},
                            **dict(
                                zip(
                                    ("theta0", "alpha0", "omega0", "nu0"),
                                    initial["x"][lane].tolist(),
                                    strict=True,
                                )
                            ),
                        }
                    )
                if case["name"] == "baseline":
                    chosen = {}
                    for category in ("low_lift", "partial_lift", "brief_visit", "captured"):
                        candidates = np.flatnonzero(
                            (initial["labels"] == "downward") & (failure == category)
                        )
                        if len(candidates) and category not in {
                            item["category"] for item in selections
                        }:
                            order = candidates[np.argsort(peak[candidates])]
                            chosen[category] = int(order[len(order) // 2])
                    if seed == 20261020:
                        chosen["exact_rest_probe"] = count - 5
                    for category, lane in chosen.items():
                        record = next(
                            row
                            for row in rows
                            if row["case"] == "baseline"
                            and row["seed"] == seed
                            and row["lane"] == lane
                        )
                        selections.append({"category": category, **record})
                        np.savez_compressed(
                            output / f"representative_{category}.npz",
                            **{key: value[:, lane] for key, value in data.items()},
                        )
                if case["name"] == "speed05" and seed == 20261020:
                    rescued = [
                        row
                        for row in rows
                        if row["case"] == "baseline"
                        and row["seed"] == seed
                        and row["stratum"] == "downward"
                        and row["failure"] == "low_lift"
                        and success[row["lane"]]
                    ]
                    rescued.sort(key=lambda row: row["peak_potential"])
                    before = rescued[len(rescued) // 2]
                    lane = before["lane"]
                    after = next(
                        row
                        for row in rows
                        if row["case"] == "speed05" and row["seed"] == seed and row["lane"] == lane
                    )
                    reference = dict(
                        np.load(
                            arguments.runs[0] / "records" / f"baseline_{seed}" / "trajectories.npz"
                        )
                    )
                    for category, record, trace in (
                        ("paired_baseline", before, reference),
                        ("paired_relaxed", after, data),
                    ):
                        selections.append({"category": category, **record})
                        np.savez_compressed(
                            output / f"representative_{category}.npz",
                            **{key: value[:, lane] for key, value in trace.items()},
                        )
                    del reference
                del data, physical

    summary = []
    for case in dict.fromkeys(row["case"] for row in rows):
        for label in ("downward", "rest", "wide_down", "moving", "near", "tight", "probe"):
            chosen = [row for row in rows if row["case"] == case and row["stratum"] == label]
            n = len(chosen)
            captured = sum(row["success"] for row in chosen)
            rate = captured / n
            center = (rate + 1.96**2 / (2 * n)) / (1 + 1.96**2 / n)
            radius = (
                1.96 * np.sqrt(rate * (1 - rate) / n + 1.96**2 / (4 * n * n)) / (1 + 1.96**2 / n)
            )
            summary.append(
                {
                    "case": case,
                    "stratum": label,
                    "episodes": n,
                    "captures": captured,
                    "capture_pct": 100 * rate,
                    "wilson95_low_pct": 100 * (center - radius),
                    "wilson95_high_pct": 100 * (center + radius),
                    "excursions": sum(row["excursion"] for row in chosen),
                    "clean_captures": sum(
                        row["success"] and not row["excursion"] for row in chosen
                    ),
                    **{
                        category: sum(row["failure"] == category for row in chosen)
                        for category in ("low_lift", "partial_lift", "brief_visit")
                    },
                }
            )
    baseline = {(row["seed"], row["lane"]): row for row in rows if row["case"] == "baseline"}
    paired = []
    for case in dict.fromkeys(row["case"] for row in rows if row["case"] != "baseline"):
        for label in ("downward", "rest", "wide_down", "moving", "near", "tight"):
            selected = [row for row in rows if row["case"] == case and row["stratum"] == label]
            pairs = [(baseline[(row["seed"], row["lane"])], row) for row in selected]
            for before, after in pairs:
                assert all(
                    before[key] == after[key] for key in ("theta0", "alpha0", "omega0", "nu0")
                )
            paired.append(
                {
                    "case": case,
                    "stratum": label,
                    "episodes": len(pairs),
                    "baseline_captures": sum(before["success"] for before, _ in pairs),
                    "case_captures": sum(after["success"] for _, after in pairs),
                    "rescued": sum(
                        not before["success"] and after["success"] for before, after in pairs
                    ),
                    "lost": sum(
                        before["success"] and not after["success"] for before, after in pairs
                    ),
                    "baseline_excursions": sum(before["excursion"] for before, _ in pairs),
                    "case_excursions": sum(after["excursion"] for _, after in pairs),
                }
            )
    for name, records in (("episodes", rows), ("summary", summary), ("paired", paired)):
        if not records:
            continue
        handle = (output / f"{name}.csv").open("w", newline="")
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        handle.close()
    (output / "integration_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    (output / "plot_selection.json").write_text(json.dumps(selections, indent=2) + "\n")
    (output / "campaign_paths.json").write_text(
        json.dumps([str(path) for path in arguments.runs], indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
