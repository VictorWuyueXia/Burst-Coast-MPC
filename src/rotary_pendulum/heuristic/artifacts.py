"""Separate numerical records and readable figures for energy-work rollouts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from rotary_pendulum.heuristic.energy import TARGET_ENERGY_J

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_artifacts(
    output: Path, traces: dict[str, np.ndarray], labels: list[str], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Audit every lane, save complete traces, and plot representative physical paths."""

    machine, human = output / "machine-scannables", output / "human-readables"
    machine.mkdir(parents=True, exist_ok=False)
    human.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(machine / "trajectories.npz", allow_pickle=False, **traces)
    (machine / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    rows: list[dict[str, Any]] = []
    for lane, label in enumerate(labels):
        active = traces["elapsed_s"][:, lane] > 0
        complete = np.isclose(traces["elapsed_s"][:, lane], 0.1)
        solved = complete & (traces["mode"][:, lane] == 0)
        sample_active = traces["physics_active"][:, lane]
        samples = traces["physics_x"][:, lane]
        beta = np.arctan2(np.sin(samples[..., 1] - np.pi), np.cos(samples[..., 1] - np.pi))
        approach = (
            (abs(beta) <= 0.35)
            & (abs(samples[..., 2]) <= 2.0)
            & (abs(samples[..., 3]) <= 2.0)
            & sample_active
        )
        energy = traces["energy_j"][:, lane] / TARGET_ENERGY_J
        cost = 0.5 * np.sum((energy - [0, 0, 1]) ** 2, axis=-1)
        speeds = np.concatenate((traces["start_x"][0, lane, 2:3], samples[..., 2][sample_active]))
        signs = np.sign(speeds[abs(speeds) > 0.15])
        in_bound_hold = traces["success"][-1, lane] and abs(traces["x"][-1, lane, 0]) <= np.pi / 2
        rows.append(
            {
                "lane": lane,
                "stratum": label,
                "success": bool(traces["success"][-1, lane]),
                "in_bound_capture": bool(in_bound_hold),
                "arm_violation": bool(traces["arm_violation"][-1, lane]),
                "duration_s": float(traces["time_s"][-1, lane]),
                "approach": bool(np.any(approach)),
                "closest_upright_rad": float(np.min(abs(beta)[sample_active])),
                "minimum_energy_cost": float(np.min(cost[active])),
                "final_energy_cost": float(cost[np.flatnonzero(active)[-1]]),
                "peak_arm_rad": float(np.max(abs(samples[..., 0])[sample_active])),
                "beyond_arm_s": float(
                    0.02 * np.sum((abs(samples[..., 0]) > np.pi / 2) & sample_active)
                ),
                "recovery_unresolved_fraction": float(np.mean(traces["mode"][active, lane] == 2)),
                "work_override_fraction": float(np.mean(traces["mode"][active, lane] == 1)),
                "numerical_reject_fraction": float(np.mean(traces["mode"][active, lane] == 3)),
                "arm_reversals": int(np.sum(signs[1:] != signs[:-1])),
                "maximum_potential_fraction": float(np.max(energy[active, 2])),
                "zero_work_nonzero_torque_count": int(
                    np.sum(
                        active
                        & (traces["requested_work_j"][:, lane] == 0)
                        & (abs(traces["torque_nm"][:, lane]) > 1e-6)
                    )
                ),
                "total_absolute_work_mismatch_j": float(
                    np.sum(abs(traces["work_mismatch_j"][active, lane]))
                ),
                "coast_fraction": float(np.mean(abs(traces["torque_nm"][active, lane]) < 1e-10)),
                "absolute_work_j": float(np.sum(abs(traces["work_j"][active, lane]))),
                "max_work_error_j": float(
                    np.max(
                        np.where(
                            solved,
                            abs(traces["work_j"][:, lane] - traces["requested_work_j"][:, lane]),
                            0,
                        )
                    )
                ),
                "max_energy_balance_j": float(
                    np.max(abs(traces["energy_balance_j"][active, lane]))
                ),
            }
        )
    handle = (machine / "episodes.csv").open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    handle.close()
    summary = []
    for label in dict.fromkeys(labels):
        selected = [row for row in rows if row["stratum"] == label]
        summary.append(
            {
                "stratum": label,
                "episodes": len(selected),
                **{
                    name: float(np.mean([row[name] for row in selected]))
                    for name in rows[0]
                    if name not in ("lane", "stratum")
                },
            }
        )
    handle = (machine / "summary.csv").open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
    handle.close()

    figure, axes = plt.subplots(6, 4, figsize=(18, 17), layout="constrained")
    selection = {}
    for column, label in enumerate(("downward", "moving", "near", "tight")):
        indices = [row["lane"] for row in rows if row["stratum"] == label]
        ordered = sorted(indices, key=lambda lane: rows[lane]["minimum_energy_cost"])
        lane = ordered[len(ordered) // 2]
        selection[label] = lane
        active = traces["elapsed_s"][:, lane] > 0
        time = traces["time_s"][active, lane]
        samples = traces["physics_x"][active, lane]
        sample_active = traces["physics_active"][active, lane]
        sample_time = (
            time[:, None]
            - traces["elapsed_s"][active, lane, None]
            + 0.02 * np.arange(1, 6)[None, :]
        )
        flattened_time = sample_time[sample_active]
        physical = samples[sample_active]
        beta = np.arctan2(np.sin(physical[:, 1] - np.pi), np.cos(physical[:, 1] - np.pi))
        axes[0, column].plot(flattened_time, abs(beta))
        axes[0, column].axhline(0.08, color="green", linestyle="--", label="Capture angle")
        axes[0, column].set_title(f"{label}: lane {lane}; capture={rows[lane]['success']}")
        axes[1, column].plot(flattened_time, physical[:, 2], label="Arm speed")
        axes[1, column].plot(flattened_time, physical[:, 3], label="Pendulum speed")
        energy = traces["energy_j"][active, lane] / TARGET_ENERGY_J
        for index, name in enumerate(("Arm kinetic", "Pendulum kinetic", "Potential")):
            axes[2, column].plot(time, energy[:, index], label=name)
        axes[2, column].plot(time, energy.sum(axis=-1), "k--", label="Total")
        axes[2, column].axhline(1, color="gray", linewidth=0.7)
        axes[3, column].plot(flattened_time, physical[:, 0])
        axes[3, column].axhline(np.pi / 2, color="red", linestyle="--", label="Soft bound")
        axes[3, column].axhline(-np.pi / 2, color="red", linestyle="--")
        start_time = time - traces["elapsed_s"][active, lane]
        axes[4, column].step(start_time, 1000 * traces["torque_nm"][active, lane], where="post")
        axes[5, column].plot(
            time, 1000 * traces["requested_work_j"][active, lane], label="Requested"
        )
        axes[5, column].plot(time, 1000 * traces["work_j"][active, lane], label="Delivered")
        axes[5, column].set_xlabel("Time [s]")
        for row in range(6):
            axes[row, column].grid(alpha=0.25)
    for row, label in enumerate(
        (
            "Distance from upright [rad]",
            "Signed speed [rad/s]",
            "Energy / upright target",
            "Arm angle [rad]",
            "Applied torque [mN m]",
            "Work per decision [mJ]",
        )
    ):
        axes[row, 0].set_ylabel(label)
    for row in (0, 1, 2, 3, 5):
        axes[row, 0].legend(fontsize=8)
    figure.suptitle(f"{settings['name']} — lossless physics, energy/work heuristic", fontsize=14)
    figure.savefig(human / "representative_sessions.png", dpi=120)
    plt.close(figure)
    (machine / "plot_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return summary
