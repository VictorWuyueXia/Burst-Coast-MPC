"""Audit sampled angle-only captures and retain compact paired-study evidence."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("pilot", "confirmation", "tuning", "fresh"), required=True
    )
    stage = parser.parse_args().stage
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    runtime = root / (
        "artifacts/rotary_pendulum/experiment-results/"
        "angle-only-goal-and-integration-step-comparison/raw-runs"
    )
    machine = (
        root
        / (
            "artifacts/rotary_pendulum/experiment-results/"
            "angle-only-goal-and-integration-step-comparison/records/run-records"
        )
        / stage
    )
    human = root / (
        "artifacts/rotary_pendulum/experiment-results/"
        "angle-only-goal-and-integration-step-comparison/figures"
    )
    machine.mkdir(parents=True, exist_ok=True)
    human.mkdir(parents=True, exist_ok=True)
    initials, summaries, episodes, audits = {}, [], [], []
    for dt in (20, 10):
        study = runtime / f"dt{dt}_{stage}"
        settings = json.loads((study / "machine-scannables/study.json").read_text())
        (machine / f"dt{dt}_study.json").write_text(json.dumps(settings, indent=2) + "\n")
        for case in settings["cases"]:
            for seed in settings["seeds"]:
                run = study / "machine-scannables/runs" / f"{case['name']}_{seed}"
                initial = np.load(run / "machine-scannables/initial_states.npz")["x"]
                if seed in initials:
                    np.testing.assert_array_equal(initial, initials[seed])
                initials[seed] = initial
                source = run / case["name"] / "machine-scannables"
                data = np.load(source / "trajectories.npz")
                active = data["physics_active"].transpose(0, 2, 1).reshape(-1, len(initial))
                state = data["physics_x"].transpose(0, 2, 1, 3).reshape(-1, len(initial), 4)
                error = np.arctan2(np.sin(state[..., 1] - np.pi), np.cos(state[..., 1] - np.pi))
                inside = active & (abs(error) <= np.deg2rad(15))
                ticks = np.arange(1, len(state) + 1)[:, None]
                streak = ticks - np.maximum.accumulate(np.where(inside, 0, ticks), axis=0)
                captured = np.any(streak >= round(100 / dt), axis=0)
                np.testing.assert_array_equal(captured, data["success"][-1])
                np.testing.assert_array_equal(
                    data["arm_violation"][-1],
                    np.any(
                        active & (abs(state[..., 0]) >= np.deg2rad(case["arm_limit_deg"])), axis=0
                    ),
                )
                np.testing.assert_allclose(data["time_s"][-1], active.sum(axis=0) * dt / 1000)
                assert np.max(abs(data["torque_nm"])) <= 0.00918 * case["torque_multiplier"] + 1e-12
                final = state[active.sum(axis=0) - 1, np.arange(len(initial))]
                handle = (source / "episodes.csv").open()
                rows = list(csv.DictReader(handle))
                handle.close()
                for lane, row in enumerate(rows):
                    episodes.append(
                        {
                            "physics_step_ms": dt,
                            "case": case["name"],
                            "seed": seed,
                            **row,
                            "final_arm_speed_rad_s": float(final[lane, 2]),
                            "final_pendulum_speed_rad_s": float(final[lane, 3]),
                            "capture_exceeds_old_speed_limits": bool(
                                captured[lane]
                                and (abs(final[lane, 2]) > 0.15 or abs(final[lane, 3]) > 0.20)
                            ),
                        }
                    )
                audits.append(
                    {
                        "physics_step_ms": dt,
                        "case": case["name"],
                        "seed": seed,
                        **json.loads((source / "integration_audit.json").read_text()),
                    }
                )
                if seed == settings["seeds"][0]:
                    selection = json.loads((source / "plot_selection.json").read_text())
                    figure, axes = plt.subplots(4, 4, figsize=(15, 10), layout="constrained")
                    for col, (label, lane) in enumerate(selection.items()):
                        selected = active[:, lane]
                        times = np.arange(1, selected.sum() + 1) * dt / 1000
                        physical = state[selected, lane]
                        applied = np.repeat(data["torque_nm"][:, lane], round(100 / dt))[selected]
                        angle_deg = np.rad2deg(physical[:, 1] % (2 * np.pi)) % 360
                        angle_deg[np.abs(np.diff(angle_deg, prepend=angle_deg[0])) > 180] = np.nan
                        axes[0, col].plot(times, angle_deg)
                        axes[0, col].axhspan(
                            165, 195, color="green", alpha=0.15, label="Goal 165–195°"
                        )
                        axes[0, col].set_ylim(0, 360)
                        axes[0, col].set_yticks([0, 90, 180, 270, 360])
                        axes[0, col].set_title(f"{label}; capture={bool(captured[lane])}")
                        axes[1, col].plot(times, physical[:, 2], label="Arm")
                        axes[1, col].plot(times, physical[:, 3], label="Pendulum")
                        axes[2, col].plot(times, np.degrees(physical[:, 0]))
                        for sign in (-1, 1):
                            axes[2, col].axhline(
                                sign * case["arm_limit_deg"], color="red", linestyle="--"
                            )
                        axes[3, col].step(
                            np.r_[times - dt / 1000, times[-1]],
                            1000 * np.r_[applied, applied[-1]],
                            where="post",
                        )
                        axes[3, col].set_xlabel("Time [s]")
                        for row in range(4):
                            axes[row, col].grid(alpha=0.2)
                    for row, label in enumerate(
                        (
                            "Pendulum [deg; 0 down, 180 up]",
                            "Speed [rad/s]",
                            "Arm angle [deg]",
                            "Torque [mN m]",
                        )
                    ):
                        axes[row, 0].set_ylabel(label)
                    axes[0, 0].legend(fontsize=8)
                    axes[1, 0].legend(fontsize=8)
                    figure.suptitle(f"{case['name']} — ±15° for 100 ms; velocities unconstrained")
                    figure_name = f"{stage}_{case['name']}".replace(
                        "pilot", "initial_parameter_screening"
                    ).replace("fresh", "independent_seed_validation")
                    figure_name = re.sub(r"dt(\d+)", r"integration_\1_milliseconds", figure_name)
                    figure_name = re.sub(r"(?<=_)d(\d+)", r"decision_\1_milliseconds", figure_name)
                    figure_name = re.sub(r"_t([\d.]+)", r"_torque_multiplier_\1", figure_name)
                    figure_name = re.sub(r"_a(\d+)", r"_arm_limit_\1_degrees", figure_name)
                    figure_name = re.sub(r"_h(\d+)", r"_prediction_steps_\1", figure_name)
                    figure_name = re.sub(r"_w([\d.]+)", r"_work_penalty_\1", figure_name)
                    figure.savefig(human / f"{figure_name}.png", dpi=140)
                    plt.close(figure)
                print(
                    f"PASS {stage} {case['name']} seed={seed}: goal, clocks, cap, range, resets",
                    flush=True,
                )
            handle = (study / "machine-scannables/summary.csv").open()
            summaries.extend(
                {"physics_step_ms": dt, **row}
                for row in csv.DictReader(handle)
                if row["case"] == case["name"]
            )
            handle.close()
    for name, rows in (("summary", summaries), ("episodes", episodes)):
        handle = (machine / f"{name}.csv").open("w", newline="")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.close()
    (machine / "integration_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    (machine / "validation.json").write_text(
        json.dumps(
            {
                "goal_recomputed_from_physics_samples": True,
                "paired_resets_including_step_sizes": True,
                "goal_hold_s": 0.1,
                "control_interval_s": 0.1,
                "deadline_s": 20,
                "torque_and_excursion_assertions": True,
                "episodes_including_probes": len(episodes),
            },
            indent=2,
        )
        + "\n"
    )
    if stage in ("pilot", "confirmation", "fresh"):
        figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        for axis, dt in zip(axes, (20, 10), strict=True):
            cases = list(
                dict.fromkeys(row["case"] for row in summaries if row["physics_step_ms"] == dt)
            )
            values = np.array(
                [
                    [
                        next(
                            float(row["captures"]) / float(row["episodes"])
                            for row in summaries
                            if row["case"] == case and row["stratum"] == label
                        )
                        for label in ("downward", "moving", "near", "tight")
                    ]
                    for case in cases
                ]
            )
            axis.imshow(100 * values, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
            axis.set_xticks(range(4), ["Downward", "Moving", "Near upright", "Tight upright"])
            labels = []
            for case in cases:
                row = next(row for row in summaries if row["case"] == case)
                label = (
                    f"{float(row['torque_limit_nm']) / 0.00918:g}× torque / "
                    f"±{float(row['arm_limit_deg']):g}°"
                )
                if stage == "fresh":
                    label += (
                        f"\n{float(row['recovery_steps']) * dt / 1000:g} s horizon; "
                        f"weight {float(row['work_weight']):g}"
                    )
                labels.append(label)
            axis.set_yticks(range(len(cases)), labels)
            axis.set_title(f"{dt} ms integration; capture [%]")
            for row in range(len(cases)):
                for col in range(4):
                    axis.text(
                        col,
                        row,
                        f"{100 * values[row, col]:.1f}",
                        ha="center",
                        va="center",
                        color="white" if values[row, col] > 0.6 else "black",
                    )
        figure.suptitle("Within ±15° for 100 ms; no velocity conditions; 1× torque = 0.00918 N m")
        figure_name = stage.replace("pilot", "initial_parameter_screening").replace(
            "fresh", "independent_seed_validation"
        )
        figure.savefig(human / f"{figure_name}_comparison.png", dpi=160)
        plt.close(figure)
    print(f"DONE {machine}", flush=True)


if __name__ == "__main__":
    main()
