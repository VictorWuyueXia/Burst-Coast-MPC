"""Independently count sustained upright intervals, arm violations and time-budget effects."""

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Audit all completed trials under one wave and plot measured live trajectories."""
    root = Path(sys.argv[1])
    human, machine = root / "human-readables", root / "machine-scannables"
    human.mkdir(parents=True, exist_ok=True)
    summaries = []
    for source in sorted(root.glob("*/machine-scannables/trajectories.npz")):
        settings = json.loads(source.with_name("settings.json").read_text())
        archive = np.load(source)
        records = {key: archive[key] for key in archive.files}
        archive.close()
        states = records["physics_x"].transpose(0, 2, 1, 3).reshape(-1, len(records["labels"]), 4)
        beta = np.arctan2(-np.sin(states[..., 1]), -np.cos(states[..., 1]))
        inside = abs(beta) < np.deg2rad(5)
        bounded = abs(states[..., 0]) < settings["arm_limit_rad"]
        count = np.zeros(inside.shape[1], int)
        longest, bounded_count, bounded_longest = count.copy(), count.copy(), count.copy()
        first = np.full((4, inside.shape[1]), np.inf)
        results = []
        print(f"Auditing {source.parents[1].name}: {len(states)} physical samples", flush=True)
        for index in range(len(states)):
            count = np.where(inside[index], count + 1, 0)
            bounded_count = np.where(inside[index] & bounded[index], bounded_count + 1, 0)
            longest, bounded_longest = (
                np.maximum(longest, count),
                np.maximum(bounded_longest, bounded_count),
            )
            for k, threshold in enumerate([1, 50, 500, 1500]):
                first[k] = np.where(
                    (count >= threshold) & np.isinf(first[k]), (index + 1) * 0.02, first[k]
                )
            if index + 1 not in {2000, 6000, 15000, len(states)}:
                continue
            crossed = ~bounded[: index + 1].all(axis=0)
            final_window = inside[max(0, index + 1 - 500) : index + 1].mean(axis=0)
            torque = records["torque_nm"][: (index + 1) // 5]
            work = np.repeat(torque, 5, axis=0) * states[: index + 1, :, 2] * 0.02
            for lane, label in enumerate(records["labels"]):
                results.append(
                    dict(
                        episode=lane,
                        stratum=str(label),
                        horizon_s=(index + 1) * 0.02,
                        first_upright_s=float(first[0, lane]),
                        first_hold1_s=float(first[1, lane]),
                        first_hold10_s=float(first[2, lane]),
                        first_hold30_s=float(first[3, lane]),
                        longest_hold_s=longest[lane] * 0.02,
                        final_hold_s=count[lane] * 0.02,
                        longest_bounded_hold_s=bounded_longest[lane] * 0.02,
                        arm_violation=bool(crossed[lane]),
                        final_10s_upright_fraction=float(final_window[lane]),
                        peak_arm_deg=float(abs(states[: index + 1, lane, 0]).max()) * 180 / np.pi,
                        torque_rms_mnm=float(np.sqrt(np.mean(torque[:, lane] ** 2)) * 1000),
                        mean_abs_torque_mnm=float(abs(torque[:, lane]).mean() * 1000),
                        final10_torque_rms_mnm=float(
                            np.sqrt(np.mean(torque[-100:, lane] ** 2)) * 1000
                        ),
                        final10_mean_abs_torque_mnm=float(abs(torque[-100:, lane]).mean() * 1000),
                        saturation_fraction=float(
                            (abs(torque[:, lane]) >= 0.999 * settings["torque_limit_nm"]).mean()
                        ),
                        mean_torque_change_mnm=float(abs(np.diff(torque[:, lane])).mean() * 1000),
                        absolute_motor_work_j=float(abs(work[:, lane]).sum()),
                        signed_motor_work_j=float(work[:, lane].sum()),
                    )
                )
        handle = source.with_name("episodes.csv").open("w")
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
        handle.close()
        for horizon in sorted({row["horizon_s"] for row in results}):
            for label in sorted(set(records["labels"]) - {"probe"}):
                rows = [
                    row
                    for row in results
                    if row["horizon_s"] == horizon and row["stratum"] == label
                ]
                summaries.append(
                    dict(
                        run=source.parents[1].name,
                        controller=settings["controller"],
                        cap_mnm=settings["torque_limit_nm"] * 1000,
                        horizon_s=horizon,
                        stratum=label,
                        first_upright_fraction=np.mean(
                            [np.isfinite(r["first_upright_s"]) for r in rows]
                        ),
                        hold1_fraction=np.mean([np.isfinite(r["first_hold1_s"]) for r in rows]),
                        hold10_fraction=np.mean([np.isfinite(r["first_hold10_s"]) for r in rows]),
                        hold30_fraction=np.mean([np.isfinite(r["first_hold30_s"]) for r in rows]),
                        bounded_hold10_fraction=np.mean(
                            [
                                np.isfinite(r["first_hold10_s"]) and not r["arm_violation"]
                                for r in rows
                            ]
                        ),
                        arm_violation_fraction=np.mean([r["arm_violation"] for r in rows]),
                        longest_hold_mean_s=np.mean([r["longest_hold_s"] for r in rows]),
                        final_10s_upright_fraction=np.mean(
                            [r["final_10s_upright_fraction"] for r in rows]
                        ),
                        bounded_settled_fraction=np.mean(
                            [r["final_hold_s"] >= 10 and not r["arm_violation"] for r in rows]
                        ),
                        **{
                            name: np.mean([r[name] for r in rows])
                            for name in [
                                "final_hold_s",
                                "torque_rms_mnm",
                                "mean_abs_torque_mnm",
                                "final10_torque_rms_mnm",
                                "final10_mean_abs_torque_mnm",
                                "saturation_fraction",
                                "mean_torque_change_mnm",
                                "absolute_motor_work_j",
                                "signed_motor_work_j",
                            ]
                        },
                    )
                )
        downward = [
            r
            for r in results
            if r["stratum"] == "downward" and r["horizon_s"] == settings["deadline_s"]
        ]
        ranked = sorted(downward, key=lambda row: row["longest_hold_s"])
        tight = [
            r
            for r in results
            if r["stratum"] == ("recovery" if "recovery" in records["labels"] else "tight")
            and r["horizon_s"] == settings["deadline_s"]
        ]
        selection = [
            ranked[len(ranked) // 2]["episode"],
            ranked[-1]["episode"],
            max(tight, key=lambda row: row["longest_hold_s"])["episode"],
        ]
        figure, axes = plt.subplots(4, 3, figsize=(16, 12), layout="constrained")
        for column, lane in enumerate(selection):
            time = np.arange(1, len(states) + 1) * 0.02
            angle = np.rad2deg(states[:, lane, 1]).copy()
            angle[abs(np.diff(angle, prepend=angle[0])) > 180] = np.nan
            axes[0, column].plot(time, angle, lw=0.8)
            axes[0, column].set_ylabel("Pendulum from down (deg)")
            axes[0, column].set_title(
                f"{records['labels'][lane]}, episode {lane}; "
                f"longest hold {longest[lane] * 0.02:.2f}s"
            )
            error = np.rad2deg(beta[:, lane]).copy()
            error[abs(np.diff(error, prepend=error[0])) > 180] = np.nan
            axes[1, column].plot(time, error, lw=0.8)
            axes[1, column].axhspan(-5, 5, color="green", alpha=0.2)
            axes[1, column].set_ylim(-15, 15)
            axes[1, column].set_ylabel("Upright error (deg), zoom")
            axes[2, column].plot(time, np.rad2deg(states[:, lane, 0]), lw=0.8)
            for bound in [-135, 135]:
                axes[2, column].axhline(bound, color="red", ls="--")
            axes[2, column].set_ylabel("Arm angle (deg)")
            axes[3, column].step(
                np.arange(len(records["torque_nm"])) * 0.1,
                records["torque_nm"][:, lane] * 1000,
                where="post",
                lw=0.8,
            )
            axes[3, column].set_ylabel("Motor torque (mN m)")
            axes[3, column].set_xlabel("Time (s)")
            for axis in axes[:, column]:
                axis.grid(alpha=0.2)
        figure.suptitle(source.parents[1].name + " · live control continues after capture")
        figure.savefig(human / f"{source.parents[1].name}.png", dpi=130)
        for axis in axes.flat:
            axis.set_xlim(settings["deadline_s"] - 10, settings["deadline_s"])
        figure.savefig(human / f"{source.parents[1].name}_last10s.png", dpi=130)
        plt.close(figure)
        source.with_name("plot_selection.json").write_text(json.dumps(selection))
    handle = (machine / "summary.csv").open("w")
    writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)
    handle.close()
    print(f"DONE: {len(summaries)} condition summaries", flush=True)


if __name__ == "__main__":
    main()
