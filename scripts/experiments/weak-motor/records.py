"""Audit recorded physical torque, work and swing motion without rerunning a policy."""

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import find_peaks

from rotary_pendulum.environment.dynamics import energy_components, rk4_step
from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Write per-episode metrics and six deterministic representative trace panels."""
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    cap = float(sys.argv[3])
    machine, human = output / "machine-scannables", output / "human-readables"
    machine.mkdir(parents=True, exist_ok=True)
    human.mkdir(parents=True, exist_ok=True)
    archive = np.load(source)
    records = {key: archive[key] for key in archive.files}
    archive.close()
    print("Loaded physical traces; auditing episodes", flush=True)
    metrics, series = [], {}
    for episode, label in enumerate(records["labels"]):
        if episode % 256 == 0:
            print(f"Auditing episode {episode}", flush=True)
        count = round(float(records["duration_s"][episode]) / 0.02)
        state = np.concatenate(
            (records["x"][:1, episode], records["physics_x"][:, episode].reshape(-1, 4)[:count])
        )
        torque = np.repeat(records["applied_nm"][:, episode], 5)[:count].astype(float)
        time = np.arange(count + 1) * 0.02
        beta = np.arctan2(-np.sin(state[:, 1]), -np.cos(state[:, 1]))
        hits = np.flatnonzero(np.abs(beta) < np.deg2rad(5))
        first = int(hits[0]) if hits.size else count
        alpha = np.unwrap(state[:, 1])
        extrema = np.sort(
            np.concatenate(
                (
                    find_peaks(alpha[: first + 1], prominence=np.deg2rad(10))[0],
                    find_peaks(-alpha[: first + 1], prominence=np.deg2rad(10))[0],
                )
            )
        )
        crossings = np.diff(np.floor(alpha[: first + 1] / (2 * np.pi)))
        cycles = max(0, max(np.count_nonzero(crossings > 0), np.count_nonzero(crossings < 0)) - 1)
        kinetic, potential, energy = energy_components(state, PHYSICAL, MODEL)
        work = torque * np.diff(state[:, 0])
        cumulative = np.concatenate(([0.0], np.cumsum(work)))
        hold, longest = 0, 0
        for inside in np.abs(beta[1:]) < np.deg2rad(5):
            hold = hold + 1 if inside else 0
            longest = max(longest, hold)
        row = {
            "episode": episode,
            "stratum": str(label),
            "success": bool(records["success"][episode]),
            "arm_violation": bool(records["arm_violation"][episode]),
            "capture_or_timeout_s": time[-1],
            "first_upright_s": time[first] if hits.size else None,
            "longest_upright_hold_s": longest * 0.02,
            "peak_torque_mnm": float(np.max(np.abs(torque))) * 1000,
            "rms_torque_mnm": float(np.sqrt(np.mean(torque**2))) * 1000,
            "saturated_time_fraction": float(np.mean(np.abs(torque) >= 0.95 * cap)),
            "pre_upright_saturation_fraction": float(np.mean(np.abs(torque[:first]) >= 0.95 * cap))
            if first
            else 0,
            "positive_work_mj": float(np.maximum(work, 0).sum()) * 1000,
            "negative_work_mj": float(np.minimum(work, 0).sum()) * 1000,
            "pre_upright_positive_work_mj": float(np.maximum(work[:first], 0).sum()) * 1000,
            "peak_motor_power_mw": float(np.max(np.abs(work / 0.02))) * 1000,
            "energy_balance_max_error_mj": float(np.max(np.abs(energy - energy[0] - cumulative)))
            * 1000,
            "pre_upright_turning_points": len(extrema),
            "complete_down_crossing_cycles": int(cycles),
            "peak_arm_deg": float(np.max(np.abs(state[:, 0]))) * 180 / np.pi,
            "peak_arm_speed_rad_s": float(np.max(np.abs(state[:, 2]))),
            "peak_pendulum_speed_rad_s": float(np.max(np.abs(state[:, 3]))),
        }
        assert row["peak_torque_mnm"] <= cap * 1000 + 1e-5
        assert row["success"] == (longest >= 50)
        metrics.append(row)
        series[episode] = (
            time,
            state,
            torque,
            beta,
            kinetic,
            potential,
            energy,
            cumulative,
            extrema,
        )
    downward = [row for row in metrics if row["stratum"] == "downward"]
    ordered = sorted(downward, key=lambda row: (row["capture_or_timeout_s"], row["episode"]))
    arm = [row for row in downward if row["arm_violation"]]
    selection = [
        ("Fastest downward", ordered[0]["episode"]),
        ("Median downward", ordered[len(ordered) // 2]["episode"]),
        ("Slowest downward", ordered[-1]["episode"]),
        ("Largest arm excursion", max(downward, key=lambda row: row["peak_arm_deg"])["episode"]),
        ("Moving start", next(row["episode"] for row in metrics if row["stratum"] == "moving")),
        (
            "Exact downward rest",
            next(row["episode"] for row in metrics if row["stratum"] == "probe"),
        ),
    ]
    writer_file = (machine / "episodes.csv").open("w")
    writer = csv.DictWriter(writer_file, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)
    writer_file.close()
    complete = np.argwhere(np.isclose(records["elapsed_s"], 0.1))
    chosen = complete[np.linspace(0, len(complete) - 1, min(256, len(complete)), dtype=int)]
    start = records["x"][chosen[:, 0], chosen[:, 1]].astype(float)
    fine = start.copy()
    applied = records["applied_nm"][chosen[:, 0], chosen[:, 1]]
    for _ in range(50):
        fine = rk4_step(fine, applied, 0.002, PHYSICAL, MODEL)
    delta = fine - records["physics_x"][chosen[:, 0], chosen[:, 1], -1]
    delta[:, 1] = np.arctan2(np.sin(delta[:, 1]), np.cos(delta[:, 1]))
    fine_balance = (
        energy_components(fine, PHYSICAL, MODEL)[2]
        - energy_components(start, PHYSICAL, MODEL)[2]
        - applied * (fine[:, 0] - start[:, 0])
    )
    summary = {
        "fine_action_replay": {
            "count": len(chosen),
            "dt_s": 0.002,
            "max_state_error": np.max(abs(delta), axis=0).tolist(),
            "max_energy_balance_mj": float(np.max(abs(fine_balance))) * 1000,
        },
        "source": str(source),
        "torque_limit_mnm": cap * 1000,
        "upright_potential_mj": 2000 * MODEL.gravity_torque_nm,
        "downward_episodes": len(downward),
        "downward_arm_crossings": len(arm),
        "selected": [{"title": title, **metrics[index]} for title, index in selection],
        "downward_mean": {
            key: float(np.mean([row[key] for row in downward]))
            for key in metrics[0]
            if key not in ("episode", "stratum") and all(row[key] is not None for row in downward)
        },
        "turning_point_histogram": {
            str(n): sum(row["pre_upright_turning_points"] == n for row in downward)
            for n in sorted({row["pre_upright_turning_points"] for row in downward})
        },
    }
    (machine / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for group in range(2):
        figure, axes = plt.subplots(5, 3, figsize=(17, 14), constrained_layout=True)
        for column, (title, episode) in enumerate(selection[group * 3 : group * 3 + 3]):
            time, state, torque, beta, kinetic, potential, energy, work, extrema = series[episode]
            row = metrics[episode]
            event = "Capture" if row["success"] else "Timeout"
            first_label = (
                str(row["first_upright_s"]) if row["first_upright_s"] is not None else "never"
            )
            axes[0, column].set_title(
                f"{title} · episode {episode}\n{event} {time[-1]:.2f} s; "
                f"first upright {first_label} s"
            )
            axes[0, column].plot(
                time, np.rad2deg(np.unwrap(state[:, 1])), label="Pendulum from down"
            )
            axes[0, column].scatter(
                time[extrema],
                np.rad2deg(np.unwrap(state[:, 1]))[extrema],
                s=22,
                color="black",
                label="Swing turning point",
            )
            bounds = np.rad2deg(np.unwrap(state[:, 1]))
            for target in range(
                int(np.floor(bounds.min() / 360)) * 360 - 180,
                int(np.ceil(bounds.max() / 360)) * 360 + 181,
                360,
            ):
                axes[0, column].axhspan(target - 5, target + 5, color="green", alpha=0.12)
            axes[0, column].set_ylabel("Pendulum angle (deg)")
            display_beta = np.rad2deg(beta).copy()
            display_beta[np.abs(np.diff(display_beta, prepend=display_beta[0])) > 180] = np.nan
            axes[1, column].plot(
                time, display_beta, color="tab:green", label="Signed upright error"
            )
            axes[1, column].axhspan(-5, 5, color="green", alpha=0.15)
            axes[1, column].set_ylim(-20, 20)
            axes[1, column].set_ylabel("Upright error (deg), zoom")
            axes[2, column].step(
                time, np.r_[torque, 0] * 1000, where="post", label="Applied torque"
            )
            axes[2, column].axhline(cap * 1000, color="red", ls="--", label="Motor cap")
            axes[2, column].axhline(-cap * 1000, color="red", ls="--")
            axes[2, column].set_ylabel("Torque (mN m)")
            axes[3, column].plot(time, np.rad2deg(state[:, 0]), label="Arm angle")
            axes[3, column].axhline(180, color="red", ls="--", label="Arm boundary")
            axes[3, column].axhline(-180, color="red", ls="--")
            axes[3, column].set_ylabel("Arm angle (deg)")
            axes[4, column].plot(time, energy * 1000, label="Total mechanical energy")
            axes[4, column].plot(time, potential * 1000, label="Pendulum potential")
            axes[4, column].plot(
                time, (work + energy[0]) * 1000, ls="--", label="Initial energy + motor work"
            )
            axes[4, column].axhline(
                2000 * MODEL.gravity_torque_nm, color="grey", ls=":", label="Upright potential"
            )
            axes[4, column].set_ylabel("Energy (mJ)")
            axes[4, column].set_xlabel("Time (s)")
            for ax in axes[:, column]:
                if row["success"]:
                    ax.axvspan(max(0, time[-1] - 1), time[-1], color="green", alpha=0.06)
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7, loc="best")
        figure.suptitle(
            f"Selected TD3 policy · motor cap {cap * 1000:.2f} mN m · "
            "pale final band = required 1 s hold",
            fontsize=15,
        )
        figure.savefig(human / f"representative_episodes_{group + 1}.png", dpi=150)
        plt.close(figure)
    print(
        json.dumps({key: value for key, value in summary.items() if key != "selected"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
