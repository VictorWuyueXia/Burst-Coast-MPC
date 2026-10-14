"""Human-readable physical trajectory panels for residual policy evaluations."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICS_DT_S
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, GOAL
from rotary_pendulum.RL.jax_residual_control import TARGET_ENERGY_J

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_trajectories(
    directory: Path,
    data: Mapping[str, Any],
    rows: list[dict[str, Any]],
    settings: Mapping[str, Any],
) -> None:
    """Select reproducible episodes and plot state, torque, energy and capture memory."""
    human, machine = directory / "human-readables", directory / "machine-scannables"
    x = data["physics_x"]
    energy = (
        0.5 * MODEL.pendulum_inertia_kg_m2 * x[..., 3] ** 2
        + MODEL.gravity_torque_nm * (1.0 - np.cos(x[..., 1]))
    ) / TARGET_ENERGY_J
    goal_degrees = np.rad2deg(
        0.08 if settings["goal_mode"] == "strict" else GOAL.beta_tolerance_rad
    )
    # Pick a median-return and best-return case at the longest deadline from each random stratum;
    # the chosen lane IDs are saved so that plots are never anonymous anecdotes.
    selected: dict[str, list[int]] = {}
    for label in ("downward", "moving", "near", "tight", "probe"):
        lanes = [
            i
            for i, row in enumerate(rows)
            if row["stratum"] == label and row["deadline_s"] == max(data["deadline_s"])
        ]
        ranked = sorted(lanes, key=lambda i: rows[i]["return"])
        chosen = ranked if label == "probe" else [ranked[len(ranked) // 2], ranked[-1]]
        selected[label] = chosen
        figure, axes = plt.subplots(6, len(chosen), figsize=(6 * len(chosen), 14), squeeze=False)
        for column, lane in enumerate(chosen):
            active = data["active"][:, lane]
            time = data["time_s"][active, lane]
            # Include all 20 ms physics samples for state traces; torque stays at 100 ms.
            fine_time = (time[:, None] + 0.02 * np.arange(1, 6)).reshape(-1)
            state = x[active, lane].reshape(-1, 4)
            angle = np.rad2deg((state[:, 1] + np.pi) % (2 * np.pi) - np.pi)
            angle[np.abs(np.diff(angle, prepend=angle[0])) > 180] = np.nan
            axes[0, column].plot(fine_time, state[:, 0], label="arm θ")
            for bound in (-float(ARM_LIMIT_RAD), float(ARM_LIMIT_RAD)):
                axes[0, column].axhline(bound, color="red", ls=":")
            axes[0, column].set_ylabel("Arm position (rad)")
            axes[1, column].plot(fine_time, angle, label="Pendulum position")
            for edge in (-180, 180):
                axes[1, column].axhspan(
                    max(-180, edge - goal_degrees),
                    min(180, edge + goal_degrees),
                    color="green",
                    alpha=0.15,
                )
            axes[1, column].set_ylim(-180, 180)
            axes[1, column].set_ylabel("Pendulum [deg; 0 down, ±180 up]")
            axes[2, column].plot(fine_time, state[:, 2], label="arm ω")
            axes[2, column].plot(fine_time, state[:, 3], label="pendulum ν")
            axes[2, column].set_ylabel("Velocity (rad/s)")
            axes[3, column].plot(
                fine_time,
                energy[active, lane].reshape(-1),
                label="Relative hinge-swing energy / target",
            )
            axes[3, column].axhline(1, color="black", ls=":")
            axes[3, column].set_ylabel("Relative swing energy / target")
            axes[4, column].step(
                time,
                1000 * data["heuristic_nm"][active, lane],
                where="post",
                alpha=0.5,
                label="heuristic",
            )
            axes[4, column].step(
                time, 1000 * data["applied_nm"][active, lane], where="post", label="applied"
            )
            axes[4, column].set_ylim(1100 * settings["torque_limit_nm"] * np.array([-1, 1]))
            axes[4, column].set_ylabel("Torque (mN m)")
            axes[5, column].step(
                time, data["filter_mode"][active, lane], where="post", label="filter mode"
            )
            hold = (
                5
                if settings["goal_mode"] == "strict"
                else round(settings["hold_duration_s"] / PHYSICS_DT_S)
            )
            axes[5, column].plot(
                time + data["elapsed_s"][active, lane],
                data["hold_count"][active, lane] / hold,
                label="hold",
            )
            axes[5, column].set_ylabel("Mode / hold fraction")
            axes[5, column].set_xlabel("Time (s)")
            axes[0, column].set_title(f"{label}, lane {lane}: success={rows[lane]['success']}")
            for axis in axes[:, column]:
                axis.grid(alpha=0.25)
                axis.legend(fontsize=8, loc="upper right")
        figure.tight_layout()
        figure.savefig(human / f"{label}_trajectories.png", dpi=130)
        for axis in axes.flat:
            axis.set_xlim(0, 5)
        figure.savefig(human / f"{label}_first5s.png", dpi=130)
        plt.close(figure)
    (machine / "plot_selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    (human / "interpretation_summary.md").write_text(
        "# Validation plots\n\n"
        "Angles: θ is the unwrapped arm position; "
        "α is pendulum position in −180–180 degrees, measured from downward; "
        "β is wrapped angle from upright. Velocities ω and ν belong to arm and pendulum. "
        "E/E* uses hinge-relative swing energy, excluding arm-carried kinetic energy; "
        "it is a control quantity, not the pendulum body's full physical energy.\n\n"
        "Filter modes: 0 accept, 1 coast, 2 brake, 3 least predicted excursion when no "
        "candidate stays inside the arm bounds. Red lines show the configured arm bounds. "
        f"Hold fraction 1 means {hold} consecutive 20 ms goal samples and early success. "
        f"Goal mode: {settings['goal_mode']}; exact tolerances are in campaign documentation.\n\n"
        "Each random stratum shows median-return and best-return longest-deadline cases. "
        "Probe plots show all deterministic probes. Complete traces, selections and "
        "episode outcomes are in ../machine-scannables/."
    )
