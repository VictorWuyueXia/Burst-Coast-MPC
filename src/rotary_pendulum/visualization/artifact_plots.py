"""Static rotary-pendulum diagnostics generated from completed episodes."""

from __future__ import annotations

from typing import Any

import numpy as np

from rotary_pendulum.environment.dynamics import ModelConstants
from rotary_pendulum.utils.config_schema import RotaryPendulumConfig
from rotary_pendulum.utils.messages import StepRecord
from rotary_pendulum.visualization.phase import PHASE_HISTORY_S, oscillator_phase_points


def create_artifact_figures(
    records: list[StepRecord],
    physical: RotaryPendulumConfig,
    model: ModelConstants,
) -> dict[str, Any]:
    """Create the five standard physics and MPC figures for one completed run."""

    if not records:
        raise ValueError("Rotary-pendulum artifact figures require at least one step record")

    import matplotlib.pyplot as plt

    # Pack dense and decision-level signals once so every figure uses aligned samples.
    replans = [record for record in records if record.replan_flag]
    data = {
        "t_sec": np.fromiter((record.t_sec for record in records), dtype=np.float64),
        "theta_rad": np.fromiter((record.theta_rad for record in records), dtype=np.float64),
        "alpha_rad": np.fromiter((record.alpha_rad for record in records), dtype=np.float64),
        "omega_rad_s": np.fromiter((record.omega_rad_s for record in records), dtype=np.float64),
        "nu_rad_s": np.fromiter((record.nu_rad_s for record in records), dtype=np.float64),
        "kinetic_energy_j": np.fromiter(
            (record.kinetic_energy_j for record in records), dtype=np.float64
        ),
        "potential_energy_j": np.fromiter(
            (record.potential_energy_j for record in records), dtype=np.float64
        ),
        "energy_j": np.fromiter((record.energy_j for record in records), dtype=np.float64),
        "energy_error_j": np.fromiter(
            (record.energy_error_j for record in records), dtype=np.float64
        ),
        "u_commanded_nm": np.fromiter(
            (record.u_commanded_nm for record in records), dtype=np.float64
        ),
        "u_applied_nm": np.fromiter((record.u_applied_nm for record in records), dtype=np.float64),
        "replan_index": np.fromiter((record.replan_index for record in records), dtype=np.int64),
        "replan_t_sec": np.fromiter((record.t_sec for record in replans), dtype=np.float64),
        "solve_time_s": np.fromiter((record.solve_time_s for record in replans), dtype=np.float64),
        "hbar": np.fromiter((record.hbar for record in replans), dtype=np.float64),
        "bbar": np.fromiter((record.bbar for record in replans), dtype=np.float64),
        "objective_value": np.fromiter(
            (record.objective_value for record in replans), dtype=np.float64
        ),
    }

    return {
        "states": _create_states_figure(plt, data),
        "energy": _create_energy_figure(plt, data, 2.0 * model.gravity_torque_nm),
        "phase": _create_phase_figure(plt, data, physical, model),
        "commands": _create_commands_figure(plt, data),
        "mpc_diagnostics": _create_mpc_diagnostics_figure(plt, data),
    }


def _create_states_figure(plt: Any, data: dict[str, np.ndarray]) -> Any:
    """Plot both generalized coordinates and angular velocities over time."""

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    axes[0].plot(data["t_sec"], data["theta_rad"], label="theta arm")
    axes[0].plot(data["t_sec"], data["alpha_rad"], label="alpha pendulum")
    axes[0].axhline(np.pi, linestyle=":", color="0.3", label="pendulum upright")
    axes[0].set_ylabel("angle rad")
    axes[1].plot(data["t_sec"], data["omega_rad_s"], label="omega arm")
    axes[1].plot(data["t_sec"], data["nu_rad_s"], label="nu pendulum")
    axes[1].set_xlabel("time s")
    axes[1].set_ylabel("angular velocity rad/s")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    figure.suptitle("Rotary-Pendulum States")
    return figure


def _create_energy_figure(
    plt: Any,
    data: dict[str, np.ndarray],
    target_energy_j: float,
) -> Any:
    """Plot pendulum-relative swing energy and its upright-target error."""

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    axes[0].plot(data["t_sec"], data["kinetic_energy_j"], label="swing kinetic")
    axes[0].plot(data["t_sec"], data["potential_energy_j"], label="swing potential")
    axes[0].plot(data["t_sec"], data["energy_j"], linewidth=2.0, label="swing total")
    axes[0].axhline(target_energy_j, linestyle=":", color="0.3", label="upright target")
    axes[0].set_ylabel("energy J")
    axes[1].plot(data["t_sec"], data["energy_error_j"], label="swing total minus target")
    axes[1].axhline(0.0, linestyle=":", color="0.3", label="target")
    axes[1].set_xlabel("time s")
    axes[1].set_ylabel("raw energy error J")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    figure.suptitle("Pendulum-Relative Swing Energy")
    return figure


def _create_phase_figure(
    plt: Any,
    data: dict[str, np.ndarray],
    physical: RotaryPendulumConfig,
    model: ModelConstants,
) -> Any:
    """Plot arm and pendulum oscillator phases at their physical radii."""

    # Convert the final ten seconds to phase coordinates while retaining physical link radii.
    trajectory_start = int(
        np.searchsorted(data["t_sec"], data["t_sec"][-1] - PHASE_HISTORY_S, side="left")
    )
    phase_states = np.column_stack(
        (
            data["theta_rad"][trajectory_start:],
            data["alpha_rad"][trajectory_start:],
            data["omega_rad_s"][trajectory_start:],
            data["nu_rad_s"][trajectory_start:],
        )
    )
    arm_phase_m, pendulum_phase_m = oscillator_phase_points(phase_states, physical, model)
    figure, axis = plt.subplots(figsize=(7, 7), layout="constrained")
    axis.plot(
        arm_phase_m[:, 0],
        arm_phase_m[:, 1],
        color="tab:blue",
        alpha=0.5,
        label="arm oscillator phase",
    )
    axis.plot(
        pendulum_phase_m[:, 0],
        pendulum_phase_m[:, 1],
        color="tab:orange",
        alpha=0.5,
        label="pendulum oscillator phase",
    )
    axis.scatter(
        [arm_phase_m[-1, 0]],
        [arm_phase_m[-1, 1]],
        color="tab:blue",
        label="arm current",
        zorder=3,
    )
    axis.scatter(
        [pendulum_phase_m[-1, 0]],
        [pendulum_phase_m[-1, 1]],
        color="tab:orange",
        label="pendulum current",
        zorder=3,
    )
    limit_m = 1.1 * max(physical.arm_length_m, physical.pendulum_length_m)
    axis.set_xlim(-limit_m, limit_m)
    axis.set_ylim(-limit_m, limit_m)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("radius cos(phase) m")
    axis.set_ylabel("radius sin(phase) m")
    axis.set_title("Arm and Pendulum Oscillator Phase Plane")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    return figure


def _create_commands_figure(plt: Any, data: dict[str, np.ndarray]) -> Any:
    """Plot commanded and applied shaft torques on one time axis."""

    figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    axis.plot(
        data["t_sec"],
        data["u_commanded_nm"],
        linestyle="--",
        label="commanded torque",
    )
    axis.plot(data["t_sec"], data["u_applied_nm"], label="applied torque")
    axis.set_xlabel("time s")
    axis.set_ylabel("shaft torque N m")
    axis.set_title("Torque Commands")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    return figure


def _create_mpc_diagnostics_figure(plt: Any, data: dict[str, np.ndarray]) -> Any:
    """Plot active plans, solve latency, selected dimensions, and objective values."""

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    axes[0, 0].step(data["t_sec"], data["replan_index"], where="post", label="active plan")
    axes[0, 0].set_xlabel("time s")
    axes[0, 0].set_ylabel("replan index")
    axes[0, 0].set_title("Plan Execution Timeline")
    axes[0, 1].plot(
        data["replan_t_sec"], data["solve_time_s"], marker="o", label="total candidate solve"
    )
    axes[0, 1].set_xlabel("replan time s")
    axes[0, 1].set_ylabel("solve time s")
    axes[0, 1].set_title("MPC Solve Time")
    axes[1, 0].scatter(data["hbar"], data["bbar"], c=data["replan_t_sec"], s=32)
    axes[1, 0].set_xlabel("hbar = H Ts / Tn")
    axes[1, 0].set_ylabel("bbar = B / H")
    axes[1, 0].set_title("Selected Burst-Coast Dimensions")
    axes[1, 1].plot(
        data["replan_t_sec"], data["objective_value"], marker="o", label="selected objective"
    )
    axes[1, 1].set_xlabel("replan time s")
    axes[1, 1].set_ylabel("objective value")
    axes[1, 1].set_title("Selected MPC Objective")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best")
    figure.suptitle("Burst-Coast MPC Diagnostics")
    return figure
