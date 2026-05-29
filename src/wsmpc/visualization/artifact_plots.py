"""Static diagnostic figures generated from completed episode records."""

from __future__ import annotations

from typing import Any

import numpy as np

from wsmpc.environment.dynamics import upright_energy
from wsmpc.mpc.ip_dynamics_natural_period.features import phase_proxy_error
from wsmpc.utils.config_schema import EnvironmentConfig
from wsmpc.utils.messages import StepRecord

DIAGNOSTIC_PHASE_EPSILON = 1.0e-6


def create_artifact_figures(
    records: list[StepRecord],
    environment: EnvironmentConfig,
) -> dict[str, Any]:
    """Create the standard static figures saved with each experiment run."""

    import matplotlib.pyplot as plt

    data = _records_to_arrays(records)
    figures = {
        "states": _create_states_figure(plt, data),
        "energy": _create_energy_figure(plt, data, upright_energy(environment.pendulum)),
        "phase": _create_phase_figure(plt, data, environment),
        "commands": _create_commands_figure(plt, data),
    }
    return figures


def _records_to_arrays(records: list[StepRecord]) -> dict[str, np.ndarray]:
    """Pack typed records into contiguous arrays for plotting and derived features."""

    return {
        "t_sec": np.fromiter((record.t_sec for record in records), dtype=np.float64),
        "theta_rad": np.fromiter((record.theta_rad for record in records), dtype=np.float64),
        "omega_rad_s": np.fromiter((record.omega_rad_s for record in records), dtype=np.float64),
        "energy_j": np.fromiter((record.energy_j for record in records), dtype=np.float64),
        "energy_error_j": np.fromiter(
            (record.energy_error_j for record in records), dtype=np.float64
        ),
        "u_commanded_nm": np.fromiter(
            (record.u_commanded_nm for record in records), dtype=np.float64
        ),
        "u_applied_nm": np.fromiter((record.u_applied_nm for record in records), dtype=np.float64),
    }


def _create_states_figure(plt: Any, data: dict[str, np.ndarray]) -> Any:
    """Plot angular position and velocity on aligned time axes."""

    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(data["t_sec"], data["theta_rad"], label="theta")
    axes[0].set_ylabel("theta rad")
    axes[1].plot(data["t_sec"], data["omega_rad_s"], label="omega", color="tab:orange")
    axes[1].set_xlabel("t sec")
    axes[1].set_ylabel("omega rad/s")
    _finish_time_axes(axes)
    figure.suptitle("States")
    figure.tight_layout()
    return figure


def _create_energy_figure(plt: Any, data: dict[str, np.ndarray], target_energy_j: float) -> Any:
    """Plot total energy and upright-relative energy error over time."""

    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(data["t_sec"], data["energy_j"], label="energy")
    axes[0].axhline(target_energy_j, linestyle="--", color="black", label="upright")
    axes[0].set_ylabel("energy J")
    axes[1].plot(data["t_sec"], data["energy_error_j"], label="energy error")
    axes[1].axhline(0.0, linestyle="--", color="black", label="upright")
    axes[1].set_xlabel("t sec")
    axes[1].set_ylabel("energy error J")
    _finish_time_axes(axes)
    figure.suptitle("Energy")
    figure.tight_layout()
    return figure


def _create_phase_figure(
    plt: Any,
    data: dict[str, np.ndarray],
    environment: EnvironmentConfig,
) -> Any:
    """Plot the MPC phase proxy error trajectory in phase coordinates."""

    states = np.column_stack((data["theta_rad"], data["omega_rad_s"]))
    phase_error = phase_proxy_error(
        states,
        environment.pendulum,
        epsilon_phi=DIAGNOSTIC_PHASE_EPSILON,
    )
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(phase_error[:, 0], phase_error[:, 1], label="trajectory")
    axis.scatter([phase_error[-1, 0]], [phase_error[-1, 1]], label="final", zorder=3)
    axis.scatter([0.0], [0.0], marker="*", s=120, label="upright", zorder=4)
    axis.set_xlabel("c_phi - 1")
    axis.set_ylabel("s_phi")
    axis.set_title("Phase")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def _create_commands_figure(plt: Any, data: dict[str, np.ndarray]) -> Any:
    """Plot commanded and applied torque histories on the same time axis."""

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(data["t_sec"], data["u_commanded_nm"], linestyle="--", label="commanded")
    axis.plot(data["t_sec"], data["u_applied_nm"], label="applied")
    axis.set_xlabel("t sec")
    axis.set_ylabel("torque N m")
    axis.set_title("Commands")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def _finish_time_axes(axes: Any) -> None:
    """Apply common grid and legend formatting to stacked time plots."""

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
