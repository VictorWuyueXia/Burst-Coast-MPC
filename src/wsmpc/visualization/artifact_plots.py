"""Static diagnostic figures generated from completed episode records."""

from __future__ import annotations

from typing import Any

import numpy as np

from wsmpc.environment.dynamics import upright_energy
from wsmpc.mpc.ip_dynamics_natural_period.features import phase_proxy_error
from wsmpc.utils.config_schema import EnvironmentConfig
from wsmpc.utils.messages import RLStepRecord, StepRecord

DIAGNOSTIC_PHASE_EPSILON = 1.0e-6


def create_artifact_figures(
    records: list[StepRecord],
    environment: EnvironmentConfig,
) -> dict[str, Any]:
    """Create the standard static figures saved with each experiment run."""

    import matplotlib.pyplot as plt

    # Pack records once so every diagnostic uses the same aligned arrays.
    data = _records_to_arrays(records)
    figures = {
        "states": _create_states_figure(plt, data),
        "energy": _create_energy_figure(plt, data, upright_energy(environment.pendulum)),
        "phase": _create_phase_figure(plt, data, environment),
        "commands": _create_commands_figure(plt, data),
    }
    return figures


def create_rl_timeseries_figure(records: list[RLStepRecord]) -> Any:
    """Create the replanning-level Monte Carlo diagnostic figure."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    # Build the Monte Carlo figure from replanning-level arrays, not dense step rows.
    return _create_rl_timeseries_figure(
        plt,
        FuncFormatter,
        MaxNLocator,
        _rl_records_to_arrays(records),
    )


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


def _rl_records_to_arrays(records: list[RLStepRecord]) -> dict[str, np.ndarray]:
    """Pack RL transition records into arrays while preserving per-segment torque traces."""

    return {
        "replan_index": np.fromiter((record.replan_index for record in records), dtype=np.int64),
        "start_t_sec": np.fromiter((record.start_t_sec for record in records), dtype=np.float64),
        "end_t_sec": np.fromiter((record.end_t_sec for record in records), dtype=np.float64),
        "bbar": np.fromiter((record.bbar for record in records), dtype=np.float64),
        "hbar": np.fromiter((record.hbar for record in records), dtype=np.float64),
        "step_cost": np.fromiter((record.step_cost for record in records), dtype=np.float64),
        "return_cost": np.fromiter((record.return_cost for record in records), dtype=np.float64),
        "solve_time_s": np.fromiter((record.solve_time_s for record in records), dtype=np.float64),
    }


def _create_states_figure(plt: Any, data: dict[str, np.ndarray]) -> Any:
    """Plot angular position and velocity on aligned time axes."""

    # Position and velocity share time alignment but keep independent physical scales.
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

    # Energy diagnostics expose both absolute energy and upright-relative error.
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

    # Phase coordinates are recomputed from dense state rows for formulation inspection.
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


def _create_rl_timeseries_figure(
    plt: Any,
    func_formatter: Any,
    max_n_locator: Any,
    data: dict[str, np.ndarray],
) -> Any:
    """Plot replanning-level Monte Carlo fields as aligned diagnostic time series."""

    figure, axes = plt.subplots(4, 1, figsize=(10, 11))
    x = data["replan_index"]
    start_t_sec = data["start_t_sec"]
    duration_s = data["end_t_sec"] - start_t_sec

    # The sampled Monte Carlo action is inspected directly in normalized Bbar-Hbar space.
    axes[0].scatter(data["bbar"], data["hbar"], c=start_t_sec, s=28)
    axes[0].set_xlabel("bbar")
    axes[0].set_ylabel("hbar")
    axes[0].set_box_aspect(1.0)

    # Horizontal floating bars show each transition interval without vertical gaps.
    axes[1].barh(x, duration_s, left=start_t_sec, height=1.0, color="tab:blue", alpha=0.75)
    axes[1].set_ylabel("replan index")
    axes[1].yaxis.set_major_locator(max_n_locator(integer=True))

    # Solver latency is compared against the action coordinates on the transition time axis.
    axes[2].plot(start_t_sec, data["bbar"], marker=".", linewidth=1.0, label="bbar")
    axes[2].plot(start_t_sec, data["hbar"], marker=".", linewidth=1.0, label="hbar")
    axes[2].set_ylabel("bbar / hbar")
    solve_axis = axes[2].twinx()
    solve_axis.set_zorder(axes[2].get_zorder() - 1)
    axes[2].patch.set_visible(False)
    solve_axis.bar(
        start_t_sec,
        data["solve_time_s"],
        width=0.8 * _time_bar_width(start_t_sec),
        label="solve-time-s",
        alpha=0.5,
    )
    solve_axis.set_ylabel("solve-time-s")

    # Immediate and Monte Carlo return costs share one scale for direct comparison.
    axes[3].plot(start_t_sec, data["step_cost"], marker=".", linewidth=1.0, label="step-cost")
    axes[3].plot(start_t_sec, data["return_cost"], marker=".", linewidth=1.0, label="return-cost")
    axes[3].set_xlabel("t sec")
    axes[3].set_ylabel("cost")
    axes[2].sharex(axes[1])
    axes[3].sharex(axes[1])
    axes[1].xaxis.set_major_locator(max_n_locator(integer=True))
    axes[1].xaxis.set_major_formatter(func_formatter(lambda value, _: f"{value:.0f}"))

    _finish_rl_axes(axes, solve_axis)
    solve_lines, solve_labels = solve_axis.get_legend_handles_labels()
    action_lines, action_labels = axes[2].get_legend_handles_labels()
    axes[2].legend(action_lines + solve_lines, action_labels + solve_labels, loc="upper right")
    axes[3].legend(loc="best")
    # figure.suptitle("RL Transition Time Series")
    figure.tight_layout()
    return figure


def _time_bar_width(t_sec: np.ndarray) -> float:
    """Choose a visible bar width from the transition spacing."""

    # A singleton transition still receives a visible finite-width bar.
    unique_t_sec = np.unique(t_sec)
    if unique_t_sec.size < 2:
        return 0.1
    return float(np.min(np.diff(unique_t_sec)))


def _finish_rl_axes(axes: Any, solve_axis: Any) -> None:
    """Disable grids across the RL diagnostic panels and their secondary axis."""

    for axis in axes:
        axis.grid(False)
    solve_axis.grid(False)


def _create_commands_figure(plt: Any, data: dict[str, np.ndarray]) -> Any:
    """Plot commanded and applied torque histories on the same time axis."""

    # Commanded and applied torque are kept separate to reveal saturation.
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


def _finish_time_axes(axes: Any, *, include_legend: bool = True) -> None:
    """Apply common grid and legend formatting to stacked time plots."""

    for axis in axes:
        axis.grid(True, alpha=0.25)
        if include_legend:
            axis.legend(loc="best")
