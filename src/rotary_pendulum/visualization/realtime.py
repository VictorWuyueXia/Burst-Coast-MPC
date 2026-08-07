"""Combined realtime mechanism and simulation diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from rotary_pendulum.environment.dynamics import ModelConstants
from rotary_pendulum.utils.config_schema import EpisodeConfig, VisualizationConfig
from rotary_pendulum.utils.messages import StateObservation, StepRecord
from rotary_pendulum.visualization.animation import RotaryPendulumAnimation
from rotary_pendulum.visualization.phase import PHASE_HISTORY_S, oscillator_phase_points


class RealtimeRotaryPendulumPlot:
    """Monitor four-state mechanics and replanning diagnostics in one window."""

    def __init__(
        self,
        config: EpisodeConfig,
        visualization: VisualizationConfig,
        model: ModelConstants,
    ) -> None:
        # Bind immutable model information and allocate dense and replan-level histories.
        self.config = config
        self.model = model
        self.update_every = visualization.update_every
        self.series: dict[str, list[float]] = {
            name: []
            for name in (
                "t_sec",
                "theta_rad",
                "alpha_rad",
                "omega_rad_s",
                "nu_rad_s",
                "kinetic_energy_j",
                "potential_energy_j",
                "energy_j",
                "u_commanded_nm",
                "u_applied_nm",
                "replan_index",
                "replan_t_sec",
                "solve_time_s",
                "hbar",
                "bbar",
            )
        }

        # Import Matplotlib only for visual runs and create persistent artists once.
        import matplotlib.pyplot as plt

        self._plt = plt
        self.figure, self.axes = self._create_figure()
        self.lines, self._hb_scatter = self._create_artists()
        for name in ("angles", "velocities", "torque", "energy", "phase", "replan", "solve"):
            self.axes[name].legend(loc="best", fontsize=8)
        self.animation = RotaryPendulumAnimation(
            config.rotary_pendulum,
            axis=self.axes["mechanism"],
        )
        plt.show(block=False)

    def start(self, observation: StateObservation) -> None:
        """Initialize the 3D mechanism from the reset state."""

        self.animation.start(observation)
        self.figure.canvas.draw_idle()
        self._plt.pause(0.001)

    def add_step(self, observation: StateObservation, record: StepRecord) -> None:
        """Append one dense sample and refresh all views on the display cadence."""

        # Store dense physical signals separately from sparse replan measurements.
        dense_values = {
            "t_sec": observation.t_sec,
            "theta_rad": observation.theta_rad,
            "alpha_rad": observation.alpha_rad,
            "omega_rad_s": observation.omega_rad_s,
            "nu_rad_s": observation.nu_rad_s,
            "kinetic_energy_j": observation.kinetic_energy_j,
            "potential_energy_j": observation.potential_energy_j,
            "energy_j": observation.energy_j,
            "u_commanded_nm": record.u_commanded_nm,
            "u_applied_nm": record.u_applied_nm,
            "replan_index": float(record.replan_index),
        }
        for name, value in dense_values.items():
            self.series[name].append(float(value))

        # Record plan-level timing and normalized action coordinates exactly once per replan.
        if record.replan_flag:
            self.series["replan_t_sec"].append(observation.t_sec)
            self.series["solve_time_s"].append(record.solve_time_s)
            self.series["hbar"].append(record.hbar)
            self.series["bbar"].append(record.bbar)

        self.animation.update(observation, record.u_applied_nm)
        if len(self.series["t_sec"]) % self.update_every == 0:
            self.update()

    def update(self) -> None:
        """Replace artist data from the accumulated histories and redraw the window."""

        if not self.series["t_sec"]:
            return

        # Map the latest ten seconds onto state-derived phase rings rather than link geometry.
        time = np.asarray(self.series["t_sec"], dtype=np.float64)
        phase_start = int(np.searchsorted(time, time[-1] - PHASE_HISTORY_S, side="left"))
        phase_states = np.column_stack(
            (
                np.asarray(self.series["theta_rad"], dtype=np.float64)[phase_start:],
                np.asarray(self.series["alpha_rad"], dtype=np.float64)[phase_start:],
                np.asarray(self.series["omega_rad_s"], dtype=np.float64)[phase_start:],
                np.asarray(self.series["nu_rad_s"], dtype=np.float64)[phase_start:],
            )
        )
        arm_phase_m, pendulum_phase_m = oscillator_phase_points(
            phase_states,
            self.config.rotary_pendulum,
            self.model,
        )

        # Update dense signals through one aligned mapping to keep display semantics auditable.
        line_data = {
            "theta": (time, self.series["theta_rad"]),
            "alpha": (time, self.series["alpha_rad"]),
            "omega": (time, self.series["omega_rad_s"]),
            "nu": (time, self.series["nu_rad_s"]),
            "commanded": (time, self.series["u_commanded_nm"]),
            "applied": (time, self.series["u_applied_nm"]),
            "kinetic": (time, self.series["kinetic_energy_j"]),
            "potential": (time, self.series["potential_energy_j"]),
            "total": (time, self.series["energy_j"]),
            "arm_phase": (arm_phase_m[:, 0], arm_phase_m[:, 1]),
            "arm_phase_current": (
                [arm_phase_m[-1, 0]],
                [arm_phase_m[-1, 1]],
            ),
            "pendulum_phase": (pendulum_phase_m[:, 0], pendulum_phase_m[:, 1]),
            "pendulum_phase_current": (
                [pendulum_phase_m[-1, 0]],
                [pendulum_phase_m[-1, 1]],
            ),
            "replan": (time, self.series["replan_index"]),
            "solve": (self.series["replan_t_sec"], self.series["solve_time_s"]),
        }
        for name, (x_values, y_values) in line_data.items():
            self.lines[name].set_data(x_values, y_values)

        # Update sparse solve-time and (h,b) distributions only at decision epochs.
        hbar = np.asarray(self.series["hbar"], dtype=np.float64)
        bbar = np.asarray(self.series["bbar"], dtype=np.float64)
        self._hb_scatter.set_offsets(np.column_stack((hbar, bbar)))

        # Rescale time-dependent axes while preserving fixed physical action bounds.
        for name, axis in self.axes.items():
            if name not in {"mechanism", "phase", "hb"}:
                axis.relim()
                axis.autoscale_view()
        self.figure.canvas.draw_idle()
        self._plt.pause(0.001)

    def finish(self) -> None:
        """Flush the final simulation sample to the interactive window."""

        self.update()
        self._plt.pause(0.001)

    def _create_figure(self) -> tuple[Any, dict[str, Any]]:
        """Allocate the nine requested views on a compact three-by-three grid."""

        # A regular grid keeps each diagnostic readable without splitting the live context.
        figure = self._plt.figure(figsize=(16, 10), layout="constrained")
        manager = figure.canvas.manager
        assert manager is not None
        manager.set_window_title("Rotary Pendulum Realtime Simulation")
        grid = figure.add_gridspec(3, 3)
        axes = {
            "mechanism": figure.add_subplot(grid[0, 0], projection="3d"),
            "angles": figure.add_subplot(grid[0, 1]),
            "velocities": figure.add_subplot(grid[0, 2]),
            "torque": figure.add_subplot(grid[1, 0]),
            "energy": figure.add_subplot(grid[1, 1]),
            "phase": figure.add_subplot(grid[1, 2]),
            "replan": figure.add_subplot(grid[2, 0]),
            "solve": figure.add_subplot(grid[2, 1]),
            "hb": figure.add_subplot(grid[2, 2]),
        }
        figure.suptitle("QUBE-Servo 3 Rotary-Pendulum Burst-Coast MPC")
        self._format_axes(axes)
        return figure, axes

    def _create_artists(self) -> tuple[dict[str, Any], Any]:
        """Create persistent line and scatter artists for all recorded signals."""

        # Command and reference styles match the established inverted-pendulum artifacts.
        lines = {
            "theta": self.axes["angles"].plot([], [], label="theta arm")[0],
            "alpha": self.axes["angles"].plot([], [], label="alpha pendulum")[0],
            "omega": self.axes["velocities"].plot([], [], label="omega arm")[0],
            "nu": self.axes["velocities"].plot([], [], label="nu pendulum")[0],
            "commanded": self.axes["torque"].plot([], [], linestyle="--", label="commanded")[0],
            "applied": self.axes["torque"].plot([], [], label="actual")[0],
            "kinetic": self.axes["energy"].plot([], [], label="swing kinetic")[0],
            "potential": self.axes["energy"].plot([], [], label="swing potential")[0],
            "total": self.axes["energy"].plot([], [], linewidth=2.0, label="swing total")[0],
            "arm_phase": self.axes["phase"].plot(
                [], [], color="tab:blue", alpha=0.5, label="arm oscillator phase"
            )[0],
            "arm_phase_current": self.axes["phase"].plot(
                [], [], marker="o", linestyle="", color="tab:blue", alpha=1.0, label="arm current"
            )[0],
            "pendulum_phase": self.axes["phase"].plot(
                [], [], color="tab:orange", alpha=0.5, label="pendulum oscillator phase"
            )[0],
            "pendulum_phase_current": self.axes["phase"].plot(
                [],
                [],
                marker="o",
                linestyle="",
                color="tab:orange",
                alpha=1.0,
                label="pendulum current",
            )[0],
            "replan": self.axes["replan"].step([], [], where="post", label="active plan")[0],
            "solve": self.axes["solve"].plot([], [], marker="o", markersize=4, label="MPC solve")[
                0
            ],
        }
        self.axes["angles"].axhline(np.pi, linestyle=":", color="0.3", label="upright")
        self.axes["energy"].axhline(
            2.0 * self.model.gravity_torque_nm,
            linestyle=":",
            color="0.3",
            label="upright target",
        )
        hb_scatter = self.axes["hb"].scatter([], [], color="tab:purple", s=30)
        return lines, hb_scatter

    def _format_axes(self, axes: dict[str, Any]) -> None:
        """Apply explicit names, units, grids, and legends to every diagnostic axis."""

        # Labels distinguish arm and pendulum states and define normalized action coordinates.
        labels = {
            "angles": ("Angular States", "t s", "angle rad"),
            "velocities": ("Angular Velocities", "t s", "rad/s"),
            "torque": ("Torque", "t s", "N m"),
            "energy": ("Pendulum-Relative Swing Energy", "t s", "J"),
            "phase": (
                "Oscillator Phase Plane (latest 10 s)",
                "radius cos(phase) m",
                "radius sin(phase) m",
            ),
            "replan": ("Replan Timeline", "t s", "replan index"),
            "solve": ("MPC Solve Time", "replan t s", "solve time s"),
            "hb": ("(h, b) Distribution", "h = H Ts / Tn", "b = B / H"),
        }
        for name, (title, x_label, y_label) in labels.items():
            axes[name].set_title(title)
            axes[name].set_xlabel(x_label)
            axes[name].set_ylabel(y_label)
            axes[name].grid(True, alpha=0.25)
            axes[name].margins(x=0.03, y=0.08)
        axes["hb"].set_xlim(-0.05, 3.05)
        axes["hb"].set_ylim(-0.03, 1.03)
        rotation_limit_m = 1.1 * max(
            self.config.rotary_pendulum.arm_length_m,
            self.config.rotary_pendulum.pendulum_length_m,
        )
        axes["phase"].set_xlim(-rotation_limit_m, rotation_limit_m)
        axes["phase"].set_ylim(-rotation_limit_m, rotation_limit_m)
        axes["phase"].set_aspect("equal", adjustable="box")
