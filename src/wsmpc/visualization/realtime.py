"""Realtime Matplotlib plots and pendulum animation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from wsmpc.mpc.ip_dynamics_natural_period.features import phase_proxy_error
from wsmpc.utils.config_schema import PendulumConfig
from wsmpc.utils.messages import StateObs, StepRecord
from wsmpc.visualization.animation import PendulumAnimation

DIAGNOSTIC_PHASE_EPSILON = 1.0e-6


@dataclass
class _SeriesBuffer:
    """In-memory time series used by the realtime diagnostics figure."""

    t_sec: list[float] = field(default_factory=list)
    theta_rad: list[float] = field(default_factory=list)
    omega_rad_s: list[float] = field(default_factory=list)
    kinetic_energy_j: list[float] = field(default_factory=list)
    potential_energy_j: list[float] = field(default_factory=list)
    energy_j: list[float] = field(default_factory=list)
    energy_error_j: list[float] = field(default_factory=list)
    phase_c_error: list[float] = field(default_factory=list)
    phase_s: list[float] = field(default_factory=list)
    constraint_margin: list[float] = field(default_factory=list)
    goal_flag: list[float] = field(default_factory=list)
    u_commanded_nm: list[float] = field(default_factory=list)
    u_applied_nm: list[float] = field(default_factory=list)


class RealtimeEpisodePlot:
    """Update grouped diagnostic plots from emitted step records."""

    def __init__(
        self,
        pendulum: PendulumConfig,
        *,
        update_every: int = 1,
        include_animation: bool = False,
    ) -> None:
        # Bind pendulum constants, refresh cadence, and the in-memory plotting buffer.
        self.pendulum = pendulum
        self.update_every = max(1, update_every)
        self.include_animation = include_animation
        self.buffer = _SeriesBuffer()

        # Import Matplotlib lazily so non-visual runs stay dependency-light at import time.
        import matplotlib.pyplot as plt

        self._plt = plt
        self.kinetic_goal_j = 0.0
        self.potential_goal_j = 2.0 * pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m
        self.phase_goal = (0.0, 0.0)
        # Build all figure artists once so step updates only replace data arrays.
        self.figure, self.axes = self._create_figure()
        self._lines = self._create_lines()
        self.animation = (
            PendulumAnimation(
                pendulum,
                update_every=update_every,
                axis=self.axes["animation"],
                plt_module=plt,
            )
            if include_animation
            else None
        )
        self.figure.tight_layout()
        plt.show(block=False)

    def add_step(self, observation: StateObs, record: StepRecord) -> None:
        """Append one record and refresh the plot on the configured cadence."""

        # 1. Append dense record fields into column-wise diagnostic buffers.
        self.buffer.t_sec.append(record.t_sec)
        self.buffer.theta_rad.append(record.theta_rad)
        self.buffer.omega_rad_s.append(record.omega_rad_s)
        self.buffer.kinetic_energy_j.append(self.kinetic_energy(record.omega_rad_s, self.pendulum))
        self.buffer.potential_energy_j.append(
            self.potential_energy(record.theta_rad, self.pendulum)
        )
        self.buffer.energy_j.append(record.energy_j)
        self.buffer.energy_error_j.append(record.energy_error_j)
        phase_error = phase_proxy_error(
            [record.theta_rad, record.omega_rad_s],
            self.pendulum,
            epsilon_phi=DIAGNOSTIC_PHASE_EPSILON,
        )
        self.buffer.phase_c_error.append(float(phase_error[0]))
        self.buffer.phase_s.append(float(phase_error[1]))
        self.buffer.constraint_margin.append(record.constraint_margin)
        self.buffer.goal_flag.append(1.0 if record.goal_flag else 0.0)
        self.buffer.u_commanded_nm.append(record.u_commanded_nm)
        self.buffer.u_applied_nm.append(record.u_applied_nm)
        # 2. Forward the same observation to the optional embedded animation.
        if self.animation is not None:
            self.animation.add_step(observation, record)

        # 3. Refresh Matplotlib on the configured sample cadence.
        if len(self.buffer.t_sec) % self.update_every == 0:
            self.update()

    def start_animation(self, observation: StateObs) -> None:
        """Initialize the embedded animation from the reset observation."""

        if self.animation is not None:
            self.animation.start(observation)

    def update(self) -> None:
        """Refresh all line data and autoscale visible axes."""

        if not self.buffer.t_sec:
            return

        # Keep line ownership fixed and replace only data arrays for responsive updates.
        self._lines["kinetic_current"].set_data(
            self.buffer.t_sec,
            self.buffer.kinetic_energy_j,
        )
        self._lines["kinetic_goal"].set_data(
            self.buffer.t_sec,
            [self.kinetic_goal_j] * len(self.buffer.t_sec),
        )
        self._lines["potential_current"].set_data(
            self.buffer.t_sec,
            self.buffer.potential_energy_j,
        )
        self._lines["potential_goal"].set_data(
            self.buffer.t_sec,
            [self.potential_goal_j] * len(self.buffer.t_sec),
        )
        self._lines["phase_path"].set_data(self.buffer.phase_c_error, self.buffer.phase_s)
        self._lines["phase_current"].set_data(
            [self.buffer.phase_c_error[-1]],
            [self.buffer.phase_s[-1]],
        )
        self._lines["action_commanded"].set_data(
            self.buffer.t_sec,
            self.buffer.u_commanded_nm,
        )
        self._lines["action_applied"].set_data(
            self.buffer.t_sec,
            self.buffer.u_applied_nm,
        )

        for name, axis in self.axes.items():
            axis.relim()
            axis.autoscale_view()
            axis.grid(True, alpha=0.25)
            if name != "animation":
                axis.legend(loc="upper right")
        self.figure.canvas.draw_idle()
        self._plt.pause(0.001)

    def finish(self) -> None:
        """Flush pending plot updates after the episode completes."""

        self.update()
        self._plt.pause(0.001)

    def _create_figure(self) -> tuple[Any, dict[str, Any]]:
        """Create one window with diagnostic subfigures and optional animation space."""

        # Allocate the animation axis only when the operator requested a visual pendulum.
        if self.include_animation:
            figure = self._plt.figure(figsize=(12, 8))
            figure.canvas.manager.set_window_title("Wake-Sleep MPC Realtime View")
            grid = figure.add_gridspec(4, 2, width_ratios=[1.0, 1.25])
            kinetic_axis = figure.add_subplot(grid[0, 0])
            potential_axis = figure.add_subplot(grid[1, 0])
            phase_axis = figure.add_subplot(grid[2, 0])
            action_axis = figure.add_subplot(grid[3, 0])
            animation_axis = figure.add_subplot(grid[:, 1])
        else:
            figure, axes = self._plt.subplots(4, 1, figsize=(8, 9))
            figure.canvas.manager.set_window_title("Wake-Sleep MPC Diagnostics")
            kinetic_axis, potential_axis, phase_axis, action_axis = axes
            animation_axis = None

        # Label each diagnostic panel with its physical quantity and units.
        kinetic_axis.set_title("Kinetic Energy")
        kinetic_axis.set_xlabel("t sec")
        kinetic_axis.set_ylabel("J")

        potential_axis.set_title("Potential Energy")
        potential_axis.set_xlabel("t sec")
        potential_axis.set_ylabel("J")

        phase_axis.set_title("Phase")
        phase_axis.set_xlabel("c_phi - 1")
        phase_axis.set_ylabel("s_phi")

        action_axis.set_title("Action")
        action_axis.set_xlabel("t sec")
        action_axis.set_ylabel("N m")

        axes_by_name = {
            "kinetic": kinetic_axis,
            "potential": potential_axis,
            "phase": phase_axis,
            "action": action_axis,
        }
        # Keep the optional animation axis addressable by name for embedded rendering.
        if animation_axis is not None:
            axes_by_name["animation"] = animation_axis
        return figure, axes_by_name

    def _create_lines(self) -> dict[str, Any]:
        """Create line artists for current values and goal references."""

        # Every line artist is created empty and populated during refresh calls.
        return {
            "kinetic_current": self.axes["kinetic"].plot([], [], label="current")[0],
            "kinetic_goal": self.axes["kinetic"].plot(
                [],
                [],
                linestyle="--",
                label="goal",
            )[0],
            "potential_current": self.axes["potential"].plot([], [], label="current")[0],
            "potential_goal": self.axes["potential"].plot(
                [],
                [],
                linestyle="--",
                label="goal",
            )[0],
            "phase_path": self.axes["phase"].plot([], [], label="trajectory")[0],
            "phase_current": self.axes["phase"].plot([], [], marker="o", label="current")[0],
            "phase_goal": self.axes["phase"].plot(
                [self.phase_goal[0]],
                [self.phase_goal[1]],
                marker="*",
                markersize=12,
                linestyle="",
                label="goal",
            )[0],
            "action_applied": self.axes["action"].plot([], [], label="applied")[0],
            "action_commanded": self.axes["action"].plot(
                [], [], linestyle="--", label="commanded"
            )[0],
        }

    @staticmethod
    def kinetic_energy(omega_rad_s: float, pendulum: PendulumConfig) -> float:
        """Return pendulum kinetic energy in joules."""
        inertia = pendulum.mass_kg * pendulum.length_m**2
        return 0.5 * inertia * omega_rad_s**2

    @staticmethod
    def potential_energy(theta_rad: float, pendulum: PendulumConfig) -> float:
        """Return potential energy with theta zero at the upright goal."""
        return pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m * (
            1.0 + math.cos(theta_rad)
        )
