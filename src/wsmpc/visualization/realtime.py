"""Realtime Matplotlib plots and pendulum animation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from wsmpc.utils.config_schema import PendulumConfig
from wsmpc.utils.messages import StateObs, StepRecord


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
        self.figure, self.axes = self._create_figure()
        self.figures = [self.figure]
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

        self.buffer.t_sec.append(record.t_sec)
        self.buffer.theta_rad.append(record.theta_rad)
        self.buffer.omega_rad_s.append(record.omega_rad_s)
        self.buffer.kinetic_energy_j.append(self.kinetic_energy(record.omega_rad_s, self.pendulum))
        self.buffer.potential_energy_j.append(
            self.potential_energy(record.theta_rad, self.pendulum)
        )
        self.buffer.energy_j.append(record.energy_j)
        self.buffer.energy_error_j.append(record.energy_error_j)
        self.buffer.constraint_margin.append(record.constraint_margin)
        self.buffer.goal_flag.append(1.0 if record.goal_flag else 0.0)
        self.buffer.u_commanded_nm.append(record.u_commanded_nm)
        self.buffer.u_applied_nm.append(record.u_applied_nm)
        if self.animation is not None:
            self.animation.add_step(observation, record)

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
        self._lines["phase_path"].set_data(self.buffer.theta_rad, self.buffer.omega_rad_s)
        self._lines["phase_current"].set_data(
            [self.buffer.theta_rad[-1]],
            [self.buffer.omega_rad_s[-1]],
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
        for figure in self.figures:
            figure.canvas.draw_idle()
        self._plt.pause(0.001)

    def finish(self) -> None:
        """Flush pending plot updates after the episode completes."""

        self.update()
        self._plt.pause(0.001)

    def _create_figure(self) -> tuple[Any, dict[str, Any]]:
        """Create one window with diagnostic subfigures and optional animation space."""

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

        kinetic_axis.set_title("Kinetic Energy")
        kinetic_axis.set_xlabel("t sec")
        kinetic_axis.set_ylabel("J")

        potential_axis.set_title("Potential Energy")
        potential_axis.set_xlabel("t sec")
        potential_axis.set_ylabel("J")

        phase_axis.set_title("Phase")
        phase_axis.set_xlabel("theta rad")
        phase_axis.set_ylabel("omega rad/s")

        action_axis.set_title("Action")
        action_axis.set_xlabel("t sec")
        action_axis.set_ylabel("N m")

        axes_by_name = {
            "kinetic": kinetic_axis,
            "potential": potential_axis,
            "phase": phase_axis,
            "action": action_axis,
        }
        if animation_axis is not None:
            axes_by_name["animation"] = animation_axis
        return figure, axes_by_name

    def _create_lines(self) -> dict[str, Any]:
        """Create line artists for current values and goal references."""

        lines = {
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
            "action_commanded": self.axes["action"].plot([], [], 
                linestyle="--", label="commanded")[0],
        }
        return lines

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


class PendulumAnimation:
    """Draw a realtime pendulum view from simulator observations."""

    def __init__(
        self,
        pendulum: PendulumConfig,
        *,
        update_every: int = 1,
        axis: Any | None = None,
        plt_module: Any | None = None,
    ) -> None:
        self.pendulum = pendulum
        self.update_every = max(1, update_every)
        self._record_count = 0
        self._torque_arrow_start = (0.0, 0.0)
        self._torque_arrow_end = (0.0, 0.0)
        self._torque_arrow_rad = 0.0

        # Import Matplotlib lazily so animation support is only required when requested.
        if plt_module is None:
            import matplotlib.pyplot as plt
        else:
            plt = plt_module
        from matplotlib.patches import FancyArrowPatch

        self._plt = plt
        if axis is None:
            self.figure, self.axis = plt.subplots(figsize=(6, 6))
            self.figure.canvas.manager.set_window_title("Wake-Sleep MPC Pendulum")
            self._owns_figure = True
        else:
            self.axis = axis
            self.figure = axis.figure
            self._owns_figure = False
        limit = self.pendulum.length_m * 1.25
        self.axis.set_xlim(-limit, limit)
        self.axis.set_ylim(-limit, limit)
        self.axis.set_aspect("equal", adjustable="box")
        self.axis.grid(True, alpha=0.25)
        self.axis.set_title("Pendulum")

        # Keep artist instances stable so frame updates are cheap and flicker-free.
        (self._rod,) = self.axis.plot([0.0, 0.0], [0.0, self.pendulum.length_m], lw=3)
        self._bob = self.axis.scatter(
            [0.0],
            [self.pendulum.length_m],
            s=self._bob_size(),
            zorder=3,
        )
        self._pivot = self.axis.scatter([0.0], [0.0], s=40, color="black", zorder=4)
        self._torque_arrow = FancyArrowPatch(
            (0.0, 0.0),
            (0.0, 0.0),
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.45",
            mutation_scale=10.0,
            linewidth=1.0,
            color="tab:red",
            alpha=0.0,
            zorder=5,
        )
        self.axis.add_patch(self._torque_arrow)
        if self._owns_figure:
            plt.show(block=False)

    @staticmethod
    def bob_position(theta_rad: float, length_m: float) -> tuple[float, float]:
        """Compute visual bob position with theta zero drawn upward."""

        return length_m * math.sin(theta_rad), length_m * math.cos(theta_rad)

    def start(self, observation: StateObs) -> None:
        """Draw the reset observation before the first completed step."""

        self._draw(observation, u_applied_nm=0.0)

    def add_step(self, observation: StateObs, record: StepRecord) -> None:
        """Refresh the animation on the configured record cadence."""

        self._record_count += 1
        if self._record_count % self.update_every == 0:
            self._draw(observation, u_applied_nm=record.u_applied_nm)

    def finish(self) -> None:
        """Flush the current animation frame after the episode completes."""

        self.figure.canvas.draw_idle()
        self._plt.pause(0.001)

    def _draw(self, observation: StateObs, *, u_applied_nm: float) -> None:
        """Update rod, bob, and torque indicator for one observation."""

        bob_x, bob_y = self.bob_position(observation.theta_rad, self.pendulum.length_m)
        self._rod.set_data([0.0, bob_x], [0.0, bob_y])
        self._bob.set_offsets([[bob_x, bob_y]])

        self._update_torque_arrow(u_applied_nm)

        self.figure.canvas.draw_idle()
        self._plt.pause(0.001)

    def _update_torque_arrow(self, u_applied_nm: float) -> None:
        """Scale and orient the torque arrow around the pivot from applied torque."""

        torque_limit = max(self.pendulum.torque_limit_nm, 1e-12)
        normalized_torque = max(-1.0, min(1.0, u_applied_nm / torque_limit))
        magnitude = abs(normalized_torque)
        if magnitude < 1e-6:
            self._torque_arrow.set_alpha(0.0)
            return

        radius = self.pendulum.length_m * (0.22 + 0.18 * magnitude)
        start_deg, end_deg = (-135.0, 135.0) if normalized_torque > 0.0 else (135.0, -135.0)
        start = self._point_on_circle(radius, start_deg)
        end = self._point_on_circle(radius, end_deg)
        rad = 0.55 if normalized_torque > 0.0 else -0.55

        self._torque_arrow_start = start
        self._torque_arrow_end = end
        self._torque_arrow_rad = rad
        self._torque_arrow.set_positions(start, end)
        self._torque_arrow.set_connectionstyle(f"arc3,rad={rad}")
        self._torque_arrow.set_mutation_scale(8.0 + 18.0 * magnitude)
        self._torque_arrow.set_linewidth(1.0 + 3.0 * magnitude)
        self._torque_arrow.set_color("tab:red" if normalized_torque > 0.0 else "tab:blue")
        self._torque_arrow.set_alpha(0.9)

    @staticmethod
    def _point_on_circle(radius: float, angle_deg: float) -> tuple[float, float]:
        """Return a data-coordinate point on a pivot-centered circle."""

        angle_rad = math.radians(angle_deg)
        return radius * math.cos(angle_rad), radius * math.sin(angle_rad)

    def _bob_size(self) -> float:
        """Scale bob area by the square root of mass for perceptible mass changes."""

        return 300.0 * math.sqrt(self.pendulum.mass_kg)
