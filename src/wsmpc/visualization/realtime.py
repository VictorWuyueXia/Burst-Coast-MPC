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
    energy_j: list[float] = field(default_factory=list)
    energy_error_j: list[float] = field(default_factory=list)
    constraint_margin: list[float] = field(default_factory=list)
    goal_flag: list[float] = field(default_factory=list)
    u_commanded_nm: list[float] = field(default_factory=list)
    u_applied_nm: list[float] = field(default_factory=list)


class RealtimeEpisodePlot:
    """Update grouped diagnostic plots from emitted step records."""

    def __init__(self, *, update_every: int = 1) -> None:
        self.update_every = max(1, update_every)
        self.buffer = _SeriesBuffer()

        # Import Matplotlib lazily so non-visual runs stay dependency-light at import time.
        import matplotlib.pyplot as plt

        self._plt = plt
        self.figure, self.axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        self.figure.canvas.manager.set_window_title("Wake-Sleep MPC Diagnostics")
        self._lines = self._create_lines()
        self.axes[-1].set_xlabel("t sec")
        self.figure.tight_layout()
        plt.show(block=False)

    def add_step(self, observation: StateObs, record: StepRecord) -> None:
        """Append one record and refresh the plot on the configured cadence."""

        self.buffer.t_sec.append(record.t_sec)
        self.buffer.theta_rad.append(record.theta_rad)
        self.buffer.omega_rad_s.append(record.omega_rad_s)
        self.buffer.energy_j.append(record.energy_j)
        self.buffer.energy_error_j.append(record.energy_error_j)
        self.buffer.constraint_margin.append(record.constraint_margin)
        self.buffer.goal_flag.append(1.0 if record.goal_flag else 0.0)
        self.buffer.u_commanded_nm.append(record.u_commanded_nm)
        self.buffer.u_applied_nm.append(record.u_applied_nm)

        if len(self.buffer.t_sec) % self.update_every == 0:
            self.update()

    def update(self) -> None:
        """Refresh all line data and autoscale visible axes."""

        t_sec = self.buffer.t_sec
        if not t_sec:
            return

        # Keep line ownership fixed and replace only data arrays for responsive updates.
        for key, line in self._lines.items():
            line.set_data(t_sec, getattr(self.buffer, key))

        for axis in self.axes:
            axis.relim()
            axis.autoscale_view()
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper right")
        self.figure.canvas.draw_idle()
        self._plt.pause(0.001)

    def finish(self) -> None:
        """Flush pending plot updates after the episode completes."""

        self.update()
        self._plt.pause(0.001)

    def _create_lines(self) -> dict[str, Any]:
        """Create grouped line artists keyed by their buffer fields."""

        state_axis, diagnostics_axis, action_axis = self.axes
        state_axis.set_ylabel("state")
        diagnostics_axis.set_ylabel("diagnostics")
        action_axis.set_ylabel("torque N m")

        lines = {
            "theta_rad": state_axis.plot([], [], label="theta rad")[0],
            "omega_rad_s": state_axis.plot([], [], label="omega rad/s")[0],
            "energy_j": diagnostics_axis.plot([], [], label="energy J")[0],
            "energy_error_j": diagnostics_axis.plot([], [], label="energy error J")[0],
            "constraint_margin": diagnostics_axis.plot([], [], label="constraint margin")[0],
            "goal_flag": diagnostics_axis.plot([], [], label="goal flag")[0],
            "u_commanded_nm": action_axis.plot([], [], label="commanded")[0],
            "u_applied_nm": action_axis.plot([], [], label="applied")[0],
        }
        return lines


class PendulumAnimation:
    """Draw a realtime pendulum view from simulator observations."""

    def __init__(self, pendulum: PendulumConfig, *, update_every: int = 1) -> None:
        self.pendulum = pendulum
        self.update_every = max(1, update_every)
        self._record_count = 0

        # Import Matplotlib lazily so animation support is only required when requested.
        import matplotlib.pyplot as plt

        self._plt = plt
        self.figure, self.axis = plt.subplots(figsize=(6, 6))
        self.figure.canvas.manager.set_window_title("Wake-Sleep MPC Pendulum")
        limit = self.pendulum.length_m * 1.25
        self.axis.set_xlim(-limit, limit)
        self.axis.set_ylim(-limit, limit)
        self.axis.set_aspect("equal", adjustable="box")
        self.axis.grid(True, alpha=0.25)

        # Keep artist instances stable so frame updates are cheap and flicker-free.
        (self._rod,) = self.axis.plot([0.0, 0.0], [0.0, self.pendulum.length_m], lw=3)
        self._bob = self.axis.scatter(
            [0.0],
            [self.pendulum.length_m],
            s=self._bob_size(),
            zorder=3,
        )
        self._pivot = self.axis.scatter([0.0], [0.0], s=40, color="black", zorder=4)
        self._torque_text = self.axis.text(
            0.02,
            0.96,
            "",
            transform=self.axis.transAxes,
            ha="left",
            va="top",
        )
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

        torque_limit = max(self.pendulum.torque_limit_nm, 1e-12)
        normalized_torque = max(-1.0, min(1.0, u_applied_nm / torque_limit))
        direction = "ccw" if normalized_torque > 0 else "cw" if normalized_torque < 0 else "zero"
        self._torque_text.set_text(f"torque {direction}: {u_applied_nm:.3f} N m")
        self._torque_text.set_color("tab:red" if normalized_torque >= 0 else "tab:blue")

        self.figure.canvas.draw_idle()
        self._plt.pause(0.001)

    def _bob_size(self) -> float:
        """Scale bob area by the square root of mass for perceptible mass changes."""

        return 300.0 * math.sqrt(self.pendulum.mass_kg)
