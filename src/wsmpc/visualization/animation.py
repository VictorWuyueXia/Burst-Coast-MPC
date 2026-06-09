"""Realtime pendulum animation drawn from simulator observations."""

from __future__ import annotations

import math
from typing import Any

from wsmpc.utils.config_schema import PendulumConfig
from wsmpc.utils.messages import StateObs, StepRecord


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
        self._torque_head = None

        if plt_module is None:
            import matplotlib.pyplot as plt
        else:
            plt = plt_module
        from matplotlib.patches import RegularPolygon

        self._plt = plt
        self._RegularPolygon = RegularPolygon
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

        # Stable artist instances keep frame updates cheap and flicker-free.
        (self._rod,) = self.axis.plot([0.0, 0.0], [0.0, self.pendulum.length_m], lw=3)
        self._bob = self.axis.scatter(
            [0.0],
            [self.pendulum.length_m],
            s=self._bob_size(),
            zorder=3,
        )
        self._pivot = self.axis.scatter([0.0], [0.0], s=40, color="black", zorder=4)
        (self._torque_arc,) = self.axis.plot([], [], color="tab:red", alpha=0.0, zorder=5)
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

        normalized_torque = max(-1.0, min(1.0, u_applied_nm / self.pendulum.torque_limit_nm))
        magnitude = abs(normalized_torque)
        if magnitude < 1e-6:
            self._torque_arc.set_alpha(0.0)
            self._remove_torque_head()
            return

        # A circular arrow maps signed torque directly to rotation direction.
        radius = self.pendulum.length_m * (0.18 + 0.22 * magnitude)
        start_deg, end_deg = (135.0, -145.0) if normalized_torque > 0.0 else (-145.0, 135.0)
        angles = [
            math.radians(start_deg + (end_deg - start_deg) * index / 48.0)
            for index in range(49)
        ]
        arc_x = [radius * math.cos(angle) for angle in angles]
        arc_y = [radius * math.sin(angle) for angle in angles]
        color = "tab:red" if normalized_torque > 0.0 else "tab:blue"

        self._torque_arc.set_data(arc_x, arc_y)
        self._torque_arc.set_linewidth(1.0 + 3.0 * magnitude)
        self._torque_arc.set_color(color)
        self._torque_arc.set_alpha(0.9)
        self._replace_torque_head(
            arc_x[-1],
            arc_y[-1],
            radius=self.pendulum.length_m * (0.025 + 0.035 * magnitude),
            orientation=angles[-1]
            + (-math.pi / 2.0 if normalized_torque > 0.0 else math.pi / 2.0),
            color=color,
        )

    def _replace_torque_head(
        self,
        x_pos: float,
        y_pos: float,
        *,
        radius: float,
        orientation: float,
        color: str,
    ) -> None:
        """Replace the arrowhead so it remains tangent to the current circular torque arc."""

        self._remove_torque_head()
        self._torque_head = self._RegularPolygon(
            (x_pos, y_pos),
            numVertices=3,
            radius=radius,
            orientation=orientation,
            color=color,
            alpha=0.9,
            zorder=6,
        )
        self.axis.add_patch(self._torque_head)

    def _remove_torque_head(self) -> None:
        """Remove the current torque arrowhead before drawing the next signed command."""

        if self._torque_head is not None:
            self._torque_head.remove()
            self._torque_head = None

    def _bob_size(self) -> float:
        """Scale bob area by the square root of mass for perceptible mass changes."""

        return 300.0 * math.sqrt(self.pendulum.mass_kg)
