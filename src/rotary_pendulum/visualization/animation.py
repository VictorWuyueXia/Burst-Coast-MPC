"""Realtime three-dimensional view of the rotary pendulum mechanism."""

from __future__ import annotations

from typing import Any

import numpy as np

from rotary_pendulum.environment.dynamics import mechanism_points
from rotary_pendulum.utils.config_schema import RotaryPendulumConfig
from rotary_pendulum.utils.messages import StateObservation


class RotaryPendulumAnimation:
    """Render arm and pendulum geometry reconstructed from the four-state model."""

    def __init__(
        self,
        physical: RotaryPendulumConfig,
        *,
        axis: Any,
    ) -> None:
        # Bind the physical geometry and create fixed artists for inexpensive updates.
        self.physical = physical
        self.axis = axis
        (self._arm_line,) = axis.plot([], [], [], linewidth=5.0, color="tab:red", label="arm")
        (self._pendulum_line,) = axis.plot(
            [], [], [], linewidth=3.0, color="tab:blue", label="pendulum"
        )
        (self._pivot_marker,) = axis.plot([], [], [], "o", color="black", markersize=6)
        (self._tip_marker,) = axis.plot([], [], [], "o", color="tab:blue", markersize=8)

        # Draw the rotary-arm sweep circle as a stationary mechanism reference.
        sweep_angle = np.linspace(0.0, 2.0 * np.pi, 160)
        axis.plot(
            physical.arm_length_m * np.cos(sweep_angle),
            physical.arm_length_m * np.sin(sweep_angle),
            np.zeros_like(sweep_angle),
            linestyle=":",
            linewidth=1.0,
            color="0.55",
        )
        axis.scatter([0.0], [0.0], [0.0], s=45, color="black", label="motor axis")
        self._format_axis()

    def start(self, observation: StateObservation) -> None:
        """Initialize the mechanism artists from the reset observation."""

        self.update(observation, 0.0)

    def update(self, observation: StateObservation, applied_torque_nm: float) -> None:
        """Move the arm and pendulum artists to the observed configuration."""

        # Reconstruct Cartesian points algebraically without integrating redundant states.
        state = np.array(
            [
                observation.theta_rad,
                observation.alpha_rad,
                observation.omega_rad_s,
                observation.nu_rad_s,
            ],
            dtype=np.float64,
        )
        origin, pivot, _, tip = mechanism_points(state, self.physical)
        self._arm_line.set_data_3d(
            [origin[0], pivot[0]],
            [origin[1], pivot[1]],
            [origin[2], pivot[2]],
        )
        self._pendulum_line.set_data_3d(
            [pivot[0], tip[0]],
            [pivot[1], tip[1]],
            [pivot[2], tip[2]],
        )
        self._pivot_marker.set_data_3d([pivot[0]], [pivot[1]], [pivot[2]])
        self._tip_marker.set_data_3d([tip[0]], [tip[1]], [tip[2]])
        self.axis.set_title(
            "3D Mechanism\n"
            f"theta={observation.theta_rad:+.2f} rad, alpha={observation.alpha_rad:+.2f} rad, "
            f"tau={applied_torque_nm:+.4f} N m"
        )

    def _format_axis(self) -> None:
        """Set fixed equal-scale bounds and physical labels for the 3D mechanism."""

        # Fixed limits prevent visual scale changes from obscuring the mechanism motion.
        radial_limit = 1.12 * (self.physical.arm_length_m + self.physical.pendulum_length_m)
        vertical_limit = 1.12 * self.physical.pendulum_length_m
        self.axis.set_xlim(-radial_limit, radial_limit)
        self.axis.set_ylim(-radial_limit, radial_limit)
        self.axis.set_zlim(-vertical_limit, vertical_limit)
        self.axis.set_box_aspect((2.0 * radial_limit, 2.0 * radial_limit, 2.0 * vertical_limit))
        self.axis.set_xlabel("x m")
        self.axis.set_ylabel("y m")
        self.axis.set_zlabel("z m")
        self.axis.view_init(elev=24.0, azim=38.0)
        self.axis.legend(loc="upper left", fontsize=8)
