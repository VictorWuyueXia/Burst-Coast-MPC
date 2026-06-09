"""Shared discrete pendulum model used by MPC prediction."""

from __future__ import annotations

import math
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from wsmpc.environment.dynamics import rk4_step
from wsmpc.mpc.types import SplitCandidate
from wsmpc.utils.config_schema import EnvironmentConfig, PendulumConfig


def pendulum_derivatives_symbolic(x: Any, u_nm: Any, pendulum: PendulumConfig) -> Any:
    """Return symbolic pendulum dynamics with theta zero at the upright."""

    # Match the numeric plant dynamics exactly inside the CasADi graph.
    theta_rad = x[0]
    omega_rad_s = x[1]
    inertia = pendulum.mass_kg * pendulum.length_m**2
    omega_dot = (
        pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m * ca.sin(theta_rad)
        - pendulum.damping_nms * omega_rad_s
        + u_nm
    ) / inertia
    return ca.vertcat(omega_rad_s, omega_dot)


def rk4_step_symbolic(x: Any, u_nm: Any, timestep_s: float, pendulum: PendulumConfig) -> Any:
    """Advance the symbolic state with one fixed RK4 step."""

    # Use the same fourth-order integration structure as the simulator.
    k1 = pendulum_derivatives_symbolic(x, u_nm, pendulum)
    k2 = pendulum_derivatives_symbolic(x + 0.5 * timestep_s * k1, u_nm, pendulum)
    k3 = pendulum_derivatives_symbolic(x + 0.5 * timestep_s * k2, u_nm, pendulum)
    k4 = pendulum_derivatives_symbolic(x + timestep_s * k3, u_nm, pendulum)
    return x + (timestep_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def natural_frequency_rad_s(pendulum: PendulumConfig) -> float:
    """Return sqrt(m g l / I), the pendulum natural frequency."""

    # Natural-period horizons use the exact pendulum inertia implied by config.
    inertia = pendulum.mass_kg * pendulum.length_m**2
    return math.sqrt(pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m / inertia)


def rollout_burst_coast(
    state: ArrayLike,
    burst_inputs_nm: ArrayLike,
    candidate: SplitCandidate,
    environment: EnvironmentConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Replay a burst-coast input sequence through the discrete RK4 model."""

    # 1. Expand the optimized burst command into the full burst-coast input sequence.
    burst_array = np.asarray(burst_inputs_nm, dtype=np.float64).reshape(candidate.burst_steps)
    full_inputs_nm = np.zeros(candidate.total_steps, dtype=np.float64)
    full_inputs_nm[: candidate.burst_steps] = burst_array

    # 2. Replay the candidate trajectory through the same discrete plant used online.
    states = np.empty((candidate.total_steps + 1, 2), dtype=np.float64)
    states[0] = np.asarray(state, dtype=np.float64).reshape(2)
    for index, torque_nm in enumerate(full_inputs_nm):
        states[index + 1] = rk4_step(
            states[index],
            float(torque_nm),
            environment.simulation.timestep_s,
            environment.pendulum,
        )
    return states, full_inputs_nm
