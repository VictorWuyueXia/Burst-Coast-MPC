"""Vectorized pendulum dynamics and state feature helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wsmpc.utils.config_schema import PendulumConfig


def wrap_angle(angle_rad: ArrayLike) -> NDArray[np.float64]:
    """Wrap angles to (-pi, pi] using NumPy batch operations."""

    angle = np.asarray(angle_rad, dtype=np.float64)
    return np.arctan2(np.sin(angle), np.cos(angle))


def pendulum_energy(
    theta_rad: ArrayLike,
    omega_rad_s: ArrayLike,
    config: PendulumConfig,
) -> NDArray[np.float64]:
    """Compute mechanical energy in joules with zero potential at downward position."""

    theta_array = np.asarray(theta_rad, dtype=np.float64)
    omega_array = np.asarray(omega_rad_s, dtype=np.float64)
    inertia = config.mass_kg * config.length_m**2
    kinetic = 0.5 * inertia * omega_array**2
    potential = config.mass_kg * config.gravity_m_s2 * config.length_m * (1.0 - np.cos(theta_array))
    return kinetic + potential


def upright_energy(config: PendulumConfig) -> float:
    """Return the pendulum energy in joules at the upright equilibrium."""

    return 2.0 * config.mass_kg * config.gravity_m_s2 * config.length_m


def state_features(state: ArrayLike, config: PendulumConfig) -> NDArray[np.float64]:
    """Return [energy error J, wrapped angle error rad, omega rad/s]."""

    state_array = np.asarray(state, dtype=np.float64)
    theta_rad = state_array[..., 0]
    omega_rad_s = state_array[..., 1]
    energy_error_j = pendulum_energy(theta_rad, omega_rad_s, config) - upright_energy(config)
    wrapped_angle_error_rad = wrap_angle(theta_rad - np.pi)
    return np.stack((energy_error_j, wrapped_angle_error_rad, omega_rad_s), axis=-1)


def clip_torque(torque_nm: ArrayLike, config: PendulumConfig) -> NDArray[np.float64]:
    """Clip torque commands in N m so scalar and batched actions share one path."""

    torque_nm_array = np.asarray(torque_nm, dtype=np.float64)
    return np.clip(torque_nm_array, -config.torque_limit_nm, config.torque_limit_nm)


def pendulum_derivatives(
    state: ArrayLike,
    torque_nm: ArrayLike,
    config: PendulumConfig,
) -> NDArray[np.float64]:
    """Compute continuous-time pendulum derivatives for scalar or batched states."""

    # Extract state components with NumPy broadcasting for efficient batch evaluation.
    state_array = np.asarray(state, dtype=np.float64)
    theta_rad = state_array[..., 0]
    omega_rad_s = state_array[..., 1]
    applied_torque_nm = clip_torque(torque_nm, config)

    # Apply the exact nonlinear underactuated pendulum model.
    inertia = config.mass_kg * config.length_m**2
    theta_dot = omega_rad_s
    omega_dot = (
        config.mass_kg * config.gravity_m_s2 * config.length_m * np.sin(theta_rad)
        - config.damping_nms * omega_rad_s
        + applied_torque_nm
    ) / inertia
    return np.stack((theta_dot, omega_dot), axis=-1)


def rk4_step(
    state: ArrayLike,
    torque_nm: ArrayLike,
    timestep_s: float,
    config: PendulumConfig,
) -> NDArray[np.float64]:
    """Advance pendulum dynamics by one fixed RK4 step."""

    # Reuse one vectorized derivative path for all RK4 stages.
    state_array = np.asarray(state, dtype=np.float64)
    k1 = pendulum_derivatives(state_array, torque_nm, config)
    k2 = pendulum_derivatives(state_array + 0.5 * timestep_s * k1, torque_nm, config)
    k3 = pendulum_derivatives(state_array + 0.5 * timestep_s * k2, torque_nm, config)
    k4 = pendulum_derivatives(state_array + timestep_s * k3, torque_nm, config)
    return state_array + (timestep_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
