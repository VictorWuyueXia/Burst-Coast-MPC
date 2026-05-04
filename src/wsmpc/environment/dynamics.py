"""Vectorized pendulum dynamics and state feature helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wsmpc.utils.config_schema import PendulumConfig


def wrap_angle(angle_rad: ArrayLike) -> NDArray[np.float64]:
    """Wrap angles to (-pi, pi] using NumPy batch operations."""

    angle = np.asarray(angle_rad, dtype=np.float64)
    return np.arctan2(np.sin(angle), np.cos(angle))


def pendulum_energy(theta: ArrayLike, omega: ArrayLike, config: PendulumConfig) -> NDArray[np.float64]:
    """Compute mechanical energy with zero potential at downward position."""

    theta_array = np.asarray(theta, dtype=np.float64)
    omega_array = np.asarray(omega, dtype=np.float64)
    inertia = config.mass_kg * config.length_m**2
    kinetic = 0.5 * inertia * omega_array**2
    potential = config.mass_kg * config.gravity_m_s2 * config.length_m * (1.0 - np.cos(theta_array))
    return kinetic + potential


def upright_energy(config: PendulumConfig) -> float:
    """Return the pendulum energy at the upright equilibrium."""

    return 2.0 * config.mass_kg * config.gravity_m_s2 * config.length_m


def state_features(state: ArrayLike, config: PendulumConfig) -> NDArray[np.float64]:
    """Return [energy error, wrapped angle error, omega] for one state or a batch."""

    state_array = np.asarray(state, dtype=np.float64)
    theta = state_array[..., 0]
    omega = state_array[..., 1]
    energy_error = pendulum_energy(theta, omega, config) - upright_energy(config)
    wrapped_angle_error = wrap_angle(theta - np.pi)
    return np.stack((energy_error, wrapped_angle_error, omega), axis=-1)


def clip_torque(torque: ArrayLike, config: PendulumConfig) -> NDArray[np.float64]:
    """Clip torque commands with NumPy so scalar and batched actions share one path."""

    torque_array = np.asarray(torque, dtype=np.float64)
    return np.clip(torque_array, -config.torque_limit_nm, config.torque_limit_nm)


def pendulum_derivatives(
    state: ArrayLike,
    torque: ArrayLike,
    config: PendulumConfig,
) -> NDArray[np.float64]:
    """Compute continuous-time pendulum derivatives for scalar or batched states."""

    # Extract state components with NumPy broadcasting for efficient batch evaluation.
    state_array = np.asarray(state, dtype=np.float64)
    theta = state_array[..., 0]
    omega = state_array[..., 1]
    applied_torque = clip_torque(torque, config)

    # Apply the exact nonlinear underactuated pendulum model.
    inertia = config.mass_kg * config.length_m**2
    theta_dot = omega
    omega_dot = (
        config.mass_kg * config.gravity_m_s2 * config.length_m * np.sin(theta)
        - config.damping_nms * omega
        + applied_torque
    ) / inertia
    return np.stack((theta_dot, omega_dot), axis=-1)


def rk4_step(
    state: ArrayLike,
    torque: ArrayLike,
    timestep_s: float,
    config: PendulumConfig,
) -> NDArray[np.float64]:
    """Advance pendulum dynamics by one fixed RK4 step."""

    # Reuse one vectorized derivative path for all RK4 stages.
    state_array = np.asarray(state, dtype=np.float64)
    k1 = pendulum_derivatives(state_array, torque, config)
    k2 = pendulum_derivatives(state_array + 0.5 * timestep_s * k1, torque, config)
    k3 = pendulum_derivatives(state_array + 0.5 * timestep_s * k2, torque, config)
    k4 = pendulum_derivatives(state_array + timestep_s * k3, torque, config)
    return state_array + (timestep_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
