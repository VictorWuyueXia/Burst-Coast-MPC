"""Energy-phase features for the natural-period dynamics MPC."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from wsmpc.environment.dynamics import pendulum_energy, upright_energy, wrap_angle
from wsmpc.mpc.discrete_model import natural_frequency_rad_s
from wsmpc.mpc.ip_dynamics_natural_period import weights
from wsmpc.utils.config_schema import PendulumConfig


def normalized_energy_error(state: ArrayLike, pendulum: PendulumConfig) -> NDArray[np.float64]:
    """Compute normalized energy error with the upright equilibrium as zero."""

    state_array = np.asarray(state, dtype=np.float64)
    theta_rad = state_array[..., 0]
    omega_rad_s = state_array[..., 1]
    target_energy_j = upright_energy(pendulum)
    return (pendulum_energy(theta_rad, omega_rad_s, pendulum) - target_energy_j) / target_energy_j


def local_upright_error(state: ArrayLike, pendulum: PendulumConfig) -> NDArray[np.float64]:
    """Return the local upright error vector [wrapped theta, normalized omega]."""

    state_array = np.asarray(state, dtype=np.float64)
    theta_rad = state_array[..., 0]
    omega_rad_s = state_array[..., 1]
    return np.stack(
        (wrap_angle(theta_rad), omega_rad_s / natural_frequency_rad_s(pendulum)),
        axis=-1,
    )


def phase_proxy_error(
    state: ArrayLike,
    pendulum: PendulumConfig,
    epsilon_phi: float = weights.EPSILON_PHI,
) -> NDArray[np.float64]:
    """Return the phase proxy error [c_phi - 1, s_phi]."""

    state_array = np.asarray(state, dtype=np.float64)
    theta_rad = state_array[..., 0]
    omega_rad_s = state_array[..., 1]
    a_theta = np.cos(0.5 * theta_rad)
    b_phi = omega_rad_s / natural_frequency_rad_s(pendulum)
    radius = np.sqrt(a_theta**2 + b_phi**2 + epsilon_phi**2)
    return np.stack(((a_theta / radius) - 1.0, b_phi / radius), axis=-1)


def energy_gate(state: ArrayLike, pendulum: PendulumConfig) -> NDArray[np.float64]:
    """Return the energy-shell gate that activates phase and local terms."""

    energy_error = normalized_energy_error(state, pendulum)
    return np.exp(-(energy_error**2) / (weights.SIGMA_ENERGY**2))


def energy_phase_value(state: ArrayLike, pendulum: PendulumConfig) -> NDArray[np.float64]:
    """Evaluate the natural-period energy-phase value."""

    energy_error = normalized_energy_error(state, pendulum)
    phase_error = phase_proxy_error(state, pendulum)
    local_error = local_upright_error(state, pendulum)
    gate = energy_gate(state, pendulum)

    # Diagonal quadratic forms keep phase and local penalties explicit.
    phase_cost = np.sum(np.asarray(weights.Q_PHASE_DIAG) * phase_error**2, axis=-1)
    local_cost = np.sum(np.asarray(weights.Q_LOCAL_DIAG) * local_error**2, axis=-1)
    return (
        weights.Q_ENERGY * energy_error**2
        + weights.Q_PHASE * gate * phase_cost
        + weights.Q_LOCAL * gate * local_cost
    )


def normalized_energy_error_symbolic(x: Any, pendulum: PendulumConfig) -> Any:
    """Compute symbolic normalized energy error."""

    theta_rad = x[0]
    omega_rad_s = x[1]
    inertia = pendulum.mass_kg * pendulum.length_m**2
    kinetic = 0.5 * inertia * omega_rad_s**2
    potential = (
        pendulum.mass_kg
        * pendulum.gravity_m_s2
        * pendulum.length_m
        * (1.0 + ca.cos(theta_rad))
    )
    target_energy = 2.0 * pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m
    return (kinetic + potential - target_energy) / target_energy


def local_upright_error_symbolic(x: Any, pendulum: PendulumConfig) -> Any:
    """Return symbolic local upright error [wrapped theta, normalized omega]."""

    theta_rad = x[0]
    omega_rad_s = x[1]
    wrapped_theta = ca.atan2(ca.sin(theta_rad), ca.cos(theta_rad))
    normalized_omega = omega_rad_s / natural_frequency_rad_s(pendulum)
    return ca.vertcat(wrapped_theta, normalized_omega)


def phase_proxy_error_symbolic(x: Any, pendulum: PendulumConfig) -> Any:
    """Return symbolic phase proxy error [c_phi - 1, s_phi]."""

    theta_rad = x[0]
    omega_rad_s = x[1]
    a_theta = ca.cos(0.5 * theta_rad)
    b_phi = omega_rad_s / natural_frequency_rad_s(pendulum)
    radius = ca.sqrt(a_theta**2 + b_phi**2 + weights.EPSILON_PHI**2)
    return ca.vertcat((a_theta / radius) - 1.0, b_phi / radius)


def energy_gate_symbolic(x: Any, pendulum: PendulumConfig) -> Any:
    """Return symbolic energy-shell gate."""

    energy_error = normalized_energy_error_symbolic(x, pendulum)
    return ca.exp(-(energy_error**2) / (weights.SIGMA_ENERGY**2))


def energy_phase_value_symbolic(x: Any, pendulum: PendulumConfig) -> Any:
    """Evaluate symbolic natural-period energy-phase value."""

    energy_error = normalized_energy_error_symbolic(x, pendulum)
    phase_error = phase_proxy_error_symbolic(x, pendulum)
    local_error = local_upright_error_symbolic(x, pendulum)
    gate = energy_gate_symbolic(x, pendulum)
    phase_cost = (
        weights.Q_PHASE_DIAG[0] * phase_error[0] ** 2
        + weights.Q_PHASE_DIAG[1] * phase_error[1] ** 2
    )
    local_cost = (
        weights.Q_LOCAL_DIAG[0] * local_error[0] ** 2
        + weights.Q_LOCAL_DIAG[1] * local_error[1] ** 2
    )
    return (
        weights.Q_ENERGY * energy_error**2
        + weights.Q_PHASE * gate * phase_cost
        + weights.Q_LOCAL * gate * local_cost
    )
