"""CasADi symbolic energy-phase features for the pendulum MPC objective."""

from __future__ import annotations

from typing import Any

import casadi as ca

from wsmpc.utils.config_schema import MPCCostConfig, PendulumConfig


def natural_velocity_scale(pendulum: PendulumConfig) -> float:
    """Return sqrt(m g l / I), matching the NumPy feature implementation."""

    inertia = pendulum.mass_kg * pendulum.length_m**2
    return (pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m / inertia) ** 0.5


def normalized_energy_error(x: Any, pendulum: PendulumConfig) -> Any:
    """Build the normalized mechanical energy error expression."""

    theta_rad = x[0]
    omega_rad_s = x[1]
    inertia = pendulum.mass_kg * pendulum.length_m**2
    kinetic = 0.5 * inertia * omega_rad_s**2
    potential = pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m * (
        1.0 + ca.cos(theta_rad)
    )
    target_energy = 2.0 * pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m
    return (kinetic + potential - target_energy) / target_energy


def local_upright_error(x: Any, pendulum: PendulumConfig) -> Any:
    """Build [atan2(sin theta, cos theta), omega / omega_s]."""

    theta_rad = x[0]
    omega_rad_s = x[1]
    wrapped_theta = ca.atan2(ca.sin(theta_rad), ca.cos(theta_rad))
    normalized_omega = omega_rad_s / natural_velocity_scale(pendulum)
    return ca.vertcat(wrapped_theta, normalized_omega)


def phase_proxy_error(x: Any, pendulum: PendulumConfig, cost: MPCCostConfig) -> Any:
    """Build the phase proxy error [c_phi - 1, s_phi]."""

    theta_rad = x[0]
    omega_rad_s = x[1]
    a_theta = ca.cos(0.5 * theta_rad)
    b_phi = omega_rad_s / natural_velocity_scale(pendulum)
    radius = ca.sqrt(a_theta**2 + b_phi**2 + cost.epsilon_phi**2)
    return ca.vertcat((a_theta / radius) - 1.0, b_phi / radius)


def energy_gate(x: Any, pendulum: PendulumConfig, cost: MPCCostConfig) -> Any:
    """Build the smooth gate that activates phase and local terms near target energy."""

    energy_error = normalized_energy_error(x, pendulum)
    return ca.exp(-(energy_error**2) / (cost.sigma_energy**2))


def weighted_diagonal_quadratic(vector: Any, diagonal: list[float]) -> Any:
    """Build a two-dimensional diagonal quadratic form."""

    return diagonal[0] * vector[0] ** 2 + diagonal[1] * vector[1] ** 2


def energy_phase_value(x: Any, pendulum: PendulumConfig, cost: MPCCostConfig) -> Any:
    """Build the complete smooth energy-phase value candidate expression."""

    energy_error = normalized_energy_error(x, pendulum)
    phase_error = phase_proxy_error(x, pendulum, cost)
    local_error = local_upright_error(x, pendulum)
    gate = energy_gate(x, pendulum, cost)
    return (
        cost.q_energy * energy_error**2
        + cost.q_phase * gate * weighted_diagonal_quadratic(phase_error, cost.q_phase_diag)
        + cost.q_local * gate * weighted_diagonal_quadratic(local_error, cost.q_local_diag)
    )
