"""NumPy energy-phase features aligned with the pendulum MPC formulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wsmpc.environment.dynamics import pendulum_energy, upright_energy, wrap_angle
from wsmpc.utils.config_schema import MPCCostConfig, PendulumConfig

DEFAULT_EPSILON_PHI = 1.0e-6


def natural_velocity_scale(pendulum: PendulumConfig) -> float:
    """Return sqrt(m g l / I), the natural angular velocity scale."""

    inertia = pendulum.mass_kg * pendulum.length_m**2
    return float(np.sqrt(pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m / inertia))


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
    omega_scale = natural_velocity_scale(pendulum)
    return np.stack((wrap_angle(theta_rad), omega_rad_s / omega_scale), axis=-1)


def phase_proxy(
    state: ArrayLike,
    pendulum: PendulumConfig,
    *,
    epsilon_phi: float = DEFAULT_EPSILON_PHI,
) -> NDArray[np.float64]:
    """Return the smooth phase proxy [c_phi, s_phi] from the raw state."""

    state_array = np.asarray(state, dtype=np.float64)
    theta_rad = state_array[..., 0]
    omega_rad_s = state_array[..., 1]
    a_theta = np.cos(0.5 * theta_rad)
    b_phi = omega_rad_s / natural_velocity_scale(pendulum)
    radius = np.sqrt(a_theta**2 + b_phi**2 + epsilon_phi**2)
    return np.stack((a_theta / radius, b_phi / radius), axis=-1)


def phase_proxy_error(
    state: ArrayLike,
    pendulum: PendulumConfig,
    *,
    epsilon_phi: float = DEFAULT_EPSILON_PHI,
) -> NDArray[np.float64]:
    """Return the phase proxy error [c_phi - 1, s_phi]."""

    proxy = phase_proxy(state, pendulum, epsilon_phi=epsilon_phi)
    return np.stack((proxy[..., 0] - 1.0, proxy[..., 1]), axis=-1)


def energy_gate(
    state: ArrayLike,
    pendulum: PendulumConfig,
    cost: MPCCostConfig,
) -> NDArray[np.float64]:
    """Return the energy-shell gate that activates phase and local terms."""

    energy_error = normalized_energy_error(state, pendulum)
    return np.exp(-(energy_error**2) / (cost.sigma_energy**2))


def energy_phase_value(
    state: ArrayLike,
    pendulum: PendulumConfig,
    cost: MPCCostConfig,
) -> NDArray[np.float64]:
    """Evaluate the smooth energy-phase value candidate."""

    energy_error = normalized_energy_error(state, pendulum)
    phase_error = phase_proxy_error(state, pendulum, epsilon_phi=cost.epsilon_phi)
    local_error = local_upright_error(state, pendulum)
    gate = energy_gate(state, pendulum, cost)

    # Diagonal quadratic forms keep the implementation transparent and vectorized.
    phase_weight = np.asarray(cost.q_phase_diag, dtype=np.float64)
    local_weight = np.asarray(cost.q_local_diag, dtype=np.float64)
    phase_cost = np.sum(phase_weight * phase_error**2, axis=-1)
    local_cost = np.sum(local_weight * local_error**2, axis=-1)
    return (
        cost.q_energy * energy_error**2
        + cost.q_phase * gate * phase_cost
        + cost.q_local * gate * local_cost
    )
