"""Simplified energy and local momentum features."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from wsmpc.environment.dynamics import pendulum_energy, upright_energy, wrap_angle
from wsmpc.mpc.discrete_model import natural_frequency_rad_s
from wsmpc.mpc.ip_energy_event_triggered import weights
from wsmpc.utils.config_schema import PendulumConfig


def normalized_energy_error(state: ArrayLike, pendulum: PendulumConfig) -> NDArray[np.float64]:
    """Compute normalized energy error with upright rest as zero."""

    state_array = np.asarray(state, dtype=np.float64)
    theta_rad = state_array[..., 0]
    omega_rad_s = state_array[..., 1]
    target_energy_j = upright_energy(pendulum)
    return (pendulum_energy(theta_rad, omega_rad_s, pendulum) - target_energy_j) / target_energy_j


def normalized_momentum_error(state: ArrayLike, pendulum: PendulumConfig) -> NDArray[np.float64]:
    """Return angular momentum error normalized by natural momentum scale."""

    state_array = np.asarray(state, dtype=np.float64)
    return state_array[..., 1] / natural_frequency_rad_s(pendulum)


def local_upright_gate(state: ArrayLike) -> NDArray[np.float64]:
    """Return the upright-local gate from wrapped angle error."""

    state_array = np.asarray(state, dtype=np.float64)
    theta_w = wrap_angle(state_array[..., 0])
    return np.exp(-(theta_w**2) / (weights.SIGMA_THETA_RAD**2))


def energy_momentum_value(state: ArrayLike, pendulum: PendulumConfig) -> NDArray[np.float64]:
    """Evaluate the state-only energy and local momentum objective."""

    energy_error = normalized_energy_error(state, pendulum)
    momentum_error = normalized_momentum_error(state, pendulum)
    gate = local_upright_gate(state)

    # Momentum capture is intentionally strong only near the upright energy shell.
    momentum_cost = gate**2 * momentum_error**2 / (
        energy_error**2 + weights.EPSILON_ENERGY**2
    )
    return weights.W_ENERGY * energy_error**2 + weights.W_MOMENTUM * momentum_cost


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


def normalized_momentum_error_symbolic(x: Any, pendulum: PendulumConfig) -> Any:
    """Return symbolic normalized angular momentum error."""

    return x[1] / natural_frequency_rad_s(pendulum)


def local_upright_gate_symbolic(x: Any) -> Any:
    """Return symbolic upright-local gate from wrapped angle error."""

    theta_w = ca.atan2(ca.sin(x[0]), ca.cos(x[0]))
    return ca.exp(-(theta_w**2) / (weights.SIGMA_THETA_RAD**2))


def energy_momentum_value_symbolic(x: Any, pendulum: PendulumConfig) -> Any:
    """Evaluate symbolic state-only energy and local momentum objective."""

    energy_error = normalized_energy_error_symbolic(x, pendulum)
    momentum_error = normalized_momentum_error_symbolic(x, pendulum)
    gate = local_upright_gate_symbolic(x)

    # The denominator suppresses momentum capture until energy is nearly correct.
    momentum_cost = gate**2 * momentum_error**2 / (
        energy_error**2 + weights.EPSILON_ENERGY**2
    )
    return weights.W_ENERGY * energy_error**2 + weights.W_MOMENTUM * momentum_cost
