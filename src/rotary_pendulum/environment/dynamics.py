"""Vectorized nonlinear dynamics and geometry for the rotary pendulum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rotary_pendulum.utils.config_schema import RotaryPendulumConfig


@dataclass(frozen=True)
class ModelConstants:
    """Derived inertial, gravitational, and natural-time-scale constants."""

    pendulum_com_length_m: float
    arm_inertia_kg_m2: float
    pendulum_com_inertia_kg_m2: float
    pendulum_inertia_kg_m2: float
    base_inertia_kg_m2: float
    coupling_inertia_kg_m2: float
    gravity_torque_nm: float
    vertical_mass_determinant_kg2_m4: float
    natural_frequency_rad_s: float
    natural_period_s: float


def derive_model(physical: RotaryPendulumConfig) -> ModelConstants:
    """Derive every model coefficient from the primitive physical configuration."""

    # Derive uniform-link inertias and the arm-pendulum coupling from measured geometry.
    pendulum_com_length_m = 0.5 * physical.pendulum_length_m
    arm_inertia_kg_m2 = physical.arm_mass_kg * physical.arm_length_m**2 / 3.0
    pendulum_com_inertia_kg_m2 = physical.pendulum_mass_kg * physical.pendulum_length_m**2 / 12.0
    pendulum_inertia_kg_m2 = pendulum_com_inertia_kg_m2 + (
        physical.pendulum_mass_kg * pendulum_com_length_m**2
    )
    base_inertia_kg_m2 = arm_inertia_kg_m2 + (physical.pendulum_mass_kg * physical.arm_length_m**2)
    coupling_inertia_kg_m2 = (
        physical.pendulum_mass_kg * physical.arm_length_m * pendulum_com_length_m
    )
    gravity_torque_nm = physical.pendulum_mass_kg * physical.gravity_m_s2 * pendulum_com_length_m

    # The vertical determinant establishes well-posedness and the coupled natural period.
    vertical_mass_determinant_kg2_m4 = (
        base_inertia_kg_m2 * pendulum_inertia_kg_m2 - coupling_inertia_kg_m2**2
    )
    natural_frequency_rad_s = np.sqrt(
        base_inertia_kg_m2 * gravity_torque_nm / vertical_mass_determinant_kg2_m4
    )
    return ModelConstants(
        pendulum_com_length_m=pendulum_com_length_m,
        arm_inertia_kg_m2=arm_inertia_kg_m2,
        pendulum_com_inertia_kg_m2=pendulum_com_inertia_kg_m2,
        pendulum_inertia_kg_m2=pendulum_inertia_kg_m2,
        base_inertia_kg_m2=base_inertia_kg_m2,
        coupling_inertia_kg_m2=coupling_inertia_kg_m2,
        gravity_torque_nm=gravity_torque_nm,
        vertical_mass_determinant_kg2_m4=vertical_mass_determinant_kg2_m4,
        natural_frequency_rad_s=float(natural_frequency_rad_s),
        natural_period_s=float(2.0 * np.pi / natural_frequency_rad_s),
    )


def state_derivative(
    state: ArrayLike,
    torque_nm: ArrayLike,
    physical: RotaryPendulumConfig,
    model: ModelConstants,
) -> NDArray[np.float64]:
    """Evaluate the coupled governing ODE for scalar or batched four-state inputs."""

    # Preserve every leading batch dimension while extracting the four physical states.
    state_array = np.asarray(state, dtype=np.float64)
    if state_array.ndim == 0 or state_array.shape[-1] != 4:
        raise ValueError(f"Rotary-pendulum state must end in four entries, got {state_array.shape}")
    _, alpha_rad, omega_rad_s, nu_rad_s = np.moveaxis(state_array, -1, 0)
    sin_alpha = np.sin(alpha_rad)
    cos_alpha = np.cos(alpha_rad)
    applied_torque_nm = np.broadcast_to(np.asarray(torque_nm, dtype=np.float64), alpha_rad.shape)

    # Assemble the symmetric inertia matrix and physically named generalized torques.
    mass_matrix = np.empty(alpha_rad.shape + (2, 2), dtype=np.float64)
    mass_matrix[..., 0, 0] = model.base_inertia_kg_m2 + model.pendulum_inertia_kg_m2 * sin_alpha**2
    mass_matrix[..., 0, 1] = model.coupling_inertia_kg_m2 * cos_alpha
    mass_matrix[..., 1, 0] = mass_matrix[..., 0, 1]
    mass_matrix[..., 1, 1] = model.pendulum_inertia_kg_m2
    generalized_torque = np.stack(
        (
            applied_torque_nm
            - physical.rotary_damping_nms * omega_rad_s
            - 2.0 * model.pendulum_inertia_kg_m2 * sin_alpha * cos_alpha * omega_rad_s * nu_rad_s
            + model.coupling_inertia_kg_m2 * sin_alpha * nu_rad_s**2,
            -physical.pendulum_damping_nms * nu_rad_s
            + model.pendulum_inertia_kg_m2 * sin_alpha * cos_alpha * omega_rad_s**2
            - model.gravity_torque_nm * sin_alpha,
        ),
        axis=-1,
    )

    # Solve every independent two-coordinate inertia system through NumPy's compiled kernel.
    accelerations = np.linalg.solve(mass_matrix, generalized_torque[..., None])[..., 0]
    return np.stack(
        (omega_rad_s, nu_rad_s, accelerations[..., 0], accelerations[..., 1]),
        axis=-1,
    )


def rk4_step(
    state: ArrayLike,
    torque_nm: ArrayLike,
    timestep_s: float,
    physical: RotaryPendulumConfig,
    model: ModelConstants,
) -> NDArray[np.float64]:
    """Advance the nonlinear plant by one fixed fourth-order Runge-Kutta step."""

    # Hold shaft torque constant across the four stages of one control interval.
    state_array = np.asarray(state, dtype=np.float64)
    k1 = state_derivative(state_array, torque_nm, physical, model)
    k2 = state_derivative(state_array + 0.5 * timestep_s * k1, torque_nm, physical, model)
    k3 = state_derivative(state_array + 0.5 * timestep_s * k2, torque_nm, physical, model)
    k4 = state_derivative(state_array + timestep_s * k3, torque_nm, physical, model)
    return state_array + timestep_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def energy_components(
    state: ArrayLike,
    physical: RotaryPendulumConfig,
    model: ModelConstants,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return kinetic, potential, and total mechanical energy in joules."""

    # Evaluate the nonlinear kinetic cross term and downward-referenced potential energy.
    state_array = np.asarray(state, dtype=np.float64)
    alpha_rad = state_array[..., 1]
    omega_rad_s = state_array[..., 2]
    nu_rad_s = state_array[..., 3]
    kinetic_energy_j = (
        0.5
        * (model.base_inertia_kg_m2 + model.pendulum_inertia_kg_m2 * np.sin(alpha_rad) ** 2)
        * omega_rad_s**2
        + model.coupling_inertia_kg_m2 * np.cos(alpha_rad) * omega_rad_s * nu_rad_s
        + 0.5 * model.pendulum_inertia_kg_m2 * nu_rad_s**2
    )
    potential_energy_j = model.gravity_torque_nm * (1.0 - np.cos(alpha_rad))
    total_energy_j = kinetic_energy_j + potential_energy_j
    return (
        np.asarray(kinetic_energy_j),
        np.asarray(potential_energy_j),
        np.asarray(total_energy_j),
    )


def mechanism_points(
    state: ArrayLike,
    physical: RotaryPendulumConfig,
) -> NDArray[np.float64]:
    """Return origin, arm pivot, pendulum center, and tip Cartesian points."""

    # Reconstruct the moving radial and tangential basis directly from both angles.
    state_array = np.asarray(state, dtype=np.float64)
    theta_rad = state_array[..., 0]
    alpha_rad = state_array[..., 1]
    zeros = np.zeros_like(theta_rad)
    radial = np.stack((np.cos(theta_rad), np.sin(theta_rad), zeros), axis=-1)
    tangent = np.stack((-np.sin(theta_rad), np.cos(theta_rad), zeros), axis=-1)
    vertical = np.stack((zeros, zeros, np.ones_like(theta_rad)), axis=-1)
    pendulum_direction = np.sin(alpha_rad)[..., None] * tangent - (
        np.cos(alpha_rad)[..., None] * vertical
    )

    # Return one contiguous point tensor for scalar rendering and batched analysis.
    origin = np.zeros_like(radial)
    pivot = physical.arm_length_m * radial
    center = pivot + 0.5 * physical.pendulum_length_m * pendulum_direction
    tip = pivot + physical.pendulum_length_m * pendulum_direction
    return np.stack((origin, pivot, center, tip), axis=-2)
