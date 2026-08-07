"""Rotary swing-energy, phase-chasing, capture, and actuator costs."""

from __future__ import annotations

from typing import Any

import casadi as ca  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import ArrayLike, NDArray

from rotary_pendulum.environment.dynamics import derive_model, state_derivative
from rotary_pendulum.mpc.discrete_model import state_derivative_symbolic
from rotary_pendulum.utils.config_schema import EpisodeConfig, MPCConfig

ARM_ANGLE_LIMIT_RAD = 0.5 * np.pi
PHASE_ORIGIN_ENERGY_RATIO = 0.05
PHASE_ORIGIN_SPEED_RATIO = 0.01


def objective_terms(
    state: ArrayLike,
    torque_nm: ArrayLike,
    physics: EpisodeConfig,
    mpc: MPCConfig,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return terminal-energy, phase, local-capture, and arm-limit costs."""

    # Derive the intrinsic pendulum swing energy without arm or moving-pivot energy.
    state_array = np.asarray(state, dtype=np.float64)
    alpha_rad = state_array[..., 1]
    omega_rad_s = state_array[..., 2]
    nu_rad_s = state_array[..., 3]
    physical = physics.rotary_pendulum
    model = derive_model(physical)
    swing_energy_j = (
        0.5 * model.pendulum_inertia_kg_m2 * nu_rad_s**2
        + model.gravity_torque_nm * (1.0 - np.cos(alpha_rad))
    )
    target_energy_j = 2.0 * model.gravity_torque_nm
    energy_error = (swing_energy_j - target_energy_j) / target_energy_j
    energy_ratio = swing_energy_j / target_energy_j
    terminal_energy_cost = mpc.terminal_swing_energy_weight * energy_error**2

    # Chase the sign-controllable arm-acceleration phase while energy is off target.
    energy_command = -np.tanh(energy_error / mpc.energy_transition_width)
    energy_speed_rad_s = 2.0 * np.sqrt(
        model.gravity_torque_nm / model.pendulum_inertia_kg_m2
    )
    phase_origin_speed_rad_s = PHASE_ORIGIN_SPEED_RATIO * energy_speed_rad_s
    phase_velocity_rad_s = nu_rad_s + phase_origin_speed_rad_s * np.exp(
        -(energy_ratio / PHASE_ORIGIN_ENERGY_RATIO) ** 2
    )
    coupling_velocity_rad_s = np.cos(alpha_rad) * phase_velocity_rad_s
    phase_signal = coupling_velocity_rad_s / np.sqrt(
        coupling_velocity_rad_s**2 + phase_origin_speed_rad_s**2
    )
    determinant = (
        model.pendulum_inertia_kg_m2
        * (
            model.base_inertia_kg_m2
            + model.pendulum_inertia_kg_m2 * np.sin(alpha_rad) ** 2
        )
        - model.coupling_inertia_kg_m2**2 * np.cos(alpha_rad) ** 2
    )
    acceleration_authority_rad_s2 = (
        model.pendulum_inertia_kg_m2 * physical.torque_limit_nm / determinant
    )
    arm_acceleration_rad_s2 = state_derivative(
        state_array,
        torque_nm,
        physical,
        model,
    )[..., 2]
    phase_error = (
        arm_acceleration_rad_s2 / acceleration_authority_rad_s2
        + energy_command * phase_signal
    )
    phase_cost = (
        mpc.phase_chasing_weight
        * energy_command**2
        * np.cos(alpha_rad) ** 2
        * phase_error**2
    )

    # Activate weak terminal state capture only near the target swing-energy shell.
    beta_rad = np.arctan2(np.sin(alpha_rad - np.pi), np.cos(alpha_rad - np.pi))
    pendulum_local_error = (beta_rad / np.pi) ** 2 + (nu_rad_s / energy_speed_rad_s) ** 2
    rotary_local_error = (state_array[..., 0] / ARM_ANGLE_LIMIT_RAD) ** 2 + (
        omega_rad_s / (ARM_ANGLE_LIMIT_RAD * model.natural_frequency_rad_s)
    ) ** 2
    local_gate = np.exp(-(energy_error / mpc.local_energy_shell_width) ** 2)
    local_capture_cost = local_gate * (
        mpc.pendulum_local_weight * pendulum_local_error
        + mpc.rotary_local_weight * rotary_local_error
    )

    # Preserve the symmetric high-penalty arm-angle operating envelope.
    theta_rad = state_array[..., 0]
    arm_excess = np.maximum(np.abs(theta_rad) - ARM_ANGLE_LIMIT_RAD, 0.0)
    arm_limit_cost = mpc.arm_angle_soft_penalty_weight * (
        arm_excess / ARM_ANGLE_LIMIT_RAD
    ) ** 2
    return tuple(
        np.asarray(term, dtype=np.float64)
        for term in (terminal_energy_cost, phase_cost, local_capture_cost, arm_limit_cost)
    )  # type: ignore[return-value]


def objective_terms_symbolic(
    x: Any,
    torque_nm: Any,
    physics: EpisodeConfig,
    mpc: MPCConfig,
) -> tuple[Any, Any, Any, Any]:
    """Return the four documented MPC state costs in CasADi form."""

    # Mirror the numeric expressions directly so symbolic and artifact audits remain exact.
    physical = physics.rotary_pendulum
    model = derive_model(physical)
    theta_rad, alpha_rad, omega_rad_s, nu_rad_s = x[0], x[1], x[2], x[3]
    swing_energy_j = (
        0.5 * model.pendulum_inertia_kg_m2 * nu_rad_s**2
        + model.gravity_torque_nm * (1.0 - ca.cos(alpha_rad))
    )
    target_energy_j = 2.0 * model.gravity_torque_nm
    energy_error = (swing_energy_j - target_energy_j) / target_energy_j
    energy_ratio = swing_energy_j / target_energy_j
    terminal_energy_cost = mpc.terminal_swing_energy_weight * energy_error**2

    energy_command = -ca.tanh(energy_error / mpc.energy_transition_width)
    energy_speed_rad_s = 2.0 * np.sqrt(
        model.gravity_torque_nm / model.pendulum_inertia_kg_m2
    )
    phase_origin_speed_rad_s = PHASE_ORIGIN_SPEED_RATIO * energy_speed_rad_s
    phase_velocity_rad_s = nu_rad_s + phase_origin_speed_rad_s * ca.exp(
        -(energy_ratio / PHASE_ORIGIN_ENERGY_RATIO) ** 2
    )
    coupling_velocity_rad_s = ca.cos(alpha_rad) * phase_velocity_rad_s
    phase_signal = coupling_velocity_rad_s / ca.sqrt(
        coupling_velocity_rad_s**2 + phase_origin_speed_rad_s**2
    )
    determinant = (
        model.pendulum_inertia_kg_m2
        * (
            model.base_inertia_kg_m2
            + model.pendulum_inertia_kg_m2 * ca.sin(alpha_rad) ** 2
        )
        - model.coupling_inertia_kg_m2**2 * ca.cos(alpha_rad) ** 2
    )
    acceleration_authority_rad_s2 = (
        model.pendulum_inertia_kg_m2 * physical.torque_limit_nm / determinant
    )
    arm_acceleration_rad_s2 = state_derivative_symbolic(x, torque_nm, physics)[2]
    phase_error = (
        arm_acceleration_rad_s2 / acceleration_authority_rad_s2
        + energy_command * phase_signal
    )
    phase_cost = (
        mpc.phase_chasing_weight
        * energy_command**2
        * ca.cos(alpha_rad) ** 2
        * phase_error**2
    )

    beta_rad = ca.atan2(ca.sin(alpha_rad - np.pi), ca.cos(alpha_rad - np.pi))
    pendulum_local_error = (beta_rad / np.pi) ** 2 + (nu_rad_s / energy_speed_rad_s) ** 2
    rotary_local_error = (theta_rad / ARM_ANGLE_LIMIT_RAD) ** 2 + (
        omega_rad_s / (ARM_ANGLE_LIMIT_RAD * model.natural_frequency_rad_s)
    ) ** 2
    local_gate = ca.exp(-(energy_error / mpc.local_energy_shell_width) ** 2)
    local_capture_cost = local_gate * (
        mpc.pendulum_local_weight * pendulum_local_error
        + mpc.rotary_local_weight * rotary_local_error
    )

    upper_excess = ca.fmax(theta_rad - ARM_ANGLE_LIMIT_RAD, 0.0)
    lower_excess = ca.fmax(-theta_rad - ARM_ANGLE_LIMIT_RAD, 0.0)
    arm_limit_cost = mpc.arm_angle_soft_penalty_weight * (
        (upper_excess / ARM_ANGLE_LIMIT_RAD) ** 2
        + (lower_excess / ARM_ANGLE_LIMIT_RAD) ** 2
    )
    return terminal_energy_cost, phase_cost, local_capture_cost, arm_limit_cost


def torque_continuation_cost(
    torque_nm: ArrayLike,
    previous_torque_nm: ArrayLike,
    physics: EpisodeConfig,
    mpc: MPCConfig,
) -> NDArray[np.float64]:
    """Return the normalized active-burst torque-continuation cost."""

    normalized_delta = (
        np.asarray(torque_nm, dtype=np.float64)
        - np.asarray(previous_torque_nm, dtype=np.float64)
    ) / physics.rotary_pendulum.torque_limit_nm
    return np.asarray(mpc.torque_slew_weight * normalized_delta**2)


def torque_continuation_cost_symbolic(
    torque_nm: Any,
    previous_torque_nm: Any,
    physics: EpisodeConfig,
    mpc: MPCConfig,
) -> Any:
    """Return the symbolic active-burst torque-continuation cost."""

    normalized_delta = (
        torque_nm - previous_torque_nm
    ) / physics.rotary_pendulum.torque_limit_nm
    return mpc.torque_slew_weight * normalized_delta**2
