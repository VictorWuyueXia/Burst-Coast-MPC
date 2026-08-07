"""State-derived oscillator phase coordinates for rotary-pendulum diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rotary_pendulum.environment.dynamics import ModelConstants
from rotary_pendulum.utils.config_schema import RotaryPendulumConfig

PHASE_EPSILON = 1.0e-6
PHASE_HISTORY_S = 10.0


def oscillator_phase_points(
    states: ArrayLike,
    physical: RotaryPendulumConfig,
    model: ModelConstants,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map arm and pendulum states onto their physical-radius phase rings."""

    # Normalize displacement-velocity pairs so one oscillation becomes one phase circuit.
    state_array = np.asarray(states, dtype=np.float64)
    theta_rad = state_array[..., 0]
    alpha_rad = state_array[..., 1]
    omega_rad_s = state_array[..., 2]
    nu_rad_s = state_array[..., 3]
    natural_frequency_rad_s = model.natural_frequency_rad_s
    arm_basis = np.stack((theta_rad, omega_rad_s / natural_frequency_rad_s), axis=-1)
    pendulum_basis = np.stack(
        (
            np.sin(0.5 * alpha_rad),
            nu_rad_s / (2.0 * natural_frequency_rad_s),
        ),
        axis=-1,
    )
    arm_norm = np.sqrt(np.sum(arm_basis**2, axis=-1, keepdims=True) + PHASE_EPSILON**2)
    pendulum_norm = np.sqrt(
        np.sum(pendulum_basis**2, axis=-1, keepdims=True) + PHASE_EPSILON**2
    )
    return (
        np.asarray(physical.arm_length_m * arm_basis / arm_norm),
        np.asarray(physical.pendulum_length_m * pendulum_basis / pendulum_norm),
    )
