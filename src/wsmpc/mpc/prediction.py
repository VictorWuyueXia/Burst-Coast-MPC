"""Prediction horizon and rollout helpers for split-ratio MPC."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wsmpc.environment.dynamics import rk4_step
from wsmpc.mpc.types import SplitCandidate
from wsmpc.utils.config_schema import EnvironmentConfig, MPCConfig, PendulumConfig


def natural_frequency_rad_s(pendulum: PendulumConfig) -> float:
    """Return sqrt(m g l / I), the small-motion frequency scale."""

    inertia = pendulum.mass_kg * pendulum.length_m**2
    return math.sqrt(pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m / inertia)


def prediction_horizon_steps(environment: EnvironmentConfig, mpc: MPCConfig) -> int:
    """Compute the fixed total prediction horizon in simulator sample counts."""

    if mpc.horizon_steps_override is not None:
        return mpc.horizon_steps_override

    omega_n = natural_frequency_rad_s(environment.pendulum)
    full_period_s = 2.0 * math.pi / omega_n
    half_period_s = 0.5 * full_period_s
    return max(1, round(half_period_s / environment.simulation.timestep_s))


def split_candidates(environment: EnvironmentConfig, mpc: MPCConfig) -> list[SplitCandidate]:
    """Enumerate fixed-dimension split candidates from configured lambda values."""

    total_steps = prediction_horizon_steps(environment, mpc)
    candidates: list[SplitCandidate] = []
    for lambda_value in mpc.split_ratios:
        burst_steps = max(1, round(lambda_value * total_steps))
        candidates.append(
            SplitCandidate(
                lambda_value=lambda_value,
                total_steps=total_steps,
                burst_steps=burst_steps,
                coast_steps=total_steps - burst_steps,
            )
        )
    return candidates


def rollout_burst_coast(
    state: ArrayLike,
    burst_inputs_nm: ArrayLike,
    candidate: SplitCandidate,
    environment: EnvironmentConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Roll out optimized burst inputs followed by the configured zero coast."""

    burst_array = np.asarray(burst_inputs_nm, dtype=np.float64).reshape(candidate.burst_steps)
    full_inputs_nm = np.zeros(candidate.total_steps, dtype=np.float64)
    full_inputs_nm[: candidate.burst_steps] = burst_array

    # The state recurrence is sequential, while each RK4 evaluation is vectorized internally.
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
