"""Shared discrete rotary-pendulum model used by MPC prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import casadi as ca  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import ArrayLike, NDArray

from rotary_pendulum.environment.dynamics import derive_model, rk4_step
from rotary_pendulum.utils.config_schema import EpisodeConfig, MPCConfig


@dataclass(frozen=True)
class SplitCandidate:
    """One fixed burst prefix within the common prediction horizon."""

    split_ratio: float
    total_steps: int
    burst_steps: int

    @property
    def coast_steps(self) -> int:
        """Return the exact zero-torque suffix length."""

        return self.total_steps - self.burst_steps


def state_derivative_symbolic(x: Any, torque_nm: Any, physics: EpisodeConfig) -> Any:
    """Evaluate the documented four-state governing ODE in CasADi form."""

    # Derive every inertial coefficient from the same primitive physics as the simulator.
    physical = physics.rotary_pendulum
    model = derive_model(physical)
    alpha_rad = x[1]
    omega_rad_s = x[2]
    nu_rad_s = x[3]
    sin_alpha = ca.sin(alpha_rad)
    cos_alpha = ca.cos(alpha_rad)

    # Solve the coupled two-coordinate inertia system directly inside the symbolic graph.
    mass_11 = model.base_inertia_kg_m2 + model.pendulum_inertia_kg_m2 * sin_alpha**2
    mass_12 = model.coupling_inertia_kg_m2 * cos_alpha
    mass_matrix = ca.vertcat(
        ca.horzcat(mass_11, mass_12),
        ca.horzcat(mass_12, model.pendulum_inertia_kg_m2),
    )
    generalized_torque = ca.vertcat(
        torque_nm
        - physical.rotary_damping_nms * omega_rad_s
        - 2.0 * model.pendulum_inertia_kg_m2 * sin_alpha * cos_alpha * omega_rad_s * nu_rad_s
        + model.coupling_inertia_kg_m2 * sin_alpha * nu_rad_s**2,
        -physical.pendulum_damping_nms * nu_rad_s
        + model.pendulum_inertia_kg_m2 * sin_alpha * cos_alpha * omega_rad_s**2
        - model.gravity_torque_nm * sin_alpha,
    )
    acceleration = ca.solve(mass_matrix, generalized_torque)
    return ca.vertcat(omega_rad_s, nu_rad_s, acceleration[0], acceleration[1])


def rk4_step_symbolic(x: Any, torque_nm: Any, physics: EpisodeConfig) -> Any:
    """Advance the symbolic state with the simulator's fixed-step RK4 map."""

    # Hold shaft torque constant through all four slopes of one control interval.
    timestep_s = physics.simulation.timestep_s
    k1 = state_derivative_symbolic(x, torque_nm, physics)
    k2 = state_derivative_symbolic(x + 0.5 * timestep_s * k1, torque_nm, physics)
    k3 = state_derivative_symbolic(x + 0.5 * timestep_s * k2, torque_nm, physics)
    k4 = state_derivative_symbolic(x + timestep_s * k3, torque_nm, physics)
    return x + timestep_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def prediction_horizon_steps(physics: EpisodeConfig, mpc: MPCConfig) -> int:
    """Convert the configured natural-period horizon into discrete prediction steps."""

    model = derive_model(physics.rotary_pendulum)
    return int(
        math.ceil(
            mpc.prediction_horizon_natural_periods
            * model.natural_period_s
            / physics.simulation.timestep_s
        )
    )


def split_candidates(physics: EpisodeConfig, mpc: MPCConfig) -> list[SplitCandidate]:
    """Build the configured burst partitions in stable configuration order."""

    total_steps = prediction_horizon_steps(physics, mpc)
    candidates = [
        SplitCandidate(
            split_ratio=split_ratio,
            total_steps=total_steps,
            burst_steps=round(split_ratio * total_steps),
        )
        for split_ratio in mpc.split_ratios
    ]
    if any(
        candidate.burst_steps <= 0 or candidate.burst_steps > total_steps
        for candidate in candidates
    ):
        raise ValueError("Every split ratio must produce a burst length in [1, H]")
    return candidates


def rollout_burst_coast(
    state: ArrayLike,
    burst_inputs_nm: ArrayLike,
    candidate: SplitCandidate,
    physics: EpisodeConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Replay one optimized torque prefix and exact zero-torque suffix numerically."""

    # Materialize the enforced burst-coast input structure before propagating the plant.
    burst_inputs = np.asarray(burst_inputs_nm, dtype=np.float64).reshape(candidate.burst_steps)
    predicted_inputs_nm: NDArray[np.float64] = np.zeros(
        candidate.total_steps,
        dtype=np.float64,
    )
    predicted_inputs_nm[: candidate.burst_steps] = burst_inputs

    # Use the simulator's numeric RK4 implementation for the accepted plan trajectory.
    physical = physics.rotary_pendulum
    model = derive_model(physical)
    predicted_states: NDArray[np.float64] = np.empty(
        (candidate.total_steps + 1, 4),
        dtype=np.float64,
    )
    predicted_states[0] = np.asarray(state, dtype=np.float64).reshape(4)
    for index, torque_nm in enumerate(predicted_inputs_nm):
        predicted_states[index + 1] = rk4_step(
            predicted_states[index],
            float(torque_nm),
            physics.simulation.timestep_s,
            physical,
            model,
        )
    return predicted_states, predicted_inputs_nm
