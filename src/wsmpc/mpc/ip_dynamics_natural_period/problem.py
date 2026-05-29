"""Natural-period split-ratio MPC problem."""

from __future__ import annotations

import math
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from wsmpc.mpc.discrete_model import (
    natural_frequency_rad_s,
    rk4_step_symbolic,
    rollout_burst_coast,
)
from wsmpc.mpc.ip_dynamics_natural_period import weights
from wsmpc.mpc.ip_dynamics_natural_period.features import energy_phase_value_symbolic
from wsmpc.mpc.types import CandidateSolution, SplitCandidate
from wsmpc.utils.config_schema import EnvironmentConfig, MPCConfig
from wsmpc.utils.time import monotonic_s

IPOPT_OPTIONS = {
    "print_time": False,
    "ipopt.print_level": 0,
    "ipopt.max_iter": 100,
    "ipopt.tol": 1.0e-6,
}


def solve_candidate_task(
    task: tuple[np.ndarray, float, SplitCandidate, EnvironmentConfig, MPCConfig],
) -> CandidateSolution:
    """Solve one candidate from a process-pool task tuple."""

    state, previous_input_nm, candidate, environment, mpc = task
    return solve_candidate(state, previous_input_nm, candidate, environment, mpc)


def solve_candidate(
    state: np.ndarray,
    previous_input_nm: float,
    candidate: SplitCandidate,
    environment: EnvironmentConfig,
    mpc: MPCConfig,
) -> CandidateSolution:
    """Solve one natural-period split-ratio candidate."""

    started_at = monotonic_s()
    solver, initial_guess, lower_bounds, upper_bounds = build_single_shooting_problem(
        state, previous_input_nm, candidate, environment
    )
    raw_solution = solver(x0=initial_guess, lbx=lower_bounds, ubx=upper_bounds)
    stats = solver.stats()
    objective_value = float(raw_solution["f"])
    message = str(stats["return_status"])
    if not bool(stats["success"]) or not math.isfinite(objective_value):
        msg = f"CasADi solver failed for lambda={candidate.lambda_value}: {message}"
        raise RuntimeError(msg)

    burst_inputs = np.asarray(raw_solution["x"], dtype=np.float64).reshape(candidate.burst_steps)
    predicted_states, predicted_inputs = rollout_burst_coast(
        state,
        burst_inputs,
        candidate,
        environment,
    )
    return CandidateSolution(
        candidate=candidate,
        objective_value=objective_value,
        burst_inputs_nm=burst_inputs,
        predicted_states=predicted_states,
        predicted_inputs_nm=predicted_inputs,
        solve_time_s=monotonic_s() - started_at,
        message=message,
    )


def build_single_shooting_problem(
    state: ArrayLike,
    previous_input_nm: float,
    candidate: SplitCandidate,
    environment: EnvironmentConfig,
) -> tuple[Any, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build the natural-period burst-coast single-shooting NLP."""

    x0 = np.asarray(state, dtype=np.float64).reshape(2)
    decision = ca.MX.sym("v", candidate.burst_steps)
    x = ca.vertcat(float(x0[0]), float(x0[1]))
    objective = 0.0

    previous_u = float(previous_input_nm)
    for index in range(candidate.burst_steps):
        u = decision[index]
        objective += _burst_stage_cost(x, u, previous_u, environment)
        x = rk4_step_symbolic(x, u, environment.simulation.timestep_s, environment.pendulum)
        previous_u = u

    for _ in range(candidate.coast_steps):
        objective += energy_phase_value_symbolic(x, environment.pendulum)
        x = rk4_step_symbolic(x, 0.0, environment.simulation.timestep_s, environment.pendulum)

    objective += weights.Q_TERMINAL * energy_phase_value_symbolic(x, environment.pendulum)

    nlp = {"x": decision, "f": objective}
    solver = ca.nlpsol("natural_period_mpc", "ipopt", nlp, IPOPT_OPTIONS)
    torque_limit = environment.pendulum.torque_limit_nm
    return (
        solver,
        np.zeros(candidate.burst_steps, dtype=np.float64),
        np.full(candidate.burst_steps, -torque_limit, dtype=np.float64),
        np.full(candidate.burst_steps, torque_limit, dtype=np.float64),
    )


def _burst_stage_cost(x: Any, u: Any, previous_u: Any, environment: EnvironmentConfig) -> Any:
    """Evaluate the actuated stage cost for the burst segment."""

    torque_limit = environment.pendulum.torque_limit_nm
    normalized_u = u / torque_limit
    normalized_delta_u = (u - previous_u) / torque_limit
    saturation_attraction = (1.0 - normalized_u**2) ** 2
    smoothness = normalized_delta_u**2
    return (
        energy_phase_value_symbolic(x, environment.pendulum)
        + weights.RHO_SATURATION * saturation_attraction
        + weights.RHO_DELTA_U * smoothness
    )


def split_candidates(environment: EnvironmentConfig, mpc: MPCConfig) -> list[SplitCandidate]:
    """Create fixed-dimension candidates from configured split ratios."""

    total_steps = prediction_horizon_steps(environment)
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


def prediction_horizon_steps(environment: EnvironmentConfig) -> int:
    """Return the fixed half-natural-period horizon in simulator steps."""

    omega_n = natural_frequency_rad_s(environment.pendulum)
    half_period_s = math.pi / omega_n
    return max(1, round(half_period_s / environment.simulation.timestep_s))
