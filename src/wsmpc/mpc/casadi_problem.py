"""CasADi split-ratio MPC formulation, solve, and open-loop replay."""

from __future__ import annotations

import math
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from wsmpc.environment.dynamics import rk4_step
from wsmpc.mpc.types import CandidateSolution, SplitCandidate
from wsmpc.utils.config_schema import EnvironmentConfig, MPCConfig, MPCCostConfig, PendulumConfig
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
    started_at = monotonic_s()
    solver, initial_guess, lower_bounds, upper_bounds = build_single_shooting_problem(
        state, previous_input_nm, candidate, environment, mpc
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
    mpc: MPCConfig,
) -> tuple[Any, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x0 = np.asarray(state, dtype=np.float64).reshape(2)
    decision = ca.MX.sym("v", candidate.burst_steps)
    x = ca.vertcat(float(x0[0]), float(x0[1]))
    objective = 0.0

    previous_u = float(previous_input_nm)
    for index in range(candidate.burst_steps):
        u = decision[index]
        objective += _burst_stage_cost(x, u, previous_u, environment, mpc)
        x = rk4_step_symbolic(
            x,
            u,
            environment.simulation.timestep_s,
            environment.pendulum,
        )
        previous_u = u

    for _ in range(candidate.coast_steps):
        objective += energy_phase_value(x, environment.pendulum, mpc.cost)
        x = rk4_step_symbolic(
            x,
            0.0,
            environment.simulation.timestep_s,
            environment.pendulum,
        )

    objective += mpc.cost.q_terminal * energy_phase_value(x, environment.pendulum, mpc.cost)

    nlp = {"x": decision, "f": objective}
    solver = ca.nlpsol("split_ratio_mpc", "ipopt", nlp, IPOPT_OPTIONS)
    torque_limit = environment.pendulum.torque_limit_nm

    return (
        solver,
        np.zeros(candidate.burst_steps, dtype=np.float64),
        np.full(candidate.burst_steps, -torque_limit, dtype=np.float64),
        np.full(candidate.burst_steps, torque_limit, dtype=np.float64),
    )


def _burst_stage_cost(
    x: Any,
    u: Any,
    previous_u: Any,
    environment: EnvironmentConfig,
    mpc: MPCConfig,
) -> Any:
    torque_limit = environment.pendulum.torque_limit_nm
    normalized_u = u / torque_limit
    normalized_delta_u = (u - previous_u) / torque_limit
    saturation_attraction = (1.0 - normalized_u**2) ** 2
    smoothness = normalized_delta_u**2
    return (
        energy_phase_value(x, environment.pendulum, mpc.cost)
        + mpc.cost.rho_saturation * saturation_attraction
        + mpc.cost.rho_delta_u * smoothness
    )


def split_candidates(environment: EnvironmentConfig, mpc: MPCConfig) -> list[SplitCandidate]:
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


def rollout_burst_coast(
    state: ArrayLike,
    burst_inputs_nm: ArrayLike,
    candidate: SplitCandidate,
    environment: EnvironmentConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    burst_array = np.asarray(burst_inputs_nm, dtype=np.float64).reshape(candidate.burst_steps)
    full_inputs_nm = np.zeros(candidate.total_steps, dtype=np.float64)
    full_inputs_nm[: candidate.burst_steps] = burst_array

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


def pendulum_derivatives_symbolic(x: Any, u_nm: Any, pendulum: PendulumConfig) -> Any:
    theta_rad = x[0]
    omega_rad_s = x[1]
    inertia = pendulum.mass_kg * pendulum.length_m**2
    omega_dot = (
        pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m * ca.sin(theta_rad)
        - pendulum.damping_nms * omega_rad_s
        + u_nm
    ) / inertia
    return ca.vertcat(omega_rad_s, omega_dot)


def rk4_step_symbolic(x: Any, u_nm: Any, timestep_s: float, pendulum: PendulumConfig) -> Any:
    k1 = pendulum_derivatives_symbolic(x, u_nm, pendulum)
    k2 = pendulum_derivatives_symbolic(x + 0.5 * timestep_s * k1, u_nm, pendulum)
    k3 = pendulum_derivatives_symbolic(x + 0.5 * timestep_s * k2, u_nm, pendulum)
    k4 = pendulum_derivatives_symbolic(x + timestep_s * k3, u_nm, pendulum)
    return x + (timestep_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def natural_frequency_rad_s(pendulum: PendulumConfig) -> float:
    inertia = pendulum.mass_kg * pendulum.length_m**2
    return math.sqrt(pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m / inertia)


def normalized_energy_error(x: Any, pendulum: PendulumConfig) -> Any:
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


def local_upright_error(x: Any, pendulum: PendulumConfig) -> Any:
    theta_rad = x[0]
    omega_rad_s = x[1]
    wrapped_theta = ca.atan2(ca.sin(theta_rad), ca.cos(theta_rad))
    normalized_omega = omega_rad_s / natural_frequency_rad_s(pendulum)
    return ca.vertcat(wrapped_theta, normalized_omega)


def phase_proxy_error(x: Any, pendulum: PendulumConfig, cost: MPCCostConfig) -> Any:
    theta_rad = x[0]
    omega_rad_s = x[1]
    a_theta = ca.cos(0.5 * theta_rad)
    b_phi = omega_rad_s / natural_frequency_rad_s(pendulum)
    radius = ca.sqrt(a_theta**2 + b_phi**2 + cost.epsilon_phi**2)
    return ca.vertcat((a_theta / radius) - 1.0, b_phi / radius)


def energy_gate(x: Any, pendulum: PendulumConfig, cost: MPCCostConfig) -> Any:
    energy_error = normalized_energy_error(x, pendulum)
    return ca.exp(-(energy_error**2) / (cost.sigma_energy**2))


def weighted_diagonal_quadratic(vector: Any, diagonal: list[float]) -> Any:
    return diagonal[0] * vector[0] ** 2 + diagonal[1] * vector[1] ** 2


def energy_phase_value(x: Any, pendulum: PendulumConfig, cost: MPCCostConfig) -> Any:
    energy_error = normalized_energy_error(x, pendulum)
    phase_error = phase_proxy_error(x, pendulum, cost)
    local_error = local_upright_error(x, pendulum)
    gate = energy_gate(x, pendulum, cost)
    return (
        cost.q_energy * energy_error**2
        + cost.q_phase * gate * weighted_diagonal_quadratic(phase_error, cost.q_phase_diag)
        + cost.q_local * gate * weighted_diagonal_quadratic(local_error, cost.q_local_diag)
    )
