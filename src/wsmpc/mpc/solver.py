"""CasADi solver wrapper for split-ratio MPC candidates."""

from __future__ import annotations

import math

import numpy as np

from wsmpc.mpc.casadi_problem import build_single_shooting_problem
from wsmpc.mpc.prediction import rollout_burst_coast
from wsmpc.mpc.types import CandidateSolution, CandidateSolveTask, SplitCandidate
from wsmpc.utils.config_schema import EnvironmentConfig, MPCConfig
from wsmpc.utils.time import monotonic_s


def solve_candidate_task(task: CandidateSolveTask) -> CandidateSolution:
    """Solve a candidate task payload, suitable for process-pool execution."""

    return solve_candidate(
        task.state,
        task.previous_input_nm,
        task.candidate,
        task.environment_config,
        task.mpc_config,
        warm_start_nm=task.warm_start_nm,
    )


def solve_candidate(
    state: np.ndarray,
    previous_input_nm: float,
    candidate: SplitCandidate,
    environment: EnvironmentConfig,
    mpc: MPCConfig,
    *,
    warm_start_nm: np.ndarray | None = None,
) -> CandidateSolution:
    """Solve one fixed-dimension CasADi nonlinear program."""

    started_at = monotonic_s()
    try:
        problem = build_single_shooting_problem(
            state,
            previous_input_nm,
            candidate,
            environment,
            mpc,
            warm_start_nm=warm_start_nm,
        )
        raw_solution = problem.solver(
            x0=problem.initial_guess,
            lbx=problem.lower_bounds,
            ubx=problem.upper_bounds,
        )
        stats = problem.solver.stats()
        burst_inputs = np.asarray(raw_solution["x"], dtype=np.float64).reshape(
            candidate.burst_steps
        )
        objective_value = float(raw_solution["f"])
        success = bool(stats.get("success", False)) and math.isfinite(objective_value)
        message = str(stats.get("return_status", "unknown"))
    except Exception as exc:  # pragma: no cover - exercised by controller fallback tests.
        return _failed_solution(
            state,
            candidate,
            environment,
            monotonic_s() - started_at,
            f"{type(exc).__name__}: {exc}",
        )

    predicted_states, predicted_inputs = rollout_burst_coast(
        state,
        burst_inputs,
        candidate,
        environment,
    )
    return CandidateSolution(
        candidate=candidate,
        success=success,
        objective_value=objective_value if success else math.inf,
        burst_inputs_nm=burst_inputs,
        predicted_states=predicted_states,
        predicted_inputs_nm=predicted_inputs,
        solve_time_s=monotonic_s() - started_at,
        message=message,
    )


def _failed_solution(
    state: np.ndarray,
    candidate: SplitCandidate,
    environment: EnvironmentConfig,
    solve_time_s: float,
    message: str,
) -> CandidateSolution:
    """Return a finite diagnostic rollout with an infinite objective."""

    burst_inputs = np.zeros(candidate.burst_steps, dtype=np.float64)
    predicted_states, predicted_inputs = rollout_burst_coast(
        state,
        burst_inputs,
        candidate,
        environment,
    )
    return CandidateSolution(
        candidate=candidate,
        success=False,
        objective_value=math.inf,
        burst_inputs_nm=burst_inputs,
        predicted_states=predicted_states,
        predicted_inputs_nm=predicted_inputs,
        solve_time_s=solve_time_s,
        message=message,
    )
