"""Single-shooting CasADi solver for rotary burst-coast candidates."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import casadi as ca  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import ArrayLike, NDArray

from rotary_pendulum.mpc.discrete_model import (
    SplitCandidate,
    rk4_step_symbolic,
    rollout_burst_coast,
)
from rotary_pendulum.mpc.features import (
    objective_terms_symbolic,
    torque_continuation_cost_symbolic,
)
from rotary_pendulum.utils.config_schema import EpisodeConfig, MPCConfig

IPOPT_OPTIONS = {
    "print_time": False,
    "ipopt.bound_relax_factor": 0.0,
    "ipopt.print_level": 0,
    "ipopt.max_iter": 300,
    "ipopt.tol": 1.0e-6,
}


@dataclass(frozen=True)
class CandidateResult:
    """Complete optimized result for one configured burst partition."""

    candidate: SplitCandidate
    objective_value: float
    burst_inputs_nm: NDArray[np.float64]
    predicted_states: NDArray[np.float64]
    predicted_inputs_nm: NDArray[np.float64]
    solve_time_s: float
    solver_status: str


@dataclass
class CandidateProblem:
    """Cache one parametric NLP, its bounds, and candidate-local initial guess."""

    candidate: SplitCandidate
    solver: Any
    initial_guess: NDArray[np.float64]
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]


def build_single_shooting_problem(
    candidate: SplitCandidate,
    physics: EpisodeConfig,
    mpc: MPCConfig,
) -> CandidateProblem:
    """Build one fixed-dimension nonlinear burst-coast optimization problem."""

    # Parameterize the measured state and prior torque so the NLP graph is built only once.
    parameter = ca.MX.sym("p", 5)
    burst_input = ca.MX.sym("tau", candidate.burst_steps)
    predicted_state = parameter[:4]
    objective = 0.0
    previous_torque = parameter[4]

    # Retain the exact RK4 and objective expressions as compact CasADi call nodes.
    step_state = ca.MX.sym("step_state", 4)
    step_torque = ca.MX.sym("step_torque")
    discrete_step = ca.Function(
        f"rotary_rk4_{candidate.burst_steps}",
        [step_state, step_torque],
        [rk4_step_symbolic(step_state, step_torque, physics)],
    )
    terminal_energy, phase, local_capture, arm_limit = objective_terms_symbolic(
        step_state,
        step_torque,
        physics,
        mpc,
    )
    active_state_cost = ca.Function(
        f"rotary_active_cost_{candidate.burst_steps}",
        [step_state, step_torque],
        [phase, arm_limit],
    )
    coast_state_cost = ca.Function(
        f"rotary_coast_cost_{candidate.burst_steps}",
        [step_state],
        [arm_limit],
    )
    terminal_state_cost = ca.Function(
        f"rotary_terminal_cost_{candidate.burst_steps}",
        [step_state],
        [terminal_energy + local_capture + arm_limit],
    )

    # Optimize mean phase tracking and continuation only where torque is actively chosen.
    for index in range(candidate.burst_steps):
        torque_nm = burst_input[index]
        phase_cost, arm_cost = active_state_cost(predicted_state, torque_nm)
        objective += phase_cost / candidate.burst_steps + arm_cost
        objective += torque_continuation_cost_symbolic(
            torque_nm,
            previous_torque,
            physics,
            mpc,
        )
        predicted_state = discrete_step(predicted_state, torque_nm)
        previous_torque = torque_nm

    # Propagate exact zero-torque coast under safety cost, then judge the terminal state.
    for _ in range(candidate.coast_steps):
        objective += coast_state_cost(predicted_state)
        predicted_state = discrete_step(predicted_state, 0.0)
    objective += terminal_state_cost(predicted_state)

    # Start each fixed-dimension candidate at the documented all-zero initial guess.
    torque_limit_nm = physics.rotary_pendulum.torque_limit_nm
    solver = ca.nlpsol(
        f"rotary_burst_coast_{candidate.burst_steps}",
        "ipopt",
        {"x": burst_input, "p": parameter, "f": objective},
        IPOPT_OPTIONS,
    )
    return CandidateProblem(
        candidate=candidate,
        solver=solver,
        initial_guess=np.zeros(candidate.burst_steps, dtype=np.float64),
        lower_bounds=np.full(candidate.burst_steps, -torque_limit_nm, dtype=np.float64),
        upper_bounds=np.full(candidate.burst_steps, torque_limit_nm, dtype=np.float64),
    )


def solve_candidate(
    state: ArrayLike,
    previous_applied_torque_nm: float,
    problem: CandidateProblem,
    physics: EpisodeConfig,
) -> CandidateResult:
    """Solve one candidate and reject every unsuccessful or nonfinite result."""

    started_at = time.perf_counter()
    state_array = np.asarray(state, dtype=np.float64).reshape(4)
    parameter = np.concatenate((state_array, np.asarray([previous_applied_torque_nm])))
    raw_solution = problem.solver(
        x0=problem.initial_guess,
        p=parameter,
        lbx=problem.lower_bounds,
        ubx=problem.upper_bounds,
    )
    statistics = problem.solver.stats()
    objective_value = float(raw_solution["f"])
    solver_status = str(statistics["return_status"])
    if not bool(statistics["success"]) or not math.isfinite(objective_value):
        raise RuntimeError(
            f"Rotary MPC failed for split ratio {problem.candidate.split_ratio}: {solver_status}"
        )

    # Replay the accepted burst and coast through the shared numeric prediction model.
    burst_inputs_nm = np.asarray(raw_solution["x"], dtype=np.float64).reshape(
        problem.candidate.burst_steps
    )
    predicted_states, predicted_inputs_nm = rollout_burst_coast(
        state_array,
        burst_inputs_nm,
        problem.candidate,
        physics,
    )
    return CandidateResult(
        candidate=problem.candidate,
        objective_value=objective_value,
        burst_inputs_nm=burst_inputs_nm,
        predicted_states=predicted_states,
        predicted_inputs_nm=predicted_inputs_nm,
        solve_time_s=time.perf_counter() - started_at,
        solver_status=solver_status,
    )
