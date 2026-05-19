"""Single-shooting CasADi NLP construction for one split candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from wsmpc.mpc.casadi_dynamics import rk4_step_symbolic
from wsmpc.mpc.symbolic_features import energy_phase_value
from wsmpc.mpc.types import SplitCandidate
from wsmpc.utils.config_schema import EnvironmentConfig, MPCConfig


@dataclass(frozen=True)
class CasadiProblem:
    """Concrete CasADi solver and bounds for one fixed split candidate."""

    solver: Any
    initial_guess: NDArray[np.float64]
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]


def build_single_shooting_problem(
    state: ArrayLike,
    previous_input_nm: float,
    candidate: SplitCandidate,
    environment: EnvironmentConfig,
    mpc: MPCConfig,
    *,
    warm_start_nm: ArrayLike | None = None,
) -> CasadiProblem:
    """Build the fixed-dimension nonlinear program for one burst-coast split."""

    x0 = np.asarray(state, dtype=np.float64).reshape(2)
    decision = ca.MX.sym("v", candidate.burst_steps)
    x = ca.vertcat(float(x0[0]), float(x0[1]))
    objective = 0.0

    # Accumulate burst cost while propagating the symbolic single-shooting state.
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

    # Coast input is fixed at zero, so only the state remains symbolic after the burst.
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
    options = {
        "print_time": False,
        "ipopt.print_level": mpc.ipopt_print_level,
        "ipopt.max_iter": mpc.solver_max_iterations,
        "ipopt.tol": mpc.solver_tolerance,
    }
    solver = ca.nlpsol("split_ratio_mpc", mpc.solver, nlp, options)
    torque_limit = environment.pendulum.torque_limit_nm
    return CasadiProblem(
        solver=solver,
        initial_guess=_initial_guess(candidate, warm_start_nm),
        lower_bounds=np.full(candidate.burst_steps, -torque_limit, dtype=np.float64),
        upper_bounds=np.full(candidate.burst_steps, torque_limit, dtype=np.float64),
    )


def _burst_stage_cost(
    x: Any,
    u: Any,
    previous_u: Any,
    environment: EnvironmentConfig,
    mpc: MPCConfig,
) -> Any:
    """Build the burst objective term from value, saturation, and input smoothness."""

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


def _initial_guess(
    candidate: SplitCandidate,
    warm_start_nm: ArrayLike | None,
) -> NDArray[np.float64]:
    """Resize a prior burst sequence to the candidate dimension, or use zeros."""

    if warm_start_nm is None:
        return np.zeros(candidate.burst_steps, dtype=np.float64)
    warm_start = np.asarray(warm_start_nm, dtype=np.float64).reshape(-1)
    if warm_start.size == 0:
        return np.zeros(candidate.burst_steps, dtype=np.float64)
    guess = np.zeros(candidate.burst_steps, dtype=np.float64)
    copied = min(candidate.burst_steps, warm_start.size)
    guess[:copied] = warm_start[:copied]
    if copied < candidate.burst_steps:
        guess[copied:] = warm_start[copied - 1]
    return guess
