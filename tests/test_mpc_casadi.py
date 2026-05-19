import casadi as ca
import numpy as np
import pytest

from wsmpc.environment.dynamics import pendulum_derivatives, rk4_step
from wsmpc.mpc.casadi_dynamics import build_rk4_function, pendulum_derivatives_symbolic
from wsmpc.mpc.controller import CasadiMPCController
from wsmpc.mpc.numeric_features import (
    energy_gate,
    energy_phase_value,
    local_upright_error,
    normalized_energy_error,
    phase_proxy_error,
)
from wsmpc.mpc.prediction import split_candidates
from wsmpc.mpc.solver import solve_candidate
from wsmpc.mpc.symbolic_features import (
    energy_gate as symbolic_energy_gate,
)
from wsmpc.mpc.symbolic_features import (
    energy_phase_value as symbolic_energy_phase_value,
)
from wsmpc.mpc.symbolic_features import (
    local_upright_error as symbolic_local_upright_error,
)
from wsmpc.mpc.symbolic_features import (
    normalized_energy_error as symbolic_normalized_energy_error,
)
from wsmpc.mpc.symbolic_features import (
    phase_proxy_error as symbolic_phase_proxy_error,
)
from wsmpc.mpc.types import MPCControllerContext
from wsmpc.utils.loaders import load_config
from wsmpc.utils.messages import StateObs


def _small_mpc_config():
    config = load_config("default")
    config.environment.simulation.pace_s = 0.0
    config.environment.goal.hold_steps = 999
    config.mpc.enabled = True
    config.mpc.horizon_steps_override = 4
    config.mpc.split_ratios = [0.5, 1.0]
    config.mpc.solver_max_iterations = 40
    config.mpc.cost.q_phase = 0.25
    config.mpc.cost.q_local = 0.1
    return config


def test_casadi_dynamics_and_features_match_numpy() -> None:
    config = _small_mpc_config()
    pendulum = config.environment.pendulum
    state = np.array([0.35, -0.2], dtype=np.float64)
    torque_nm = 0.4

    symbolic_state = ca.MX.sym("x", 2)
    symbolic_torque = ca.MX.sym("u")
    derivative_function = ca.Function(
        "derivative",
        [symbolic_state, symbolic_torque],
        [pendulum_derivatives_symbolic(symbolic_state, symbolic_torque, pendulum)],
    )
    rk4_function = build_rk4_function(config.environment.simulation.timestep_s, pendulum)

    assert np.asarray(derivative_function(state, torque_nm)).reshape(2) == pytest.approx(
        pendulum_derivatives(state, torque_nm, pendulum)
    )
    assert np.asarray(rk4_function(state, torque_nm)).reshape(2) == pytest.approx(
        rk4_step(state, torque_nm, config.environment.simulation.timestep_s, pendulum)
    )

    feature_function = ca.Function(
        "features",
        [symbolic_state],
        [
            symbolic_normalized_energy_error(symbolic_state, pendulum),
            symbolic_phase_proxy_error(symbolic_state, pendulum, config.mpc.cost),
            symbolic_local_upright_error(symbolic_state, pendulum),
            symbolic_energy_gate(symbolic_state, pendulum, config.mpc.cost),
            symbolic_energy_phase_value(symbolic_state, pendulum, config.mpc.cost),
        ],
    )
    energy_error, phase_error, local_error, gate, value = feature_function(state)

    assert float(energy_error) == pytest.approx(normalized_energy_error(state, pendulum))
    assert np.asarray(phase_error).reshape(2) == pytest.approx(
        phase_proxy_error(state, pendulum, epsilon_phi=config.mpc.cost.epsilon_phi)
    )
    assert np.asarray(local_error).reshape(2) == pytest.approx(local_upright_error(state, pendulum))
    assert float(gate) == pytest.approx(energy_gate(state, pendulum, config.mpc.cost))
    assert float(value) == pytest.approx(energy_phase_value(state, pendulum, config.mpc.cost))


def test_casadi_solver_returns_bounded_finite_candidate_solution() -> None:
    config = _small_mpc_config()
    candidate = split_candidates(config.environment, config.mpc)[0]

    solution = solve_candidate(
        np.array([0.6, 0.0], dtype=np.float64),
        0.0,
        candidate,
        config.environment,
        config.mpc,
    )

    assert solution.success is True, solution.message
    assert np.isfinite(solution.objective_value)
    assert solution.predicted_states.shape == (candidate.total_steps + 1, 2)
    assert solution.predicted_inputs_nm.shape == (candidate.total_steps,)
    assert np.max(np.abs(solution.burst_inputs_nm)) <= (
        config.environment.pendulum.torque_limit_nm + 1.0e-8
    )


def test_controller_selects_plan_and_falls_back_when_solver_fails(monkeypatch) -> None:
    config = _small_mpc_config()
    context = MPCControllerContext(
        environment=config.environment,
        experiment=config.experiment,
        mpc=config.mpc,
        runtime=config.runtime,
    )
    controller = CasadiMPCController(context)
    observation = StateObs(
        run_id=config.experiment.run_id,
        episode_id=config.experiment.episode_id,
        t_index=0,
        t_sec=0.0,
        theta_rad=0.7,
        omega_rad_s=0.0,
        energy_j=0.0,
        energy_error_j=0.0,
        wrapped_angle_error_rad=0.7,
        constraint_margin=1.0,
        goal_reached=False,
    )

    action = controller.select_action(observation)

    assert action.source in {"mpc_burst", "mpc_coast"}
    assert action.plan_id is not None
    assert abs(action.u_nm) <= config.environment.pendulum.torque_limit_nm + 1.0e-8

    def fail_all_candidates(tasks):
        return []

    monkeypatch.setattr(controller, "_solve_tasks", fail_all_candidates)
    controller._active_plan = None
    fallback = controller.select_action(observation)

    assert fallback.source == "mpc_failure_fallback"
    assert fallback.u_nm == pytest.approx(config.experiment.default_action.u_nm)


def test_parallel_candidate_solving_matches_sequential_first_action() -> None:
    sequential_config = _small_mpc_config()
    parallel_config = _small_mpc_config()
    parallel_config.mpc.solve_candidates_in_parallel = True
    parallel_config.mpc.max_parallel_workers = 2
    observation = StateObs(
        run_id=sequential_config.experiment.run_id,
        episode_id=sequential_config.experiment.episode_id,
        t_index=0,
        t_sec=0.0,
        theta_rad=0.65,
        omega_rad_s=0.1,
        energy_j=0.0,
        energy_error_j=0.0,
        wrapped_angle_error_rad=0.65,
        constraint_margin=1.0,
        goal_reached=False,
    )

    sequential = CasadiMPCController(
        MPCControllerContext(
            environment=sequential_config.environment,
            experiment=sequential_config.experiment,
            mpc=sequential_config.mpc,
            runtime=sequential_config.runtime,
        )
    )
    parallel = CasadiMPCController(
        MPCControllerContext(
            environment=parallel_config.environment,
            experiment=parallel_config.experiment,
            mpc=parallel_config.mpc,
            runtime=parallel_config.runtime,
        )
    )

    sequential_action = sequential.select_action(observation)
    parallel_action = parallel.select_action(observation)

    assert parallel_action.u_nm == pytest.approx(sequential_action.u_nm, abs=1.0e-7)
    assert parallel._active_plan is not None
    assert sequential._active_plan is not None
    assert parallel._active_plan.plan.candidate.lambda_value == pytest.approx(
        sequential._active_plan.plan.candidate.lambda_value
    )
