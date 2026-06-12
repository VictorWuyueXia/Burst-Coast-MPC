import logging

import casadi as ca
import numpy as np
import pytest

from wsmpc.environment.dynamics import pendulum_derivatives, rk4_step
from wsmpc.mpc.controller import CasadiMPCController
from wsmpc.mpc.discrete_model import pendulum_derivatives_symbolic, rk4_step_symbolic
from wsmpc.mpc.ip_dynamics_natural_period.controller import NaturalPeriodMPCController
from wsmpc.mpc.ip_dynamics_natural_period.features import (
    energy_gate,
    energy_gate_symbolic,
    energy_phase_value,
    energy_phase_value_symbolic,
    local_upright_error,
    local_upright_error_symbolic,
    normalized_energy_error,
    normalized_energy_error_symbolic,
    phase_proxy_error,
    phase_proxy_error_symbolic,
)
from wsmpc.mpc.ip_dynamics_natural_period.problem import (
    prediction_horizon_steps,
)
from wsmpc.mpc.ip_dynamics_natural_period.problem import (
    solve_candidate as solve_natural_candidate,
)
from wsmpc.mpc.ip_dynamics_natural_period.problem import (
    split_candidates as natural_split_candidates,
)
from wsmpc.utils.config_schema import load_config
from wsmpc.utils.messages import StateObs
from wsmpc.utils.monte_carlo import MonteCarloAction


def _small_mpc_config():
    config = load_config()
    config.environment.simulation.pace_s = 0.0
    config.environment.goal.hold_steps = 999
    config.environment.simulation.timestep_s = 0.25
    config.mpc.split_ratios = [0.5, 1.0]
    return config


def _observation(config):
    return StateObs(
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


def test_casadi_dynamics_and_natural_period_features_match_numpy() -> None:
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
    rk4_function = ca.Function(
        "rk4",
        [symbolic_state, symbolic_torque],
        [
            rk4_step_symbolic(
                symbolic_state,
                symbolic_torque,
                config.environment.simulation.timestep_s,
                pendulum,
            )
        ],
    )

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
            normalized_energy_error_symbolic(symbolic_state, pendulum),
            phase_proxy_error_symbolic(symbolic_state, pendulum),
            local_upright_error_symbolic(symbolic_state, pendulum),
            energy_gate_symbolic(symbolic_state, pendulum),
            energy_phase_value_symbolic(symbolic_state, pendulum),
        ],
    )
    energy_error, phase_error, local_error, gate, value = feature_function(state)

    assert float(energy_error) == pytest.approx(normalized_energy_error(state, pendulum))
    assert np.asarray(phase_error).reshape(2) == pytest.approx(
        phase_proxy_error(state, pendulum)
    )
    assert np.asarray(local_error).reshape(2) == pytest.approx(local_upright_error(state, pendulum))
    assert float(gate) == pytest.approx(energy_gate(state, pendulum))
    assert float(value) == pytest.approx(energy_phase_value(state, pendulum))


def test_natural_period_solver_returns_bounded_finite_candidate_solution() -> None:
    config = _small_mpc_config()
    candidate = natural_split_candidates(config.environment, config.mpc)[0]

    solution = solve_natural_candidate(
        np.array([0.6, 0.0], dtype=np.float64),
        0.0,
        candidate,
        config.environment,
    )

    assert np.isfinite(solution.objective_value)
    assert solution.predicted_states.shape == (candidate.total_steps + 1, 2)
    assert solution.predicted_inputs_nm.shape == (candidate.total_steps,)
    assert np.max(np.abs(solution.burst_inputs_nm)) <= (
        config.environment.pendulum.torque_limit_nm + 1.0e-8
    )


def test_controller_uses_natural_period_formulation() -> None:
    config = _small_mpc_config()
    natural_controller = CasadiMPCController(
        config.environment,
        config.mpc,
        logger=logging.getLogger("test"),
    )

    assert natural_controller.controller.__class__ is NaturalPeriodMPCController


def test_controller_selects_plan_and_raises_when_solver_fails(monkeypatch) -> None:
    config = _small_mpc_config()
    controller = CasadiMPCController(
        config.environment,
        config.mpc,
        logger=logging.getLogger("test"),
    )
    observation = _observation(config)

    action = controller.select_action(observation, force_replan=False)

    assert config.mpc.controller == "IP-dynamics-naturalPeriod"
    assert action.source in {"mpc_burst", "mpc_coast"}
    assert action.plan_id is not None
    assert abs(action.u_nm) <= config.environment.pendulum.torque_limit_nm + 1.0e-8

    def fail_candidate(*args):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(
        "wsmpc.mpc.ip_dynamics_natural_period.controller.solve_candidate",
        fail_candidate,
    )
    controller.controller._active_plan = None
    with pytest.raises(RuntimeError, match="solver failed"):
        controller.select_action(observation, force_replan=False)


def test_controller_reuses_selected_plan_until_inputs_are_exhausted() -> None:
    config = _small_mpc_config()
    observation = _observation(config)
    observation = observation.model_copy(update={"theta_rad": 0.65, "omega_rad_s": 0.1})

    controller = CasadiMPCController(
        config.environment,
        config.mpc,
        logger=logging.getLogger("test"),
    )
    first_action = controller.select_action(observation, force_replan=False)
    next_observation = observation.model_copy(
        update={
            "t_index": 1,
            "t_sec": config.environment.simulation.timestep_s,
        }
    )
    second_action = controller.select_action(next_observation, force_replan=False)

    assert first_action.plan_id == second_action.plan_id
    assert second_action.t_index == 1


def test_controller_force_replan_starts_new_plan() -> None:
    config = _small_mpc_config()
    observation = _observation(config)
    observation = observation.model_copy(update={"theta_rad": 0.65, "omega_rad_s": 0.1})
    controller = CasadiMPCController(
        config.environment,
        config.mpc,
        logger=logging.getLogger("test"),
    )

    first_action = controller.select_action(observation, force_replan=False)
    next_observation = observation.model_copy(
        update={
            "t_index": 1,
            "t_sec": config.environment.simulation.timestep_s,
        }
    )
    second_action = controller.select_action(next_observation, force_replan=True)

    assert first_action.plan_id != second_action.plan_id
    assert second_action.early_wake_flag is True


def test_controller_starts_direct_monte_carlo_plan_without_candidate_enumeration(
    monkeypatch,
) -> None:
    config = _small_mpc_config()
    observation = _observation(config)
    controller = CasadiMPCController(
        config.environment,
        config.mpc,
        logger=logging.getLogger("test"),
    )
    horizon_steps = prediction_horizon_steps(config.environment)
    monte_carlo_action = MonteCarloAction(
        bbar=1.0,
        hbar=1.0,
        horizon_steps=horizon_steps,
        burst_steps=round(0.5 * horizon_steps),
        coast_steps=horizon_steps - round(0.5 * horizon_steps),
    )

    def fail_split_candidates(*args):
        raise RuntimeError("split enumeration should not run")

    monkeypatch.setattr(
        "wsmpc.mpc.ip_dynamics_natural_period.controller.split_candidates",
        fail_split_candidates,
    )

    plan = controller.start_monte_carlo_plan(observation, monte_carlo_action)
    action = controller.select_action(observation, force_replan=False)

    assert plan.candidate.total_steps == horizon_steps
    assert plan.candidate.burst_steps == monte_carlo_action.burst_steps
    assert action.plan_id == plan.plan_id
