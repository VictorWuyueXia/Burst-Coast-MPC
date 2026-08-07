from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from rotary_pendulum.environment.dynamics import derive_model, rk4_step, state_derivative
from rotary_pendulum.mpc.controller import RotaryMPCController
from rotary_pendulum.mpc.discrete_model import (
    prediction_horizon_steps,
    rk4_step_symbolic,
    split_candidates,
    state_derivative_symbolic,
)
from rotary_pendulum.mpc.features import (
    ARM_ANGLE_LIMIT_RAD,
    objective_terms,
    objective_terms_symbolic,
    torque_continuation_cost,
    torque_continuation_cost_symbolic,
)
from rotary_pendulum.mpc.solver import (
    CandidateResult,
    build_single_shooting_problem,
    solve_candidate,
)
from rotary_pendulum.utils.config_schema import EPISODE_CONFIG_PATHS, load_episode_config


def _configs(*, timestep_s: float = 0.05, horizon_periods: float = 0.3):
    physical = SimpleNamespace(
        gravity_m_s2=9.81,
        arm_mass_kg=0.095,
        arm_length_m=0.085,
        pendulum_mass_kg=0.024,
        pendulum_length_m=0.129,
        rotary_damping_nms=0.001,
        pendulum_damping_nms=0.00005,
        torque_limit_nm=0.0204,
    )
    physics = SimpleNamespace(
        simulation=SimpleNamespace(timestep_s=timestep_s),
        rotary_pendulum=physical,
    )
    mpc = SimpleNamespace(
        prediction_horizon_natural_periods=horizon_periods,
        split_ratios=[0.25, 0.5],
        terminal_swing_energy_weight=1.0,
        phase_chasing_weight=0.25,
        energy_transition_width=0.25,
        local_energy_shell_width=0.25,
        pendulum_local_weight=0.01,
        rotary_local_weight=0.001,
        arm_angle_soft_penalty_weight=1.0e4,
        torque_slew_weight=1.0e-3,
    )
    return physics, mpc


def test_symbolic_dynamics_rk4_and_objective_terms_match_numpy() -> None:
    physics, mpc = _configs()
    state = np.array([0.3, 0.7, 1.1, -0.4], dtype=np.float64)
    torque_nm, previous_torque_nm = 0.006, -0.002
    symbolic_state = ca.MX.sym("x", 4)
    symbolic_torque = ca.MX.sym("u")
    symbolic_previous = ca.MX.sym("u_previous")
    symbolic_terms = objective_terms_symbolic(symbolic_state, symbolic_torque, physics, mpc)
    function = ca.Function(
        "rotary_parity",
        [symbolic_state, symbolic_torque, symbolic_previous],
        [
            state_derivative_symbolic(symbolic_state, symbolic_torque, physics),
            rk4_step_symbolic(symbolic_state, symbolic_torque, physics),
            *symbolic_terms,
            torque_continuation_cost_symbolic(
                symbolic_torque, symbolic_previous, physics, mpc
            ),
        ],
    )

    derivative, next_state, *costs = function(state, torque_nm, previous_torque_nm)
    model = derive_model(physics.rotary_pendulum)
    np.testing.assert_allclose(
        np.asarray(derivative).reshape(4),
        state_derivative(state, torque_nm, physics.rotary_pendulum, model),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(next_state).reshape(4),
        rk4_step(state, torque_nm, physics.simulation.timestep_s, physics.rotary_pendulum, model),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    numeric_costs = (
        *objective_terms(state, torque_nm, physics, mpc),
        torque_continuation_cost(torque_nm, previous_torque_nm, physics, mpc),
    )
    assert [float(cost) for cost in costs] == pytest.approx(
        [float(cost) for cost in numeric_costs]
    )


def test_swing_energy_excludes_arm_motion_and_phase_breaks_downward_symmetry() -> None:
    physics, mpc = _configs()
    same_swing_states = np.array(
        [[0.0, 0.8, 0.0, 0.4], [1.2, 0.8, -3.0, 0.4]], dtype=np.float64
    )
    terminal_energy, _, _, arm_limit = objective_terms(same_swing_states, 0.0, physics, mpc)
    assert terminal_energy[0] == pytest.approx(terminal_energy[1])
    assert arm_limit[0] == pytest.approx(0.0)
    assert arm_limit[1] == pytest.approx(0.0)

    torque_nm = ca.MX.sym("phase_torque")
    downward = ca.DM.zeros(4)
    phase_cost = objective_terms_symbolic(downward, torque_nm, physics, mpc)[1]
    phase_gradient = ca.Function(
        "phase_gradient",
        [torque_nm],
        [ca.gradient(phase_cost, torque_nm)],
    )
    assert abs(float(phase_gradient(0.0))) > 1.0

    horizontal = np.array([0.0, 0.5 * np.pi, 0.0, 1.0], dtype=np.float64)
    assert objective_terms(horizontal, 0.01, physics, mpc)[1] == pytest.approx(0.0, abs=1e-14)


def test_arm_limit_horizon_and_splits_follow_the_documented_contract() -> None:
    physics, mpc = _configs(timestep_s=0.01, horizon_periods=2.0)
    states = np.array(
        [
            [ARM_ANGLE_LIMIT_RAD, 0.0, 0.0, 0.0],
            [1.5 * ARM_ANGLE_LIMIT_RAD, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    arm_cost = objective_terms(states, 0.0, physics, mpc)[3]
    assert arm_cost == pytest.approx([0.0, 0.25 * mpc.arm_angle_soft_penalty_weight])

    model = derive_model(physics.rotary_pendulum)
    horizon_steps = prediction_horizon_steps(physics, mpc)
    candidates = split_candidates(physics, mpc)
    assert horizon_steps == int(np.ceil(2.0 * model.natural_period_s / 0.01))
    assert [candidate.burst_steps for candidate in candidates] == [
        round(split_ratio * horizon_steps) for split_ratio in mpc.split_ratios
    ]


def test_single_shooting_uses_zero_guess_and_exact_documented_objective() -> None:
    physics, mpc = _configs(horizon_periods=0.3)
    candidate = split_candidates(physics, mpc)[0]
    problem = build_single_shooting_problem(candidate, physics, mpc)
    np.testing.assert_array_equal(problem.initial_guess, 0.0)

    result = solve_candidate(np.zeros(4, dtype=np.float64), 0.0, problem, physics)
    states, inputs = result.predicted_states, result.predicted_inputs_nm
    all_terms = objective_terms(states, np.append(inputs, 0.0), physics, mpc)
    previous_inputs = np.concatenate(([0.0], result.burst_inputs_nm[:-1]))
    continuation = np.sum(
        torque_continuation_cost(result.burst_inputs_nm, previous_inputs, physics, mpc)
    )
    expected = (
        all_terms[0][-1]
        + np.mean(all_terms[1][: candidate.burst_steps])
        + all_terms[2][-1]
        + np.sum(all_terms[3])
        + continuation
    )
    assert result.objective_value == pytest.approx(float(expected), rel=2.0e-6)
    np.testing.assert_array_equal(inputs[candidate.burst_steps :], 0.0)
    assert np.max(np.abs(inputs)) <= physics.rotary_pendulum.torque_limit_nm + 1e-9


def test_default_longest_burst_converges_from_zero_initial_guess() -> None:
    config = load_episode_config(EPISODE_CONFIG_PATHS, runtime_mode="headless")
    candidate = split_candidates(config, config.mpc)[-1]
    problem = build_single_shooting_problem(candidate, config, config.mpc)
    result = solve_candidate(np.zeros(4, dtype=np.float64), 0.0, problem, config)
    assert result.solver_status == "Solve_Succeeded"
    assert np.max(np.abs(result.predicted_states[:, 0])) <= ARM_ANGLE_LIMIT_RAD + 1.0e-5


def test_controller_commits_candidate_local_last_solutions_after_all_succeed(monkeypatch) -> None:
    physics, mpc = _configs()
    guesses: list[np.ndarray] = []

    def build_stub(candidate, *_):
        return SimpleNamespace(
            candidate=candidate,
            initial_guess=np.zeros(candidate.burst_steps, dtype=np.float64),
        )

    def solve_stub(_state, _previous, problem, _physics):
        guesses.append(problem.initial_guess.copy())
        candidate = problem.candidate
        burst = np.full(candidate.burst_steps, candidate.split_ratio, dtype=np.float64)
        inputs = np.pad(burst, (0, candidate.coast_steps))
        return CandidateResult(
            candidate, 2.0, burst, np.zeros((candidate.total_steps + 1, 4)), inputs, 0.01,
            "Solve_Succeeded",
        )

    monkeypatch.setattr("rotary_pendulum.mpc.controller.build_single_shooting_problem", build_stub)
    monkeypatch.setattr("rotary_pendulum.mpc.controller.solve_candidate", solve_stub)
    observation = SimpleNamespace(theta_rad=0.0, alpha_rad=0.0, omega_rad_s=0.0, nu_rad_s=0.0)
    controller = RotaryMPCController(physics, mpc)
    plan = controller.solve_plan(observation, 0.0, 4)
    controller.solve_plan(observation, 0.0, 5)

    assert all(np.count_nonzero(guess) == 0 for guess in guesses[:2])
    assert [np.unique(guess).item() for guess in guesses[2:]] == mpc.split_ratios
    assert plan.plan_id == "rotary-mpc-000004-lambda-0.25"
    assert [candidate.selected for candidate in plan.candidates] == [True, False]


def test_controller_does_not_partially_commit_initial_guesses_on_failure(monkeypatch) -> None:
    physics, mpc = _configs()

    def build_stub(candidate, *_):
        return SimpleNamespace(candidate=candidate, initial_guess=np.zeros(candidate.burst_steps))

    def solve_stub(_state, _previous, problem, _physics):
        if problem.candidate.split_ratio == mpc.split_ratios[1]:
            raise RuntimeError("candidate failed")
        candidate = problem.candidate
        burst = np.ones(candidate.burst_steps)
        return CandidateResult(candidate, 1.0, burst, np.zeros((candidate.total_steps + 1, 4)),
                               np.pad(burst, (0, candidate.coast_steps)), 0.01, "Solve_Succeeded")

    monkeypatch.setattr("rotary_pendulum.mpc.controller.build_single_shooting_problem", build_stub)
    monkeypatch.setattr("rotary_pendulum.mpc.controller.solve_candidate", solve_stub)
    controller = RotaryMPCController(physics, mpc)
    observation = SimpleNamespace(theta_rad=0.0, alpha_rad=0.0, omega_rad_s=0.0, nu_rad_s=0.0)
    with pytest.raises(RuntimeError, match="candidate failed"):
        controller.solve_plan(observation, 0.0, 0)
    assert all(np.count_nonzero(problem.initial_guess) == 0 for problem in controller._problems)


def test_invalid_zero_length_burst_fails_loudly() -> None:
    physics, mpc = _configs(timestep_s=1.0, horizon_periods=0.1)
    mpc.split_ratios = [0.1]
    with pytest.raises(ValueError, match="burst length"):
        split_candidates(physics, mpc)
