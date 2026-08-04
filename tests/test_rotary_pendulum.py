import numpy as np
import pytest

from rotary_pendulum.environment import RotaryPendulumEnvironment
from rotary_pendulum.environment.dynamics import (
    derive_model,
    energy_components,
    mechanism_points,
    rk4_step,
    state_derivative,
)
from rotary_pendulum.utils.config_schema import load_config
from rotary_pendulum.utils.monte_carlo import sample_action_plan


def test_default_config_reproduces_documented_model_constants() -> None:
    config = load_config()
    model = derive_model(config.rotary_pendulum)

    assert config.experiment.run_id == "rotary_pendulum_simulation"
    assert config.simulation.timestep_s == pytest.approx(0.01)
    assert config.visualization.update_every == 10
    assert model.arm_inertia_kg_m2 == pytest.approx(2.2879167e-4)
    assert model.pendulum_com_inertia_kg_m2 == pytest.approx(3.3282e-5)
    assert model.pendulum_inertia_kg_m2 == pytest.approx(1.33128e-4)
    assert model.base_inertia_kg_m2 == pytest.approx(4.0219167e-4)
    assert model.coupling_inertia_kg_m2 == pytest.approx(1.3158e-4)
    assert model.gravity_torque_nm == pytest.approx(0.01518588)
    assert model.vertical_mass_determinant_kg2_m4 == pytest.approx(3.62296758e-8)
    assert model.natural_frequency_rad_s == pytest.approx(12.983874, rel=1e-7)
    assert model.natural_period_s == pytest.approx(0.483922, rel=1e-6)


def test_vectorized_ode_matches_explicit_documented_accelerations() -> None:
    config = load_config()
    physical = config.rotary_pendulum
    model = derive_model(physical)
    state = np.array([0.2, 0.7, 1.1, -0.4], dtype=np.float64)
    torque_nm = 0.01

    derivative = state_derivative(state, torque_nm, physical, model)
    _, alpha_rad, omega_rad_s, nu_rad_s = state
    sin_alpha = np.sin(alpha_rad)
    cos_alpha = np.cos(alpha_rad)
    mass_11 = model.base_inertia_kg_m2 + model.pendulum_inertia_kg_m2 * sin_alpha**2
    mass_12 = model.coupling_inertia_kg_m2 * cos_alpha
    mass_22 = model.pendulum_inertia_kg_m2
    s_theta = (
        torque_nm
        - physical.rotary_damping_nms * omega_rad_s
        - 2.0 * model.pendulum_inertia_kg_m2 * sin_alpha * cos_alpha * omega_rad_s * nu_rad_s
        + model.coupling_inertia_kg_m2 * sin_alpha * nu_rad_s**2
    )
    s_alpha = (
        -physical.pendulum_damping_nms * nu_rad_s
        + model.pendulum_inertia_kg_m2 * sin_alpha * cos_alpha * omega_rad_s**2
        - model.gravity_torque_nm * sin_alpha
    )
    determinant = mass_11 * mass_22 - mass_12**2
    expected = np.array(
        [
            omega_rad_s,
            nu_rad_s,
            (mass_22 * s_theta - mass_12 * s_alpha) / determinant,
            (-mass_12 * s_theta + mass_11 * s_alpha) / determinant,
        ]
    )

    np.testing.assert_allclose(derivative, expected, rtol=1e-13, atol=1e-13)
    batch = np.stack((state, state + np.array([0.1, -0.2, 0.3, 0.1])))
    batch_derivative = state_derivative(batch, np.array([torque_nm, -torque_nm]), physical, model)
    assert batch_derivative.shape == (2, 4)
    np.testing.assert_allclose(batch_derivative[0], derivative)


def test_energy_rate_geometry_and_equilibrium_match_the_mechanical_model() -> None:
    config = load_config()
    physical = config.rotary_pendulum
    model = derive_model(physical)
    state = np.array([0.4, 0.8, 1.2, -0.7], dtype=np.float64)
    torque_nm = -0.006
    derivative = state_derivative(state, torque_nm, physical, model)

    # Verify the exact mechanical power identity through an independent directional derivative.
    epsilon = 1e-7
    energy_plus = energy_components(state + epsilon * derivative, physical, model)[2]
    energy_minus = energy_components(state - epsilon * derivative, physical, model)[2]
    numerical_energy_rate = float((energy_plus - energy_minus) / (2.0 * epsilon))
    expected_energy_rate = (
        torque_nm * state[2]
        - physical.rotary_damping_nms * state[2] ** 2
        - physical.pendulum_damping_nms * state[3] ** 2
    )
    assert numerical_energy_rate == pytest.approx(expected_energy_rate, rel=1e-7, abs=1e-9)

    # Downward geometry and energy retain the document's global angle convention.
    downward = np.zeros(4, dtype=np.float64)
    kinetic_j, potential_j, total_j = energy_components(downward, physical, model)
    origin, pivot, center, tip = mechanism_points(downward, physical)
    assert float(kinetic_j) == pytest.approx(0.0)
    assert float(potential_j) == pytest.approx(0.0)
    assert float(total_j) == pytest.approx(0.0)
    np.testing.assert_allclose(origin, [0.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(pivot, [physical.arm_length_m, 0.0, 0.0], atol=1e-15)
    assert np.linalg.norm(center - pivot) == pytest.approx(0.5 * physical.pendulum_length_m)
    np.testing.assert_allclose(tip, [physical.arm_length_m, 0.0, -physical.pendulum_length_m])
    np.testing.assert_allclose(
        rk4_step(downward, 0.0, config.simulation.timestep_s, physical, model),
        downward,
        atol=1e-15,
    )


def test_mass_matrix_is_positive_and_lossless_rk4_energy_converges() -> None:
    config = load_config()
    physical = config.rotary_pendulum
    model = derive_model(physical)

    # Verify the inertia determinant over a complete angular revolution in one batch.
    alpha_rad = np.linspace(-np.pi, np.pi, 2049)
    mass_11 = model.base_inertia_kg_m2 + model.pendulum_inertia_kg_m2 * np.sin(alpha_rad) ** 2
    mass_12 = model.coupling_inertia_kg_m2 * np.cos(alpha_rad)
    determinant = mass_11 * model.pendulum_inertia_kg_m2 - mass_12**2
    assert np.all(determinant > 0.0)
    assert determinant.min() == pytest.approx(model.vertical_mass_determinant_kg2_m4)

    # Halving RK4 step size must reduce zero-torque energy drift in the lossless plant.
    lossless = physical.model_copy(update={"rotary_damping_nms": 0.0, "pendulum_damping_nms": 0.0})
    initial_state = np.array([0.3, 0.9, 1.0, -0.5], dtype=np.float64)
    initial_energy_j = float(energy_components(initial_state, lossless, model)[2])
    energy_drifts_j = []
    for timestep_s in (0.01, 0.005):
        state = initial_state.copy()
        for _ in range(round(2.0 / timestep_s)):
            state = rk4_step(state, 0.0, timestep_s, lossless, model)
        final_energy_j = float(energy_components(state, lossless, model)[2])
        energy_drifts_j.append(abs(final_energy_j - initial_energy_j))
    assert energy_drifts_j[1] < energy_drifts_j[0]


def test_monte_carlo_plan_samples_requested_open_magnitude_and_discrete_horizon() -> None:
    config = load_config()
    model = derive_model(config.rotary_pendulum)
    rng = np.random.default_rng(config.experiment.seed)
    plans = [sample_action_plan(rng, config, model, index) for index in range(512)]
    torques_nm = np.array([plan.torques_nm[0] for plan in plans])
    horizons = np.array([plan.horizon_steps for plan in plans])
    max_horizon = int(np.floor(3.0 * model.natural_period_s / config.simulation.timestep_s))
    torque_limit_nm = config.rotary_pendulum.torque_limit_nm

    assert np.all((horizons >= 1) & (horizons <= max_horizon))
    assert np.all((np.abs(torques_nm) > 0.0) & (np.abs(torques_nm) < torque_limit_nm))
    assert 0.44 < np.mean(torques_nm > 0.0) < 0.56
    assert np.mean(horizons) == pytest.approx(0.5 * (max_horizon + 1), rel=0.06)
    for plan, torque_nm in zip(plans, torques_nm, strict=True):
        assert plan.plan_id == f"monte-carlo-{plan.replan_index:06d}"
        assert plan.torques_nm.shape == (plan.horizon_steps,)
        np.testing.assert_array_equal(plan.torques_nm, torque_nm)
        assert plan.hbar == pytest.approx(
            plan.horizon_steps * config.simulation.timestep_s / model.natural_period_s
        )
        assert plan.bbar == pytest.approx(abs(torque_nm) / torque_limit_nm)


def test_environment_emits_complete_observation_and_explicit_replan_record() -> None:
    config = load_config()
    config.simulation.pace_s = 0.0
    environment = RotaryPendulumEnvironment(config)
    observation = environment.reset()
    plan = sample_action_plan(
        np.random.default_rng(config.experiment.seed),
        config,
        environment.model,
        replan_index=0,
    )
    torque_limit_nm = config.rotary_pendulum.torque_limit_nm

    next_observation, record = environment.step(
        2.0 * torque_limit_nm,
        plan,
        0.003,
        replan_flag=True,
    )

    assert observation.t_index == 0
    assert next_observation.t_index == environment.t_index == 1
    assert next_observation.t_sec == environment.t_sec == pytest.approx(0.01)
    assert np.isfinite(environment.state).all()
    assert record.u_commanded_nm == pytest.approx(2.0 * torque_limit_nm)
    assert record.u_applied_nm == pytest.approx(torque_limit_nm)
    assert record.plan_id == plan.plan_id
    assert record.replan_flag is True
    assert record.solve_time_s == pytest.approx(0.003)
    assert record.energy_j == next_observation.energy_j
    with pytest.raises(ValueError, match="zero away"):
        environment.step(0.0, plan, 0.001, replan_flag=False)


def test_environment_requires_held_complete_upright_goal() -> None:
    config = load_config()
    config.simulation.pace_s = 0.0
    config.experiment.initial_state.alpha_rad = np.pi
    config.goal.hold_steps = 2
    environment = RotaryPendulumEnvironment(config)
    first_observation = environment.reset()
    plan = sample_action_plan(np.random.default_rng(2), config, environment.model, 0)

    second_observation, _ = environment.step(0.0, plan, 0.0, replan_flag=True)

    assert first_observation.beta_rad == pytest.approx(0.0)
    assert first_observation.goal_reached is False
    assert second_observation.goal_reached is True
    assert second_observation.energy_error_j == pytest.approx(0.0, abs=1e-14)
