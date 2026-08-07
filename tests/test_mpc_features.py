import math

import numpy as np
import pytest

from inverted_pendulum.environment.dynamics import pendulum_derivatives, pendulum_energy, rk4_step
from inverted_pendulum.mpc.controller import (
    prediction_horizon_steps,
    split_candidates,
)
from inverted_pendulum.mpc.discrete_model import natural_frequency_rad_s, rollout_burst_coast
from inverted_pendulum.mpc.features import (
    energy_gate,
    energy_phase_value,
    local_upright_error,
    normalized_energy_error,
    phase_proxy_error,
)
from inverted_pendulum.utils.config_schema import PendulumConfig, load_mpc_only_config


def test_natural_period_features_match_upright_convention() -> None:
    config = load_mpc_only_config()
    pendulum = config.environment.pendulum
    upright = np.array([0.0, 0.0])
    bottom = np.array([math.pi, 0.0])

    assert pendulum_energy(math.pi, 0.0, pendulum) == pytest.approx(0.0)
    assert pendulum_energy(0.0, 0.0, pendulum) == pytest.approx(
        2.0 * pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m
    )
    assert normalized_energy_error(upright, pendulum) == pytest.approx(0.0)
    assert normalized_energy_error(bottom, pendulum) == pytest.approx(-1.0)
    assert phase_proxy_error(upright, pendulum) == pytest.approx([0.0, 0.0])
    assert local_upright_error(upright, pendulum) == pytest.approx([0.0, 0.0])
    assert energy_gate(upright, pendulum) == pytest.approx(1.0)
    assert energy_phase_value(upright, pendulum) == pytest.approx(0.0)


def test_current_gravity_sign_matches_theta_zero_upright_formulation() -> None:
    config = load_mpc_only_config()
    pendulum = config.environment.pendulum

    positive_derivative = pendulum_derivatives(np.array([0.1, 0.0]), 0.0, pendulum)
    negative_derivative = pendulum_derivatives(np.array([-0.1, 0.0]), 0.0, pendulum)

    assert positive_derivative[1] > 0.0
    assert negative_derivative[1] < 0.0


def test_unforced_undamped_rk4_approximately_conserves_energy() -> None:
    config = load_mpc_only_config()
    pendulum = PendulumConfig(
        mass_kg=config.environment.pendulum.mass_kg,
        gravity_m_s2=config.environment.pendulum.gravity_m_s2,
        length_m=config.environment.pendulum.length_m,
        damping_nms=0.0,
        torque_limit_nm=config.environment.pendulum.torque_limit_nm,
        theta_limit_abs_rad=config.environment.pendulum.theta_limit_abs_rad,
        omega_limit_abs_rad_s=config.environment.pendulum.omega_limit_abs_rad_s,
    )
    state = np.array([0.75, 0.3], dtype=np.float64)
    initial_energy = float(pendulum_energy(state[0], state[1], pendulum))

    for _ in range(50):
        state = rk4_step(state, 0.0, config.environment.simulation.timestep_s, pendulum)

    final_energy = float(pendulum_energy(state[0], state[1], pendulum))
    assert abs(final_energy - initial_energy) < 1.0e-6


def test_prediction_horizon_split_dimensions_and_zero_coast_rollout() -> None:
    config = load_mpc_only_config()
    config.mpc.split_ratios = [0.2, 0.5, 1.0]
    config.mpc.prediction_horizon_natural_periods = 1.0
    half_period_s = math.pi / natural_frequency_rad_s(config.environment.pendulum)
    config.environment.simulation.timestep_s = half_period_s / 6.0

    total_steps = prediction_horizon_steps(
        config.environment,
        config.mpc.prediction_horizon_natural_periods,
    )
    candidates = split_candidates(config.environment, config.mpc)

    assert total_steps == 12
    assert [candidate.burst_steps for candidate in candidates] == [2, 6, 12]
    for candidate in candidates:
        assert candidate.burst_steps >= 1
        assert candidate.coast_steps >= 0
        assert candidate.burst_steps + candidate.coast_steps == candidate.total_steps

    states, inputs = rollout_burst_coast(
        np.array([0.2, -0.1]),
        np.ones(candidates[1].burst_steps),
        candidates[1],
        config.environment,
    )
    assert states.shape == (13, 2)
    assert inputs.shape == (12,)
    assert inputs[:6] == pytest.approx([1.0] * 6)
    assert inputs[6:] == pytest.approx([0.0] * 6)
    assert np.isfinite(states).all()
