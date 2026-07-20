import math

import numpy as np
import pytest

from inverted_pendulum.mpc.controller import prediction_horizon_steps
from inverted_pendulum.utils.config_schema import DataGenerationConfig, load_data_generation_config
from inverted_pendulum.utils.monte_carlo import (
    sample_uniform_initial_state,
    sample_uniform_monte_carlo_action,
)


def test_uniform_monte_carlo_action_maps_to_positive_candidate_dimensions() -> None:
    config = load_data_generation_config()
    generation = DataGenerationConfig(
        episodes=1,
        seed=1,
        visual_artifacts=False,
        theta_rad_sample_range=[-math.pi, math.pi],
        omega_eq_scale_sample_range=[-1.5, 1.5],
        gamma=0.99,
        bbar_min=1.0,
        bbar_max=1.0,
        hbar_min=1.0,
        hbar_max=1.0,
        time_weight=1.0,
        action_weight=1.0,
        compute_weight=0.0,
        fail_penalty=1.0,
    )

    action = sample_uniform_monte_carlo_action(
        np.random.default_rng(1),
        generation,
        config.environment,
    )

    horizon_steps = prediction_horizon_steps(config.environment, 1.0)
    assert action.bbar == pytest.approx(1.0)
    assert action.hbar == pytest.approx(1.0)
    assert action.horizon_steps == horizon_steps
    assert action.burst_steps == horizon_steps
    assert action.coast_steps == action.horizon_steps - action.burst_steps


def test_uniform_monte_carlo_action_wraps_zero_dimensions_to_one_step() -> None:
    config = load_data_generation_config()
    generation = DataGenerationConfig(
        episodes=1,
        seed=1,
        visual_artifacts=False,
        theta_rad_sample_range=[-math.pi, math.pi],
        omega_eq_scale_sample_range=[-1.5, 1.5],
        gamma=0.99,
        bbar_min=0.0,
        bbar_max=0.0,
        hbar_min=0.0,
        hbar_max=0.0,
        time_weight=1.0,
        action_weight=1.0,
        compute_weight=0.0,
        fail_penalty=1.0,
    )

    action = sample_uniform_monte_carlo_action(
        np.random.default_rng(1),
        generation,
        config.environment,
    )

    assert action.bbar == pytest.approx(0.0)
    assert action.hbar == pytest.approx(0.0)
    assert action.horizon_steps == 1
    assert action.burst_steps == 1
    assert action.coast_steps == 0


def test_uniform_initial_state_samples_declared_physical_domain() -> None:
    config = load_data_generation_config()
    initial_state = sample_uniform_initial_state(
        np.random.default_rng(1),
        config.data_generation,
        config.environment,
    )
    omega_eq = 2.0 * math.sqrt(
        config.environment.pendulum.gravity_m_s2 / config.environment.pendulum.length_m
    )

    assert config.data_generation.theta_rad_sample_range[0] <= initial_state.theta_rad <= (
        config.data_generation.theta_rad_sample_range[1]
    )
    assert -1.5 * omega_eq <= initial_state.omega_rad_s <= 1.5 * omega_eq
