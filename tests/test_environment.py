import logging

import numpy as np
import pytest

from wsmpc.environment import Environment
from wsmpc.utils.config_schema import InitialStateConfig
from wsmpc.utils.loaders import load_config
from wsmpc.utils.messages import ActionCommand


def test_environment_step_returns_finite_observation_and_record() -> None:
    config = load_config("standard")
    config.environment.simulation.pace_s = 0.0
    environment = Environment(
        config.environment,
        run_id=config.experiment.run_id,
        episode_id=config.experiment.episode_id,
        logger=logging.getLogger("test"),
    )
    observation = environment.reset(config.experiment.initial_state)

    next_observation, record = environment.step(
        ActionCommand(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            t_index=observation.t_index,
            t_sec=observation.t_sec,
            u_nm=0.0,
            source="test",
        )
    )

    assert next_observation.t_index == 1
    assert record.t_index == 1
    assert np.isfinite([next_observation.theta_rad, next_observation.omega_rad_s]).all()


def test_environment_rollout_shape() -> None:
    config = load_config("standard")
    config.environment.simulation.pace_s = 0.0
    environment = Environment(
        config.environment,
        run_id=config.experiment.run_id,
        episode_id=config.experiment.episode_id,
        logger=logging.getLogger("test"),
    )

    trajectory = environment.rollout(np.array([0.0, 0.0]), np.zeros(4))

    assert trajectory.shape == (5, 2)
    assert np.isfinite(trajectory).all()


def test_environment_goal_targets_zero_angle_upright() -> None:
    config = load_config("standard")
    config.environment.simulation.pace_s = 0.0
    environment = Environment(
        config.environment,
        run_id=config.experiment.run_id,
        episode_id=config.experiment.episode_id,
        logger=logging.getLogger("test"),
    )

    upright_observation = environment.reset(InitialStateConfig(theta_rad=0.0, omega_rad_s=0.0))
    downward_observation = environment.reset(InitialStateConfig(theta_rad=np.pi, omega_rad_s=0.0))

    assert upright_observation.goal_reached is True
    assert upright_observation.wrapped_angle_error_rad == pytest.approx(0.0)
    assert upright_observation.energy_error_j == pytest.approx(0.0)
    assert downward_observation.goal_reached is False
    assert downward_observation.energy_error_j < 0.0
