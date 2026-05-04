import numpy as np

from wsmpc.utils.loaders import load_config
from wsmpc.utils.messages import ActionCommand
from wsmpc.environment import Environment


def test_environment_step_returns_finite_observation_and_record() -> None:
    config = load_config()
    config.environment.simulation.pace_s = 0.0
    environment = Environment(
        config.environment,
        run_id=config.experiment.run_id,
        episode_id=config.experiment.episode_id,
    )
    observation = environment.reset(config.experiment.initial_state)

    next_observation, record = environment.step(
        ActionCommand(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            t_index=observation.t_index,
            t_sec=observation.t_sec,
            u=0.0,
            source="test",
        )
    )

    assert next_observation.t_index == 1
    assert record.t_index == 1
    assert np.isfinite([next_observation.theta, next_observation.omega]).all()


def test_environment_rollout_shape() -> None:
    config = load_config()
    config.environment.simulation.pace_s = 0.0
    environment = Environment(
        config.environment,
        run_id=config.experiment.run_id,
        episode_id=config.experiment.episode_id,
    )

    trajectory = environment.rollout(np.array([0.0, 0.0]), np.zeros(4))

    assert trajectory.shape == (5, 2)
    assert np.isfinite(trajectory).all()
