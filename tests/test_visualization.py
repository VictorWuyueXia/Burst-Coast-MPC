import math

import pytest

from wsmpc.environment import Environment
from wsmpc.utils.loaders import load_config
from wsmpc.utils.messages import ActionCommand


def _require_matplotlib() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


def _one_record():
    config = load_config("default")
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
            u_nm=0.25,
            source="test",
        )
    )
    return config, next_observation, record


def test_realtime_episode_plot_accepts_records() -> None:
    _require_matplotlib()
    from wsmpc.visualization.realtime import RealtimeEpisodePlot

    config, observation, record = _one_record()
    plot = RealtimeEpisodePlot(update_every=1)

    plot.add_step(observation, record)
    plot.finish()

    assert plot.buffer.t_sec == [record.t_sec]
    assert plot.buffer.u_applied_nm == [record.u_applied_nm]


def test_pendulum_animation_bob_position_and_update() -> None:
    _require_matplotlib()
    from wsmpc.visualization.realtime import PendulumAnimation

    config, observation, record = _one_record()
    animation = PendulumAnimation(config.environment.pendulum, update_every=1)

    x_zero, y_zero = animation.bob_position(0.0, config.environment.pendulum.length_m)
    x_pi, y_pi = animation.bob_position(math.pi, config.environment.pendulum.length_m)
    animation.start(observation)
    animation.add_step(observation, record)
    animation.finish()

    assert x_zero == pytest.approx(0.0)
    assert y_zero == pytest.approx(config.environment.pendulum.length_m)
    assert x_pi == pytest.approx(0.0)
    assert y_pi == pytest.approx(-config.environment.pendulum.length_m)
