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
    plot = RealtimeEpisodePlot(config.environment.pendulum, update_every=1)

    plot.add_step(observation, record)
    plot.finish()

    assert plot.buffer.t_sec == [record.t_sec]
    assert plot.buffer.u_applied_nm == [record.u_applied_nm]
    assert plot.buffer.kinetic_energy_j[0] >= 0.0
    assert plot.buffer.kinetic_energy_j[0] + plot.buffer.potential_energy_j[0] == pytest.approx(
        record.energy_j
    )
    assert plot.kinetic_goal_j == pytest.approx(0.0)
    assert plot.potential_goal_j > 0.0
    assert set(plot.axes) == {"kinetic", "potential", "phase", "action"}
    assert plot.buffer.u_commanded_nm == [record.u_commanded_nm]
    assert plot.buffer.u_applied_nm == [record.u_applied_nm]
    assert plot.animation is None


def test_realtime_episode_plot_embeds_animation_when_requested() -> None:
    _require_matplotlib()
    from wsmpc.visualization.realtime import RealtimeEpisodePlot

    config, observation, record = _one_record()
    plot = RealtimeEpisodePlot(
        config.environment.pendulum,
        update_every=1,
        include_animation=True,
    )

    plot.start_animation(observation)
    plot.add_step(observation, record)
    plot.finish()

    assert set(plot.axes) == {"kinetic", "potential", "phase", "action", "animation"}
    assert plot.animation is not None
    assert plot.animation.figure is plot.figure
    assert plot.animation.axis is plot.axes["animation"]


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


def test_pendulum_animation_torque_arrow_tracks_action() -> None:
    _require_matplotlib()
    from wsmpc.visualization.realtime import PendulumAnimation

    config, observation, record = _one_record()
    animation = PendulumAnimation(config.environment.pendulum, update_every=1)

    animation._update_torque_arrow(0.0)
    assert animation._torque_arrow.get_alpha() == pytest.approx(0.0)

    animation._update_torque_arrow(config.environment.pendulum.torque_limit_nm * 0.25)
    positive_start = animation._torque_arrow_start
    positive_end = animation._torque_arrow_end
    positive_rad = animation._torque_arrow_rad
    positive_scale = animation._torque_arrow.get_mutation_scale()
    assert animation._torque_arrow.get_alpha() == pytest.approx(0.9)

    animation._update_torque_arrow(-config.environment.pendulum.torque_limit_nm * 0.25)
    negative_start = animation._torque_arrow_start
    negative_end = animation._torque_arrow_end
    negative_rad = animation._torque_arrow_rad
    assert negative_start == pytest.approx(positive_end)
    assert negative_end == pytest.approx(positive_start)
    assert negative_rad == pytest.approx(-positive_rad)

    animation._update_torque_arrow(config.environment.pendulum.torque_limit_nm)
    assert animation._torque_arrow.get_mutation_scale() > positive_scale

    animation.add_step(observation, record)
    assert animation._torque_arrow.get_alpha() == pytest.approx(0.9)
