import logging
import math

import pytest

from wsmpc.environment import Environment
from wsmpc.mpc.ip_dynamics_natural_period.features import phase_proxy_error
from wsmpc.utils.config_schema import load_config
from wsmpc.utils.messages import ActionCommand
from wsmpc.visualization.realtime import DIAGNOSTIC_PHASE_EPSILON


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
        logger=logging.getLogger("test"),
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
            early_wake_flag=False,
        )
    )
    return config, next_observation, record


def test_realtime_episode_plot_accepts_records() -> None:
    _require_matplotlib()
    from wsmpc.visualization.realtime import RealtimeEpisodePlot

    config, observation, record = _one_record()
    plot = RealtimeEpisodePlot(
        config.environment.pendulum,
        update_every=1,
    )

    plot.add_step(observation, record)
    plot.finish()

    assert plot.buffer.t_sec == [record.t_sec]
    assert plot.buffer.u_applied_nm == [record.u_applied_nm]
    assert plot.buffer.kinetic_energy_j[0] >= 0.0
    assert plot.buffer.kinetic_energy_j[0] + plot.buffer.potential_energy_j[0] == pytest.approx(
        record.energy_j
    )
    phase_error = phase_proxy_error(
        [record.theta_rad, record.omega_rad_s],
        config.environment.pendulum,
        DIAGNOSTIC_PHASE_EPSILON,
    )
    assert plot.buffer.phase_c_error == pytest.approx([phase_error[0]])
    assert plot.buffer.phase_s == pytest.approx([phase_error[1]])
    assert plot._lines["phase_path"].get_xdata()[0] == pytest.approx(phase_error[0])
    assert plot._lines["phase_path"].get_ydata()[0] == pytest.approx(phase_error[1])
    assert plot.kinetic_goal_j == pytest.approx(0.0)
    assert plot.potential_goal_j > 0.0
    assert set(plot.axes) == {"kinetic", "potential", "phase", "action"}
    assert plot.buffer.u_commanded_nm == [record.u_commanded_nm]
    assert plot.buffer.u_applied_nm == [record.u_applied_nm]
    assert plot.animation is None


def test_artifact_figures_cover_static_diagnostics() -> None:
    _require_matplotlib()
    from matplotlib import pyplot as plt

    from wsmpc.visualization.artifact_plots import create_artifact_figures

    config, _, record = _one_record()
    figures = create_artifact_figures([record], config.environment)

    assert set(figures) == {"states", "energy", "phase", "commands"}
    assert [axis.get_ylabel() for axis in figures["states"].axes] == [
        "theta rad",
        "omega rad/s",
    ]
    assert figures["energy"].axes[0].get_ylabel() == "energy J"
    assert figures["energy"].axes[1].get_ylabel() == "energy error J"
    assert figures["phase"].axes[0].get_xlabel() == "c_phi - 1"
    assert figures["phase"].axes[0].get_ylabel() == "s_phi"
    command_labels = {line.get_label() for line in figures["commands"].axes[0].lines}
    assert command_labels == {"commanded", "applied"}

    for figure in figures.values():
        plt.close(figure)


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
    assert animation._torque_arc.get_alpha() == pytest.approx(0.0)
    assert animation._torque_head is None

    animation._update_torque_arrow(config.environment.pendulum.torque_limit_nm * 0.25)
    positive_x, positive_y = animation._torque_arc.get_data()
    positive_width = animation._torque_arc.get_linewidth()
    positive_radius = math.hypot(positive_x[0], positive_y[0])
    assert animation._torque_arc.get_alpha() == pytest.approx(0.9)
    assert animation._torque_head is not None

    animation._update_torque_arrow(-config.environment.pendulum.torque_limit_nm * 0.25)
    negative_x, negative_y = animation._torque_arc.get_data()
    assert negative_x[0] == pytest.approx(positive_x[-1])
    assert negative_y[0] == pytest.approx(positive_y[-1])
    assert negative_x[-1] == pytest.approx(positive_x[0])
    assert negative_y[-1] == pytest.approx(positive_y[0])

    animation._update_torque_arrow(config.environment.pendulum.torque_limit_nm)
    full_x, full_y = animation._torque_arc.get_data()
    full_radius = math.hypot(full_x[0], full_y[0])
    assert full_radius > positive_radius
    assert animation._torque_arc.get_linewidth() > positive_width

    animation.add_step(observation, record)
    assert animation._torque_arc.get_alpha() == pytest.approx(0.9)
