import numpy as np
import pytest

from rotary_pendulum.environment.dynamics import derive_model
from rotary_pendulum.environment.environment import RotaryPendulumEnvironment
from rotary_pendulum.utils.config_schema import load_config
from rotary_pendulum.utils.monte_carlo import sample_action_plan


def _require_matplotlib() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_realtime_rotary_plot_covers_mechanics_and_requested_signals() -> None:
    _require_matplotlib()
    from matplotlib import pyplot as plt

    from rotary_pendulum.visualization.realtime import RealtimeRotaryPendulumPlot

    config = load_config()
    config.simulation.pace_s = 0.0
    config.visualization.update_every = 1
    model = derive_model(config.rotary_pendulum)
    environment = RotaryPendulumEnvironment(config)
    observation = environment.reset()
    plan = sample_action_plan(
        np.random.default_rng(4),
        config,
        model,
        0,
    )
    observation, record = environment.step(
        float(plan.torques_nm[0]),
        plan,
        1.0e-4,
        replan_flag=True,
    )

    plot = RealtimeRotaryPendulumPlot(config, model)
    plot.start(observation)
    plot.add_step(observation, record)
    plot.finish()

    assert set(plot.axes) == {
        "mechanism",
        "angles",
        "velocities",
        "torque",
        "energy",
        "phase",
        "replan",
        "solve",
        "hb",
    }
    assert plot.axes["mechanism"].name == "3d"
    assert plot.series["theta_rad"] == pytest.approx([observation.theta_rad])
    assert plot.series["u_commanded_nm"] == pytest.approx([record.u_commanded_nm])
    assert plot.series["solve_time_s"] == pytest.approx([record.solve_time_s])
    assert plot.lines["theta"].get_xdata()[0] == pytest.approx(record.t_sec)
    arm_length_m = config.rotary_pendulum.arm_length_m
    pendulum_length_m = config.rotary_pendulum.pendulum_length_m
    assert plot.lines["arm_rotation"].get_xdata()[0] == pytest.approx(
        arm_length_m * np.cos(record.theta_rad)
    )
    assert plot.lines["arm_rotation"].get_ydata()[0] == pytest.approx(
        arm_length_m * np.sin(record.theta_rad)
    )
    assert plot.lines["pendulum_rotation"].get_xdata()[0] == pytest.approx(
        pendulum_length_m * np.cos(record.alpha_rad)
    )
    assert plot.lines["pendulum_rotation"].get_ydata()[0] == pytest.approx(
        pendulum_length_m * np.sin(record.alpha_rad)
    )
    assert plot.lines["arm_rotation"].get_alpha() == pytest.approx(0.5)
    assert plot.lines["pendulum_rotation"].get_alpha() == pytest.approx(0.5)
    assert plot.lines["arm_rotation_current"].get_alpha() == pytest.approx(1.0)
    assert plot.lines["pendulum_rotation_current"].get_alpha() == pytest.approx(1.0)
    phase_xlim = plot.axes["phase"].get_xlim()
    phase_ylim = plot.axes["phase"].get_ylim()
    assert phase_xlim[0] < -pendulum_length_m < pendulum_length_m < phase_xlim[1]
    assert phase_ylim[0] < -pendulum_length_m < pendulum_length_m < phase_ylim[1]
    assert plot.figure.get_layout_engine() is not None
    assert plot._hb_scatter.get_offsets().shape == (1, 2)
    assert plot.lines["solve"].get_ydata()[0] == pytest.approx(record.solve_time_s)

    # Retain only rotation-plane samples within ten seconds of the latest simulated time.
    plot.series["t_sec"] = [0.0, 2.2, 12.1]
    plot.series["theta_rad"] = [0.0, 0.5 * np.pi, np.pi]
    plot.series["alpha_rad"] = [0.0, np.pi, 1.5 * np.pi]
    for name in (
        "omega_rad_s",
        "nu_rad_s",
        "kinetic_energy_j",
        "potential_energy_j",
        "energy_j",
        "beta_rad",
        "u_commanded_nm",
        "u_applied_nm",
        "replan_index",
    ):
        plot.series[name] = [0.0, 0.0, 0.0]
    plot.update()

    np.testing.assert_allclose(
        plot.lines["arm_rotation"].get_xdata(), [0.0, -arm_length_m], atol=1e-15
    )
    np.testing.assert_allclose(
        plot.lines["arm_rotation"].get_ydata(), [arm_length_m, 0.0], atol=1e-15
    )
    np.testing.assert_allclose(
        plot.lines["pendulum_rotation"].get_xdata(), [-pendulum_length_m, 0.0], atol=1e-15
    )
    np.testing.assert_allclose(
        plot.lines["pendulum_rotation"].get_ydata(), [0.0, -pendulum_length_m], atol=1e-15
    )
    plt.close(plot.figure)
