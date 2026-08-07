import numpy as np
import pytest

from rotary_pendulum.environment.dynamics import derive_model
from rotary_pendulum.environment.environment import RotaryPendulumEnvironment
from rotary_pendulum.utils.config_schema import (
    EPISODE_CONFIG_PATHS,
    load_episode_config,
    load_visualization_config,
)
from rotary_pendulum.utils.messages import ActionPlan, CandidateRecord
from rotary_pendulum.visualization.phase import oscillator_phase_points


def _test_plan() -> ActionPlan:
    """Create a minimal selected burst-coast result for display tests."""

    candidate = CandidateRecord(0.5, 2, 1, 1, 1.0, 1.0e-4, "Solve_Succeeded", True)
    return ActionPlan(
        plan_id="test-plan",
        replan_index=0,
        hbar=1.0,
        bbar=0.5,
        horizon_steps=2,
        burst_steps=1,
        coast_steps=1,
        objective_value=1.0,
        solve_time_s=1.0e-4,
        solver_status="Solve_Succeeded",
        torques_nm=np.array([0.01, 0.0], dtype=np.float64),
        predicted_states=np.zeros((3, 4), dtype=np.float64),
        candidates=(candidate,),
    )


def _require_matplotlib() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_realtime_rotary_plot_covers_mechanics_and_requested_signals() -> None:
    _require_matplotlib()
    from matplotlib import pyplot as plt

    from rotary_pendulum.visualization.realtime import RealtimeRotaryPendulumPlot

    config = load_episode_config(EPISODE_CONFIG_PATHS, runtime_mode="headless")
    visualization = load_visualization_config()
    visualization.update_every = 1
    model = derive_model(config.rotary_pendulum)
    environment = RotaryPendulumEnvironment(config)
    observation = environment.reset()
    plan = _test_plan()
    observation, record = environment.step(
        float(plan.torques_nm[0]),
        plan,
        replan_flag=True,
    )

    plot = RealtimeRotaryPendulumPlot(config, visualization, model)
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
    pendulum_length_m = config.rotary_pendulum.pendulum_length_m
    expected_arm, expected_pendulum = oscillator_phase_points(
        np.array(
            [[record.theta_rad, record.alpha_rad, record.omega_rad_s, record.nu_rad_s]],
            dtype=np.float64,
        ),
        config.rotary_pendulum,
        model,
    )
    assert plot.lines["arm_phase"].get_xdata()[0] == pytest.approx(expected_arm[0, 0])
    assert plot.lines["arm_phase"].get_ydata()[0] == pytest.approx(expected_arm[0, 1])
    assert plot.lines["pendulum_phase"].get_xdata()[0] == pytest.approx(expected_pendulum[0, 0])
    assert plot.lines["pendulum_phase"].get_ydata()[0] == pytest.approx(expected_pendulum[0, 1])
    assert plot.lines["arm_phase"].get_alpha() == pytest.approx(0.5)
    assert plot.lines["pendulum_phase"].get_alpha() == pytest.approx(0.5)
    assert plot.lines["arm_phase_current"].get_alpha() == pytest.approx(1.0)
    assert plot.lines["pendulum_phase_current"].get_alpha() == pytest.approx(1.0)
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
        "u_commanded_nm",
        "u_applied_nm",
        "replan_index",
    ):
        plot.series[name] = [0.0, 0.0, 0.0]
    plot.update()

    recent_states = np.array(
        [[0.5 * np.pi, np.pi, 0.0, 0.0], [np.pi, 1.5 * np.pi, 0.0, 0.0]],
        dtype=np.float64,
    )
    expected_arm, expected_pendulum = oscillator_phase_points(
        recent_states,
        config.rotary_pendulum,
        model,
    )
    np.testing.assert_allclose(plot.lines["arm_phase"].get_xdata(), expected_arm[:, 0])
    np.testing.assert_allclose(plot.lines["arm_phase"].get_ydata(), expected_arm[:, 1])
    np.testing.assert_allclose(plot.lines["pendulum_phase"].get_xdata(), expected_pendulum[:, 0])
    np.testing.assert_allclose(plot.lines["pendulum_phase"].get_ydata(), expected_pendulum[:, 1])
    plt.close(plot.figure)


def test_oscillator_phase_points_map_one_cycle_to_each_physical_radius() -> None:
    """Map displacement-velocity quadrants to phase-ring quadrants."""

    config = load_episode_config(EPISODE_CONFIG_PATHS, runtime_mode="headless")
    model = derive_model(config.rotary_pendulum)
    omega_n = model.natural_frequency_rad_s
    states = np.array(
        [
            [1.0, np.pi, 0.0, 0.0],
            [0.0, 0.0, omega_n, 2.0 * omega_n],
            [-1.0, -np.pi, 0.0, 0.0],
            [0.0, 0.0, -omega_n, -2.0 * omega_n],
        ],
        dtype=np.float64,
    )

    arm_phase, pendulum_phase = oscillator_phase_points(
        states,
        config.rotary_pendulum,
        model,
    )
    arm_length_m = config.rotary_pendulum.arm_length_m
    pendulum_length_m = config.rotary_pendulum.pendulum_length_m

    np.testing.assert_allclose(
        arm_phase,
        [[arm_length_m, 0.0], [0.0, arm_length_m], [-arm_length_m, 0.0], [0.0, -arm_length_m]],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        pendulum_phase,
        [
            [pendulum_length_m, 0.0],
            [0.0, pendulum_length_m],
            [-pendulum_length_m, 0.0],
            [0.0, -pendulum_length_m],
        ],
        atol=1.0e-12,
    )
