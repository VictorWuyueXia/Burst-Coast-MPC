"""Static artifact-figure checks for the rotary-pendulum scenario."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")


def _records() -> list[Any]:
    """Return aligned dense records with two explicit MPC replans."""

    time = (0.01, 0.02, 0.03)
    return [
        SimpleNamespace(
            t_sec=t_sec,
            theta_rad=theta_rad,
            alpha_rad=alpha_rad,
            omega_rad_s=0.1 * index,
            nu_rad_s=-0.2 * index,
            kinetic_energy_j=0.001 * index,
            potential_energy_j=0.002 * index,
            energy_j=0.003 * index,
            energy_error_j=-0.03 + 0.003 * index,
            u_commanded_nm=0.01 * (-1.0) ** index,
            u_applied_nm=0.009 * (-1.0) ** index,
            plan_id=f"mpc-{index}",
            replan_index=index // 2,
            hbar=3.0,
            bbar=(0.1, 0.1, 0.3)[index],
            solve_time_s=(0.1, 0.0, 0.2)[index],
            objective_value=(4.0, 4.0, 1.5)[index],
            replan_flag=index in {0, 2},
        )
        for index, (t_sec, theta_rad, alpha_rad) in enumerate(
            zip(time, (0.0, 0.5, 1.0), (0.0, np.pi / 2.0, np.pi), strict=True)
        )
    ]


def _long_records() -> list[Any]:
    """Return thirteen seconds of samples for the fixed phase-history window."""

    template = vars(_records()[0])
    return [
        SimpleNamespace(
            **{
                **template,
                "t_sec": float(index),
                "theta_rad": 0.1 * index,
                "alpha_rad": 0.2 * index,
                "replan_index": index,
                "replan_flag": True,
            }
        )
        for index in range(13)
    ]


def test_rotary_artifact_figures_cover_physics_and_mpc_diagnostics() -> None:
    """Create every required figure and verify its primary signal contract."""

    from matplotlib import pyplot as plt

    from rotary_pendulum.environment.dynamics import derive_model
    from rotary_pendulum.utils.config_schema import RotaryPendulumConfig
    from rotary_pendulum.visualization.artifact_plots import create_artifact_figures

    physical = RotaryPendulumConfig(
        gravity_m_s2=9.81,
        arm_mass_kg=0.095,
        arm_length_m=0.085,
        pendulum_mass_kg=0.024,
        pendulum_length_m=0.129,
        rotary_damping_nms=1.0e-3,
        pendulum_damping_nms=5.0e-5,
        torque_limit_nm=0.0204,
    )
    figures = create_artifact_figures(_records(), physical, derive_model(physical))

    assert set(figures) == {
        "states",
        "energy",
        "phase",
        "commands",
        "mpc_diagnostics",
    }
    assert len(figures["states"].axes) == 2
    assert len(figures["energy"].axes) == 2
    assert len(figures["mpc_diagnostics"].axes) == 4
    assert figures["commands"].axes[0].lines[0].get_linestyle() == "--"
    assert np.allclose(figures["mpc_diagnostics"].axes[3].lines[0].get_ydata(), [4.0, 1.5])
    for figure in figures.values():
        assert figure.get_layout_engine() is not None
        plt.close(figure)


def test_phase_uses_physical_rings_and_opaque_current_markers() -> None:
    """Map both oscillator states to physical-radius phase rings."""

    from matplotlib import pyplot as plt

    from rotary_pendulum.environment.dynamics import derive_model
    from rotary_pendulum.utils.config_schema import RotaryPendulumConfig
    from rotary_pendulum.visualization.artifact_plots import create_artifact_figures
    from rotary_pendulum.visualization.phase import oscillator_phase_points

    physical = RotaryPendulumConfig(
        gravity_m_s2=9.81,
        arm_mass_kg=0.095,
        arm_length_m=0.085,
        pendulum_mass_kg=0.024,
        pendulum_length_m=0.129,
        rotary_damping_nms=1.0e-3,
        pendulum_damping_nms=5.0e-5,
        torque_limit_nm=0.0204,
    )
    figures = create_artifact_figures(_records(), physical, derive_model(physical))
    axis = figures["phase"].axes[0]
    arm_path, pendulum_path = axis.lines

    model = derive_model(physical)
    expected_arm, expected_pendulum = oscillator_phase_points(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, np.pi / 2.0, 0.1, -0.2],
                [1.0, np.pi, 0.2, -0.4],
            ]
        ),
        physical,
        model,
    )
    assert np.allclose(arm_path.get_xdata(), expected_arm[:, 0])
    assert np.allclose(pendulum_path.get_ydata(), expected_pendulum[:, 1])
    assert arm_path.get_alpha() == 0.5
    assert pendulum_path.get_alpha() == 0.5
    assert all(collection.get_alpha() is None for collection in axis.collections)
    assert axis.get_aspect() == 1.0
    assert axis.get_xlim() == axis.get_ylim()
    assert axis.get_xlim()[1] > physical.pendulum_length_m
    for figure in figures.values():
        plt.close(figure)


def test_phase_discards_trajectory_samples_older_than_ten_seconds() -> None:
    """Keep a fixed ten-second path while marking the final episode state."""

    from matplotlib import pyplot as plt

    from rotary_pendulum.environment.dynamics import derive_model
    from rotary_pendulum.utils.config_schema import RotaryPendulumConfig
    from rotary_pendulum.visualization.artifact_plots import create_artifact_figures
    from rotary_pendulum.visualization.phase import oscillator_phase_points

    physical = RotaryPendulumConfig(
        gravity_m_s2=9.81,
        arm_mass_kg=0.095,
        arm_length_m=0.085,
        pendulum_mass_kg=0.024,
        pendulum_length_m=0.129,
        rotary_damping_nms=1.0e-3,
        pendulum_damping_nms=5.0e-5,
        torque_limit_nm=0.0204,
    )
    figures = create_artifact_figures(_long_records(), physical, derive_model(physical))
    axis = figures["phase"].axes[0]
    arm_path, pendulum_path = axis.lines

    model = derive_model(physical)
    phase_states = np.array(
        [
            [0.1 * index, 0.2 * index, 0.0, 0.0]
            for index in range(2, 13)
        ],
        dtype=np.float64,
    )
    expected_arm, expected_pendulum = oscillator_phase_points(phase_states, physical, model)
    assert len(arm_path.get_xdata()) == 11
    assert np.isclose(arm_path.get_xdata()[0], expected_arm[0, 0])
    assert np.isclose(pendulum_path.get_ydata()[0], expected_pendulum[0, 1])
    assert np.allclose(
        axis.collections[0].get_offsets()[0],
        expected_arm[-1],
    )
    assert np.allclose(
        axis.collections[1].get_offsets()[0],
        expected_pendulum[-1],
    )
    for figure in figures.values():
        plt.close(figure)
