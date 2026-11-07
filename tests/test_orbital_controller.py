"""Verify orbital mechanics and discrete supervisor actions on the Furuta simulator."""

# ruff: noqa: E402, I001
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from rotary_pendulum.environment.dynamics import energy_components, state_derivative
from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL, rk4_step
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD
from rotary_pendulum.heuristic.orbital_controller import (
    CAPTURE_TICKS,
    OrbitalMemory,
    controller_step,
)
from rotary_pendulum.heuristic.orbital_phase import orbital_invariants, orbital_phase
from rotary_pendulum.heuristic.orbital_prediction import predict_candidates
from rotary_pendulum.heuristic.phase_pulse_controller import refined_step

PLANT = (0.00459, np.array([-0.07571436, 15.36034805, -0.12182625, 1.22115647]))
TUNING = (0.35, np.pi / 6, 8.0)
REFINED_STEP = jax.jit(lambda x, u: rk4_step(x, u, substeps=40))


@pytest.mark.parametrize("torque", [-0.00459, 0.00459])
def test_invariants(torque: float) -> None:
    x = np.array([[0.3, 1.2, 1.8, -3.2], [0.8, -2.8, -1.0, 4.5]])
    physical = PHYSICAL.model_copy(
        update={"rotary_damping_nms": 0.0003, "pendulum_damping_nms": 0.0001}
    )
    H, p = orbital_invariants(x, PHYSICAL, MODEL)
    np.testing.assert_allclose(H, energy_components(x, physical, MODEL)[2], atol=1e-15)
    _, alpha, omega, nu = x.T
    A = MODEL.base_inertia_kg_m2 + MODEL.pendulum_inertia_kg_m2 * np.sin(alpha) ** 2
    c = MODEL.coupling_inertia_kg_m2 * np.cos(alpha)
    arm = MODEL.arm_inertia_kg_m2 * omega**2 / 2
    carried = (A - MODEL.arm_inertia_kg_m2) * omega**2 / 2 + c * omega * nu
    pivot = MODEL.pendulum_inertia_kg_m2 * nu**2 / 2
    potential = MODEL.gravity_torque_nm * (1 - np.cos(alpha))
    for group in (
        (arm, carried + pivot, potential),
        (arm, carried, pivot + potential),
        (arm, carried, pivot, potential),
    ):
        np.testing.assert_allclose(np.sum(group, axis=0), H, atol=1e-15)
    np.testing.assert_allclose((p - c * nu) / A, omega, atol=1e-14)
    D = MODEL.pendulum_inertia_kg_m2 - c**2 / A
    np.testing.assert_allclose(H, D * nu**2 / 2 + potential + p**2 / (2 * A), atol=1e-15)

    # Check both balance laws independently by differentiating along the full plant vector field.
    derivative = state_derivative(x, torque, physical, MODEL)
    dt = 1e-6
    ahead = np.asarray(orbital_invariants(x + dt * derivative, physical, MODEL))
    behind = np.asarray(orbital_invariants(x - dt * derivative, physical, MODEL))
    expected = np.array(
        [
            torque * omega
            - physical.rotary_damping_nms * omega**2
            - physical.pendulum_damping_nms * nu**2,
            torque - physical.rotary_damping_nms * omega,
        ]
    )
    scale_h = 2 * MODEL.gravity_torque_nm
    scale_p = np.sqrt(2 * MODEL.base_inertia_kg_m2 * scale_h)
    scales = np.array([scale_h, scale_p])[:, None] / MODEL.natural_period_s
    np.testing.assert_allclose(((ahead - behind) / (2 * dt) - expected) / scales, 0, atol=1e-8)
    reversed_x = x * np.array([1, 1, -1, -1])
    np.testing.assert_allclose(orbital_invariants(reversed_x, PHYSICAL, MODEL)[0], H, atol=1e-15)
    assert np.all(torque * reversed_x[:, 2] != torque * omega)


@pytest.mark.parametrize(
    "alpha,omega,nu,chart",
    [
        (0.0, -2.0, 5.0, 1),
        (1.0, 0.0, 0.0, 1),
        (0.7, 13.00641693, 0.5, 2),
        (-0.7, -13.00641693, -0.5, 3),
        (0.0, -10.0, 26.0, 4),
        (0.0, 10.0, -26.0, 5),
        (0.0, 0.0, 0.0, 0),
        (-np.pi, 0.0, 0.0, 0),
    ],
)
def test_regular_orbits(alpha: float, omega: float, nu: float, chart: int) -> None:
    x = np.array([0.0, alpha, omega, nu])
    H, p = orbital_invariants(x, PHYSICAL, MODEL)
    branch, period, phase = orbital_phase(x, H, p, MODEL)
    assert branch == chart
    if not chart:
        assert np.isnan(period) and np.isnan(phase)
        return
    # Integrate an independent adaptive solution and measure its directed return.
    reference = solve_ivp(
        lambda _, y: state_derivative(y, 0, PHYSICAL, MODEL),
        (0, 1.1 * float(period)),
        x,
        rtol=1e-11,
        atol=1e-12,
        dense_output=True,
        max_step=0.002,
    )
    assert reference.sol is not None and reference.success
    times = np.linspace(0, float(period), 257)
    trajectory = reference.sol(times).T
    h_path, p_path = orbital_invariants(trajectory, PHYSICAL, MODEL)
    branches, _, phases = orbital_phase(trajectory, h_path, p_path, MODEL)
    np.testing.assert_array_equal(branches, chart)
    np.testing.assert_allclose(h_path, H, atol=1e-5 * 2 * MODEL.gravity_torque_nm, rtol=0)
    np.testing.assert_allclose(
        p_path,
        p,
        atol=1e-5 * np.sqrt(4 * MODEL.base_inertia_kg_m2 * MODEL.gravity_torque_nm),
        rtol=0,
    )
    np.testing.assert_allclose(
        np.angle(np.exp(1j * (phases - phase - 2 * np.pi * times / period))), 0, atol=0.02
    )
    # Return to the same velocity at a turn, or the same unwrapped directed section otherwise.
    index = 3 if nu == 0 else 1
    section = x[index] + (np.sign(nu) * 2 * np.pi if chart >= 4 else 0)
    return_time = brentq(lambda t: reference.sol(t)[index] - section, 0.98 * period, 1.02 * period)
    assert abs(return_time - period) / period < 0.005


@pytest.mark.parametrize(
    "state,torque_sign,energy_sign",
    [([0, 0, -1.7, 5], -1, 1), ([0, 0, 1.7, -5], 1, 1), ([0, 1.5, -6.75, 25], 1, -1)],
)
def test_pulse_events(state: list[float], torque_sign: int, energy_sign: int) -> None:
    x = np.array(state, dtype=float)
    torque, memory, record = controller_step(x, OrbitalMemory(), PLANT, TUNING, "gated")
    assert torque is not None and torque_sign * torque > 0 and record["mode"] == "pulse"
    assert record["window_entry"] and energy_sign * record["requested_j"] > 0
    initial_H, _ = orbital_invariants(x, PHYSICAL, MODEL)
    held = memory.pulse_ticks + 1
    for _ in range(held):
        x = np.asarray(REFINED_STEP(x, torque))
    end_H, _ = orbital_invariants(x, PHYSICAL, MODEL)
    assert energy_sign * (end_H - initial_H) > 0
    np.testing.assert_allclose(end_H - initial_H, record["predicted_delta_j"], atol=1e-9)
    _, _, next_record = controller_step(x, memory._replace(pulse_ticks=0), PLANT, TUNING, "gated")
    assert next_record["window_entry"] == 0 and next_record["mode"] == "coast"

    # A held command can enter and leave the excluded separatrix band inside one tick.
    x = np.array([-0.364232, 1.66117920, -0.38922708, -14.42086196])
    active = OrbitalMemory(mode="pulse", chart=1, window=2, pulse_torque=-0.00057375, pulse_ticks=1)
    torque, memory, record = controller_step(x, active, PLANT, TUNING, "gated")
    assert torque == 0 and memory.mode == "coast" and record["rejected_phase"] == 1


def test_screen_and_brake() -> None:
    cap, gain = PLANT
    x = np.array([2.25, 0.0, 0.7, 0.0])
    torque, memory, record = controller_step(
        x, OrbitalMemory(pulse_ticks=2), PLANT, TUNING, "gated"
    )
    assert torque is not None and torque < 0 and memory.mode == "brake"
    assert record["coast_half_peak_rad"] > ARM_LIMIT_RAD > record["selected_peak_rad"]
    x[0] = 2.35
    torque, memory, _ = controller_step(x, memory, PLANT, TUNING, "gated")
    assert torque is None and memory.mode == "no_screened_action"

    # A conservative libration returns inside the bound after an interior arm excursion.
    nu = -5.0
    x = np.array([2.28, 0.0, -MODEL.coupling_inertia_kg_m2 * nu / MODEL.base_inertia_kg_m2, nu])
    states, _, peaks, _, _ = predict_candidates(
        x,
        np.zeros((1, 3)),
        np.array([13, 25]),
        (cap, gain, TUNING[2]),
    )
    assert abs(x[0]) < ARM_LIMIT_RAD and abs(states[24, 0, 0]) < ARM_LIMIT_RAD
    assert peaks[0, 1] > ARM_LIMIT_RAD

    x = np.array([-1.9, 0.0, -1.7, 5.0])
    _, _, record = controller_step(x, OrbitalMemory(), PLANT, TUNING, "gated")
    assert record["rejected_arm"] > 0 and record["selected_peak_rad"] < ARM_LIMIT_RAD


def test_launch_and_capture() -> None:
    x = np.zeros(4)
    torque, memory, record = controller_step(x, OrbitalMemory(), PLANT, TUNING, "gated")
    assert torque == PLANT[0] and memory.launch_used and record["mode"] == "launch"
    assert controller_step(x, memory, PLANT, TUNING, "gated")[0] == 0
    x[1] = -np.pi
    _, memory, _ = controller_step(
        x, OrbitalMemory(capture_dwell=CAPTURE_TICKS), PLANT, TUNING, "gated"
    )
    assert memory.mode == "local" and memory.local_ticks == 4
    x[0] = 0.09
    torque, next_memory, _ = controller_step(x, memory, PLANT, TUNING, "gated")
    assert (
        next_memory.mode == "local"
        and torque == memory.local_torque
        and next_memory.local_ticks == 3
    )
    x[0] = 0.17
    _, next_memory, _ = controller_step(x, next_memory, PLANT, TUNING, "gated")
    assert next_memory.mode != "local" and next_memory.capture_dwell == 0


def test_refined_phase_timing_recovery_and_local_entry() -> None:
    tuning = (*TUNING, 0.5, 0.0)
    x = np.array([0.335774218359, 0.464087066169, 2.413072209244, -10.927920874826])
    torque, memory, first = refined_step(x, OrbitalMemory(), PLANT, tuning)
    assert torque is not None and memory.mode == "pulse" and first["planned_ticks"] == 4
    center = 3 * np.pi / 2
    assert first["phase_rad"] < center
    for _ in range(first["planned_ticks"]):
        x = np.asarray(REFINED_STEP(x, torque))
    H, p = orbital_invariants(x, PHYSICAL, MODEL)
    _, _, end_phase = orbital_phase(x, H, p, MODEL)
    assert end_phase > center
    assert abs(first["phase_rad"] + end_phase - 2 * center) <= 2 * np.pi * 0.02 / first["period_s"]
    _, _, next_record = refined_step(x, memory._replace(pulse_ticks=0), PLANT, tuning)
    assert not next_record["window_entry"]

    # Replay the state where the previous braking rule predicted a 160-degree excursion.
    x = np.array([1.66579617, 2.14920975, 1.70307210, 10.35701347])
    torque, memory, record = refined_step(x, OrbitalMemory(), PLANT, tuning)
    assert torque is not None and memory.mode == "brake"
    assert record["coast_full_peak_rad"] > ARM_LIMIT_RAD > record["selected_peak_rad"]

    # Enter local feedback before the evaluation's one-second holding requirement.
    x = np.array([0.02, -np.pi + 0.008, 0.0, 0.0])
    _, memory, _ = refined_step(x, OrbitalMemory(), PLANT, tuning)
    assert memory.mode == "local"
