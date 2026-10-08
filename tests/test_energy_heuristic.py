"""Behavioral contracts for the lossless energy-work controller."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rotary_pendulum.environment.dynamics import energy_components
from rotary_pendulum.environment.dynamics import rk4_step as numpy_step
from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL, rk4_step, state_derivative
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, reset, step
from rotary_pendulum.heuristic.artifacts import write_artifacts
from rotary_pendulum.heuristic.decoder import POLICY_TORQUE_NM, WORK_TOLERANCE_J, decode
from rotary_pendulum.heuristic.energy import TARGET_ENERGY_J, WORK_LIMIT_J, encode, policy
from rotary_pendulum.heuristic.evaluation import evaluate
from rotary_pendulum.RL.jax_residual_evaluation import validation_resets

# Numerical parity is audited in float64, independently of test collection order.
jax.config.update("jax_enable_x64", True)


def test_energy_encoding_and_policy_information_contract() -> None:
    x = jnp.array([[0.0, 0.7, 1.0, 2.0], [1.5, -0.7, 1.0, 2.0]])
    energy = encode(x)
    np.testing.assert_allclose(energy[0], energy[1], rtol=1e-6)
    np.testing.assert_allclose(
        energy.sum(axis=-1),
        energy_components(np.asarray(x), PHYSICAL, MODEL)[2],
        rtol=1e-6,
    )
    assert np.all(energy >= 0)
    np.testing.assert_allclose(encode(jnp.zeros(4)), [0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(
        encode(jnp.array([1.2, jnp.pi, 0, 0])),
        [0, 0, TARGET_ENERGY_J],
        atol=1e-8,
    )
    # Independent rigid-body check: COM translation plus rotation about each COM.
    _, alpha, omega, nu = np.moveaxis(np.asarray(x), -1, 0)
    center = MODEL.pendulum_com_length_m
    center_velocity = np.stack(
        (
            -center * np.sin(alpha) * omega,
            PHYSICAL.arm_length_m * omega + center * np.cos(alpha) * nu,
            center * np.sin(alpha) * nu,
        ),
        axis=-1,
    )
    pendulum_body = 0.5 * PHYSICAL.pendulum_mass_kg * np.sum(center_velocity**2, axis=-1)
    pendulum_body += 0.5 * MODEL.pendulum_com_inertia_kg_m2 * (nu**2 + (omega * np.sin(alpha)) ** 2)
    np.testing.assert_allclose(energy[:, 1], pendulum_body, rtol=1e-6)
    np.testing.assert_allclose(
        energy[:, 0],
        PHYSICAL.arm_mass_kg * PHYSICAL.arm_length_m**2 * omega**2 / 6,
        rtol=1e-6,
    )
    request = policy(energy, jnp.array(0.02))
    np.testing.assert_allclose(request[0], request[1])
    assert abs(float(policy(jnp.zeros(3), jnp.array(1.0)))) <= WORK_LIMIT_J * 1.000001
    assert policy(jnp.array([0, 0, TARGET_ENERGY_J]), 0.02) == 0
    pendulum_only = jnp.array([0.0, 0.01, 0.01])
    arm_only = jnp.array([0.01, 0.0, 0.01])
    assert policy(pendulum_only, 0.02) == policy(arm_only, 0.02)
    assert policy(jnp.array([TARGET_ENERGY_J, 0, TARGET_ENERGY_J]), 0.02) < 0


def test_decoder_predicts_only_one_uniform_action_without_speed_cap() -> None:
    initial = jnp.array([[1.4, 0.3, 1, -1], [-1.4, -0.3, -1, 1]], dtype=float)
    requested = policy(encode(initial), 0.04)
    compiled = jax.jit(lambda x, w: decode(x, w, 0.01, 10000.0))
    torque, diagnostic = compiled(initial, requested)
    assert np.all(abs(torque) <= POLICY_TORQUE_NM)
    actual = initial
    peak = jnp.abs(initial[:, 0])
    for _ in range(5):
        actual = rk4_step(actual, torque)
        peak = jnp.maximum(peak, jnp.abs(actual[:, 0]))
    # These must match exactly five constant-torque steps, with no braking suffix.
    np.testing.assert_allclose(diagnostic["predicted_terminal_arm_speed"], actual[:, 2], atol=1e-8)
    np.testing.assert_allclose(diagnostic["predicted_energy_j"], encode(actual), atol=1e-9)
    np.testing.assert_allclose(diagnostic["predicted_peak_arm_rad"], peak, atol=1e-8)
    np.testing.assert_allclose(
        diagnostic["predicted_work_j"], torque * (actual[:, 0] - initial[:, 0]), atol=1e-9
    )
    assert np.all(abs(actual[:, 2]) > 0.15)
    np.testing.assert_allclose(torque[0], -torque[1], atol=2e-6)
    fine = np.asarray(initial).copy()
    for _ in range(50):
        fine = numpy_step(fine, np.asarray(torque), 0.002, PHYSICAL, MODEL)
    np.testing.assert_allclose(
        np.asarray(torque) * (fine[:, 0] - np.asarray(initial[:, 0])),
        diagnostic["predicted_work_j"],
        atol=WORK_TOLERANCE_J,
    )
    rest, status = compiled(jnp.array([1.0, jnp.pi, 0.0, 0.0]), jnp.array(0.0))
    assert status["mode"] == 0 and abs(float(rest)) < 1e-10
    invalid, status = compiled(initial[0], jnp.array(jnp.nan))
    assert status["mode"] == 3 and jnp.isnan(invalid)


def test_lossless_environment_soft_boundary_and_noncentered_capture() -> None:
    assert POLICY_TORQUE_NM == pytest.approx(0.01836)
    assert PHYSICAL.rotary_damping_nms == PHYSICAL.pendulum_damping_nms == 0.0
    x = jnp.array([0.3, 1.2, -0.5, 3.0])
    derivative = jax.grad(lambda state: encode(state).sum())(x) @ state_derivative(x, 0.003)
    np.testing.assert_allclose(derivative, 0.003 * x[2], atol=1e-8)
    initial = reset(jax.random.PRNGKey(0), 0)
    crossed = step(initial._replace(x=jnp.array([ARM_LIMIT_RAD - 0.02, 0, 1, 0])), 0)
    assert crossed.arm_violation and not crossed.success and not crossed.timeout
    assert crossed.x[0] > ARM_LIMIT_RAD and crossed.physics_steps == 5
    captured = step(initial._replace(x=jnp.array([1.0, jnp.pi, 0, 0])), 0)
    assert captured.success and captured.goal_count == 5
    fast = step(initial._replace(x=jnp.array([1.0, jnp.pi, 0.5, 1.0])), 0)
    assert fast.success and abs(fast.x[2]) > 0.15 and abs(fast.x[3]) > 0.2
    outside_capture = step(initial._replace(x=jnp.array([ARM_LIMIT_RAD + 0.1, jnp.pi, 0, 0])), 0)
    assert outside_capture.success and outside_capture.arm_violation
    # The soft penalty starts at pi itself, with no inward margin and no rejection.
    for angle in (np.pi - 0.02, np.pi + 0.1):
        state = jnp.array([angle, jnp.pi, 0, 0])
        torque, status = decode(state, jnp.array(0.0), 0.01, 10000.0)
        assert np.isfinite(torque)
        expected = 10000.0 * (max(float(status["predicted_peak_arm_rad"]) - np.pi, 0) / np.pi) ** 2
        assert float(status["selected_arm_limit_cost"]) == pytest.approx(expected)
        if angle < np.pi:
            assert torque == 0 and expected == 0


def test_rollout_actual_work_terminal_masking_and_artifact_split(tmp_path: Path) -> None:
    initial, _, labels = validation_resets(123, 1, [20])
    params = jnp.array([0.04, 0.01, 10000.0])
    final, traces = jax.jit(lambda state: evaluate(state, params, 2, "energy"))(initial)
    data = jax.device_get(traces)
    assert all(np.isfinite(value).all() for value in data.values())
    assert data["physics_x"].shape == (2, 8, 5, 4)
    assert final.success[5] and data["elapsed_s"][1, 5] == 0
    previous = np.concatenate((np.asarray(initial.x)[None, ...], data["x"][:-1]), axis=0)
    np.testing.assert_allclose(
        data["work_j"],
        data["torque_nm"] * (data["x"][..., 0] - previous[..., 0]),
        atol=1e-9,
    )
    active = data["elapsed_s"] > 0
    np.testing.assert_allclose(
        data["work_mismatch_j"],
        np.where(active, data["work_j"] - data["requested_work_j"], 0),
        atol=1e-9,
    )
    summary = write_artifacts(tmp_path, data, labels, {"name": "contract-check"})
    assert len(summary) == 5
    assert (tmp_path / "machine-scannables/episodes.csv").is_file()
    assert (tmp_path / "machine-scannables/trajectories.npz").is_file()
    assert (tmp_path / "human-readables/representative_sessions.png").is_file()
