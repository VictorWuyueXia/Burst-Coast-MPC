"""Behavioral contracts for the lossless energy-work controller."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.dynamics import energy_components
from rotary_pendulum.environment.dynamics import rk4_step as numpy_step
from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL, state_derivative
from rotary_pendulum.environment.jax_environment import reset, step
from rotary_pendulum.heuristic.artifacts import write_artifacts
from rotary_pendulum.heuristic.decoder import POLICY_TORQUE_NM, WORK_TOLERANCE_J, decode
from rotary_pendulum.heuristic.energy import TARGET_ENERGY_J, WORK_LIMIT_J, encode, policy
from rotary_pendulum.heuristic.evaluation import evaluate
from rotary_pendulum.RL.jax_residual_evaluation import validation_resets


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
    request = policy(energy, jnp.array(0.02), jnp.array(1.0))
    np.testing.assert_allclose(request[0], request[1])
    assert (
        abs(float(policy(jnp.zeros(3), jnp.array(1.0), jnp.array(1.0)))) <= WORK_LIMIT_J * 1.000001
    )
    assert policy(jnp.array([0, 0, TARGET_ENERGY_J]), 0.02, 1.0) == 0
    pendulum_only = jnp.array([0.0, 0.01, 0.01])
    assert policy(pendulum_only, 0.02, 4.0) == policy(pendulum_only, 0.02, 1.0)
    arm_only = jnp.array([0.01, 0.0, 0.01])
    assert policy(arm_only, 0.02, 4.0) < 0 < policy(arm_only, 0.02, 1.0)


def test_decoder_reverses_with_zero_work_and_preserves_mirrored_physics() -> None:
    initial = jnp.array([[1.4, 0, 1, 0], [-1.4, 0, -1, 0]], dtype=float)
    requested = jnp.zeros(2)
    compiled = jax.jit(lambda x, w: decode(x, w, jnp.array(0.1), 40))
    torque, diagnostic = compiled(initial, requested)
    assert np.all(diagnostic["mode"] == 0)
    assert np.all(abs(torque) <= POLICY_TORQUE_NM)
    actual = np.asarray(initial).copy()
    for _ in range(50):
        actual = numpy_step(actual, np.asarray(torque), 0.002, PHYSICAL, MODEL)
        assert np.all(abs(actual[:, 0]) < np.pi / 2)
    work = np.asarray(torque) * (actual[:, 0] - np.asarray(initial[:, 0]))
    np.testing.assert_allclose(work, requested, rtol=0, atol=WORK_TOLERANCE_J)
    assert torque[0] < 0 < torque[1]
    assert actual[0, 2] < 0 < actual[1, 2]
    np.testing.assert_allclose(torque[0], -torque[1], atol=2e-6)
    single, _ = compiled(initial[1], requested[1])
    np.testing.assert_allclose(single, torque[1], atol=1e-8)
    rest, status = compiled(jnp.array([1.0, jnp.pi, 0.0, 0.0]), jnp.array(0.0))
    assert status["mode"] == 0 and abs(float(rest)) < 1e-10
    invalid, status = compiled(initial[0], jnp.array(jnp.nan))
    assert status["mode"] == 3 and jnp.isnan(invalid)


def test_lossless_environment_soft_boundary_and_noncentered_capture() -> None:
    assert PHYSICAL.rotary_damping_nms == PHYSICAL.pendulum_damping_nms == 0.0
    x = jnp.array([0.3, 1.2, -0.5, 3.0])
    derivative = jax.grad(lambda state: encode(state).sum())(x) @ state_derivative(x, 0.003)
    np.testing.assert_allclose(derivative, 0.003 * x[2], atol=1e-8)
    initial = reset(jax.random.PRNGKey(0), 0)
    crossed = step(initial._replace(x=jnp.array([1.55, 0, 1, 0])), 0)
    assert crossed.arm_violation and not crossed.success and not crossed.timeout
    assert crossed.x[0] > jnp.pi / 2 and crossed.physics_steps == 5
    captured = step(initial._replace(x=jnp.array([1.0, jnp.pi, 0, 0])), 0)
    assert captured.success and captured.goal_count == 5
    outside_capture = step(initial._replace(x=jnp.array([2.0, jnp.pi, 0, 0])), 0)
    assert outside_capture.success and outside_capture.arm_violation
    # Positive requested work must yield to braking, before the physical bound.
    early = jnp.array([1.4, 0.0, 3.0, 0.0])
    torque, diagnostic = decode(early, jnp.array(0.001), jnp.array(0.1), 20)
    assert torque < 0 and diagnostic["mode"] in (1, 2)
    actual = np.asarray(early).copy()
    for _ in range(50):
        actual = numpy_step(actual, float(torque), 0.002, PHYSICAL, MODEL)
        assert abs(actual[0]) < np.pi / 2
    assert float(torque) * (actual[0] - float(early[0])) < 0
    outside, status = decode(jnp.array([2.0, 0, 1, 0]), jnp.array(0.0), jnp.array(0.1), 20)
    assert outside < 0 and status["mode"] == 2


def test_rollout_actual_work_terminal_masking_and_artifact_split(tmp_path: Path) -> None:
    initial, _, labels = validation_resets(123, 1, [20])
    params = jnp.array([0.02, 1.0, 0.1])
    final, traces = jax.jit(lambda state: evaluate(state, params, 20, 2, "energy"))(initial)
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
