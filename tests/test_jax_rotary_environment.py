# ruff: noqa: E402, I001

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from rotary_pendulum.environment.dynamics import derive_model
from rotary_pendulum.environment.dynamics import rk4_step as numpy_rk4_step
from rotary_pendulum.environment.dynamics import state_derivative as numpy_state_derivative
from rotary_pendulum.environment.jax_dynamics import (
    MODEL,
    PHYSICAL,
    PHYSICS_DT_S,
    TORQUE_LIMIT_NM,
    rk4_step,
    state_derivative,
)
from rotary_pendulum.environment.jax_environment import (
    ARM_LIMIT_RAD,
    GOAL,
    MAX_PHYSICS_STEPS,
    EnvState,
    reset,
    step,
)
from rotary_pendulum.utils.config_schema import EPISODE_CONFIG_PATHS, load_episode_config


VALIDATION_SEED = 20261005


def _validation_states(count: int = 4096) -> np.ndarray:
    """Generate the frozen parity set across three strata and explicit edge cases."""

    rng = np.random.default_rng(VALIDATION_SEED)
    strata = np.arange(count) % 3
    states = np.empty((count, 4), dtype=np.float64)
    downward = strata == 0
    moving = strata == 1
    upright = strata == 2
    states[downward] = rng.uniform(
        [-0.20, -0.20, -0.5, -0.5], [0.20, 0.20, 0.5, 0.5], (downward.sum(), 4)
    )
    states[moving] = rng.uniform(
        [-0.50, 0.40, -2.0, -6.0], [0.50, 2.60, 2.0, 6.0], (moving.sum(), 4)
    )
    signs = rng.choice([-1.0, 1.0], moving.sum())
    states[moving, 1] *= signs
    states[upright] = rng.uniform(
        [-0.25, np.pi - 0.25, -1.0, -1.0],
        [0.25, np.pi + 0.25, 1.0, 1.0],
        (upright.sum(), 4),
    )
    states[:8] = np.array(
        [
            [np.pi / 2 - 1e-6, 0.0, 2.0, 6.0],
            [-np.pi / 2 + 1e-6, 0.0, -2.0, -6.0],
            [0.5, 2.6, 2.0, 6.0],
            [-0.5, -2.6, -2.0, -6.0],
            [0.0, np.pi, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.4, np.pi - 0.25, 2.0, -6.0],
            [-1.4, np.pi + 0.25, -2.0, 6.0],
        ]
    )
    return states


def test_constants_match_the_validated_nominal_configuration() -> None:
    config = load_episode_config(EPISODE_CONFIG_PATHS, runtime_mode="headless")
    expected_model = derive_model(config.rotary_pendulum)

    assert PHYSICAL == config.rotary_pendulum
    assert MODEL == expected_model
    assert PHYSICS_DT_S == pytest.approx(0.02)
    assert TORQUE_LIMIT_NM == pytest.approx(0.0204)
    assert GOAL.theta_tolerance_rad == pytest.approx(0.08)
    assert GOAL.beta_tolerance_rad == pytest.approx(0.08)
    assert GOAL.omega_tolerance_rad_s == pytest.approx(0.15)
    assert GOAL.nu_tolerance_rad_s == pytest.approx(0.20)
    assert GOAL.hold_steps == 5
    assert MAX_PHYSICS_STEPS * PHYSICS_DT_S == pytest.approx(20.0)


def test_float64_derivative_parity_over_4096_states_and_torque_extremes() -> None:
    states = _validation_states()
    tiled_states = np.repeat(states, 3, axis=0)
    torques = np.tile([-TORQUE_LIMIT_NM, 0.0, TORQUE_LIMIT_NM], len(states))
    expected = numpy_state_derivative(tiled_states, torques, PHYSICAL, MODEL)
    actual = np.asarray(jax.jit(state_derivative)(jnp.asarray(tiled_states), jnp.asarray(torques)))

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)


def test_float64_rk4_parity_and_20ms_resolution_gate() -> None:
    states = _validation_states()
    tiled_states = np.repeat(states, 3, axis=0)
    torques = np.tile([-TORQUE_LIMIT_NM, 0.0, TORQUE_LIMIT_NM], len(states))
    expected_20ms = numpy_rk4_step(tiled_states, torques, 0.02, PHYSICAL, MODEL)
    actual_20ms = np.asarray(jax.jit(rk4_step)(jnp.asarray(tiled_states), jnp.asarray(torques)))

    np.testing.assert_allclose(actual_20ms[:, :2], expected_20ms[:, :2], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(actual_20ms[:, 2:], expected_20ms[:, 2:], rtol=1e-12, atol=1e-8)

    reference_2ms = tiled_states.copy()
    candidate_20ms = jnp.asarray(tiled_states)
    for _ in range(5):
        candidate_20ms = rk4_step(candidate_20ms, jnp.asarray(torques))
    for _ in range(50):
        reference_2ms = numpy_rk4_step(reference_2ms, torques, 0.002, PHYSICAL, MODEL)
    error = np.abs(np.asarray(candidate_20ms) - reference_2ms)

    assert np.isfinite(candidate_20ms).all()
    assert error[:, :2].max() <= 0.02
    assert error[:, 2:].max() <= 0.5


@pytest.mark.parametrize("stratum", [0, 1, 2])
def test_reset_support_variance_signs_and_seed_replay(stratum: int) -> None:
    keys = jax.random.split(jax.random.key(VALIDATION_SEED + stratum), 10_000)
    strata = jnp.full((10_000,), stratum)
    states = np.asarray(jax.jit(jax.vmap(reset))(keys, strata).x)
    replay = np.asarray(jax.jit(jax.vmap(reset))(keys, strata).x)

    np.testing.assert_array_equal(states, replay)
    assert np.all(np.var(states, axis=0) > 0.0)
    if stratum == 0:
        assert np.all(states >= [-0.20, -0.20, -0.5, -0.5])
        assert np.all(states <= [0.20, 0.20, 0.5, 0.5])
    elif stratum == 1:
        assert np.all(states[:, [0, 2, 3]] >= [-0.50, -2.0, -6.0])
        assert np.all(states[:, [0, 2, 3]] <= [0.50, 2.0, 6.0])
        assert np.all((np.abs(states[:, 1]) >= 0.40) & (np.abs(states[:, 1]) <= 2.60))
        assert np.any(states[:, 1] < 0.0) and np.any(states[:, 1] > 0.0)
    else:
        assert np.all(states >= [-0.25, np.pi - 0.25, -1.0, -1.0])
        assert np.all(states <= [0.25, np.pi + 0.25, 1.0, 1.0])


def test_action_clipping_boundary_detection_and_terminal_idempotence() -> None:
    initial = EnvState(
        jnp.array([float(ARM_LIMIT_RAD) - 0.005, 0.0, 1.0, 0.0]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(False),
        jnp.array(False),
        jnp.array(False),
    )
    positive = jax.jit(step)(initial, 10.0 * TORQUE_LIMIT_NM)
    clipped = jax.jit(step)(initial, TORQUE_LIMIT_NM)

    np.testing.assert_allclose(positive.x, clipped.x)
    assert bool(positive.arm_violation)
    assert not bool(positive.success) and not bool(positive.timeout)
    assert int(positive.physics_steps) == 1
    for before, after in zip(positive, step(positive, -TORQUE_LIMIT_NM), strict=True):
        np.testing.assert_array_equal(before, after)


def test_goal_hold_reset_and_success_over_timeout_precedence() -> None:
    upright = EnvState(
        jnp.array([0.0, jnp.pi, 0.0, 0.0]),
        jnp.array(0),
        jnp.array(0),
        jnp.array(False),
        jnp.array(False),
        jnp.array(False),
    )
    success = step(upright, 0.0)
    assert bool(success.success) and int(success.goal_count) == 5
    assert int(success.physics_steps) == 5

    departed = step(upright._replace(goal_count=jnp.array(3), x=jnp.zeros(4)), 0.0)
    assert int(departed.goal_count) == 0 and not bool(departed.success)
    at_horizon = upright._replace(physics_steps=jnp.array(999), goal_count=jnp.array(4))
    simultaneous = step(at_horizon, 0.0)
    assert bool(simultaneous.success) and not bool(simultaneous.timeout)
    assert int(simultaneous.physics_steps) == 1000


def test_timeout_transformations_full_scan_and_raw_jacobians() -> None:
    base = reset(jax.random.key(7), 0)._replace(x=jnp.zeros(4), physics_steps=jnp.array(995))
    timed_out = jax.jit(step)(base, 0.0)
    assert bool(timed_out.timeout) and int(timed_out.physics_steps) == 1000
    for before, after in zip(timed_out, step(timed_out, 0.0), strict=True):
        np.testing.assert_array_equal(before, after)

    keys = jax.random.split(jax.random.key(8), 12)
    batch = jax.vmap(reset)(keys, jnp.arange(12) % 3)
    actions = jnp.linspace(-TORQUE_LIMIT_NM, TORQUE_LIMIT_NM, 12)
    eager = jax.vmap(step)(batch, actions)
    compiled = jax.jit(jax.vmap(step))(batch, actions)
    np.testing.assert_allclose(eager.x, compiled.x)

    def decision_scan(states: EnvState, action: jax.Array) -> tuple[EnvState, None]:
        return jax.vmap(step)(states, jnp.full((12,), action)), None

    completed = jax.jit(lambda carry: jax.lax.scan(decision_scan, carry, jnp.zeros(200))[0])(batch)
    assert np.isfinite(completed.x).all()
    assert np.all(np.asarray(completed.physics_steps) <= MAX_PHYSICS_STEPS)

    state_jacobian = jax.jacfwd(rk4_step, 0)(batch.x[0], jnp.array(0.0))
    torque_jacobian = jax.jacfwd(rk4_step, 1)(batch.x[0], jnp.array(0.0))
    assert state_jacobian.shape == (4, 4) and torque_jacobian.shape == (4,)
    assert jnp.isfinite(state_jacobian).all() and jnp.isfinite(torque_jacobian).all()


def test_raw_transition_does_not_clip_or_hide_nonfinite_inputs() -> None:
    state = jnp.array([0.1, 0.2, 0.3, 0.4])
    bounded = rk4_step(state, TORQUE_LIMIT_NM)
    unbounded = rk4_step(state, 2.0 * TORQUE_LIMIT_NM)
    invalid = rk4_step(state, jnp.nan)

    assert not np.allclose(bounded, unbounded)
    assert not np.isfinite(invalid).all()
