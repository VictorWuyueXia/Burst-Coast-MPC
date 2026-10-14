"""Weak-motor tasks must share a physical action limit and valid replay normalization."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_environment import reset, step
from rotary_pendulum.heuristic.decoder import decode
from rotary_pendulum.RL.jax_residual_control import residual_action
from rotary_pendulum.RL.jax_residual_task import observe, transition

SETTINGS = json.loads(
    Path("artifacts/rotary_pendulum/reference-data/9-RL/11-rl-continuous-residual/machine-scannables/training_template.json").read_text()
) | {"action_mode": "direct_unfiltered", "torque_limit_nm": 0.002295}


def test_weak_cap_action_gradient_and_physical_enforcement() -> None:
    state = reset(jax.random.PRNGKey(7), 0)
    observation = observe(state, jnp.array(2000))
    commands = jnp.array([-3.0, -0.4, 0.4, 3.0])
    applied = residual_action(observation, commands, jnp.zeros(4), SETTINGS)[0]
    np.testing.assert_allclose(applied / SETTINGS["torque_limit_nm"], [-1, -0.4, 0.4, 1])
    derivative = jax.grad(
        lambda action: residual_action(observation, action, jnp.array(0), SETTINGS)[0]
    )(jnp.array(0.4))
    np.testing.assert_allclose(derivative, SETTINGS["torque_limit_nm"])
    following = transition(state, jnp.array(0.01836), jnp.array(2000), SETTINGS)[0]
    expected = step(state, SETTINGS["torque_limit_nm"], max_physics_steps=2000)
    np.testing.assert_allclose(following.x, expected.x, atol=1e-7)


def test_weak_teacher_and_student_follow_identical_physical_action() -> None:
    state = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(9), 8), jnp.zeros(8, int))
    cap = SETTINGS["torque_limit_nm"]
    torque, _ = decode(state.x, jnp.full(8, 0.0001), jnp.array(0.0), jnp.array(1.0), cap)
    assert np.max(np.abs(torque)) <= cap + 1e-9
    student = transition(state, torque, jnp.full(8, 2000), SETTINGS)[0]
    teacher = step(state, torque, max_physics_steps=2000)
    np.testing.assert_allclose(student.x, teacher.x, atol=1e-7)
    command = torque / cap
    restored = residual_action(observe(state, jnp.full(8, 2000)), command, jnp.zeros(8), SETTINGS)[
        0
    ]
    np.testing.assert_allclose(restored, torque, atol=1e-9)
