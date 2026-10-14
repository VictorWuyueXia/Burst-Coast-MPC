"""Campaign semantics: explicit goal, direct actions and nondefault networks."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_environment import reset
from rotary_pendulum.RL.jax_residual_control import POLICY_TORQUE_NM, residual_action
from rotary_pendulum.RL.jax_residual_task import observe, transition
from rotary_pendulum.RL.jax_td3 import Actor, Critic, initialize, update

SETTINGS = json.loads(
    Path("artifacts/rotary_pendulum/reference-data/9-RL/11-rl-continuous-residual/machine-scannables/training_template.json").read_text()
)


def test_strict_goal_rejects_angle_capture_and_freezes_true_success() -> None:
    state = reset(jax.random.PRNGKey(0), 0)._replace(x=jnp.array([0.4, -jnp.pi, 0, 0]))
    angle = transition(
        state._replace(goal_count=jnp.array(45)), jnp.array(0), jnp.array(1000), SETTINGS
    )[0]
    strict = SETTINGS | {"goal_mode": "strict"}
    result = transition(state, jnp.array(0), jnp.array(1000), strict)[0]
    assert angle.success and not result.success and result.goal_count == 0
    state = state._replace(x=jnp.array([0, -jnp.pi, 0, 0]))
    result = transition(state, jnp.array(0), jnp.array(1000), strict)[0]
    assert result.success and result.physics_steps == 5
    final, reward, *_ = transition(result, jnp.array(0.005), jnp.array(1000), strict)
    np.testing.assert_array_equal(final.x, result.x)
    assert reward == 0


def test_direct_action_gradient_and_larger_network_update() -> None:
    settings = SETTINGS | {
        "action_mode": "direct_unfiltered",
        "hidden_widths": [64, 64, 64],
        "minibatch_size": 8,
    }
    state = reset(jax.random.PRNGKey(1), 0)
    observation = observe(state, jnp.array(1000))
    derivative = jax.grad(
        lambda action: residual_action(observation, action, jnp.array(0), settings)[0]
    )(jnp.array(0.1))
    np.testing.assert_allclose(derivative, POLICY_TORQUE_NM)
    learner = initialize(jax.random.PRNGKey(2), settings)
    observation = jnp.tile(observation, (8, 1))
    assert Actor((64, 64, 64)).apply(learner["actor"], observation).shape == (8,)
    assert Critic((64, 64, 64)).apply(learner["critics"][0], observation, jnp.zeros(8)).shape == (
        8,
    )
    replay = {
        "observation": observation,
        "next_observation": observation,
        "action": jnp.zeros(8),
        "reward": -jnp.ones(8),
        "done": jnp.ones(8, bool),
        "size": jnp.array(8),
    }
    learner, metrics = jax.jit(lambda model: update(model, replay, settings))(learner)
    assert metrics["target_mean"] == -1
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(learner))
