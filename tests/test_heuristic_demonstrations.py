"""Demonstration learning must preserve teacher action and terminal-return semantics."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_environment import GOAL, reset, step
from rotary_pendulum.heuristic.decoder import POLICY_TORQUE_NM as TEACHER_TORQUE_NM
from rotary_pendulum.RL.jax_demonstrations import pretrain
from rotary_pendulum.RL.jax_residual_control import POLICY_TORQUE_NM
from rotary_pendulum.RL.jax_td3 import Actor, Critic, initialize, update

SETTINGS = json.loads(
    Path("artifacts/rotary_pendulum/reference-data/9-RL/11-rl-continuous-residual/machine-scannables/training_template.json").read_text()
) | {
    "hidden_widths": [16, 16],
    "minibatch_size": 16,
    "learning_rate": 0.003,
    "action_mode": "direct_unfiltered",
    "critic_pretrain": True,
    "demonstration_fraction": 1.0,
    "cloning_weight": 100.0,
}


def test_supervised_actor_and_complete_return_learning() -> None:
    learner = initialize(jax.random.PRNGKey(0), SETTINGS)
    demonstrations = {
        "observation": jnp.ones((32, 8)),
        "action": jnp.full(32, 0.3),
        "return": jnp.full(32, -2.0),
    }
    learner, losses = jax.jit(
        lambda model: jax.lax.scan(
            lambda carry, _: pretrain(carry, demonstrations, SETTINGS), model, None, length=200
        )
    )(learner)
    np.testing.assert_allclose(
        Actor((16, 16)).apply(learner["actor"], demonstrations["observation"]), 0.3, atol=0.005
    )
    for critic in learner["critics"]:
        np.testing.assert_allclose(
            Critic((16, 16)).apply(critic, demonstrations["observation"], demonstrations["action"]),
            -2.0,
            atol=0.02,
        )
    assert losses["cloning_mse"][-1] < losses["cloning_mse"][0] * 0.01
    assert losses["return_mse"][-1] < losses["return_mse"][0] * 0.01


def test_retained_demonstrations_control_targets_and_policy_anchor() -> None:
    learner = initialize(jax.random.PRNGKey(1), SETTINGS)
    replay = {
        "observation": jnp.ones((32, 8)),
        "next_observation": jnp.ones((32, 8)),
        "action": jnp.full(32, -0.5),
        "reward": jnp.full(32, -3.0),
        "done": jnp.zeros(32, bool),
        "size": jnp.array(32),
    }
    demonstrations = replay | {
        "action": jnp.full(32, 0.8),
        "reward": jnp.full(32, -7.0),
        "done": jnp.ones(32, bool),
    }
    train = jax.jit(lambda model: update(model, replay, SETTINGS, demonstrations))
    learner, losses = train(learner)
    assert losses["target_mean"] == -7.0
    learner, losses = train(learner)
    assert losses["actor_updated"]
    assert jnp.mean(Actor((16, 16)).apply(learner["actor"], replay["observation"])) > 0


def test_teacher_student_torque_and_sustained_angle_task() -> None:
    assert POLICY_TORQUE_NM == TEACHER_TORQUE_NM
    assert GOAL.hold_steps == 50
    np.testing.assert_allclose(GOAL.beta_tolerance_rad, np.deg2rad(5))
    initial = reset(jax.random.PRNGKey(2), 0)._replace(x=jnp.array([1.0, -jnp.pi, 0.0, 0.0]))
    almost = step(initial, 0, physics_steps=49)
    assert not almost.success and almost.goal_count == 49
    final = step(almost, 0, physics_steps=1)
    assert final.success and final.physics_steps == 50
    outside = almost._replace(x=jnp.array([1.0, -jnp.pi + 0.1, 0.0, 0.0]))
    failed = step(outside, 0, physics_steps=1)
    assert not failed.success and failed.goal_count == 0
