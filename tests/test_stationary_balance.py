"""Continuing physical-state contract, truncation bootstrap and targeted reset checks."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.RL.jax_residual_task import observe
from rotary_pendulum.RL.jax_stationary_task import collect, targeted_resets, transition
from rotary_pendulum.RL.jax_td3 import initialize, update

BASE = json.loads(
    Path("artifacts/rotary_pendulum/reference-data/9-RL/11-rl-continuous-residual/machine-scannables/training_template.json").read_text()
)
SETTINGS = BASE | dict(
    contract_id="rotary-stationary-td3-v3",
    discount=0.995,
    torque_limit_nm=0.00459,
    deadline_s=120,
    hidden_widths=[16, 16],
    target_noise=0.0,
    exploration_noise=0.0,
    minibatch_size=4,
    demonstration_fraction=0.0,
    cloning_weight=0.0,
)


def test_stationary_features_and_reward_ignore_time_and_success_memory() -> None:
    state, _ = targeted_resets(jax.random.PRNGKey(0), 4)
    changed = state._replace(
        physics_steps=jnp.full(4, 1500), goal_count=jnp.full(4, 500), success=jnp.ones(4, bool)
    )
    a = observe(state, jnp.full(4, 6000), stationary=True)
    b = observe(changed, jnp.full(4, 12000), stationary=True)
    assert a.shape == (4, 5)
    np.testing.assert_array_equal(a, b)
    first = transition(state, jnp.full(4, 0.2), SETTINGS)
    second = transition(changed, jnp.full(4, 0.2), SETTINGS)
    for index in [1, 2, 3]:
        np.testing.assert_array_equal(first[index], second[index])
    assert np.any(np.asarray(second[0].x) != np.asarray(changed.x))
    assert np.all(np.asarray(second[1]) <= 0)


def test_sampling_timeout_bootstraps_real_next_state_and_discounted_target() -> None:
    settings = SETTINGS | dict(deadline_s=1)
    learner = initialize(jax.random.PRNGKey(2), settings)
    state, _ = targeted_resets(jax.random.PRNGKey(1), 4)
    state = state._replace(physics_steps=jnp.full(4, 45))
    replay = dict(
        observation=jnp.zeros((8, 5)),
        next_observation=jnp.zeros((8, 5)),
        action=jnp.zeros(8),
        reward=jnp.zeros(8),
        done=jnp.zeros(8, bool),
        position=jnp.array(0),
        size=jnp.array(0),
    )
    following = transition(state, jnp.zeros(4), settings)
    run, memory, metrics = collect(
        learner, dict(state=state, key=jax.random.PRNGKey(3)), replay, settings
    )
    assert int(metrics["resets"]) == 4
    assert not np.asarray(memory["done"]).any()
    np.testing.assert_allclose(memory["next_observation"][:4], following[3], atol=1e-6)
    assert np.all(np.asarray(run["state"].physics_steps) == 0)
    critics = jax.tree.map(jnp.zeros_like, learner["target_critics"])
    for critic in critics:
        critic["params"]["Dense_2"]["bias"] = jnp.array([2.0])
    learner["target_critics"] = critics
    memory["reward"] = jnp.full(8, -0.1)
    _, result = update(learner, memory, settings)
    np.testing.assert_allclose(result["target_mean"], -0.1 + 2 * 0.995, atol=1e-6)


def test_targeted_reset_distribution_and_five_feature_network() -> None:
    state, groups = targeted_resets(jax.random.PRNGKey(5), 400)
    x = np.asarray(state.x)
    np.testing.assert_array_equal(np.bincount(groups), [100] * 4)
    braking = x[np.asarray(groups) == 2]
    assert np.all(braking[:, 0] * braking[:, 2] > 0)
    assert np.max(abs(x[:, 0])) < 0.75 * np.pi
    recovery = x[np.asarray(groups) == 3]
    beta = np.arctan2(-np.sin(recovery[:, 1]), -np.cos(recovery[:, 1]))
    assert np.max(abs(beta)) <= 0.120001
    learner = initialize(jax.random.PRNGKey(4), SETTINGS)
    assert learner["actor"]["params"]["Dense_0"]["kernel"].shape == (5, 16)
    assert learner["critics"][0]["params"]["Dense_0"]["kernel"].shape == (6, 16)
