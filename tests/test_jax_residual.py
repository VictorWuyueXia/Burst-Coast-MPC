"""Behavioral tests for continuous control; no training campaign is launched."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_dynamics import MODEL, state_derivative
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, reset, step
from rotary_pendulum.RL.jax_residual_control import (
    POLICY_TORQUE_NM,
    TARGET_ENERGY_J,
    filter_torque,
    heuristic_torque,
    residual_action,
)
from rotary_pendulum.RL.jax_residual_evaluation import evaluate, validation_resets
from rotary_pendulum.RL.jax_residual_task import collect, observe, transition
from rotary_pendulum.RL.jax_td3 import Actor, initialize, update

SETTINGS = json.loads(
    Path("docs/11-rl-continuous-residual/machine-scannables/heuristic_campaign.json").read_text()
)["shared"] | {
    "learning_rate": 3e-4,
    "minibatch_size": 8,
    "deadline_s": 20,
    "exploration_noise": 0.1,
    "evaluation_max_s": 60,
}


def test_analytical_inverse_matches_energy_derivative() -> None:
    x = jnp.array([[0.1, 0.7, 0.3, 2.0], [-0.2, 2.1, -0.4, -3.0]])
    inertia, gravity = MODEL.pendulum_inertia_kg_m2, MODEL.gravity_torque_nm
    energy = 0.5 * inertia * x[:, 3] ** 2 + gravity * (1 - jnp.cos(x[:, 1]))
    drift = x[:, 3] * (inertia * state_derivative(x, 0)[:, 3] + gravity * jnp.sin(x[:, 1]))
    sensitivity = x[:, 3] * inertia * (state_derivative(x, 1)[:, 3] - state_derivative(x, 0)[:, 3])
    expected = sensitivity * ((TARGET_ENERGY_J - energy) / 2.0 - drift) / (sensitivity**2 + 4)
    np.testing.assert_allclose(
        heuristic_torque(x, 2.0, 2.0),
        jnp.clip(expected, -POLICY_TORQUE_NM, POLICY_TORQUE_NM),
        rtol=2e-6,
    )
    torque = 0.001
    derivative = x[:, 3] * (
        inertia * state_derivative(x, torque)[:, 3] + gravity * jnp.sin(x[:, 1])
    )
    np.testing.assert_allclose(derivative, drift + sensitivity * torque, atol=1e-8)


def test_singular_states_are_finite_and_rest_starts() -> None:
    x = jnp.array([[0, 0, 0, 0], [0, jnp.pi / 2, 0, 1], [0, jnp.pi, 0, 0]])
    torque = heuristic_torque(x, 1, 0.5)
    assert np.isfinite(torque).all()
    assert 0 < torque[0] < POLICY_TORQUE_NM
    assert abs(float(torque[2])) < 1e-8


def test_residual_can_cancel_and_reverse_heuristic() -> None:
    state = reset(jax.random.PRNGKey(0), 0)._replace(x=jnp.array([0, 0.1, 0, 1.0]))
    observation = observe(state, jnp.array(1000))
    heuristic = heuristic_torque(state.x, 1, 0.5)
    cancelled = residual_action(
        observation, -heuristic / (2 * POLICY_TORQUE_NM), jnp.array(0), SETTINGS
    )
    reversed_action = residual_action(
        observation, -heuristic / POLICY_TORQUE_NM, jnp.array(0), SETTINGS
    )
    assert abs(float(cancelled[2])) < 1e-8
    np.testing.assert_allclose(reversed_action[2], -heuristic, atol=1e-8)
    assert abs(float(reversed_action[0])) <= POLICY_TORQUE_NM + 1e-8


def test_arm_filter_intervenes_symmetrically_and_bounds_remain_nonterminal() -> None:
    x = jnp.array([[ARM_LIMIT_RAD - 0.07, 0, 3, 0], [-ARM_LIMIT_RAD + 0.07, 0, -3, 0]])
    torque, mode = filter_torque(x, jnp.array([POLICY_TORQUE_NM, -POLICY_TORQUE_NM]), 20)
    assert np.all(np.asarray(mode) != 0)
    assert float(torque[0]) < 0 < float(torque[1])
    np.testing.assert_allclose(torque[0], -torque[1], atol=1e-8)
    state = reset(jax.random.PRNGKey(0), 0)._replace(x=jnp.array([ARM_LIMIT_RAD + 0.1, 0, 0, 0]))
    following = step(state, 0)
    assert following.arm_violation and not following.timeout and not following.success


def test_deadlines_and_success_freeze_at_physics_resolution() -> None:
    state = reset(jax.random.PRNGKey(0), 0)._replace(x=jnp.zeros(4))
    for deadline in (1000, 2000, 3000):
        late = state._replace(physics_steps=jnp.array(deadline - 2))
        result, reward, components, _, observation = transition(
            late, jnp.array(0), jnp.array(deadline), SETTINGS
        )
        assert result.timeout and result.physics_steps == deadline
        assert observation[6] == 0
        np.testing.assert_allclose(reward, components.sum())
    upright = state._replace(x=jnp.array([0, jnp.pi, 0, 0]))
    result, *_ = transition(upright, jnp.array(0), jnp.array(3000), SETTINGS)
    assert result.success and result.physics_steps == 5 and not result.timeout
    frozen, reward, *_ = transition(result, jnp.array(POLICY_TORQUE_NM), jnp.array(3000), SETTINGS)
    np.testing.assert_array_equal(result.x, frozen.x)
    assert reward == 0


def test_observation_distinguishes_deadlines_and_hold_memory() -> None:
    state = reset(jax.random.PRNGKey(1), 0)
    short, long = observe(state, jnp.array(1000)), observe(state, jnp.array(3000))
    assert short.shape == (8,) and short[6] != long[6]
    changed = observe(state._replace(goal_count=jnp.array(3)), jnp.array(1000))
    assert changed[7] == 0.6


def test_zero_actor_replay_and_delayed_td3_update() -> None:
    learner = initialize(jax.random.PRNGKey(1), SETTINGS)
    state = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(2), 8), jnp.arange(8) % 3)
    observation = observe(state, jnp.full(8, 1000))
    np.testing.assert_array_equal(Actor().apply(learner["actor"], observation), np.zeros(8))
    replay = {
        "observation": jnp.zeros((16, 8)),
        "next_observation": jnp.zeros((16, 8)),
        "action": jnp.zeros(16),
        "reward": jnp.zeros(16),
        "done": jnp.zeros(16, bool),
        "position": jnp.array(0),
        "size": jnp.array(0),
    }
    rollout, replay, metrics = collect(
        learner, {"state": state, "key": jax.random.PRNGKey(3)}, replay, SETTINGS
    )
    assert replay["size"] == 8 and rollout["state"].x.shape == (8, 4)
    np.testing.assert_allclose(
        replay["action"][:8] * POLICY_TORQUE_NM, metrics["applied_nm"], atol=1e-9
    )
    # Synthetic terminal targets test both timeout/success semantics without any campaign.
    replay = replay | {"done": jnp.ones(16, bool), "reward": jnp.full(16, -3.0)}
    compiled = jax.jit(lambda model, memory: update(model, memory, SETTINGS))
    first, diagnostic = compiled(learner, replay)
    assert diagnostic["target_mean"] == -3 and not diagnostic["actor_updated"]
    for before, after in zip(
        jax.tree.leaves(learner["actor"]), jax.tree.leaves(first["actor"]), strict=True
    ):
        np.testing.assert_array_equal(before, after)
    second, diagnostic = compiled(first, replay)
    assert second["updates"] == 2 and diagnostic["actor_updated"]
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(second))
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(first["actor"]), jax.tree.leaves(second["actor"]), strict=True
        )
    )
    for old, online, target in zip(
        jax.tree.leaves(first["target_actor"]),
        jax.tree.leaves(second["actor"]),
        jax.tree.leaves(second["target_actor"]),
        strict=True,
    ):
        np.testing.assert_allclose(target, 0.995 * old + 0.005 * online, atol=1e-7)


def test_matched_deadlines_and_heuristic_rollout() -> None:
    initial, deadlines, labels = validation_resets(11, 2, [20, 40, 60])
    assert len(labels) == 36
    np.testing.assert_array_equal(initial.x[:12], initial.x[12:24])
    # Full deadlines are exercised using zero torque, including early upright success.
    result = jax.jit(lambda state: evaluate(state, deadlines, SETTINGS, "zero"))(initial)
    np.testing.assert_allclose(np.asarray(result["duration_s"])[[8, 20, 32]], [20, 40, 60])
    np.testing.assert_allclose(np.asarray(result["duration_s"])[[9, 21, 33]], [0.1, 0.1, 0.1])
    assert np.isfinite(result["physics_x"]).all()


def test_twin_minimum_continues_only_nonterminal_returns() -> None:
    learner = initialize(jax.random.PRNGKey(4), SETTINGS)
    critics = tuple(jax.tree.map(jnp.zeros_like, params) for params in learner["target_critics"])
    critics[0]["params"]["Dense_2"]["bias"] = jnp.array([5.0])
    critics[1]["params"]["Dense_2"]["bias"] = jnp.array([3.0])
    learner = learner | {"target_critics": critics}
    state = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(5), 8), jnp.zeros(8, jnp.int32))
    observation = observe(state, jnp.full(8, 1000))
    replay = {
        "observation": observation,
        "next_observation": observation,
        "action": jnp.zeros(8),
        "reward": jnp.full(8, -2.0),
        "done": jnp.zeros(8, bool),
        "position": jnp.array(0),
        "size": jnp.array(8),
    }
    _, metrics = jax.jit(lambda model: update(model, replay, SETTINGS))(learner)
    assert metrics["target_mean"] == 1.0  # -2 + min(5, 3), gamma exactly one.


def test_terminal_collection_preserves_terminal_observation_before_reset() -> None:
    learner = initialize(jax.random.PRNGKey(6), SETTINGS)
    state = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(7), 8), jnp.zeros(8, jnp.int32))
    state = state._replace(physics_steps=jnp.full(8, 999))
    replay = {
        "observation": jnp.zeros((8, 8)),
        "next_observation": jnp.zeros((8, 8)),
        "action": jnp.zeros(8),
        "reward": jnp.zeros(8),
        "done": jnp.zeros(8, bool),
        "position": jnp.array(0),
        "size": jnp.array(0),
    }
    rollout, replay, metrics = collect(
        learner,
        {"state": state, "key": jax.random.PRNGKey(8)},
        replay,
        SETTINGS,
    )
    assert metrics["completed"] == 8 and np.all(replay["done"])
    np.testing.assert_array_equal(replay["next_observation"][:, 6], np.zeros(8))
    np.testing.assert_array_equal(rollout["state"].physics_steps, np.zeros(8))
    assert replay["position"] == 0 and replay["size"] == 8


def test_policy_gradient_and_zero_actor_validation_match_controller() -> None:
    state = reset(jax.random.PRNGKey(9), 0)._replace(x=jnp.array([0, 0.1, 0, 1.0]))
    observation = observe(state, jnp.array(1000))
    heuristic = heuristic_torque(state.x, 1, 0.5)
    cancelling = -heuristic / (2 * POLICY_TORQUE_NM)
    derivative = jax.grad(
        lambda residual: residual_action(
            observation,
            residual,
            jnp.array(0),
            SETTINGS,
        )[0]
    )(cancelling)
    np.testing.assert_allclose(derivative, 2 * POLICY_TORQUE_NM, rtol=1e-6)
    initial, deadlines, _ = validation_resets(10, 2, [20])
    settings = SETTINGS | {"evaluation_max_s": 0.2}
    learner = initialize(jax.random.PRNGKey(11), settings)
    expected = evaluate(initial, deadlines, settings, "heuristic")
    actual = evaluate(initial, deadlines, settings, "policy", learner["actor"])
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
