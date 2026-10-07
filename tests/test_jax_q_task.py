"""Semantic checks for the finite-action rotary Q task and replay collection."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_dynamics import MODEL, TORQUE_LIMIT_NM
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, EnvState, reset, step
from rotary_pendulum.RL.jax_q import QNetwork
from rotary_pendulum.RL.jax_task import ACTION_TORQUES_NM, collect, observe, transition

REWARD = {
    "energy_cost_per_s": 1.0,
    "torque_cost_per_s": 0.02,
    "time_cost_per_s": 1.05,
    "arm_cost_per_s": 0.2,
    "upright_reward_per_s": 1.0,
    "upright_widths": [0.16, 0.4, 0.3],
}


def test_observation_order_includes_time_and_hold_memory() -> None:
    active = EnvState(
        jnp.zeros((4,), dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
        jnp.array(False),
        jnp.array(False),
    )
    observation = observe(active)
    np.testing.assert_allclose(observation, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    assert observation.dtype == jnp.float32
    upright = active._replace(x=jnp.array([0.0, jnp.pi, 0.0, 0.0]))
    observation = observe(upright._replace(goal_count=jnp.array(4), physics_steps=jnp.array(50)))
    np.testing.assert_allclose(observation[-2:], [0.95, 0.8], atol=1e-7)


def test_boundary_crossing_continues_and_success_masks_post_terminal_reward() -> None:
    near_boundary = EnvState(
        jnp.array([ARM_LIMIT_RAD - 1e-4, 0.0, 2.0, 0.0]),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
        jnp.array(False),
        jnp.array(False),
    )
    crossed, _, base, train_reward, components, done = transition(
        near_boundary, jnp.array(0), REWARD
    )
    assert not bool(done) and bool(crossed.arm_violation)
    assert int(crossed.physics_steps) == 5
    assert float(crossed.x[0]) > float(ARM_LIMIT_RAD)
    np.testing.assert_allclose(base, components.sum(), atol=1e-7)
    np.testing.assert_allclose(train_reward, base, atol=1e-7)
    assert float(components[3]) < -0.02

    almost_held = near_boundary._replace(
        x=jnp.array([0.0, jnp.pi, 0.0, 0.0]),
        goal_count=jnp.array(4),
        arm_violation=jnp.array(True),
    )
    held, _, reward, _, terms, done = transition(almost_held, jnp.array(0), REWARD)
    assert bool(done) and bool(held.success) and int(held.physics_steps) == 1
    np.testing.assert_allclose(terms, [0, 0, -0.02 * 1.05 * 1.001, 0, 0.02], atol=1e-7)
    assert float(reward) < 0

    repeated, _, repeated_base, repeated_train, repeated_components, repeated_done = transition(
        held, jnp.array(2), REWARD
    )
    np.testing.assert_array_equal(repeated.x, held.x)
    assert bool(repeated_done)
    np.testing.assert_allclose([repeated_base, repeated_train], 0.0, atol=1e-7)
    np.testing.assert_allclose(repeated_components, 0.0, atol=1e-7)


def test_action_order_has_measured_pump_and_fine_symmetric_pairs() -> None:
    normalized = ACTION_TORQUES_NM / TORQUE_LIMIT_NM
    np.testing.assert_allclose(normalized, [0.0, -0.45, 0.45, -0.02, 0.02], atol=1e-7)


def test_dense_reward_matches_independent_physics_sum_for_all_actions() -> None:
    for action in range(5):
        state = reset(jax.random.fold_in(jax.random.key(7), action), 1)._replace(
            physics_steps=jnp.array(998, dtype=jnp.int32)
        )
        torque = float(ACTION_TORQUES_NM[action])
        expected = np.zeros(5)
        reference = state
        for _ in range(2):
            reference = step(reference, torque, physics_steps=1)
            theta, alpha, omega, nu = np.asarray(reference.x)
            energy = 0.5 * MODEL.pendulum_inertia_kg_m2 * nu**2
            energy += MODEL.gravity_torque_nm * (1 - np.cos(alpha))
            beta = np.arctan2(np.sin(alpha - np.pi), np.cos(alpha - np.pi))
            score = np.exp(
                -0.5 * np.sum((np.array([beta, nu, omega]) / REWARD["upright_widths"]) ** 2)
            )
            expected += 0.02 * np.array(
                [
                    -abs(energy / (2 * MODEL.gravity_torque_nm) - 1),
                    -0.02 * abs(torque) / float(ACTION_TORQUES_NM[2]),
                    -1.05 * (1 + int(reference.physics_steps) / 1000),
                    -0.2 * (theta / float(ARM_LIMIT_RAD)) ** 2,
                    score,
                ]
            )
        result = jax.jit(lambda s, a: transition(s, a, REWARD))(state, jnp.array(action))
        assert bool(result[0].timeout) and bool(result[5])
        np.testing.assert_allclose(result[4], expected, atol=1e-6)
        np.testing.assert_allclose(result[2], expected.sum(), atol=1e-6)
        np.testing.assert_allclose(transition(result[0], jnp.array(action), REWARD)[4], 0)


def test_dense_reward_symmetry_time_and_arm_magnitude_and_batch_parity() -> None:
    state = reset(jax.random.key(9), 0)._replace(x=jnp.zeros(4))
    positive = state._replace(x=jnp.array([2.0, 0.1, 0.2, 0.3]))
    negative = positive._replace(x=-positive.x)
    first = transition(positive, jnp.array(2), REWARD)
    second = transition(negative, jnp.array(1), REWARD)
    np.testing.assert_allclose(first[4], second[4], atol=1e-6)
    near = transition(state._replace(x=jnp.array([2.0, 0, 0, 0])), jnp.array(0), REWARD)
    far = transition(state._replace(x=jnp.array([4.0, 0, 0, 0])), jnp.array(0), REWARD)
    np.testing.assert_allclose(far[4][3], 4 * near[4][3], atol=1e-6)
    late = transition(state._replace(physics_steps=jnp.array(500)), jnp.array(0), REWARD)
    early = transition(state, jnp.array(0), REWARD)
    assert float(late[4][2]) < float(early[4][2]) < 0
    fine = transition(state, jnp.array(4), REWARD)
    pump = transition(state, jnp.array(2), REWARD)
    np.testing.assert_allclose(pump[4][1] / fine[4][1], 22.5, rtol=1e-6)
    batch = jax.tree.map(lambda a, b: jnp.stack((a, b)), positive, negative)
    compiled = jax.jit(lambda s: transition(s, jnp.array([2, 1]), REWARD))(batch)
    np.testing.assert_allclose(compiled[4], jnp.stack((first[4], second[4])), atol=1e-6)
    assert np.all(np.asarray(compiled[2]) < 0)


def test_collection_inserts_terminal_next_state_and_wraps_replay() -> None:
    experiment = {
        **REWARD,
        "hidden_widths": (8, 8),
        "activation_name": "tanh",
        "heuristic_fraction": 0.5,
        "epsilon": jnp.array(1.0),
        "stage": 2,
        "stage_zero_tight_fraction": 0.5,
        "collection_decisions": 3,
    }
    model = QNetwork((8, 8), "tanh")
    params = model.init(jax.random.key(1), jnp.zeros((4, 7), dtype=jnp.float32))
    states = jax.vmap(reset)(jax.random.split(jax.random.key(2), 4), jnp.zeros((4,), jnp.int32))
    states = states._replace(physics_steps=states.physics_steps.at[0].set(999))
    rollout = {
        "env_state": states,
        "key": jax.random.split(jax.random.key(3), 2),
        "episode_totals": jnp.zeros((4, 9), dtype=jnp.float32),
        "completed_totals": jnp.zeros((12,), dtype=jnp.float32),
    }
    replay = {
        "observation": jnp.zeros((16, 7), dtype=jnp.float32),
        "action": jnp.zeros((16,), dtype=jnp.int32),
        "reward": jnp.zeros((16,), dtype=jnp.float32),
        "next_observation": jnp.zeros((16, 7), dtype=jnp.float32),
        "done": jnp.zeros((16,), dtype=jnp.bool_),
        "position": jnp.array(0, dtype=jnp.int32),
        "size": jnp.array(0, dtype=jnp.int32),
    }
    collect_compiled = jax.jit(
        lambda rollout_, replay_: collect({"params": params}, rollout_, replay_, experiment)
    )
    rollout, replay, metrics = collect_compiled(rollout, replay)
    assert int(replay["position"]) == 12 and int(replay["size"]) == 12
    terminal_slots = np.flatnonzero(np.asarray(replay["done"]))
    assert terminal_slots.size >= 1
    terminal_remaining = np.asarray(replay["next_observation"])[terminal_slots, 5]
    assert np.any(terminal_remaining == 0.0)
    assert np.all(terminal_remaining < 1.0)
    assert float(metrics["completed"]) >= 1.0

    _, replay, metrics = collect_compiled(rollout, replay)
    assert int(replay["position"]) == 8 and int(replay["size"]) == 16
    action_count = (
        metrics["off_actions"]
        + metrics["negative_pump_actions"]
        + metrics["positive_pump_actions"]
        + metrics["negative_fine_actions"]
        + metrics["positive_fine_actions"]
    )
    assert float(action_count) == 12
