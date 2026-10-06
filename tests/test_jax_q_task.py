"""Semantic checks for the finite-action rotary Q task and replay collection."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, EnvState, reset
from rotary_pendulum.RL.jax_q import QNetwork
from rotary_pendulum.RL.jax_task import collect, observe, transition

REWARD = {
    "capture_weight": 1.0,
    "success_reward": 5.0,
    "arm_failure_cost": 5.0,
    "timeout_cost": 2.0,
    "on_cost_per_s": 0.05,
    "time_cost_per_s": 0.005,
}


def test_observation_order_scales_and_terminal_potential_mask() -> None:
    active = EnvState(
        jnp.zeros((4,), dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
        jnp.array(False),
        jnp.array(False),
    )
    observation, potential = observe(active, REWARD)
    np.testing.assert_allclose(observation, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    assert observation.dtype == jnp.float32
    np.testing.assert_allclose(potential[0], -0.5, atol=1e-7)
    assert -1.0 < float(potential[1]) < -0.99

    upright = active._replace(x=jnp.array([0.0, jnp.pi, 0.0, 0.0]))
    _, upright_potential = observe(upright, REWARD)
    np.testing.assert_allclose(upright_potential, 0.0, atol=1e-6)
    _, terminal_potential = observe(upright._replace(success=jnp.array(True)), REWARD)
    np.testing.assert_array_equal(terminal_potential, jnp.zeros((2,), dtype=jnp.float32))


def test_transition_uses_actual_terminal_duration_and_no_post_terminal_reward() -> None:
    near_boundary = EnvState(
        jnp.array([ARM_LIMIT_RAD - 1e-4, 0.0, 2.0, 0.0]),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
        jnp.array(False),
        jnp.array(False),
    )
    failed, _, base, train_reward, components, done = transition(
        near_boundary, jnp.array(0), REWARD
    )
    assert bool(done) and bool(failed.arm_violation)
    assert int(failed.physics_steps) < 5
    elapsed = 0.02 * int(failed.physics_steps)
    np.testing.assert_allclose(base, -5.0 - 0.005 * elapsed, atol=1e-6)
    expected_train = base - sum(observe(near_boundary, REWARD)[1])
    np.testing.assert_allclose(train_reward, expected_train, atol=1e-6)
    np.testing.assert_allclose(components[2:], [0.0, -0.005 * elapsed, -5.0], atol=1e-6)

    repeated, _, repeated_base, repeated_train, repeated_components, repeated_done = transition(
        failed, jnp.array(2), REWARD
    )
    np.testing.assert_array_equal(repeated.x, failed.x)
    assert bool(repeated_done)
    np.testing.assert_allclose([repeated_base, repeated_train], 0.0, atol=1e-7)
    np.testing.assert_allclose(repeated_components, 0.0, atol=1e-7)


def test_shaping_telescopes_through_timeout_for_all_action_indices() -> None:
    for action in range(3):
        state = reset(jax.random.fold_in(jax.random.key(7), action), 0)._replace(
            physics_steps=jnp.array(990, dtype=jnp.int32)
        )
        initial_potential = sum(observe(state, REWARD)[1])
        base_total = jnp.array(0.0)
        train_total = jnp.array(0.0)
        for _ in range(2):
            state, _, base, train_reward, _, _ = transition(state, jnp.array(action), REWARD)
            base_total += base
            train_total += train_reward
        assert bool(state.timeout | state.arm_violation | state.success)
        np.testing.assert_allclose(train_total, base_total - initial_potential, atol=1e-5)


def test_collection_inserts_terminal_next_state_and_wraps_replay() -> None:
    experiment = {
        **REWARD,
        "hidden_widths": (8, 8),
        "activation_name": "tanh",
        "heuristic_fraction": 0.5,
        "epsilon": jnp.array(1.0),
        "stage": 2,
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
        metrics["off_actions"] + metrics["negative_actions"] + metrics["positive_actions"]
    )
    assert float(action_count) == 12
