"""Reward, observation, exploration, replay, and reset logic for JAX Q learning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array

from rotary_pendulum.environment.jax_dynamics import (
    MODEL,
    PHYSICS_DT_S,
    TORQUE_LIMIT_NM,
)
from rotary_pendulum.environment.jax_environment import (
    ARM_LIMIT_RAD,
    GOAL,
    HOLD_PHYSICS_STEPS,
    MAX_PHYSICS_STEPS,
    EnvState,
    reset,
    step,
)
from rotary_pendulum.RL.jax_q import ACTION_COUNT, QNetwork

PUMP_TORQUE_FRACTION = 0.45
FINE_TORQUE_FRACTION = 0.02
ACTION_TORQUES_NM = jnp.array(
    [
        0.0,
        -PUMP_TORQUE_FRACTION * TORQUE_LIMIT_NM,
        PUMP_TORQUE_FRACTION * TORQUE_LIMIT_NM,
        -FINE_TORQUE_FRACTION * TORQUE_LIMIT_NM,
        FINE_TORQUE_FRACTION * TORQUE_LIMIT_NM,
    ],
    dtype=jnp.float32,
)
TARGET_ENERGY_J = 2.0 * MODEL.gravity_torque_nm
ARM_SPEED_SCALE = float(ARM_LIMIT_RAD) * MODEL.natural_frequency_rad_s
PENDULUM_SPEED_SCALE = 2.0 * jnp.sqrt(MODEL.gravity_torque_nm / MODEL.pendulum_inertia_kg_m2)


def observe(env_state: EnvState) -> Array:
    """Return seven Markov features, including elapsed-horizon and hold memory."""

    theta, alpha, omega, nu = jnp.moveaxis(env_state.x, -1, 0)
    observation = jnp.stack(
        (
            theta / ARM_LIMIT_RAD,
            jnp.sin(alpha),
            jnp.cos(alpha),
            omega / ARM_SPEED_SCALE,
            nu / PENDULUM_SPEED_SCALE,
            (MAX_PHYSICS_STEPS - env_state.physics_steps) / float(MAX_PHYSICS_STEPS),
            env_state.goal_count / float(GOAL.hold_steps),
        ),
        axis=-1,
    ).astype(jnp.float32)
    return observation


def transition(
    env_state: EnvState, action_index: Array, experiment: Mapping[str, Any]
) -> tuple[EnvState, Array, Array, Array, Array, Array]:
    """Integrate five dense components at 20 ms; base and train returns coincide."""

    torque = ACTION_TORQUES_NM[jnp.asarray(action_index, dtype=jnp.int32)]
    widths = jnp.asarray(experiment["upright_widths"], dtype=env_state.x.dtype)

    def advance_reward(state: EnvState, _: None) -> tuple[EnvState, Array]:
        active = ~(state.success | state.timeout)
        updated = step(state, torque, physics_steps=1)
        theta, alpha, omega, nu = jnp.moveaxis(updated.x, -1, 0)
        energy = 0.5 * MODEL.pendulum_inertia_kg_m2 * nu**2 + MODEL.gravity_torque_nm * (
            1.0 - jnp.cos(alpha)
        )
        error = (energy - TARGET_ENERGY_J) / TARGET_ENERGY_J
        beta = jnp.arctan2(jnp.sin(alpha - jnp.pi), jnp.cos(alpha - jnp.pi))
        upright = jnp.exp(
            -0.5 * jnp.sum((jnp.stack((beta, nu, omega), axis=-1) / widths) ** 2, axis=-1)
        )
        rates = jnp.stack(
            (
                -experiment["energy_cost_per_s"] * jnp.abs(error),
                jnp.broadcast_to(
                    -experiment["torque_cost_per_s"] * jnp.abs(torque) / ACTION_TORQUES_NM[2],
                    error.shape,
                ),
                -experiment["time_cost_per_s"]
                * (1.0 + updated.physics_steps / float(MAX_PHYSICS_STEPS)),
                -experiment["arm_cost_per_s"] * (theta / ARM_LIMIT_RAD) ** 2,
                experiment["upright_reward_per_s"] * upright,
            ),
            axis=-1,
        )
        return updated, jnp.where(active[..., None], PHYSICS_DT_S * rates, 0.0)

    next_state, sample_components = jax.lax.scan(
        advance_reward, env_state, None, length=HOLD_PHYSICS_STEPS
    )
    components = jnp.sum(sample_components, axis=0).astype(jnp.float32)
    reward = jnp.sum(components, axis=-1)
    done = next_state.success | next_state.timeout
    return next_state, observe(next_state), reward, reward, components, done


def collect(
    learner: Mapping[str, Any],
    rollout: Mapping[str, Any],
    replay: Mapping[str, Array],
    experiment: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Array], dict[str, Array]]:
    """Collect one fixed on-device block, auto-reset lanes, and append uniform replay."""

    network = QNetwork(tuple(experiment["hidden_widths"]), experiment["activation_name"])
    environment_count = rollout["env_state"].x.shape[0]

    def advance(
        carry: Mapping[str, Any], _: None
    ) -> tuple[dict[str, Any], tuple[Array, Array, Array, Array, Array, Array]]:
        behavior_key, reset_key = carry["key"]
        behavior_key, branch_key, random_action_key, heuristic_key = jax.random.split(
            behavior_key, 4
        )
        observation = observe(carry["env_state"])
        q_values = cast(Array, network.apply(learner["params"], observation))
        greedy_action = jnp.argmax(q_values, axis=-1).astype(jnp.int32)
        random_action = jax.random.randint(
            random_action_key, (environment_count,), 0, ACTION_COUNT, dtype=jnp.int32
        )

        exploratory_action = random_action
        if experiment["heuristic_fraction"] > 0.0:
            candidate_state = jax.tree.map(
                lambda value: jnp.repeat(value[:, None, ...], ACTION_COUNT, axis=1),
                carry["env_state"],
            )
            candidate_action = jnp.broadcast_to(
                jnp.arange(ACTION_COUNT, dtype=jnp.int32),
                (environment_count, ACTION_COUNT),
            )
            candidate_reward = transition(candidate_state, candidate_action, experiment)[3]
            heuristic_action = jnp.argmax(candidate_reward, axis=-1).astype(jnp.int32)
            guided = (
                jax.random.uniform(heuristic_key, (environment_count,))
                < experiment["heuristic_fraction"]
            )
            exploratory_action = jnp.where(guided, heuristic_action, random_action)
        explore = jax.random.uniform(branch_key, (environment_count,)) < experiment["epsilon"]
        action = jnp.where(explore, exploratory_action, greedy_action)
        next_state, next_observation, base_reward, train_reward, components, done = transition(
            carry["env_state"], action, experiment
        )

        elapsed = PHYSICS_DT_S * (
            next_state.physics_steps - carry["env_state"].physics_steps
        ).astype(jnp.float32)
        previous_action = carry["episode_totals"][:, 6].astype(jnp.int32)
        on_time = elapsed * (action != 0)
        off_to_on = (previous_action == 0) & (action != 0)
        reversal = ACTION_TORQUES_NM[previous_action] * ACTION_TORQUES_NM[action] < 0.0
        theta, alpha, _, nu = jnp.moveaxis(next_state.x, -1, 0)
        energy_ratio = (
            0.5 * MODEL.pendulum_inertia_kg_m2 * nu**2
            + MODEL.gravity_torque_nm * (1.0 - jnp.cos(alpha))
        ).astype(jnp.float32) / TARGET_ENERGY_J
        starting_theta, starting_alpha, _, starting_nu = jnp.moveaxis(carry["env_state"].x, -1, 0)
        starting_energy_ratio = (
            0.5 * MODEL.pendulum_inertia_kg_m2 * starting_nu**2
            + MODEL.gravity_torque_nm * (1.0 - jnp.cos(starting_alpha))
        ).astype(jnp.float32) / TARGET_ENERGY_J
        energy_ratio = jnp.maximum(energy_ratio, starting_energy_ratio)
        episode_totals = carry["episode_totals"].at[:, 0].add(base_reward)
        episode_totals = episode_totals.at[:, 1].add(train_reward)
        episode_totals = episode_totals.at[:, 2].add(on_time)
        episode_totals = episode_totals.at[:, 3].add(elapsed)
        episode_totals = episode_totals.at[:, 4].add(off_to_on)
        episode_totals = episode_totals.at[:, 5].add(reversal)
        episode_totals = episode_totals.at[:, 6].set(action)
        episode_totals = episode_totals.at[:, 7].max(energy_ratio)
        peak_theta = jnp.maximum(jnp.abs(theta), jnp.abs(starting_theta)).astype(jnp.float32)
        episode_totals = episode_totals.at[:, 8].max(peak_theta)
        completed_add = jnp.stack(
            (
                done,
                next_state.success,
                done & next_state.arm_violation,
                next_state.timeout,
                done * episode_totals[:, 0],
                done * episode_totals[:, 1],
                done * episode_totals[:, 2],
                done * episode_totals[:, 3],
                done * episode_totals[:, 4],
                done * episode_totals[:, 5],
                done * episode_totals[:, 7],
                done * episode_totals[:, 8],
            ),
            axis=-1,
        ).astype(jnp.float32)

        reset_key, stratum_key, tight_key, sample_key = jax.random.split(reset_key, 4)
        lane_keys = jax.random.split(sample_key, environment_count)
        uniforms = jax.random.uniform(stratum_key, (environment_count,), dtype=jnp.float32)
        stage = jnp.asarray(experiment["stage"], dtype=jnp.int32)
        stage_one = jnp.where(uniforms < 0.50, 2, jnp.where(uniforms < 0.75, 1, 0))
        stage_two = jnp.floor(3.0 * uniforms).astype(jnp.int32)
        strata = jnp.where(stage == 0, 2, jnp.where(stage == 1, stage_one, stage_two))
        reset_state = jax.vmap(reset)(lane_keys, strata)
        tight_sample = jax.random.uniform(
            tight_key,
            (environment_count, 4),
            minval=jnp.array([-0.08, -0.12, -0.15, -0.30]),
            maxval=jnp.array([0.08, 0.12, 0.15, 0.30]),
            dtype=jnp.float32,
        )
        tight_x = tight_sample.at[:, 1].add(jnp.asarray(jnp.pi, dtype=jnp.float32))
        tight_mask = (stage == 0) & (uniforms < experiment["stage_zero_tight_fraction"])
        reset_state = reset_state._replace(x=jnp.where(tight_mask[:, None], tight_x, reset_state.x))
        continued_state = jax.tree.map(
            lambda fresh, current: jnp.where(
                done.reshape(done.shape + (1,) * (current.ndim - done.ndim)),
                fresh,
                current,
            ),
            reset_state,
            next_state,
        )
        next_episode_totals = jnp.where(done[:, None], 0.0, episode_totals)
        metrics = jnp.concatenate(
            (
                jnp.sum(completed_add, axis=0),
                jnp.sum(components, axis=0),
                jnp.bincount(action, length=ACTION_COUNT),
            )
        )
        return {
            "env_state": continued_state,
            "key": jnp.stack((behavior_key, reset_key)),
            "episode_totals": next_episode_totals,
            "completed_totals": carry["completed_totals"] + jnp.sum(completed_add, axis=0),
        }, (observation, action, train_reward, next_observation, done, metrics)

    updated_rollout, records = jax.lax.scan(
        advance,
        rollout,
        None,
        length=experiment["collection_decisions"],
    )
    observation, action, reward, next_observation, done, block_metrics = records
    transition_count = observation.shape[0] * observation.shape[1]
    indices = (replay["position"] + jnp.arange(transition_count)) % replay["observation"].shape[0]
    updated_replay = {
        "observation": replay["observation"].at[indices].set(observation.reshape((-1, 7))),
        "action": replay["action"].at[indices].set(action.reshape(-1)),
        "reward": replay["reward"].at[indices].set(reward.reshape(-1)),
        "next_observation": replay["next_observation"]
        .at[indices]
        .set(next_observation.reshape((-1, 7))),
        "done": replay["done"].at[indices].set(done.reshape(-1)),
        "position": (replay["position"] + transition_count) % replay["observation"].shape[0],
        "size": jnp.minimum(replay["size"] + transition_count, replay["observation"].shape[0]),
    }
    totals = jnp.sum(block_metrics, axis=0)
    return (
        dict(updated_rollout),
        updated_replay,
        {
            "completed": totals[0],
            "successes": totals[1],
            "completed_arm_excursions": totals[2],
            "timeouts": totals[3],
            "base_return_sum": totals[4],
            "train_return_sum": totals[5],
            "energy_cost_sum": totals[12],
            "torque_cost_sum": totals[13],
            "time_cost_sum": totals[14],
            "arm_cost_sum": totals[15],
            "upright_reward_sum": totals[16],
            "off_actions": totals[17],
            "negative_pump_actions": totals[18],
            "positive_pump_actions": totals[19],
            "negative_fine_actions": totals[20],
            "positive_fine_actions": totals[21],
        },
    )
