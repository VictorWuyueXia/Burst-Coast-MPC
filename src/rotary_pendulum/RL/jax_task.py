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
    EnvState,
    reset,
    step,
)
from rotary_pendulum.RL.jax_q import QNetwork

ACTION_TORQUES_NM = jnp.array([0.0, -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM], dtype=jnp.float32)
TARGET_ENERGY_J = 2.0 * MODEL.gravity_torque_nm
ARM_SPEED_SCALE = float(ARM_LIMIT_RAD) * MODEL.natural_frequency_rad_s
PENDULUM_SPEED_SCALE = 2.0 * jnp.sqrt(MODEL.gravity_torque_nm / MODEL.pendulum_inertia_kg_m2)


def observe(env_state: EnvState, experiment: Mapping[str, Any]) -> tuple[Array, Array]:
    """Return the seven Markov features and two terminal-masked reward potentials."""

    theta, alpha, omega, nu = jnp.moveaxis(env_state.x, -1, 0)
    swing_energy = 0.5 * MODEL.pendulum_inertia_kg_m2 * nu**2 + MODEL.gravity_torque_nm * (
        1.0 - jnp.cos(alpha)
    )
    energy_error = (swing_energy - TARGET_ENERGY_J) / TARGET_ENERGY_J
    beta = jnp.arctan2(jnp.sin(alpha - jnp.pi), jnp.cos(alpha - jnp.pi))
    capture_error = 0.25 * (
        (theta / GOAL.theta_tolerance_rad) ** 2
        + (beta / GOAL.beta_tolerance_rad) ** 2
        + (omega / GOAL.omega_tolerance_rad_s) ** 2
        + (nu / GOAL.nu_tolerance_rad_s) ** 2
    )
    done = env_state.success | env_state.arm_violation | env_state.timeout
    active = (~done).astype(env_state.x.dtype)
    potentials = jnp.stack(
        (
            -active * energy_error**2 / (1.0 + energy_error**2),
            -active * experiment["capture_weight"] * capture_error / (1.0 + capture_error),
        ),
        axis=-1,
    )
    observation = jnp.stack(
        (
            theta / ARM_LIMIT_RAD,
            jnp.sin(alpha),
            jnp.cos(alpha),
            omega / ARM_SPEED_SCALE,
            nu / PENDULUM_SPEED_SCALE,
            (1000.0 - env_state.physics_steps) / 1000.0,
            env_state.goal_count / float(GOAL.hold_steps),
        ),
        axis=-1,
    ).astype(jnp.float32)
    return observation, potentials.astype(jnp.float32)


def transition(
    env_state: EnvState, action_index: Array, experiment: Mapping[str, Any]
) -> tuple[EnvState, Array, Array, Array, Array, Array]:
    """Advance one decision and return base, shaped, and decomposed rewards."""

    _, starting_potential = observe(env_state, experiment)
    torque = ACTION_TORQUES_NM[jnp.asarray(action_index, dtype=jnp.int32)]
    next_state = step(env_state, torque)
    next_observation, ending_potential = observe(next_state, experiment)
    starting_done = env_state.success | env_state.arm_violation | env_state.timeout
    done = next_state.success | next_state.arm_violation | next_state.timeout
    elapsed = PHYSICS_DT_S * (next_state.physics_steps - env_state.physics_steps).astype(
        jnp.float32
    )
    terminal = (
        experiment["success_reward"] * (next_state.success & ~starting_done)
        - experiment["arm_failure_cost"] * (next_state.arm_violation & ~starting_done)
        - experiment["timeout_cost"] * (next_state.timeout & ~starting_done)
    )
    on_cost = -experiment["on_cost_per_s"] * elapsed * (action_index != 0)
    time_cost = -experiment["time_cost_per_s"] * elapsed
    base_reward = terminal + on_cost + time_cost
    shaping = ending_potential - starting_potential
    components = jnp.concatenate(
        (shaping, on_cost[..., None], time_cost[..., None], terminal[..., None]),
        axis=-1,
    )
    train_reward = base_reward + jnp.sum(shaping, axis=-1)
    return next_state, next_observation, base_reward, train_reward, components, done


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
        observation, _potential = observe(carry["env_state"], experiment)
        q_values = cast(Array, network.apply(learner["params"], observation))
        greedy_action = jnp.argmax(q_values, axis=-1).astype(jnp.int32)
        random_action = jax.random.randint(
            random_action_key, (environment_count,), 0, 3, dtype=jnp.int32
        )

        exploratory_action = random_action
        if experiment["heuristic_fraction"] > 0.0:
            candidate_state = jax.tree.map(
                lambda value: jnp.repeat(value[:, None, ...], 3, axis=1),
                carry["env_state"],
            )
            candidate_action = jnp.broadcast_to(
                jnp.arange(3, dtype=jnp.int32), (environment_count, 3)
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
        reversal = (previous_action != 0) & (action != 0) & (previous_action != action)
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
                next_state.arm_violation,
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
                jnp.bincount(action, length=3),
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
            "arm_failures": totals[2],
            "timeouts": totals[3],
            "base_return_sum": totals[4],
            "train_return_sum": totals[5],
            "energy_shaping_sum": totals[12],
            "capture_shaping_sum": totals[13],
            "on_cost_sum": totals[14],
            "time_cost_sum": totals[15],
            "terminal_reward_sum": totals[16],
            "off_actions": totals[17],
            "negative_actions": totals[18],
            "positive_actions": totals[19],
        },
    )
