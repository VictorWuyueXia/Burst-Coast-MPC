"""Deadline-aware continuous task and uniform replay collection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array

from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICS_DT_S
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, GOAL, EnvState, reset, step
from rotary_pendulum.RL.jax_residual_control import (
    ARM_SPEED_SCALE,
    PENDULUM_SPEED_SCALE,
    POLICY_TORQUE_NM,
    TARGET_ENERGY_J,
    residual_action,
)


def observe(state: EnvState, deadline_steps: Array) -> Array:
    """Eight features: physical state, elapsed/remaining seconds over 60, hold memory."""

    theta, alpha, omega, nu = jnp.moveaxis(state.x, -1, 0)
    return jnp.stack(
        (
            theta / ARM_LIMIT_RAD,
            jnp.sin(alpha),
            jnp.cos(alpha),
            omega / ARM_SPEED_SCALE,
            nu / PENDULUM_SPEED_SCALE,
            state.physics_steps / 3000.0,
            (deadline_steps - state.physics_steps) / 3000.0,
            state.goal_count / float(GOAL.hold_steps),
        ),
        axis=-1,
    ).astype(jnp.float32)


def transition(
    state: EnvState, torque: Array, deadline_steps: Array, settings: Mapping[str, Any]
) -> tuple[EnvState, Array, Array, Array, Array]:
    """Five dense components, integrated at physics rate; retain substep states for audits."""

    applied = jnp.broadcast_to(
        jnp.clip(torque, -POLICY_TORQUE_NM, POLICY_TORQUE_NM), state.x.shape[:-1]
    )
    widths = jnp.asarray(settings["upright_widths"])

    def advance(current: EnvState, _: None) -> tuple[EnvState, tuple[Array, Array]]:
        active = ~(current.success | current.timeout)
        following = step(current, applied, physics_steps=1, max_physics_steps=deadline_steps)
        theta, alpha, omega, nu = jnp.moveaxis(following.x, -1, 0)
        energy = 0.5 * MODEL.pendulum_inertia_kg_m2 * nu**2 + MODEL.gravity_torque_nm * (
            1.0 - jnp.cos(alpha)
        )
        beta = jnp.arctan2(jnp.sin(alpha - jnp.pi), jnp.cos(alpha - jnp.pi))
        upright = jnp.exp(
            -0.5
            * jnp.sum(
                (jnp.stack((beta, nu, omega), axis=-1) / widths) ** 2,
                axis=-1,
            )
        )
        rates = jnp.stack(
            (
                -settings["energy_cost_per_s"] * jnp.abs(energy / TARGET_ENERGY_J - 1.0),
                -settings["torque_cost_per_s"] * jnp.abs(applied) / POLICY_TORQUE_NM,
                -settings["time_cost_per_s"] * (1.0 + following.physics_steps / deadline_steps),
                -settings["arm_cost_per_s"] * (theta / ARM_LIMIT_RAD) ** 2,
                settings["upright_reward_per_s"] * upright,
            ),
            axis=-1,
        )
        components = jnp.where(active[..., None], PHYSICS_DT_S * rates, 0.0)
        return following, (components, following.x)

    following, (samples, states) = jax.lax.scan(advance, state, None, length=5)
    components = jnp.sum(samples, axis=0)
    return (
        following,
        jnp.sum(components, axis=-1),
        components,
        states,
        observe(following, deadline_steps),
    )


def collect(
    learner: Mapping[str, Any],
    rollout: Mapping[str, Any],
    replay: Mapping[str, Array],
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Array], dict[str, Array]]:
    """Collect one parallel decision, retain applied actions, then reset completed lanes."""

    # Local import avoids making the TD3 update depend on the collection implementation.
    from rotary_pendulum.RL.jax_td3 import Actor

    state = rollout["state"]
    count = state.x.shape[0]
    key, noise_key, reset_key, stratum_key = jax.random.split(rollout["key"], 4)
    deadline = jnp.full((count,), round(settings["deadline_s"] / PHYSICS_DT_S), jnp.int32)
    observation = observe(state, deadline)
    residual = cast(Array, Actor().apply(learner["actor"], observation))
    noise = settings["exploration_noise"] * jax.random.normal(noise_key, (count,))
    torque, heuristic, proposed, mode = residual_action(observation, residual, noise, settings)
    following, reward, components, _, next_observation = transition(
        state, torque, deadline, settings
    )
    done = following.success | following.timeout
    indices = (replay["position"] + jnp.arange(count)) % replay["observation"].shape[0]
    entries = {
        "observation": observation,
        "action": torque / POLICY_TORQUE_NM,
        "reward": reward,
        "next_observation": next_observation,
        "done": done,
    }
    updated = {name: replay[name].at[indices].set(value) for name, value in entries.items()}
    updated["position"] = (replay["position"] + count) % replay["observation"].shape[0]
    updated["size"] = jnp.minimum(replay["size"] + count, replay["observation"].shape[0])
    fresh = jax.vmap(reset)(
        jax.random.split(reset_key, count), jax.random.randint(stratum_key, (count,), 0, 3)
    )
    continued = jax.tree.map(
        lambda new, old: jnp.where(done.reshape(done.shape + (1,) * (old.ndim - 1)), new, old),
        fresh,
        following,
    )
    return (
        {"state": continued, "key": key},
        updated,
        {
            "completed": jnp.sum(done),
            "successes": jnp.sum(following.success),
            "reward_mean": jnp.mean(reward),
            "components_mean": jnp.mean(components, axis=0),
            "filter_fraction": jnp.mean(mode != 0),
            "heuristic_nm": heuristic,
            "proposed_nm": proposed,
            "applied_nm": torque,
            "filter_mode": mode,
        },
    )
