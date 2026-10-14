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
    TARGET_ENERGY_J,
    residual_action,
)


def observe(
    state: EnvState,
    deadline_steps: Array,
    hold_steps: int = GOAL.hold_steps,
    *,
    stationary: bool = False,
) -> Array:
    """Eight features: physical state, elapsed/remaining seconds over 60, hold memory."""

    theta, alpha, omega, nu = jnp.moveaxis(state.x, -1, 0)
    observation = jnp.stack(
        (
            theta / ARM_LIMIT_RAD,
            jnp.sin(alpha),
            jnp.cos(alpha),
            omega / ARM_SPEED_SCALE,
            nu / PENDULUM_SPEED_SCALE,
            state.physics_steps / 3000.0,
            (deadline_steps - state.physics_steps) / 3000.0,
            jnp.minimum(state.goal_count / float(hold_steps), 1.0),
        ),
        axis=-1,
    ).astype(jnp.float32)
    return observation[..., :5] if stationary else observation


def transition(
    state: EnvState, torque: Array, deadline_steps: Array, settings: Mapping[str, Any]
) -> tuple[EnvState, Array, Array, Array, Array]:
    """Five dense components, integrated at physics rate; retain substep states for audits."""

    applied = jnp.broadcast_to(
        jnp.clip(torque, -settings["torque_limit_nm"], settings["torque_limit_nm"]),
        state.x.shape[:-1],
    )
    widths = jnp.asarray(settings["upright_widths"])

    def advance(current: EnvState, _: None) -> tuple[EnvState, tuple[Array, Array]]:
        active = ~(current.success | current.timeout)
        following = step(
            current,
            applied,
            physics_steps=1,
            max_physics_steps=deadline_steps,
            goal_hold_steps=round(settings["hold_duration_s"] / PHYSICS_DT_S),
        )
        theta, alpha, omega, nu = jnp.moveaxis(following.x, -1, 0)
        energy = 0.5 * MODEL.pendulum_inertia_kg_m2 * nu**2 + MODEL.gravity_torque_nm * (
            1.0 - jnp.cos(alpha)
        )
        beta = jnp.arctan2(jnp.sin(alpha - jnp.pi), jnp.cos(alpha - jnp.pi))
        if settings["goal_mode"] == "strict":
            inside = (
                (jnp.abs(theta) <= 0.08)
                & (jnp.abs(beta) <= 0.08)
                & (jnp.abs(omega) <= 0.15)
                & (jnp.abs(nu) <= 0.20)
            )
            hold = jnp.where(
                active, jnp.where(inside, current.goal_count + 1, 0), current.goal_count
            )
            success = current.success | (active & (hold >= 5))
            following = following._replace(
                goal_count=hold,
                success=success,
                timeout=current.timeout
                | (active & (following.physics_steps >= deadline_steps) & ~success),
            )
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
                -settings["torque_cost_per_s"] * jnp.abs(applied) / settings["torque_limit_nm"],
                -settings["time_cost_per_s"] * (1.0 + following.physics_steps / deadline_steps),
                -settings["arm_cost_per_s"]
                * (
                    (theta / ARM_LIMIT_RAD) ** 2
                    + (jnp.maximum(jnp.abs(theta) - 0.8 * ARM_LIMIT_RAD, 0) / (0.2 * ARM_LIMIT_RAD))
                    ** 2
                ),
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
        observe(following, deadline_steps, round(settings["hold_duration_s"] / PHYSICS_DT_S)),
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
    observation = observe(state, deadline, round(settings["hold_duration_s"] / PHYSICS_DT_S))
    residual = cast(
        Array, Actor(tuple(settings["hidden_widths"])).apply(learner["actor"], observation)
    )
    noise = settings["exploration_noise"] * jax.random.normal(noise_key, (count,))
    torque, heuristic, proposed, mode = residual_action(observation, residual, noise, settings)
    following, reward, components, _, next_observation = transition(
        state, torque, deadline, settings
    )
    done = following.success | following.timeout
    indices = (replay["position"] + jnp.arange(count)) % replay["observation"].shape[0]
    entries = {
        "observation": observation,
        "action": torque / settings["torque_limit_nm"],
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
    if settings["reset_mode"] == "local":
        widths = jnp.where(
            (jnp.arange(count) % 2 == 0)[:, None],
            jnp.array([0.08, 0.12, 0.15, 0.30]),
            jnp.array([0.25, 0.25, 1.0, 1.0]),
        )
        local = jax.random.uniform(reset_key, (count, 4), minval=-1.0, maxval=1.0) * widths
        local = local.at[:, 1].set((local[:, 1] + 2 * jnp.pi) % (2 * jnp.pi) - jnp.pi)
        fresh = fresh._replace(x=local)
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
            "saturation_fraction": jnp.mean(
                jnp.abs(proposed) >= 0.999 * settings["torque_limit_nm"]
            ),
            "controllable_fraction": jnp.mean(
                (mode == 0) & (jnp.abs(proposed) < 0.999 * settings["torque_limit_nm"])
            ),
            "replay_terminal_fraction": jnp.sum(updated["done"]) / updated["size"],
            "heuristic_nm": heuristic,
            "proposed_nm": proposed,
            "applied_nm": torque,
            "filter_mode": mode,
        },
    )
