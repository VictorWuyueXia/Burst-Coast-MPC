"""Physical-state-only continuing task, targeted resets and genuine truncation bootstrap."""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from rotary_pendulum.environment.jax_dynamics import MODEL
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, EnvState, reset, step
from rotary_pendulum.RL.jax_residual_control import TARGET_ENERGY_J
from rotary_pendulum.RL.jax_residual_task import observe
from rotary_pendulum.RL.jax_td3 import Actor


def targeted_resets(key: Array, count: int) -> tuple[EnvState, Array]:
    """Interleave downward, moving capture, outward braking and local recovery starts."""
    keys = jax.random.split(key, 3)
    state = jax.vmap(reset)(jax.random.split(keys[0], count), jnp.zeros(count, jnp.int32))
    group = jnp.arange(count) % 4
    sample = jax.random.uniform(keys[1], (count, 4), minval=-1.0, maxval=1.0)
    sign = jnp.where(jax.random.bernoulli(keys[2], shape=(count,)), 1.0, -1.0)
    capture = sample * jnp.array([0.6, 0.3, 0.4, 1.2])
    braking = sample * jnp.array([0.0, 0.12, 0.0, 0.3])
    braking = braking.at[:, 0].set(sign * ARM_LIMIT_RAD * (0.55 + 0.1 * (sample[:, 0] + 1)))
    braking = braking.at[:, 2].set(sign * (0.1 + 0.15 * (sample[:, 2] + 1)))
    recovery = sample * jnp.array([0.35, 0.12, 0.25, 0.5])
    local = jnp.where((group == 1)[:, None], capture, braking)
    local = jnp.where((group == 3)[:, None], recovery, local)
    local = local.at[:, 1].set((local[:, 1] + 2 * jnp.pi) % (2 * jnp.pi) - jnp.pi)
    return state._replace(x=jnp.where((group == 0)[:, None], state.x, local)), group


def transition(
    state: EnvState, action: Array, settings: Mapping[str, Any]
) -> tuple[EnvState, Array, Array, Array]:
    """Continue after success; stationary nonpositive reward integrated over five samples."""
    torque = settings["torque_limit_nm"] * jnp.clip(action, -1, 1)

    def advance(current: EnvState, unused: None) -> tuple[EnvState, tuple[Array, Array]]:
        following = step(
            current,
            torque,
            physics_steps=1,
            max_physics_steps=settings["deadline_s"] * 50,
            goal_hold_steps=500,
            stop_on_success=False,
        )
        theta, alpha, omega, nu = jnp.moveaxis(following.x, -1, 0)
        beta = jnp.arctan2(-jnp.sin(alpha), -jnp.cos(alpha))
        upright = jnp.exp(
            -0.5
            * jnp.sum(
                (jnp.stack((beta, nu, omega), axis=-1) / jnp.asarray(settings["upright_widths"]))
                ** 2,
                axis=-1,
            )
        )
        energy = 0.5 * MODEL.pendulum_inertia_kg_m2 * nu**2 + MODEL.gravity_torque_nm * (
            1 - jnp.cos(alpha)
        )
        arm = (theta / ARM_LIMIT_RAD) ** 2 + (
            jnp.maximum(abs(theta) - 0.8 * ARM_LIMIT_RAD, 0) / (0.2 * ARM_LIMIT_RAD)
        ) ** 2
        rate = (
            -settings["upright_reward_per_s"] * (1 - upright)
            - settings["energy_cost_per_s"] * abs(energy / TARGET_ENERGY_J - 1)
            - settings["torque_cost_per_s"] * jnp.clip(action, -1, 1) ** 2
            - settings["arm_cost_per_s"] * arm
        )
        return following, (jnp.where(current.timeout, 0, 0.02 * rate), following.x)

    following, (reward, physics) = jax.lax.scan(advance, state, None, length=5)
    observation = observe(following, jnp.zeros_like(following.physics_steps), stationary=True)
    return following, reward.sum(axis=0), jnp.moveaxis(physics, 0, 1), observation


def collect(
    learner: Mapping[str, Any],
    rollout: Mapping[str, Any],
    replay: Mapping[str, Array],
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Array], dict[str, Array]]:
    """Reset at sampling boundaries; store the pre-reset next state with done=False."""
    state = rollout["state"]
    key, noise_key, reset_key = jax.random.split(rollout["key"], 3)
    count, capacity = state.x.shape[0], replay["observation"].shape[0]
    observation = observe(state, jnp.zeros_like(state.physics_steps), stationary=True)
    command = Actor(tuple(settings["hidden_widths"])).apply(learner["actor"], observation)
    action = jnp.clip(
        command + settings["exploration_noise"] * jax.random.normal(noise_key, (count,)), -1, 1
    )
    following, reward, _, next_observation = transition(state, action, settings)
    indices = (replay["position"] + jnp.arange(count)) % capacity
    entries = dict(
        observation=observation,
        next_observation=next_observation,
        action=action,
        reward=reward,
        done=jnp.zeros(count, jnp.bool_),
    )
    updated = {name: replay[name].at[indices].set(value) for name, value in entries.items()}
    updated.update(
        position=(replay["position"] + count) % capacity,
        size=jnp.minimum(replay["size"] + count, capacity),
    )
    fresh, _ = targeted_resets(reset_key, count)
    continued = jax.tree.map(
        lambda new, old: jnp.where(
            following.timeout.reshape((count,) + (1,) * (old.ndim - 1)), new, old
        ),
        fresh,
        following,
    )
    return (
        dict(state=continued, key=key),
        updated,
        dict(
            reward_mean=reward.mean(),
            torque_rms=jnp.sqrt(jnp.mean(action**2)),
            arm_crossing_fraction=following.arm_violation.mean(),
            resets=following.timeout.sum(),
        ),
    )
