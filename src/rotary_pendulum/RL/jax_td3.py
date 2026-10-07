"""Residual TD3 with physical-action critics and finite-horizon terminal masking."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax import Array

from rotary_pendulum.RL.jax_residual_control import POLICY_TORQUE_NM, residual_action


class Actor(nn.Module):
    """Eight Markov features to a residual in [-1, 1], scaled to twice the cap."""

    @nn.compact
    def __call__(self, observation: Array) -> Array:
        hidden = nn.silu(nn.Dense(128)(observation))
        hidden = nn.silu(nn.Dense(128)(hidden))
        output = nn.Dense(
            1, kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.zeros_init()
        )(hidden)
        return jnp.tanh(output[..., 0])


class Critic(nn.Module):
    """Scalar return from observation and actual applied torque divided by the cap."""

    @nn.compact
    def __call__(self, observation: Array, action: Array) -> Array:
        hidden = jnp.concatenate((observation, action[..., None]), axis=-1)
        hidden = nn.silu(nn.Dense(128)(hidden))
        hidden = nn.silu(nn.Dense(128)(hidden))
        return nn.Dense(1)(hidden)[..., 0]


def initialize(key: Array, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Initialize independent critics, zero residual, target copies and Adam states."""

    actor_key, first_key, second_key, learner_key = jax.random.split(key, 4)
    observation, action = jnp.zeros((1, 8)), jnp.zeros((1,))
    actor = Actor().init(actor_key, observation)
    critics = (
        Critic().init(first_key, observation, action),
        Critic().init(second_key, observation, action),
    )
    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(settings["learning_rate"]))
    return {
        "actor": actor,
        "critics": critics,
        "target_actor": actor,
        "target_critics": critics,
        "actor_opt": optimizer.init(actor),
        "critic_opt": optimizer.init(critics),
        "updates": jnp.array(0, jnp.int32),
        "key": learner_key,
    }


def update(
    learner: Mapping[str, Any],
    replay: Mapping[str, Array],
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Array]]:
    """One twin-critic update; every second update also changes actor and targets."""

    key, sample_key, noise_key = jax.random.split(learner["key"], 3)
    indices = jax.random.randint(sample_key, (settings["minibatch_size"],), 0, replay["size"])
    observation, next_observation = (
        replay["observation"][indices],
        replay["next_observation"][indices],
    )
    noise = jnp.clip(0.2 * jax.random.normal(noise_key, (settings["minibatch_size"],)), -0.5, 0.5)
    residual = cast(Array, Actor().apply(learner["target_actor"], next_observation))
    applied = residual_action(next_observation, residual, noise, settings)[0] / POLICY_TORQUE_NM
    target_values = jnp.stack(
        [
            cast(Array, Critic().apply(params, next_observation, applied))
            for params in learner["target_critics"]
        ]
    )
    continuation = jnp.min(target_values, axis=0)
    target = jax.lax.stop_gradient(
        jnp.where(
            replay["done"][indices],
            replay["reward"][indices],
            replay["reward"][indices] + continuation,
        )
    )
    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(settings["learning_rate"]))

    def critic_loss(params: Any) -> Array:
        values = jnp.stack(
            [cast(Array, Critic().apply(p, observation, replay["action"][indices])) for p in params]
        )
        return jnp.mean((values - target[None, :]) ** 2)

    loss, gradients = jax.value_and_grad(critic_loss)(learner["critics"])
    updates, critic_opt = optimizer.update(gradients, learner["critic_opt"], learner["critics"])
    critics = optax.apply_updates(learner["critics"], updates)

    def actor_loss(params: Any) -> Array:
        command = cast(Array, Actor().apply(params, observation))
        action = residual_action(observation, command, jnp.zeros(command.shape), settings)[0]
        return -jnp.mean(
            cast(Array, Critic().apply(critics[0], observation, action / POLICY_TORQUE_NM))
        )

    def delayed(_: None) -> tuple[Any, Any, Any, Any, Array]:
        policy_loss, policy_gradient = jax.value_and_grad(actor_loss)(learner["actor"])
        changes, actor_opt = optimizer.update(
            policy_gradient, learner["actor_opt"], learner["actor"]
        )
        actor = optax.apply_updates(learner["actor"], changes)
        target_actor = jax.tree.map(
            lambda old, new: 0.995 * old + 0.005 * new, learner["target_actor"], actor
        )
        target_critics = jax.tree.map(
            lambda old, new: 0.995 * old + 0.005 * new, learner["target_critics"], critics
        )
        return actor, actor_opt, target_actor, target_critics, policy_loss

    count = learner["updates"] + 1
    actor, actor_opt, target_actor, target_critics, policy_loss = jax.lax.cond(
        count % 2 == 0,
        delayed,
        lambda _: (
            learner["actor"],
            learner["actor_opt"],
            learner["target_actor"],
            learner["target_critics"],
            jnp.array(0.0),
        ),
        None,
    )
    return {
        "actor": actor,
        "critics": critics,
        "target_actor": target_actor,
        "target_critics": target_critics,
        "actor_opt": actor_opt,
        "critic_opt": critic_opt,
        "key": key,
        "updates": count,
    }, {
        "critic_loss": loss,
        "actor_loss": policy_loss,
        "actor_updated": count % 2 == 0,
        "target_mean": jnp.mean(target),
        "critic_gradient_norm": optax.tree.norm(gradients),
    }
