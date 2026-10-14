"""Shared TD3 with physical-action critics and explicit finite or continuing contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax import Array

from rotary_pendulum.RL.jax_residual_control import residual_action


class Actor(nn.Module):
    """Declared physical or deadline-aware features to a normalized action or residual."""

    hidden_widths: tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(self, observation: Array) -> Array:
        hidden = observation
        for width in self.hidden_widths:
            hidden = nn.silu(nn.Dense(width)(hidden))
        output = nn.Dense(
            1, kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.zeros_init()
        )(hidden)
        return jnp.tanh(output[..., 0])


class Critic(nn.Module):
    """Scalar return from observation and actual applied torque divided by the cap."""

    hidden_widths: tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(self, observation: Array, action: Array) -> Array:
        hidden = jnp.concatenate((observation, action[..., None]), axis=-1)
        for width in self.hidden_widths:
            hidden = nn.silu(nn.Dense(width)(hidden))
        return nn.Dense(1)(hidden)[..., 0]


def initialize(key: Array, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Initialize independent critics, zero residual, target copies and Adam states."""

    actor_key, first_key, second_key, learner_key = jax.random.split(key, 4)
    dimensions = 5 if settings["contract_id"] == "rotary-stationary-td3-v3" else 8
    observation, action = jnp.zeros((1, dimensions)), jnp.zeros((1,))
    widths = tuple(settings["hidden_widths"])
    actor = Actor(widths).init(actor_key, observation)
    critics = (
        Critic(widths).init(first_key, observation, action),
        Critic(widths).init(second_key, observation, action),
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
    demonstrations: Mapping[str, Array] | None = None,
) -> tuple[dict[str, Any], dict[str, Array]]:
    """One twin-critic update; every second update also changes actor and targets."""

    key, sample_key, noise_key = jax.random.split(learner["key"], 3)
    actor_network = Actor(tuple(settings["hidden_widths"]))
    critic_network = Critic(tuple(settings["hidden_widths"]))
    indices = jax.random.randint(sample_key, (settings["minibatch_size"],), 0, replay["size"])
    demonstration_count = 0
    if demonstrations is not None:
        demonstration_count = round(settings["demonstration_fraction"] * settings["minibatch_size"])
        teacher_indices = jax.random.randint(
            jax.random.fold_in(sample_key, 1),
            (demonstration_count,),
            0,
            demonstrations["action"].shape[0],
        )
        replay = {
            name: jnp.concatenate(
                (demonstrations[name][teacher_indices], replay[name][indices[demonstration_count:]])
            )
            for name in ("observation", "next_observation", "action", "reward", "done")
        }
        indices = jnp.arange(settings["minibatch_size"])
    observation, next_observation = (
        replay["observation"][indices],
        replay["next_observation"][indices],
    )
    noise = jnp.clip(
        settings["target_noise"] * jax.random.normal(noise_key, (settings["minibatch_size"],)),
        -2.5 * settings["target_noise"],
        2.5 * settings["target_noise"],
    )
    residual = cast(Array, actor_network.apply(learner["target_actor"], next_observation))
    applied = (
        residual_action(next_observation, residual, noise, settings)[0]
        / settings["torque_limit_nm"]
    )
    target_values = jnp.stack(
        [
            cast(Array, critic_network.apply(params, next_observation, applied))
            for params in learner["target_critics"]
        ]
    )
    continuation = jnp.min(target_values, axis=0)
    if settings["contract_id"] == "rotary-stationary-td3-v3":
        continuation = settings["discount"] * continuation
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
            [
                cast(Array, critic_network.apply(p, observation, replay["action"][indices]))
                for p in params
            ]
        )
        return jnp.mean((values - target[None, :]) ** 2)

    loss, gradients = jax.value_and_grad(critic_loss)(learner["critics"])
    updates, critic_opt = optimizer.update(gradients, learner["critic_opt"], learner["critics"])
    critics = optax.apply_updates(learner["critics"], updates)

    def actor_loss(params: Any) -> Array:
        command = cast(Array, actor_network.apply(params, observation))
        action = residual_action(observation, command, jnp.zeros(command.shape), settings)[0]
        objective = -jnp.mean(
            cast(
                Array,
                critic_network.apply(critics[0], observation, action / settings["torque_limit_nm"]),
            )
        )
        if demonstration_count:
            objective += settings["cloning_weight"] * jnp.mean(
                (command[:demonstration_count] - replay["action"][:demonstration_count]) ** 2
            )
        return objective

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
