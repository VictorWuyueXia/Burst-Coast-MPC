"""Supervised initialization from complete, independently qualified teacher episodes."""

from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from jax import Array

from rotary_pendulum.RL.jax_td3 import Actor, Critic


def pretrain(
    learner: Mapping[str, Any], demonstrations: Mapping[str, Array], settings: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Array]]:
    """Fit teacher actions and the return targets declared by the demonstration contract."""
    key, sample_key = jax.random.split(learner["key"])
    indices = jax.random.randint(
        sample_key, (settings["minibatch_size"],), 0, demonstrations["action"].shape[0]
    )
    observation, action = demonstrations["observation"][indices], demonstrations["action"][indices]
    actor_network = Actor(tuple(settings["hidden_widths"]))
    critic_network = Critic(tuple(settings["hidden_widths"]))
    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(settings["learning_rate"]))

    def actor_loss(params: Any) -> Array:
        prediction = cast(Array, actor_network.apply(params, observation))
        return jnp.mean((prediction - action) ** 2)

    def critic_loss(params: Any) -> Array:
        predictions = jnp.stack(
            [cast(Array, critic_network.apply(p, observation, action)) for p in params]
        )
        return jnp.mean((predictions - demonstrations["return"][indices]) ** 2)

    loss, gradients = jax.value_and_grad(actor_loss)(learner["actor"])
    changes, actor_opt = optimizer.update(gradients, learner["actor_opt"], learner["actor"])
    actor = optax.apply_updates(learner["actor"], changes)
    critics, critic_opt, value_loss = learner["critics"], learner["critic_opt"], jnp.array(0.0)
    if settings["critic_pretrain"]:
        value_loss, gradients = jax.value_and_grad(critic_loss)(critics)
        changes, critic_opt = optimizer.update(gradients, critic_opt, critics)
        critics = optax.apply_updates(critics, changes)
    return dict(learner) | {
        "actor": actor,
        "target_actor": actor,
        "actor_opt": actor_opt,
        "critics": critics,
        "target_critics": critics,
        "critic_opt": critic_opt,
        "key": key,
    }, {"cloning_mse": loss, "return_mse": value_loss}
