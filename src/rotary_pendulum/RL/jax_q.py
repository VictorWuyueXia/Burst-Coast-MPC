"""Small Flax Q-network and replay-based Double DQN updates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax import Array

ACTION_COUNT = 5


class QNetwork(nn.Module):
    """Map one normalized Markov observation to the five discrete action values."""

    hidden_widths: tuple[int, ...] = (128, 128)
    activation_name: str = "tanh"

    @nn.compact
    def __call__(self, observation: Array) -> Array:
        if self.activation_name == "tanh":
            activation = jnp.tanh
        elif self.activation_name == "silu":
            activation = jax.nn.silu
        elif self.activation_name == "relu":
            activation = jax.nn.relu
        else:
            raise ValueError(f"Unsupported Q-network activation: {self.activation_name}")
        hidden = jnp.asarray(observation, dtype=jnp.float32)
        for index, width in enumerate(self.hidden_widths):
            hidden = activation(
                nn.Dense(
                    width,
                    kernel_init=nn.initializers.glorot_uniform(),
                    bias_init=nn.initializers.zeros_init(),
                    dtype=jnp.float32,
                    param_dtype=jnp.float32,
                    name=f"hidden_{index}",
                )(hidden)
            )
        return nn.Dense(
            ACTION_COUNT,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros_init(),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            name="q_values",
        )(hidden)


def update(
    learner: Mapping[str, Any],
    replay: Mapping[str, Array],
    experiment: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Array]]:
    """Apply the configured number of uniform-replay Double DQN updates."""

    network = QNetwork(tuple(experiment["hidden_widths"]), experiment["activation_name"])
    optimizer = optax.chain(
        optax.clip_by_global_norm(experiment["gradient_norm_limit"]),
        optax.adam(
            experiment["learning_rate"],
            b1=experiment["adam_betas"][0],
            b2=experiment["adam_betas"][1],
            eps=experiment["adam_epsilon"],
        ),
    )

    def update_one(carry: Mapping[str, Any], _: None) -> tuple[dict[str, Any], Array]:
        sample_key, next_key = jax.random.split(carry["key"])
        indices = jax.random.randint(
            sample_key,
            (experiment["minibatch_size"],),
            0,
            replay["size"],
        )
        observation = replay["observation"][indices]
        action = replay["action"][indices]
        reward = replay["reward"][indices]
        next_observation = replay["next_observation"][indices]
        done = replay["done"][indices]
        online_next = cast(Array, network.apply(carry["params"], next_observation))
        selected_next = jnp.argmax(online_next, axis=-1)
        target_next = cast(Array, network.apply(carry["target_params"], next_observation))
        continuation = jnp.take_along_axis(target_next, selected_next[:, None], axis=-1)[:, 0]
        target = jax.lax.stop_gradient(jnp.where(done, reward, reward + continuation))

        def loss_fn(params: Any) -> tuple[Array, tuple[Array, Array]]:
            action_values = cast(Array, network.apply(params, observation))
            predicted = jnp.take_along_axis(action_values, action[:, None], axis=-1)[:, 0]
            residual = predicted - target
            if experiment["loss_name"] == "huber":
                losses = optax.huber_loss(residual, delta=experiment["huber_delta"])
            elif experiment["loss_name"] == "mse":
                losses = 0.5 * residual**2
            else:
                raise ValueError(f"Unsupported TD loss: {experiment['loss_name']}")
            return jnp.mean(losses), (predicted, residual)

        (loss, (predicted, residual)), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            carry["params"]
        )
        updates, opt_state = optimizer.update(gradients, carry["opt_state"], carry["params"])
        params = optax.apply_updates(carry["params"], updates)
        update_count = carry["updates"] + 1
        if experiment["target_update"] == "hard":
            target_params = jax.lax.cond(
                update_count % experiment["target_copy_updates"] == 0,
                lambda _: params,
                lambda _: carry["target_params"],
                operand=None,
            )
        elif experiment["target_update"] == "polyak":
            coefficient = experiment["target_polyak"]
            target_params = jax.tree.map(
                lambda target_leaf, online_leaf: (
                    (1.0 - coefficient) * target_leaf + coefficient * online_leaf
                ),
                carry["target_params"],
                params,
            )
        else:
            raise ValueError(f"Unsupported target update: {experiment['target_update']}")
        metrics = jnp.array(
            [
                loss,
                jnp.mean(jnp.abs(residual)),
                jnp.mean(predicted),
                jnp.min(predicted),
                jnp.max(predicted),
                jnp.mean(target),
                jnp.min(target),
                jnp.max(target),
                optax.tree.norm(gradients),
            ],
            dtype=jnp.float32,
        )
        return {
            "params": params,
            "target_params": target_params,
            "opt_state": opt_state,
            "key": next_key,
            "updates": update_count,
        }, metrics

    updated, history = jax.lax.scan(
        update_one,
        learner,
        None,
        length=experiment["updates_per_collection"],
    )
    means = jnp.mean(history, axis=0)
    return dict(updated), {
        "loss": means[0],
        "mean_absolute_td_error": means[1],
        "predicted_mean": means[2],
        "predicted_min": jnp.min(history[:, 3]),
        "predicted_max": jnp.max(history[:, 4]),
        "target_mean": means[5],
        "target_min": jnp.min(history[:, 6]),
        "target_max": jnp.max(history[:, 7]),
        "gradient_norm": means[8],
    }
