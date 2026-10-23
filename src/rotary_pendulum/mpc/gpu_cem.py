"""Compiled CEM over simulation, initial-plan and population axes on one GPU."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from rotary_pendulum.mpc.inner_model import evaluate, rollout


@partial(jax.jit, static_argnames=("horizon",))
def initial_plans(
    states: Array,
    networks: dict[str, Any],
    plant: dict[str, Any],
    horizon: int,
) -> Array:
    """Prepare the actor rollout and the same two constant-torque seeds for every solver."""

    def advance(state: Array, unused: None) -> tuple[Array, Array]:
        action = evaluate(jnp.zeros(1), state, networks, plant, "policy")
        return rollout(state, action[:, None], plant)[:, -1, -1], action

    _, actions = jax.lax.scan(advance, states, None, length=horizon)
    actions = actions.T
    return jnp.stack((actions, jnp.full_like(actions, 0.5), jnp.full_like(actions, -0.5)), axis=1)


@partial(jax.jit, static_argnames=("population", "elites", "generations"))
def sample_gpu(
    states: Array,
    seeds: Array,
    networks: dict[str, Any],
    plant: dict[str, Any],
    key: Array,
    population: int,
    elites: int,
    generations: int,
    initial_std: float,
) -> dict[str, Any]:
    """Keep sampling, feasible-only selection and refitting on device until the final audit.

    Save random keys, distribution histories and counts instead of transferring entire
    populations. Exact draws can be regenerated from the frozen sources and recorded key.
    """
    scored = evaluate(
        jnp.pad(seeds, ((0, 0), (0, 0), (0, 1))), states[:, None, :], networks, plant, "score"
    )
    best = jnp.where(scored["feasible"], scored["objective"], jnp.inf)
    carry = (
        seeds,
        jnp.full_like(seeds, initial_std),
        jnp.ones(seeds.shape[:2], bool),
        seeds,
        best,
        key,
    )

    def generation(state: tuple[Array, ...], unused: None) -> tuple[Any, Any]:
        means, spreads, active, incumbent, best, key = state
        key, draw = jax.random.split(key)
        samples = jnp.clip(
            means[..., None, :]
            + spreads[..., None, :]
            * jax.random.normal(
                draw, (*seeds.shape[:2], population, seeds.shape[-1]), dtype=seeds.dtype
            ),
            -1,
            1,
        )
        scored = evaluate(
            jnp.pad(samples, ((0, 0), (0, 0), (0, 0), (0, 1))),
            states[:, None, None, :],
            networks,
            plant,
            "score",
        )
        costs = jnp.where(scored["feasible"], scored["objective"], jnp.inf)
        elite_cost, indices = jax.lax.top_k(-costs, elites)
        selected = jnp.take_along_axis(samples, indices[..., None], axis=-2)
        improved = active & (-elite_cost[..., 0] < best)
        incumbent = jnp.where(improved[..., None], selected[..., 0, :], incumbent)
        best = jnp.where(improved, -elite_cost[..., 0], best)
        count = scored["feasible"].sum(axis=-1)
        next_spreads = selected.std(axis=-2)
        following = active & (count >= elites) & jnp.all(next_spreads > 0, axis=-1)
        history = dict(
            means=means,
            spreads=spreads,
            key=draw,
            active=active,
            feasible_count=count,
            objective=best,
        )
        means = jnp.where(following[..., None], selected.mean(axis=-2), means)
        spreads = jnp.where(following[..., None], next_spreads, spreads)
        return (means, spreads, following, incumbent, best, key), history

    final, history = jax.lax.scan(generation, carry, None, length=generations)
    scored = evaluate(
        jnp.pad(final[3], ((0, 0), (0, 0), (0, 1))), states[:, None, :], networks, plant, "score"
    )
    plans = jnp.concatenate((final[3], scored["costs"].max(axis=-1, keepdims=True)), axis=-1)
    audit = evaluate(plans, states[:, None, :], networks, plant, "audit")
    accepted = audit["physical_pass"] & jnp.isfinite(final[4])
    return dict(plans=plans, audit=audit, accepted=accepted, history=history)
