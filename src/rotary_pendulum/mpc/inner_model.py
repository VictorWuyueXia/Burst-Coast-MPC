"""Shared differentiable prediction, frozen terminal value and physical plan audit."""

from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array

from rotary_pendulum.environment.jax_dynamics import MODEL, rk4_step
from rotary_pendulum.RL.jax_td3 import Actor, Critic


@partial(jax.jit, static_argnames=("substeps",))
def rollout(x0: Array, actions: Array, settings: dict[str, Any], substeps: int = 10) -> Array:
    """Return (..., decisions, 5 * substeps, 4) states without clipping or episode latches."""

    actions = jnp.asarray(actions)
    state = jnp.broadcast_to(jnp.asarray(x0), (*actions.shape[:-1], 4))
    torques = jnp.repeat(jnp.moveaxis(actions, -1, 0), 5, axis=0) * settings["torque_limit_nm"]

    def advance(current: Array, torque: Array) -> tuple[Array, Array]:
        history = rk4_step(current, torque, record_substeps=True, substeps=substeps)
        return history[-1], history

    _, history = jax.lax.scan(advance, state, torques)
    history = jnp.moveaxis(history, (0, 1), (-3, -2))
    return history.reshape((*actions.shape, 5 * substeps, 4))


@partial(jax.jit, static_argnames=("output",))
def evaluate(
    z: Array, x0: Array, networks: dict[str, Any], settings: dict[str, Any], output: str
) -> Any:
    """One physical/value contract for optimization, diagnostics and finer-grid acceptance.

    `z` always contains normalized controls followed by a free epigraph scalar.
    Policy/value inference uses x0 directly and ignores z. Constraint residuals
    follow SciPy's nonnegative convention. Frozen weights retain input gradients.
    """

    if output not in (
        "policy",
        "value",
        "objective",
        "rl_objective",
        "constraints",
        "score",
        "audit",
        "sqp",
    ):
        raise ValueError(f"Unknown inner-MPC evaluation output: {output}")
    actions = z[..., :-1]
    powers = settings["discount"] ** jnp.arange(actions.shape[-1])
    effort = 0.1 * settings["torque_cost_per_s"] * jnp.sum(actions**2 * powers, axis=-1)
    trajectory: Any = None
    state = x0
    if output not in ("policy", "value"):
        substeps = 20 if "lqr_state_weights" in settings else 10
        trajectory = rollout(x0, actions, settings, substeps=substeps)
        state = trajectory[..., -1, -1, :]
        upright_tracking = jnp.zeros_like(effort)
        if "lqr_state_weights" in settings:
            # Discrete LQR cost uses x_0,...,x_{N-1}, not quadrature or x_1,...,x_N.
            start = jnp.broadcast_to(x0, (*actions.shape[:-1], 4))
            nodes = jnp.concatenate((start[..., None, :], trajectory[..., :-1, -1, :]), axis=-2)
            error = nodes.at[..., 1].set(
                jnp.arctan2(-jnp.sin(nodes[..., 1]), -jnp.cos(nodes[..., 1]))
            )
            components = jnp.sum(error**2 * jnp.asarray(settings["lqr_state_weights"]), axis=-2)
            centering, upright_tracking, speed_arm, speed_pendulum = jnp.moveaxis(components, -1, 0)
            effort = jnp.sum(actions**2, axis=-1)  # R = 1 on normalized torque, per decision.
        else:
            centering = (
                0.002
                * settings["centering_cost_per_s"]
                * jnp.sum(
                    (trajectory[..., 0] / settings["observation_scales"][0]) ** 2
                    * powers[..., None],
                    axis=(-2, -1),
                )
            )
            # The angle only gates the speed cost; there is no added angle-error cost.
            proximity = jnp.exp(
                -(1 + jnp.cos(trajectory[..., 1])) / settings["speed_width_rad"] ** 2
            )
            speed_components = (
                0.001
                * settings["speed_cost_per_s"]
                * jnp.sum(
                    proximity[..., None]
                    * (
                        1
                        - jnp.exp(
                            -0.5
                            * (
                                trajectory[..., 2:]
                                / jnp.asarray(settings["upright_widths"])[jnp.array([2, 1])]
                            )
                            ** 2
                        )
                    )
                    * powers[..., None, None],
                    axis=(-3, -2),
                )
            )
            speed_arm, speed_pendulum = speed_components[..., 0], speed_components[..., 1]
        primary = effort + centering + upright_tracking + speed_arm + speed_pendulum
        if output == "objective":
            return primary + settings["discount"] ** actions.shape[-1] * z[..., -1]
    theta, alpha, omega, nu = jnp.moveaxis(state, -1, 0)
    observation = jnp.stack(
        (theta, jnp.sin(alpha), jnp.cos(alpha), omega, nu), axis=-1
    ) / jnp.asarray(settings["observation_scales"])
    widths = tuple(
        networks["actor"]["params"][f"Dense_{i}"]["bias"].shape[0]
        for i in range(len(networks["actor"]["params"]) - 1)
    )
    policy = Actor(widths).apply(networks["actor"], observation)
    if output == "policy":
        return policy
    costs = -jnp.stack(
        [
            cast(Array, Critic(widths).apply(params, observation, policy))
            for params in networks["critics"]
        ],
        axis=-1,
    )
    if output == "value":
        return dict(policy=policy, costs=costs, observation=observation)

    arm_limit = settings["observation_scales"][0]
    arm = trajectory[..., 0].reshape((*actions.shape[:-1], -1)) / arm_limit
    constraints = jnp.concatenate((1 - arm, 1 + arm, z[..., -1:] - costs), axis=-1)
    if output == "constraints":
        return constraints
    if output == "sqp":
        # Exact grid-feasible set, with only the worst upper/lower node per interval.
        arm_nodes = arm.reshape((*actions.shape, 5 * substeps))
        compact = jnp.concatenate(
            (
                1 - arm_nodes.max(axis=-1),
                1 + arm_nodes.min(axis=-1),
                z[..., -1:] - costs,
                1 - actions,
                1 + actions,
            ),
            axis=-1,
        )
        value = primary + settings["discount"] ** actions.shape[-1] * z[..., -1]
        return jnp.concatenate((value[..., None], compact), axis=-1)
    observed = trajectory[..., substeps - 1 :: substeps, :]
    theta, alpha, omega, nu = jnp.moveaxis(observed, -1, 0)
    beta = jnp.arctan2(-jnp.sin(alpha), -jnp.cos(alpha))
    upright = jnp.exp(
        -0.5
        * jnp.sum(
            (jnp.stack((beta, nu, omega), axis=-1) / jnp.asarray(settings["upright_widths"])) ** 2,
            axis=-1,
        )
    )
    energy = 0.5 * MODEL.pendulum_inertia_kg_m2 * nu**2 + MODEL.gravity_torque_nm * (
        1 - jnp.cos(alpha)
    )
    arm_cost = (theta / arm_limit) ** 2 + (
        jnp.maximum(jnp.abs(theta) - 0.8 * arm_limit, 0) / (0.2 * arm_limit)
    ) ** 2
    rates = (
        settings["upright_reward_per_s"] * (1 - upright)
        + settings["energy_cost_per_s"] * jnp.abs(energy / (2 * MODEL.gravity_torque_nm) - 1)
        + settings["torque_cost_per_s"] * actions[..., None] ** 2
        + settings["arm_cost_per_s"] * arm_cost
    )
    stage_costs = 0.02 * rates.sum(axis=-1)
    running = jnp.where(settings["rl_cost"], jnp.sum(stage_costs * powers, axis=-1), primary)
    if output == "rl_objective":
        return running + settings["discount"] ** actions.shape[-1] * z[..., -1]
    objective = running + settings["discount"] ** actions.shape[-1] * costs.max(axis=-1)
    physical = (
        jnp.all(jnp.isfinite(trajectory), axis=(-3, -2, -1))
        & jnp.all(jnp.isfinite(costs), axis=-1)
        & jnp.all(jnp.isfinite(x0), axis=-1)
        & (jnp.abs(x0[..., 0]) < arm_limit)
        & jnp.all(jnp.abs(actions) <= 1, axis=-1)
        & jnp.all(jnp.abs(arm) < 1, axis=-1)
    )
    if output == "score":
        return dict(objective=objective, feasible=physical, costs=costs)

    refined = rollout(x0, actions, settings, substeps=2 * substeps)
    difference = refined[..., 1::2, :] - trajectory
    difference = difference.at[..., 1].set(
        jnp.arctan2(jnp.sin(difference[..., 1]), jnp.cos(difference[..., 1]))
    )
    errors = jnp.max(jnp.abs(difference), axis=(-3, -2))
    physical &= (
        jnp.all(jnp.isfinite(refined), axis=(-3, -2, -1))
        & jnp.all(jnp.abs(refined[..., 0]) < arm_limit, axis=(-2, -1))
        & jnp.all(errors <= jnp.array([1e-3, 1e-3, 1e-2, 1e-2]), axis=-1)
    )
    return dict(
        objective=objective,
        effort=effort,
        centering=centering,
        upright_tracking=upright_tracking,
        speed_pendulum=speed_pendulum,
        speed_arm=speed_arm,
        costs=costs,
        trajectory=trajectory,
        refined=refined,
        refinement_error=errors,
        physical_pass=physical,
        min_constraint=jnp.min(constraints, axis=-1),
        arm_margin_rad=arm_limit
        - jnp.maximum(jnp.abs(x0[..., 0]), jnp.max(jnp.abs(refined[..., 0]), axis=(-2, -1))),
        rl_stage_costs=stage_costs,
    )


# Reuse compiled transformations across all states, seeds and solver-budget settings.
OBJECTIVE_GRAD = jax.jit(jax.grad(evaluate), static_argnames=("output",))
CONSTRAINT_JAC = jax.jit(jax.jacfwd(evaluate), static_argnames=("output",))
