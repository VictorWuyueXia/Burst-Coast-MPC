"""Device-resident batched SQP; this is not a port of Kraft/SciPy SLSQP."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_solve

from rotary_pendulum.mpc.inner_model import evaluate


def quadratic_step(
    hessian: Array,
    gradient: Array,
    matrix: Array,
    residual: Array,
    rho: float,
    iterations: int,
) -> tuple[Array, Array, Array]:
    """ADMM for a positive-definite QP with matrix @ step + residual >= 0.

    Row scaling and the fixed penalty belong to the numerical QP, not the MPC cost.
    A fixed budget returns residuals, never a false convergence certificate.
    """
    scale = jnp.maximum(jnp.linalg.norm(matrix, axis=-1), 1.0)
    matrix, residual = matrix / scale[:, None], residual / scale
    factor = jnp.linalg.cholesky(hessian + rho * matrix.T @ matrix)
    initial = (jnp.zeros_like(gradient), jnp.maximum(residual, 0), jnp.zeros_like(residual))

    def update(unused: int, state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        _, slack, dual = state
        step = cho_solve((factor, True), -gradient + rho * matrix.T @ (slack - residual - dual))
        mapped = matrix @ step + residual
        slack = jnp.maximum(mapped + dual, 0)
        return step, slack, dual + mapped - slack

    step, slack, dual = jax.lax.fori_loop(0, iterations, update, initial)
    error = jnp.maximum(
        jnp.max(jnp.abs(matrix @ step + residual - slack)),
        jnp.max(jnp.abs(hessian @ step + gradient + rho * matrix.T @ dual)),
    )
    return step, -rho * dual / scale, error


@partial(jax.jit, static_argnames=("iterations", "qp_iterations"))
def solve_gpu(
    states: Array,
    seeds: Array,
    networks: dict[str, Any],
    plant: dict[str, Any],
    iterations: int,
    qp_iterations: int,
    step_limit: float,
    rho: float,
) -> dict[str, Any]:
    """Optimize all states and initial plans together, retaining feasible incumbents.

    Damped BFGS, linearized hard constraints, ADMM QPs and eight trial lengths
    execute inside one compiled scan. Budget-limited feasible plans are usable;
    stationarity and QP residuals are separate from physical acceptance.
    """
    shape, n = seeds.shape[:2], seeds.shape[-1] + 1
    x = jnp.broadcast_to(states[:, None, :], (*shape, 4)).reshape((-1, 4))
    z = jnp.pad(seeds.reshape((-1, n - 1)), ((0, 0), (0, 1)))
    initial = evaluate(z, x, networks, plant, "score")
    z = z.at[:, -1].set(initial["costs"].max(axis=-1))
    best = jnp.where(initial["feasible"], initial["objective"], jnp.inf)
    identity = jnp.eye(n, dtype=z.dtype)
    matrices = jnp.broadcast_to(identity, (len(z), n, n))
    values = evaluate(z, x, networks, plant, "sqp")
    jacobian = jax.vmap(jax.jacfwd(evaluate), in_axes=(0, 0, None, None, None))
    jac = jacobian(z, x, networks, plant, "sqp")
    dual = jnp.zeros_like(values[:, 1:])
    carry = (z, matrices, z, jac, dual, z, best)

    def iteration(state: tuple[Array, ...], unused: None) -> tuple[Any, Any]:
        z, hessian, previous, old_jac, multiplier, incumbent, best = state
        values = evaluate(z, x, networks, plant, "sqp")
        jac = jacobian(z, x, networks, plant, "sqp")
        gradient, constraint_jac = jac[:, 0], jac[:, 1:]
        change = z - previous
        delta = (
            jac[:, 0]
            - old_jac[:, 0]
            - jnp.einsum("bmi,bm->bi", constraint_jac - old_jac[:, 1:], multiplier)
        )
        product = jnp.einsum("bij,bj->bi", hessian, change)
        curvature = jnp.sum(change * product, axis=-1)
        observed = jnp.sum(change * delta, axis=-1)
        fraction = jnp.where(
            observed >= 0.2 * curvature,
            1.0,
            0.8 * curvature / jnp.maximum(curvature - observed, 1e-20),
        )
        delta = fraction[:, None] * delta + (1 - fraction[:, None]) * product
        updated = (
            hessian
            - product[:, :, None]
            * product[:, None, :]
            / jnp.maximum(curvature[:, None, None], 1e-20)
            + delta[:, :, None]
            * delta[:, None, :]
            / jnp.maximum(jnp.sum(change * delta, axis=-1)[:, None, None], 1e-20)
        )
        hessian = jnp.where((curvature > 1e-14)[:, None, None], updated, hessian)
        # Explicit numerical trust box: normalized torque step <= step_limit, epigraph <= 10.
        bound = jnp.full((n,), step_limit).at[-1].set(10.0)
        matrix = jnp.concatenate(
            (
                constraint_jac,
                jnp.broadcast_to(identity, matrices.shape),
                -jnp.broadcast_to(identity, matrices.shape),
            ),
            axis=1,
        )
        residual = jnp.concatenate(
            (values[:, 1:], jnp.broadcast_to(bound, z.shape), jnp.broadcast_to(bound, z.shape)),
            axis=-1,
        )
        step, multipliers, qp_error = jax.vmap(quadratic_step, in_axes=(0, 0, 0, 0, None, None))(
            hessian, gradient, matrix, residual, rho, qp_iterations
        )
        multiplier = multipliers[:, : values.shape[-1] - 1]
        penalty = jnp.maximum(10.0, 2 * jnp.max(jnp.abs(multiplier), axis=-1))
        lengths = jnp.concatenate((0.5 ** jnp.arange(8), jnp.zeros(1)))
        trials = z[:, None, :] + lengths[None, :, None] * step[:, None, :]
        trials = trials.at[..., :-1].set(jnp.clip(trials[..., :-1], -1, 1))
        trial_values = evaluate(trials, x[:, None, :], networks, plant, "sqp")
        merit = trial_values[..., 0] + penalty[:, None] * jnp.sum(
            jnp.maximum(-trial_values[..., 1:], 0), axis=-1
        )
        choice = jnp.argmin(jnp.where(jnp.isfinite(merit), merit, jnp.inf), axis=-1)
        following = jnp.take_along_axis(trials, choice[:, None, None], axis=1)[:, 0]
        scored = evaluate(trials, x[:, None, :], networks, plant, "score")
        costs = jnp.where(scored["feasible"], scored["objective"], jnp.inf)
        selected = jnp.argmin(costs, axis=-1)
        candidate = jnp.take_along_axis(trials, selected[:, None, None], axis=1)[:, 0]
        candidate_cost = jnp.min(costs, axis=-1)
        terminal = jnp.take_along_axis(scored["costs"].max(axis=-1), selected[:, None], axis=1)[
            :, 0
        ]
        candidate = candidate.at[:, -1].set(terminal)
        improved = candidate_cost < best
        incumbent = jnp.where(improved[:, None], candidate, incumbent)
        best = jnp.minimum(best, candidate_cost)
        stationarity = jnp.max(
            jnp.abs(gradient - jnp.einsum("bmi,bm->bi", constraint_jac, multiplier)), axis=-1
        )
        history = dict(
            objective=best,
            qp_error=qp_error,
            stationarity=stationarity,
            step_length=lengths[choice],
            violation=jnp.maximum(-values[:, 1:].min(-1), 0),
        )
        return (following, hessian, z, jac, multiplier, incumbent, best), history

    final, history = jax.lax.scan(iteration, carry, None, length=iterations)
    plans = final[-2].reshape((*shape, n))
    audit = evaluate(plans, states[:, None, :], networks, plant, "audit")
    accepted = audit["physical_pass"] & jnp.isfinite(final[-1].reshape(shape))
    return dict(plans=plans, audit=audit, accepted=accepted, history=history)
