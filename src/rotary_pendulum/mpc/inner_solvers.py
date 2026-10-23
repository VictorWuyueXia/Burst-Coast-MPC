"""SLSQP and batched cross-entropy search with identical physical acceptance."""

import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import minimize  # type: ignore[import-untyped]

from rotary_pendulum.mpc.inner_model import CONSTRAINT_JAC, OBJECTIVE_GRAD, evaluate


def solve_inner(
    x0: Array, seeds: Array, networks: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    """Solve every declared seed; accept only converged, independently replayed plans."""

    started = time.perf_counter()
    plant, tolerance = settings["plant"], settings["tolerance"]
    if not jax.config.x64_enabled:
        raise ValueError("Inner optimization requires JAX_ENABLE_X64=true")
    if settings["max_iterations"] < 1 or tolerance <= 0:
        raise ValueError("SLSQP iteration budget and tolerance must be positive")
    if seeds.ndim != 2 or seeds.shape[0] < 1 or seeds.shape[1] < 1:
        raise ValueError("Supply at least one nonempty initial action sequence")
    if not np.isfinite(seeds).all() or np.any(np.abs(seeds) > 1):
        raise ValueError("Initial actions must be finite and within the actuator bounds")
    objective_output = "rl_objective" if plant["rl_cost"] else "objective"
    objective = partial(evaluate, x0=x0, networks=networks, settings=plant, output=objective_output)
    gradient = partial(
        OBJECTIVE_GRAD, x0=x0, networks=networks, settings=plant, output=objective_output
    )
    constraints = partial(evaluate, x0=x0, networks=networks, settings=plant, output="constraints")
    jacobian = partial(
        CONSTRAINT_JAC, x0=x0, networks=networks, settings=plant, output="constraints"
    )
    scored = evaluate(jnp.pad(seeds, ((0, 0), (0, 1))), x0, networks, plant, "score")
    guesses = np.column_stack((seeds, np.asarray(scored["costs"]).max(axis=-1)))
    records: list[dict[str, Any]] = []
    plans = []
    for index, guess in enumerate(guesses):
        print(f"SLSQP seed {index + 1}/{len(guesses)}: starting", flush=True)
        raw = minimize(
            objective,
            guess,
            method="SLSQP",
            jac=gradient,
            bounds=[(-1.0, 1.0)] * seeds.shape[1] + [(None, None)],
            constraints=[dict(type="ineq", fun=constraints, jac=jacobian)],
            options=dict(maxiter=settings["max_iterations"], ftol=tolerance, disp=True, iprint=2),
        )
        audit = jax.device_get(evaluate(jnp.asarray(raw.x), x0, networks, plant, "audit"))
        accepted = bool(
            raw.success
            and np.isfinite(raw.x).all()
            and np.isfinite(raw.fun)
            and audit["physical_pass"]
            and audit["min_constraint"] >= -tolerance
        )
        records.append(
            dict(
                seed=index,
                accepted=accepted,
                status=str(raw.message),
                iterations=int(raw.nit),
                objective=float(audit["objective"]),
                min_constraint=float(audit["min_constraint"]),
                physical_pass=bool(audit["physical_pass"]),
            )
        )
        plans.append((np.asarray(raw.x), audit))
        print(
            f"SLSQP seed {index + 1}: accepted={accepted}, cost={audit['objective']:.6g}",
            flush=True,
        )
    accepted_indices = [index for index, row in enumerate(records) if row["accepted"]]
    selected = (
        min(accepted_indices, key=lambda i: records[i]["objective"]) if accepted_indices else -1
    )
    return dict(
        accepted=selected >= 0,
        selected=selected,
        records=records,
        plans=plans,
        elapsed_s=time.perf_counter() - started,
        solver="slsqp",
    )


def sample_inner(
    x0: Array, seeds: Array, networks: dict[str, Any], settings: dict[str, Any], key: Array
) -> dict[str, Any]:
    """Fit one bounded CEM distribution per seed, retaining evaluated feasible samples.

    All populations and network evaluations are batched on the current JAX device.
    Insufficient feasible elites and collapsed variance stop that distribution;
    neither introduces a penalty, variance floor or substitute controller.
    """

    started = time.perf_counter()
    plant = settings["plant"]
    population, elites = settings["population"], settings["elite_count"]
    if not jax.config.x64_enabled:
        raise ValueError("Inner optimization requires JAX_ENABLE_X64=true")
    if not 2 <= elites <= population or settings["generations"] < 1 or settings["initial_std"] <= 0:
        raise ValueError("Require population >= elites >= 2, positive generations and spread")
    if seeds.ndim != 2 or min(seeds.shape) < 1:
        raise ValueError("Supply a matrix of nonempty initial action sequences")
    if not np.isfinite(seeds).all() or np.any(np.abs(seeds) > 1):
        raise ValueError("Initial actions must be finite and within the actuator bounds")
    means = jnp.asarray(seeds)
    spreads = jnp.full_like(means, settings["initial_std"])
    active = np.ones(len(seeds), dtype=bool)
    records: list[dict[str, Any]] = []
    populations, plans = [], []
    best_cost = np.full(len(seeds), np.inf)
    best_actions = jnp.asarray(seeds)
    initial = evaluate(jnp.pad(means, ((0, 0), (0, 1))), x0, networks, plant, "score")
    best_cost = np.where(
        np.asarray(initial["feasible"]), np.asarray(initial["objective"]), best_cost
    )
    for generation in range(settings["generations"]):
        key, draw = jax.random.split(key)
        samples = jnp.clip(
            means[:, None, :]
            + spreads[:, None, :]
            * jax.random.normal(draw, (len(seeds), population, seeds.shape[1]), dtype=means.dtype),
            -1,
            1,
        )
        scored = evaluate(jnp.pad(samples, ((0, 0), (0, 0), (0, 1))), x0, networks, plant, "score")
        scores = jnp.where(scored["feasible"], scored["objective"], jnp.inf)
        order = jnp.argsort(scores, axis=-1)
        elite_samples = jnp.take_along_axis(samples, order[..., None], axis=1)
        costs = np.asarray(jnp.take_along_axis(scores, order[:, :1], axis=-1)[:, 0])
        improved = active & (costs < best_cost)
        best_actions = jnp.where(jnp.asarray(improved)[:, None], elite_samples[:, 0], best_actions)
        best_cost = np.where(improved, costs, best_cost)
        feasible_count = np.asarray(scored["feasible"].sum(axis=-1))
        next_spreads = elite_samples[:, :elites].std(axis=1)
        collapsed = np.asarray(jnp.any(next_spreads == 0, axis=-1))
        next_active = active & (feasible_count >= elites) & ~collapsed
        records.extend(
            dict(
                generation=generation,
                seed=index,
                active=bool(active[index]),
                feasible_count=int(feasible_count[index]),
                best_objective=float(best_cost[index]),
                status=(
                    "inactive"
                    if not active[index]
                    else "insufficient_elites"
                    if feasible_count[index] < elites
                    else "collapsed"
                    if collapsed[index]
                    else "updated"
                ),
            )
            for index in range(len(seeds))
        )
        populations.append(
            dict(
                samples=np.asarray(samples),
                objective=np.asarray(scored["objective"]),
                feasible=np.asarray(scored["feasible"]),
                key=np.asarray(draw),
            )
        )
        means = jnp.where(
            jnp.asarray(next_active)[:, None], elite_samples[:, :elites].mean(axis=1), means
        )
        spreads = jnp.where(jnp.asarray(next_active)[:, None], next_spreads, spreads)
        active = next_active
        print(
            f"CEM generation {generation + 1}: feasible={feasible_count.tolist()}, "
            f"best={best_cost.tolist()}",
            flush=True,
        )
        if not active.any():
            break
    scored = evaluate(jnp.pad(best_actions, ((0, 0), (0, 1))), x0, networks, plant, "score")
    decisions = jnp.concatenate(
        (best_actions, scored["costs"].max(axis=-1, keepdims=True)), axis=-1
    )
    audits = jax.device_get(evaluate(decisions, x0, networks, plant, "audit"))
    accepted = np.asarray(audits["physical_pass"]) & np.isfinite(best_cost)
    accepted_indices = np.flatnonzero(accepted).tolist()
    for index in range(len(seeds)):
        plans.append(
            (np.asarray(decisions[index]), {name: value[index] for name, value in audits.items()})
        )
    selected = (
        min(accepted_indices, key=lambda i: float(audits["objective"][i]))
        if accepted_indices
        else -1
    )
    return dict(
        accepted=selected >= 0,
        selected=selected,
        records=records,
        plans=plans,
        populations=populations,
        seed_accepted=accepted.tolist(),
        elapsed_s=time.perf_counter() - started,
        solver="cem",
    )
