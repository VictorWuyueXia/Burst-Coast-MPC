"""Continuous-control validation traces, numerical records and trajectory plots."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICS_DT_S
from rotary_pendulum.environment.jax_environment import EnvState, reset
from rotary_pendulum.RL.jax_residual_control import (
    TARGET_ENERGY_J,
    residual_action,
)
from rotary_pendulum.RL.jax_residual_plotting import plot_trajectories
from rotary_pendulum.RL.jax_residual_task import observe, transition
from rotary_pendulum.RL.jax_td3 import Actor


def validation_resets(
    seed: int, count: int, deadlines_s: list[int]
) -> tuple[EnvState, Array, list[str]]:
    """Pair four reset strata and four deterministic probes across every deadline."""

    key, tight_key = jax.random.split(jax.random.PRNGKey(seed))
    strata = jnp.repeat(jnp.array([0, 1, 2, 2]), count)
    state = jax.vmap(reset)(
        jax.random.split(key, 4 * count + 4), jnp.concatenate((strata, jnp.zeros(4, jnp.int32)))
    )
    tight = jax.random.uniform(
        tight_key,
        (count, 4),
        minval=jnp.array([-0.08, -0.12, -0.15, -0.30]),
        maxval=jnp.array([0.08, 0.12, 0.15, 0.30]),
    )
    tight = tight.at[:, 1].add(jnp.pi)
    probes = jnp.array(
        [[0, 0, 0, 0], [0, jnp.pi, 0, 0], [1.45, 0.4, 2, 0], [-1.45, -0.4, -2, 0]],
        dtype=jnp.float32,
    )
    state = state._replace(x=state.x.at[3 * count : 4 * count].set(tight).at[-4:].set(probes))
    state = state._replace(x=state.x.at[:, 1].set((state.x[:, 1] + jnp.pi) % (2 * jnp.pi) - jnp.pi))
    lanes = 4 * count + 4
    repeated = jax.tree.map(
        lambda value: jnp.tile(value, (len(deadlines_s),) + (1,) * (value.ndim - 1)), state
    )
    deadline_steps = jnp.repeat(jnp.asarray(deadlines_s, jnp.int32) * 50, lanes)
    labels = (
        [label for label in ("downward", "moving", "near", "tight") for _ in range(count)]
        + ["probe"] * 4
    ) * len(deadlines_s)
    return repeated, deadline_steps, labels


def evaluate(
    initial: EnvState,
    deadline_steps: Array,
    settings: Mapping[str, Any],
    mode: str,
    actor: Any = None,
) -> dict[str, Array]:
    """Evaluate matched lanes through the longest deadline, freezing completed episodes."""

    if mode not in ("heuristic", "unfiltered", "zero", "policy", "linear"):
        raise ValueError(f"Unknown validation mode: {mode}")
    if mode == "policy" and actor is None:
        raise ValueError("Policy validation requires explicit actor parameters")

    def advance(state: EnvState, unused: None) -> tuple[EnvState, dict[str, Array]]:
        observation = observe(
            state, deadline_steps, round(settings["hold_duration_s"] / PHYSICS_DT_S)
        )
        residual = (
            cast(Array, Actor(tuple(settings["hidden_widths"])).apply(actor, observation))
            if mode == "policy"
            else jnp.zeros(state.x.shape[0])
        )
        applied, heuristic, proposed, intervention = residual_action(
            observation,
            residual,
            jnp.zeros(residual.shape),
            settings,
        )
        if mode == "unfiltered":
            applied, intervention = heuristic, jnp.zeros_like(intervention)
        elif mode == "zero":
            applied, intervention = jnp.zeros_like(applied), jnp.zeros_like(intervention)
        elif mode == "linear":
            error = state.x.at[:, 1].set(jnp.arctan2(-observation[:, 1], -observation[:, 2]))
            applied = settings["torque_limit_nm"] * jnp.clip(-error @ actor, -1.0, 1.0)
            intervention = jnp.zeros_like(intervention)
        active = ~(state.success | state.timeout)
        applied = jnp.where(active, applied, 0.0)
        following, reward, components, physics_x, _ = transition(
            state, applied, deadline_steps, settings
        )
        return following, {
            "x": state.x,
            "physics_x": jnp.moveaxis(physics_x, 0, 1),
            "active": active,
            "time_s": state.physics_steps * PHYSICS_DT_S,
            "heuristic_nm": jnp.where(active, heuristic, 0.0),
            "proposed_nm": jnp.where(active, proposed, 0.0),
            "applied_nm": applied,
            "filter_mode": intervention,
            "reward": reward,
            "components": components,
            "hold_count": following.goal_count,
            "elapsed_s": (following.physics_steps - state.physics_steps) * PHYSICS_DT_S,
        }

    final, traces = jax.lax.scan(
        advance, initial, None, length=round(settings["evaluation_max_s"] / 0.1)
    )
    return {
        **traces,
        "success": final.success,
        "arm_violation": final.arm_violation,
        "timeout": final.timeout,
        "final_x": final.x,
        "duration_s": final.physics_steps * PHYSICS_DT_S,
        "deadline_s": deadline_steps * PHYSICS_DT_S,
    }


def write_validation(
    directory: Path,
    traces: Mapping[str, Any],
    labels: list[str],
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Save full physics traces, episode summaries and reproducible representative figures."""

    human, machine = directory / "human-readables", directory / "machine-scannables"
    human.mkdir(parents=True, exist_ok=True)
    machine.mkdir(parents=True, exist_ok=True)
    data = {key: np.asarray(value) for key, value in traces.items()}
    np.savez_compressed(
        machine / "trajectories.npz", allow_pickle=False, **data, labels=np.asarray(labels)
    )
    (machine / "settings.json").write_text(
        json.dumps(
            {
                **settings,
                "energy_metric_definition": "hinge-relative swing energy, not full body energy",
            },
            indent=2,
        )
        + "\n"
    )
    x = data["physics_x"]
    energy = (
        0.5 * MODEL.pendulum_inertia_kg_m2 * x[..., 3] ** 2
        + MODEL.gravity_torque_nm * (1.0 - np.cos(x[..., 1]))
    ) / TARGET_ENERGY_J
    beta = np.arctan2(np.sin(x[..., 1] - np.pi), np.cos(x[..., 1] - np.pi))
    valid = data["active"][..., None]
    totals = data["components"].sum(axis=0)
    rows: list[dict[str, Any]] = []
    for lane, label in enumerate(labels):
        active = data["active"][:, lane]
        rows.append(
            {
                "lane": lane,
                "stratum": label,
                "deadline_s": float(data["deadline_s"][lane]),
                "success": bool(data["success"][lane]),
                "timeout": bool(data["timeout"][lane]),
                "arm_violation": bool(data["arm_violation"][lane]),
                "duration_s": float(data["duration_s"][lane]),
                "return": float(data["reward"][:, lane].sum()),
                "energy_error_mean": float(np.abs(energy[active, lane] - 1.0).mean()),
                "target_energy_reached": bool(np.any((energy[:, lane] >= 0.9) & valid[:, lane])),
                "upright_visited": bool(np.any((np.abs(beta[:, lane]) < 0.16) & valid[:, lane])),
                "peak_arm_rad": float(np.abs(x[:, lane, :, 0]).max()),
                "peak_pendulum_speed_rad_s": float(np.abs(x[:, lane, :, 3]).max()),
                "filter_fraction": float(np.mean(data["filter_mode"][active, lane] != 0)),
                "best_effort_fraction": float(np.mean(data["filter_mode"][active, lane] == 3)),
                "on_time_s": float(
                    np.sum(
                        data["elapsed_s"][:, lane] * (np.abs(data["applied_nm"][:, lane]) > 1e-8)
                    )
                ),
                **{
                    name: float(totals[lane, index])
                    for index, name in enumerate(
                        ("energy_cost", "torque_cost", "time_cost", "arm_cost", "upright_credit")
                    )
                },
            }
        )
    handle = (machine / "episodes.csv").open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    handle.close()

    plot_trajectories(directory, data, rows, settings)
    return rows
