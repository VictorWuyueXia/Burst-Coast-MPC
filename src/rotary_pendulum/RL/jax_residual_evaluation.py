"""Continuous-control validation traces, numerical records and trajectory plots."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax import Array

from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICS_DT_S
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, GOAL, EnvState, reset
from rotary_pendulum.RL.jax_residual_control import (
    POLICY_TORQUE_NM,
    TARGET_ENERGY_J,
    residual_action,
)
from rotary_pendulum.RL.jax_residual_task import observe, transition
from rotary_pendulum.RL.jax_td3 import Actor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

    if mode not in ("heuristic", "unfiltered", "zero", "policy"):
        raise ValueError(f"Unknown validation mode: {mode}")
    if mode == "policy" and actor is None:
        raise ValueError("Policy validation requires explicit actor parameters")

    def advance(state: EnvState, unused: None) -> tuple[EnvState, dict[str, Array]]:
        observation = observe(state, deadline_steps)
        residual = (
            cast(Array, Actor().apply(actor, observation))
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

    # Pick a median-return and best-return case at 60 s from each random stratum;
    # the chosen lane IDs are saved so that plots are never anonymous anecdotes.
    selected: dict[str, list[int]] = {}
    for label in ("downward", "moving", "near", "tight", "probe"):
        lanes = [
            i
            for i, row in enumerate(rows)
            if row["stratum"] == label and row["deadline_s"] == max(data["deadline_s"])
        ]
        ranked = sorted(lanes, key=lambda i: rows[i]["return"])
        chosen = ranked if label == "probe" else [ranked[len(ranked) // 2], ranked[-1]]
        selected[label] = chosen
        figure, axes = plt.subplots(6, len(chosen), figsize=(6 * len(chosen), 14), squeeze=False)
        for column, lane in enumerate(chosen):
            active = data["active"][:, lane]
            time = data["time_s"][active, lane]
            # Include all 20 ms physics samples for state traces; torque stays at 100 ms.
            fine_time = (time[:, None] + 0.02 * np.arange(1, 6)).reshape(-1)
            state = x[active, lane].reshape(-1, 4)
            angle = np.rad2deg((state[:, 1] + np.pi) % (2 * np.pi) - np.pi)
            angle[np.abs(np.diff(angle, prepend=angle[0])) > 180] = np.nan
            axes[0, column].plot(fine_time, state[:, 0], label="arm θ")
            for bound in (-float(ARM_LIMIT_RAD), float(ARM_LIMIT_RAD)):
                axes[0, column].axhline(bound, color="red", ls=":")
            axes[0, column].set_ylabel("Arm position (rad)")
            axes[1, column].plot(fine_time, angle, label="Pendulum position")
            for edge in (-180, 180):
                axes[1, column].axhspan(
                    max(-180, edge - np.rad2deg(GOAL.beta_tolerance_rad)),
                    min(180, edge + np.rad2deg(GOAL.beta_tolerance_rad)),
                    color="green",
                    alpha=0.15,
                )
            axes[1, column].set_ylim(-180, 180)
            axes[1, column].set_ylabel("Pendulum [deg; 0 down, ±180 up]")
            axes[2, column].plot(fine_time, state[:, 2], label="arm ω")
            axes[2, column].plot(fine_time, state[:, 3], label="pendulum ν")
            axes[2, column].set_ylabel("Velocity (rad/s)")
            axes[3, column].plot(
                fine_time,
                energy[active, lane].reshape(-1),
                label="Relative hinge-swing energy / target",
            )
            axes[3, column].axhline(1, color="black", ls=":")
            axes[3, column].set_ylabel("Relative swing energy / target")
            axes[4, column].step(
                time,
                1000 * data["heuristic_nm"][active, lane],
                where="post",
                alpha=0.5,
                label="heuristic",
            )
            axes[4, column].step(
                time, 1000 * data["applied_nm"][active, lane], where="post", label="applied"
            )
            axes[4, column].set_ylim(-1100 * POLICY_TORQUE_NM, 1100 * POLICY_TORQUE_NM)
            axes[4, column].set_ylabel("Torque (mN m)")
            axes[5, column].step(
                time, data["filter_mode"][active, lane], where="post", label="filter mode"
            )
            axes[5, column].plot(time, data["hold_count"][active, lane] / 5, label="hold fraction")
            axes[5, column].set_ylabel("Mode / hold fraction")
            axes[5, column].set_xlabel("Time (s)")
            axes[0, column].set_title(f"{label}, lane {lane}: success={rows[lane]['success']}")
            for axis in axes[:, column]:
                axis.grid(alpha=0.25)
                axis.legend(fontsize=8, loc="upper right")
        figure.tight_layout()
        figure.savefig(human / f"{label}_trajectories.png", dpi=130)
        for axis in axes.flat:
            axis.set_xlim(0, 5)
        figure.savefig(human / f"{label}_first5s.png", dpi=130)
        plt.close(figure)
    (machine / "plot_selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    (human / "interpretation_summary.md").write_text(
        "# Validation plots\n\n"
        "Angles: θ is the unwrapped arm position; "
        "α is pendulum position in −180–180 degrees, measured from downward; "
        "β is wrapped angle from upright. Velocities ω and ν belong to arm and pendulum. "
        "E/E* uses hinge-relative swing energy, excluding arm-carried kinetic energy; "
        "it is a control quantity, not the pendulum body's full physical energy.\n\n"
        "Filter modes: 0 accept, 1 coast, 2 brake, 3 least predicted excursion when no "
        "candidate stays inside the arm bounds. Red lines show the configured arm bounds. "
        "Hold fraction 1 means five consecutive 20 ms goal samples and early success. "
        "The goal checks only pendulum angle within [-180, -165] or [165, 180) degrees.\n\n"
        "Each random stratum shows median-return and best-return longest-deadline cases. "
        "Probe plots show all deterministic probes. Complete traces, selections and "
        "episode outcomes are in ../machine-scannables/. Campaign interpretation is "
        "written separately after visual inspection.\n"
    )
    return rows
