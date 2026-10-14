"""Continued validation of physical-state actors, with full-duration constraint/effort scores."""

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, EnvState
from rotary_pendulum.RL.jax_residual_task import observe
from rotary_pendulum.RL.jax_stationary_task import transition
from rotary_pendulum.RL.jax_td3 import Actor


def evaluate(initial: EnvState, actor: Any, settings: Mapping[str, Any]) -> dict[str, jax.Array]:
    """Run actual feedback until sampling horizon, never freezing after a successful hold."""

    def advance(state: EnvState, unused: None) -> tuple[EnvState, dict[str, jax.Array]]:
        observation = observe(state, jnp.zeros_like(state.physics_steps), stationary=True)
        action = cast(jax.Array, Actor(tuple(settings["hidden_widths"])).apply(actor, observation))
        following, reward, physics, _ = transition(state, action, settings)
        return following, dict(
            start_x=state.x,
            physics_x=physics,
            reward=reward,
            torque_nm=settings["torque_limit_nm"] * action,
            goal_count=following.goal_count,
            arm_violation=following.arm_violation,
        )

    _, traces = jax.lax.scan(advance, initial, None, length=settings["deadline_s"] * 10)
    return traces


def write_evaluation(
    directory: Path, traces: Mapping[str, Any], settings: Mapping[str, Any]
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Count uninterrupted holds independently and rank full-duration compliant settling."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    machine, human = directory / "machine-scannables", directory / "human-readables"
    machine.mkdir(parents=True, exist_ok=False)
    human.mkdir(parents=True, exist_ok=False)
    records = {key: np.asarray(value) for key, value in traces.items()}
    if not all(np.isfinite(value).all() for value in records.values()):
        raise FloatingPointError("Nonfinite stationary evaluation")
    count = records["start_x"].shape[1]
    labels = np.asarray(["downward", "capture", "braking", "recovery"])[np.arange(count) % 4]
    np.savez_compressed(machine / "trajectories.npz", allow_pickle=False, **records, labels=labels)
    (machine / "settings.json").write_text(json.dumps(dict(settings), indent=2) + "\n")
    states = records["physics_x"].transpose(0, 2, 1, 3).reshape(-1, count, 4)
    beta = np.arctan2(-np.sin(states[..., 1]), -np.cos(states[..., 1]))
    inside = abs(beta) < np.deg2rad(5)
    hold, longest = np.zeros(count, int), np.zeros(count, int)
    for sample in inside:
        hold = np.where(sample, hold + 1, 0)
        longest = np.maximum(longest, hold)
    crossed = np.asarray((abs(states[..., 0]) >= float(ARM_LIMIT_RAD)).any(axis=0))
    torque = records["torque_nm"]
    final_rms = np.sqrt(np.mean(torque[-100:] ** 2, axis=0))
    settled = np.asarray(hold >= 500)
    rows = []
    for lane in range(count):
        rows.append(
            dict(
                episode=lane,
                stratum=labels[lane],
                settled=bool(settled[lane]),
                arm_violation=bool(crossed[lane]),
                bounded_settled=bool(settled[lane] and not crossed[lane]),
                longest_hold_s=float(longest[lane] * 0.02),
                final_hold_s=float(hold[lane] * 0.02),
                final10_upright_fraction=float(inside[-500:, lane].mean()),
                peak_arm_deg=float(np.rad2deg(abs(states[:, lane, 0]).max())),
                torque_rms_mnm=float(np.sqrt(np.mean(torque[:, lane] ** 2)) * 1000),
                final10_torque_rms_mnm=float(final_rms[lane] * 1000),
                final10_mean_abs_torque_mnm=float(abs(torque[-100:, lane]).mean() * 1000),
                absolute_motor_work_j=float(
                    (abs(torque[:, lane, None] * records["physics_x"][:, lane, :, 2]) * 0.02).sum()
                ),
                episode_return=float(records["reward"][:, lane].sum()),
            )
        )
    handle = (machine / "episodes.csv").open("w")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    handle.close()
    summary = {
        group: {
            key: float(
                np.mean([row[key] for row in rows if group == "all" or row["stratum"] == group])
            )
            for key in rows[0]
            if key not in ("episode", "stratum")
        }
        for group in ["all", "downward", "capture", "braking", "recovery"]
    }
    (machine / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    score = (float(np.mean(settled & ~crossed)), float(settled.mean()), -float(final_rms.mean()))
    figure, axes = plt.subplots(3, 4, figsize=(15, 8), layout="constrained")
    for column, group in enumerate(["downward", "capture", "braking", "recovery"]):
        selected = np.flatnonzero(labels == group)
        lane = selected[np.argsort(longest[selected])[len(selected) // 2]]
        time = np.arange(1, len(states) + 1) * 0.02
        error = np.rad2deg(beta[:, lane]).copy()
        error[abs(np.diff(error, prepend=error[0])) > 180] = np.nan
        axes[0, column].plot(time, error, lw=0.7)
        axes[0, column].axhspan(-5, 5, color="green", alpha=0.2)
        axes[0, column].set(
            title=f"{group}, episode {lane}", ylabel="Upright error (deg)", ylim=(-15, 15)
        )
        axes[1, column].plot(time, np.rad2deg(states[:, lane, 0]), lw=0.7)
        for bound in [-135, 135]:
            axes[1, column].axhline(bound, color="red", ls="--")
        axes[1, column].set_ylabel("Arm angle (deg)")
        axes[2, column].step(
            np.arange(len(torque)) * 0.1, torque[:, lane] * 1000, where="post", lw=0.7
        )
        axes[2, column].set(xlabel="Time (s)", ylabel="Torque (mN m)")
        for axis in axes[:, column]:
            axis.grid(alpha=0.2)
    figure.suptitle("Stationary actor: real feedback throughout the full rollout")
    figure.savefig(human / "episodes.png", dpi=120)
    plt.close(figure)
    return score, summary
