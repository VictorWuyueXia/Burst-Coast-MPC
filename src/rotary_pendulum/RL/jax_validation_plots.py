"""State, dwell, and direct reward diagnostics for fixed validation trajectories."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, GOAL
from rotary_pendulum.RL.jax_task import ACTION_TORQUES_NM


def plot_validation(human_dir: Path, trajectories: Mapping[str, NDArray[Any]]) -> None:
    """Plot retained successes and timeouts with exact sampled elapsed times."""

    state_figure, state_axes = plt.subplots(2, 7, figsize=(23, 7), sharex=True)
    reward_figure, reward_axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True)
    for row, stratum in enumerate(("tight", "near")):
        states = trajectories[f"{stratum}_state"]
        initial = trajectories[f"{stratum}_initial_state"]
        valid = trajectories[f"{stratum}_valid"]
        times = trajectories[f"{stratum}_time_s"]
        actions = trajectories[f"{stratum}_action"]
        success = trajectories[f"{stratum}_success"]
        components = trajectories[f"{stratum}_reward_components"]
        counts = trajectories[f"{stratum}_goal_count"]
        outcome_seen: set[str] = set()
        for episode in range(states.shape[1]):
            mask = valid[:, episode]
            x = np.concatenate((initial[episode][None], states[mask, episode]), axis=0)
            end_times = times[mask, episode]
            elapsed = np.concatenate(([0.0], end_times))
            beta = np.arctan2(np.sin(x[:, 1] - np.pi), np.cos(x[:, 1] - np.pi))
            outcome = "success" if success[episode] else "timeout"
            color = "#2A9D46" if success[episode] else "#777777"
            label = outcome if outcome not in outcome_seen else None
            outcome_seen.add(outcome)
            for column, values in enumerate((x[:, 0], beta, x[:, 1], x[:, 2], x[:, 3])):
                state_axes[row, column].plot(elapsed, values, color=color, alpha=0.8, label=label)
            torque = np.asarray(ACTION_TORQUES_NM)[actions[mask, episode]]
            if torque.size:
                state_axes[row, 5].stairs(torque, elapsed, color=color, alpha=0.8)
            state_axes[row, 6].plot(
                end_times,
                counts[mask, episode] / float(GOAL.hold_steps),
                color=color,
                alpha=0.8,
                marker="o" if success[episode] else None,
                markersize=3,
                zorder=3 if success[episode] else 2,
            )
            for column in range(5):
                reward_axes[row, column].plot(
                    end_times,
                    np.cumsum(components[mask, episode, column]),
                    color=color,
                    alpha=0.8,
                    label=label,
                    marker="o" if success[episode] else None,
                    markersize=3,
                    zorder=3 if success[episode] else 2,
                )
        for column, limit in (
            (0, GOAL.theta_tolerance_rad),
            (1, GOAL.beta_tolerance_rad),
            (3, GOAL.omega_tolerance_rad_s),
            (4, GOAL.nu_tolerance_rad_s),
        ):
            state_axes[row, column].axhspan(-limit, limit, color="#2A9D46", alpha=0.08)
        for sign in (-1, 1):
            state_axes[row, 0].axhline(
                sign * float(ARM_LIMIT_RAD), color="black", linestyle="--", linewidth=0.8
            )
        state_axes[row, 0].set_ylabel(stratum)
        reward_axes[row, 0].set_ylabel(f"{stratum}\ncumulative reward")
        state_axes[row, 0].legend(frameon=False, fontsize=8)
        reward_axes[row, 0].legend(frameon=False, fontsize=8)
    for axes, titles in (
        (
            state_axes,
            (
                "arm θ [rad]",
                "upright error β [rad]",
                "unwrapped α [rad]",
                "arm ω [rad/s]",
                "pendulum ν [rad/s]",
                "torque [N m]",
                "100 ms hold fraction",
            ),
        ),
        (
            reward_axes,
            (
                "energy error cost",
                "torque magnitude cost",
                "elapsed time cost",
                "arm position cost",
                "smooth upright credit",
            ),
        ),
    ):
        for axis, title in zip(axes[0], titles, strict=True):
            axis.set_title(title)
        for axis in axes[1]:
            axis.set_xlabel("Elapsed time [s]")
    state_figure.suptitle("Validation trajectories; dashed arm thresholds are nonterminal")
    reward_figure.suptitle("Five direct reward components, accumulated over active physics samples")
    for figure, filename in (
        (state_figure, "validation_trajectories.png"),
        (reward_figure, "validation_rewards.png"),
    ):
        figure.tight_layout()
        figure.savefig(human_dir / filename, dpi=160)
        plt.close(figure)
