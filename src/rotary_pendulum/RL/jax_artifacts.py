"""Separated machine records, checkpoints, and readable figures for Q training."""

from __future__ import annotations

import csv
import importlib.metadata
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import matplotlib.pyplot as plt
import numpy as np

from rotary_pendulum.RL.jax_task import ACTION_TORQUES_NM

plt.switch_backend("Agg")


def write_artifacts(
    run_dir: Path,
    metrics: Mapping[str, Any],
    trajectories: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> None:
    """Materialize one checkpoint and its machine/human evaluation evidence."""

    human_dir = run_dir / "human-readables"
    machine_dir = run_dir / "machine-scannables"
    checkpoint_dir = machine_dir / "checkpoints"
    human_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_rows = list(metrics["history"])
    serializable_metrics = {
        key: (
            np.asarray(value).item() if np.asarray(value).ndim == 0 else np.asarray(value).tolist()
        )
        for key, value in metrics.items()
        if key != "history"
    }
    action_order = [
        "off",
        "negative_pump",
        "positive_pump",
        "negative_fine",
        "positive_fine",
    ]
    metadata = {
        "contract_id": "rotary-q-prior-v2",
        "network_id": experiment["network_id"],
        "reward_revision": experiment["reward_revision"],
        "experiment_id": experiment["experiment_id"],
        "seed": experiment["seed"],
        "checkpoint_eligible": metrics.get("checkpoint_eligible", True),
        "evaluation_split": metrics.get("evaluation_split", "provided"),
        "feature_order": (
            "theta_over_limit sin_alpha cos_alpha omega_scaled nu_scaled "
            "remaining_fraction goal_hold_fraction"
        ).split(),
        "action_order": action_order,
        "action_torques_nm": np.asarray(ACTION_TORQUES_NM).tolist(),
        "action_units": "N m; 45% pump and 2% fine magnitudes from physical configuration",
        "hidden_widths": list(experiment["hidden_widths"]),
        "activation_name": experiment["activation_name"],
        "loss_name": experiment["loss_name"],
        "output_meaning": "five shaped finite-deadline action returns in reward units",
        "potential_formula": "Phi=-(e_E^2/(1+e_E^2)+w_c*q_c/(1+q_c)); zero at terminal",
        "base_reward": {
            key: experiment[key]
            for key in (
                "success_reward",
                "arm_failure_cost",
                "timeout_cost",
                "on_cost_per_s",
                "time_cost_per_s",
            )
        },
        "terminal_mask": "success, arm violation, and timeout all have zero continuation",
        "gamma": 1.0,
        "deadline_s": 20.0,
        "jax_version": importlib.metadata.version("jax"),
        "flax_version": importlib.metadata.version("flax"),
        "optax_version": importlib.metadata.version("optax"),
        "prng_key_encoding": "uint32-key-data",
        "resolved_experiment": dict(experiment),
    }
    (machine_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n"
    )
    (machine_dir / "metrics.json").write_text(
        json.dumps(serializable_metrics, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    history_buffer = io.StringIO()
    history_writer = csv.DictWriter(history_buffer, fieldnames=list(history_rows[0]))
    history_writer.writeheader()
    history_writer.writerows(history_rows)
    (machine_dir / "history.csv").write_text(history_buffer.getvalue())

    evaluation_rows = []
    controllers = ("greedy", "lookahead_zero", "lookahead_potential", "lookahead_q")
    for controller in controllers:
        for stratum in ("overall", "downward", "moving", "near", "tight"):
            prefix = f"{controller}_{stratum}_"
            row = {
                key.removeprefix(prefix): value
                for key, value in serializable_metrics.items()
                if key.startswith(prefix)
            }
            if row:
                evaluation_rows.append({"controller": controller, "stratum": stratum, **row})
    evaluation_buffer = io.StringIO()
    evaluation_writer = csv.DictWriter(
        evaluation_buffer, fieldnames=list(evaluation_rows[0]), extrasaction="ignore"
    )
    evaluation_writer.writeheader()
    evaluation_writer.writerows(evaluation_rows)
    (machine_dir / "evaluation.csv").write_text(evaluation_buffer.getvalue())

    all_arrays = {key: np.asarray(jax.device_get(value)) for key, value in trajectories.items()}
    for stratum in ("tight", "near"):
        success = all_arrays[f"{stratum}_success"]
        selected = np.concatenate(
            (np.flatnonzero(success)[:5], np.flatnonzero(~success)[:5])
        )
        episode_count = success.size
        for key, value in list(all_arrays.items()):
            if not key.startswith(f"{stratum}_"):
                continue
            if value.shape[0] == episode_count:
                all_arrays[key] = value[selected]
            elif value.ndim > 1 and value.shape[1] == episode_count:
                all_arrays[key] = value[:, selected]
        all_arrays[f"{stratum}_sample_index"] = selected
    initial_arrays = {
        key.removeprefix("initial_"): value
        for key, value in all_arrays.items()
        if key.startswith("initial_")
    }
    audit_arrays = {
        key.removeprefix("audit_"): value
        for key, value in all_arrays.items()
        if key.startswith("audit_")
    }
    trajectory_arrays = {
        key: value
        for key, value in all_arrays.items()
        if not key.startswith(("initial_", "audit_"))
    }
    np.savez_compressed(machine_dir / "initial_states.npz", **initial_arrays)  # type: ignore[arg-type]
    np.savez_compressed(machine_dir / "trajectories.npz", **trajectory_arrays)  # type: ignore[arg-type]
    if audit_arrays:
        np.savez_compressed(machine_dir / "value_audit.npz", **audit_arrays)  # type: ignore[arg-type]
    for checkpoint_name, learner in checkpoint.items():
        checkpoint_payload = dict(learner)
        checkpoint_payload["key"] = jax.random.key_data(checkpoint_payload["key"])
        (checkpoint_dir / f"{checkpoint_name}.msgpack").write_bytes(
            flax.serialization.to_bytes(checkpoint_payload)
        )

    if history_rows:
        transitions = [row["transitions"] for row in history_rows]
        figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        axes[0].plot(transitions, [row["tight_success"] for row in history_rows])
        axes[0].plot(transitions, [row["near_success"] for row in history_rows])
        axes[0].set(ylabel="Success rate", title="Held-out validation during training")
        axes[0].legend(("tight", "near"))
        axes[1].plot(transitions, [row["tight_arm_violation"] for row in history_rows])
        axes[1].plot(transitions, [row["near_arm_violation"] for row in history_rows])
        axes[1].set(ylabel="Arm-violation rate")
        axes[1].legend(("tight", "near"))
        axes[2].plot(transitions, [row["loss"] for row in history_rows])
        axes[2].set(xlabel="Collected transitions", ylabel="Huber loss")
        figure.tight_layout()
        figure.savefig(human_dir / "learning.png", dpi=160)
        plt.close(figure)

    greedy_rows = [
        row
        for row in evaluation_rows
        if row["controller"] == "greedy" and row["stratum"] != "overall"
    ]
    if greedy_rows and "success_rate" in greedy_rows[0]:
        labels = [str(row["stratum"]) for row in greedy_rows]
        positions = np.arange(len(labels))
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(positions - 0.2, [row["success_rate"] for row in greedy_rows], 0.4)
        axis.bar(positions + 0.2, [row["arm_violation_rate"] for row in greedy_rows], 0.4)
        axis.set(xticks=positions, xticklabels=labels, ylabel="Episode fraction")
        axis.legend(("Success", "Arm violation"))
        figure.tight_layout()
        figure.savefig(human_dir / "outcomes.png", dpi=160)
        plt.close(figure)

    deployment_rows = [row for row in evaluation_rows if row["stratum"] == "overall"]
    if len(deployment_rows) > 1 and "mean_powered_s" in deployment_rows[0]:
        labels = [str(row["controller"]).replace("lookahead_", "") for row in deployment_rows]
        figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        axes[0].bar(labels, [row["success_rate"] for row in deployment_rows])
        axes[0].set_ylabel("Success rate")
        axes[1].bar(labels, [row["mean_powered_s"] for row in deployment_rows])
        axes[1].set(ylabel="Mean powered time [s]", xlabel="Controller")
        figure.tight_layout()
        figure.savefig(human_dir / "deployment.png", dpi=160)
        plt.close(figure)

    if "tight_state" in trajectory_arrays:
        figure, axes = plt.subplots(2, 5, figsize=(17, 7), sharex=True)
        for row, stratum in enumerate(("tight", "near")):
            states = trajectory_arrays[f"{stratum}_state"]
            initial = trajectory_arrays[f"{stratum}_initial_state"]
            valid = trajectory_arrays[f"{stratum}_valid"]
            actions = trajectory_arrays[f"{stratum}_action"]
            successes = trajectory_arrays[f"{stratum}_success"]
            violations = trajectory_arrays[f"{stratum}_arm_violation"]
            states = np.concatenate((initial[None], states), axis=0)
            states = np.where(
                np.concatenate((np.ones_like(valid[:1]), valid), axis=0)[..., None],
                states,
                np.nan,
            )
            time_s = 0.1 * np.arange(states.shape[0])
            upright_error = np.arctan2(
                np.sin(states[..., 1] - np.pi), np.cos(states[..., 1] - np.pi)
            )
            outcome_seen: set[str] = set()
            for episode in range(states.shape[1]):
                if successes[episode]:
                    outcome = "success"
                elif violations[episode]:
                    outcome = "violation"
                else:
                    outcome = "timeout"
                color = {"success": "#2A9D46", "violation": "#D95319", "timeout": "#777777"}[
                    outcome
                ]
                label = outcome if outcome not in outcome_seen else None
                outcome_seen.add(outcome)
                axes[row, 0].plot(
                    time_s,
                    states[:, episode, 0],
                    color=color,
                    alpha=0.8,
                    label=label,
                )
                axes[row, 1].plot(time_s, upright_error[:, episode], color=color, alpha=0.8)
                axes[row, 2].plot(time_s, states[:, episode, 2], color=color, alpha=0.8)
                axes[row, 3].plot(time_s, states[:, episode, 3], color=color, alpha=0.8)
                torque = np.where(
                    valid[:, episode],
                    np.asarray(ACTION_TORQUES_NM)[actions[:, episode]],
                    np.nan,
                )
                axes[row, 4].step(time_s[:-1], torque, where="post", color=color, alpha=0.8)
            axes[row, 0].axhspan(-0.08, 0.08, color="#2A9D46", alpha=0.08)
            axes[row, 0].axhline(np.pi / 2.0, color="black", linestyle="--", linewidth=0.8)
            axes[row, 0].axhline(-np.pi / 2.0, color="black", linestyle="--", linewidth=0.8)
            for column, limit in ((1, 0.08), (2, 0.15), (3, 0.20)):
                axes[row, column].axhspan(-limit, limit, color="#2A9D46", alpha=0.08)
            axes[row, 0].set_ylabel(f"{stratum}\nstate value")
            axes[row, 0].legend(frameon=False, fontsize=8)
        for axis, title in zip(
            axes[0],
            (
                "arm angle θ [rad]",
                "upright error β [rad]",
                "arm speed ω [rad/s]",
                "pendulum speed ν [rad/s]",
                "torque [N m]",
            ),
            strict=True,
        ):
            axis.set_title(title)
        for axis in axes[1]:
            axis.set_xlabel("Time [s]")
        figure.suptitle("Held-out greedy validation: first five successes and nonsuccesses")
        figure.tight_layout()
        figure.savefig(human_dir / "validation_trajectories.png", dpi=160)
        plt.close(figure)

    if "predicted_base" in audit_arrays:
        predicted = audit_arrays["predicted_base"].reshape(-1)
        realized = audit_arrays["realized_base"].reshape(-1)
        lower = float(min(predicted.min(), realized.min()))
        upper = float(max(predicted.max(), realized.max()))
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.scatter(realized, predicted, s=8, alpha=0.5)
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        axis.set(
            xlabel="Realized base return",
            ylabel="Predicted base return",
            title="Q-value calibration",
        )
        figure.tight_layout()
        figure.savefig(human_dir / "q_calibration.png", dpi=160)
        plt.close(figure)
