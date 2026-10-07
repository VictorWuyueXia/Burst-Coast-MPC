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
    metadata = {
        "contract_id": "rotary-q-prior-v1",
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
        "action_order": ["off", "negative_full", "positive_full"],
        "action_units": "N m; magnitude from the physical configuration",
        "hidden_widths": list(experiment["hidden_widths"]),
        "activation_name": experiment["activation_name"],
        "loss_name": experiment["loss_name"],
        "output_meaning": "three shaped finite-deadline action returns in reward units",
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
        figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(transitions, [row["minimum_success"] for row in history_rows])
        axes[0].plot(transitions, [row["overall_success"] for row in history_rows])
        axes[0].set(ylabel="Success rate", title="Validation learning progress")
        axes[0].legend(("Minimum stratum", "Stage mean"))
        axes[1].plot(transitions, [row["loss"] for row in history_rows])
        axes[1].set(xlabel="Collected transitions", ylabel="Huber loss")
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

    if "state" in trajectory_arrays and "action" in trajectory_arrays:
        states = trajectory_arrays["state"]
        actions = trajectory_arrays["action"]
        if states.ndim == 4:
            states = states[:, 0, 0]
            actions = actions[:, 0, 0]
        else:
            states = states[:, 0]
            actions = actions[:, 0]
        time_s = 0.1 * np.arange(states.shape[0])
        figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(time_s, states[:, 0], label="arm angle")
        axes[0].plot(time_s, states[:, 1], label="pendulum angle")
        axes[0].set_ylabel("Angle [rad]")
        axes[0].legend()
        axes[1].plot(time_s, states[:, 2], label="arm velocity")
        axes[1].plot(time_s, states[:, 3], label="pendulum velocity")
        axes[1].set_ylabel("Velocity [rad/s]")
        axes[1].legend()
        axes[2].step(time_s, actions, where="post")
        axes[2].set(xlabel="Time [s]", ylabel="Action index")
        figure.suptitle("Representative discrete-Q trajectory")
        figure.tight_layout()
        figure.savefig(human_dir / "trajectories.png", dpi=160)
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
