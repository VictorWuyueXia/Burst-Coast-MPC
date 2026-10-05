"""Separated machine and human artifacts for rotary PPO training."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rotary_pendulum.utils.config_schema import EpisodeConfig, PPOConfig

UPDATE_HEADERS = (
    "update",
    "decisions",
    "stage",
    "learning_rate",
    "policy_loss",
    "value_loss",
    "entropy",
    "approximate_kl",
    "clip_fraction",
    "explained_variance",
    "mean_reward",
    "progress_reward",
    "phase_reward",
    "torque_slew_reward",
    "torque_effort_reward",
    "arm_boundary_reward",
    "terminal_reward",
    "successes",
    "arm_failures",
    "timeouts",
)
EVALUATION_HEADERS = (
    "update",
    "exact_success",
    "success_rate",
    "arm_violation_rate",
    "mean_return",
    "median_success_time_s",
    "mean_normalized_effort",
    "critic_return_mse",
)


class PPOArtifactWriter:
    """Persist reproducible numeric training data and shallow interpretive outputs."""

    def __init__(self, episode: EpisodeConfig, ppo: PPOConfig) -> None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
        self.run_dir = Path(ppo.artifact_root) / f"ppo_{timestamp}"
        self.machine_dir = self.run_dir / "machine"
        self.human_dir = self.run_dir / "human"
        self.machine_dir.mkdir(parents=True, exist_ok=False)
        self.human_dir.mkdir()
        self.update_rows: list[dict[str, float | int]] = []
        self.evaluation_rows: list[dict[str, float | int]] = []
        self.best_score = (-1.0, -1.0, -float("inf"), -float("inf"))

        config_file = (self.machine_dir / "resolved_config.json").open("w", encoding="utf-8")
        json.dump(
            {
                "episode": episode.model_dump(mode="json", by_alias=True),
                "ppo": ppo.model_dump(mode="json", by_alias=True),
            },
            config_file,
            indent=2,
            sort_keys=True,
        )
        config_file.write("\n")
        config_file.close()
        artifact_tables = (
            ("updates.csv", UPDATE_HEADERS),
            ("evaluations.csv", EVALUATION_HEADERS),
        )
        for name, headers in artifact_tables:
            file = (self.machine_dir / name).open("w", newline="", encoding="utf-8")
            csv.DictWriter(file, fieldnames=headers).writeheader()
            file.close()

    def record_update(self, metrics: dict[str, float | int]) -> None:
        """Append one complete PPO-update diagnostic row."""

        self.update_rows.append(metrics)
        file = (self.machine_dir / "updates.csv").open("a", newline="", encoding="utf-8")
        csv.DictWriter(file, fieldnames=UPDATE_HEADERS).writerow(metrics)
        file.close()

    def record_evaluation(
        self,
        metrics: dict[str, float | int],
        model: torch.nn.Module,
    ) -> None:
        """Append deterministic evaluation metrics and retain the best checkpoint."""

        self.evaluation_rows.append(metrics)
        file = (self.machine_dir / "evaluations.csv").open("a", newline="", encoding="utf-8")
        csv.DictWriter(file, fieldnames=EVALUATION_HEADERS).writerow(metrics)
        file.close()
        score = (
            float(metrics["exact_success"]),
            float(metrics["success_rate"]),
            -float(metrics["median_success_time_s"]),
            -float(metrics["mean_normalized_effort"]),
        )
        if score > self.best_score:
            self.best_score = score
            torch.save(model.state_dict(), self.machine_dir / "best_actor_critic.pt")

    def finalize(
        self,
        model: torch.nn.Module,
        summary: dict[str, Any],
        trajectory: list[dict[str, float | int]],
    ) -> None:
        """Write the final checkpoint, evaluation trajectory, plots, and interpretation."""

        if not self.update_rows or not self.evaluation_rows or not trajectory:
            raise ValueError(
                "Completed PPO artifacts require updates, evaluations, and a trajectory"
            )
        torch.save(model.state_dict(), self.machine_dir / "final_actor_critic.pt")
        summary_file = (self.machine_dir / "summary.json").open("w", encoding="utf-8")
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
        summary_file.close()
        trajectory_headers = tuple(trajectory[0])
        trajectory_file = (self.machine_dir / "evaluation_trajectory.csv").open(
            "w", newline="", encoding="utf-8"
        )
        trajectory_writer = csv.DictWriter(trajectory_file, fieldnames=trajectory_headers)
        trajectory_writer.writeheader()
        trajectory_writer.writerows(trajectory)
        trajectory_file.close()

        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        updates = np.asarray([row["update"] for row in self.update_rows], dtype=np.float64)
        figure, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
        axes[0].plot(updates, [row["mean_reward"] for row in self.update_rows], label="reward")
        axes[0].plot(updates, [row["value_loss"] for row in self.update_rows], label="value loss")
        axes[0].set(xlabel="PPO update", ylabel="Batch metric", title="Training progress")
        axes[0].legend()
        eval_updates = [row["update"] for row in self.evaluation_rows]
        axes[1].plot(
            eval_updates,
            [row["success_rate"] for row in self.evaluation_rows],
            label="success rate",
        )
        axes[1].plot(
            eval_updates,
            [row["arm_violation_rate"] for row in self.evaluation_rows],
            label="arm violation rate",
        )
        axes[1].set(xlabel="PPO update", ylabel="Episode fraction", ylim=(-0.02, 1.02))
        axes[1].legend()
        figure.savefig(self.human_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(figure)

        time_s = np.asarray([row["time_s"] for row in trajectory], dtype=np.float64)
        trace, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
        axes[0].plot(time_s, [row["theta_rad"] for row in trajectory], label="arm theta")
        axes[0].plot(time_s, [row["beta_rad"] for row in trajectory], label="upright error beta")
        axes[0].set(ylabel="Angle [rad]", title="Deterministic exact-downward evaluation")
        axes[0].legend()
        axes[1].plot(time_s, [row["omega_rad_s"] for row in trajectory], label="arm omega")
        axes[1].plot(time_s, [row["nu_rad_s"] for row in trajectory], label="pendulum nu")
        axes[1].set(ylabel="Angular velocity [rad/s]")
        axes[1].legend()
        axes[2].plot(time_s, [row["normalized_action"] for row in trajectory], label="action")
        axes[2].plot(time_s, [row["value"] for row in trajectory], label="critic value")
        axes[2].set(xlabel="Simulated time [s]", ylabel="Normalized signal")
        axes[2].legend()
        trace.savefig(self.human_dir / "evaluation_trace.png", dpi=150, bbox_inches="tight")
        plt.close(trace)

        final = self.evaluation_rows[-1]
        report = (
            "# Rotary-Pendulum PPO Interpretation\n\n"
            f"Training completed after {summary['updates']} PPO updates and "
            f"{summary['decisions']} collected decisions.\n\n"
            f"- Strict-dwell success rate: {float(final['success_rate']):.3f}\n"
            f"- Exact-downward success: {bool(final['exact_success'])}\n"
            f"- Arm-limit violation rate: {float(final['arm_violation_rate']):.3f}\n"
            f"- Median successful time: {float(final['median_success_time_s']):.3f} s\n"
            f"- Mean normalized effort: {float(final['mean_normalized_effort']):.3f}\n"
            f"- Critic return MSE: {float(final['critic_return_mse']):.6f}\n\n"
            "Use `training_curves.png` to judge optimization stability and `evaluation_trace.png` "
            "to distinguish energy acquisition, capture, cable-limit failure, and critic bias. "
            "Machine CSV files preserve the exact values needed to reproduce both figures.\n"
        )
        report_file = (self.human_dir / "interpretation_summary.md").open("w", encoding="utf-8")
        report_file.write(report)
        report_file.close()
