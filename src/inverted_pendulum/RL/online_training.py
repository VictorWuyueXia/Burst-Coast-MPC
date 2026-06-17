"""Online replay-buffer fitted-Q training for the structured critic policy."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from inverted_pendulum.RL.policy import REPO_ROOT, StructuredCriticPolicy
from inverted_pendulum.utils.config_schema import RLConfig
from inverted_pendulum.utils.messages import RLStepRecord

REPLAY_CAPACITY = 4096
TARGET_UPDATE_INTERVAL = 32


class ReplayBuffer:
    """Fixed-capacity transition store for off-policy fitted-Q updates."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.records: list[RLStepRecord] = []
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def extend(self, records: list[RLStepRecord]) -> None:
        """Append new transitions and keep the most recent fixed-capacity window."""

        self.records.extend(records)
        if len(self.records) > self.capacity:
            self.records = self.records[-self.capacity:]

    def sample(self, batch_size: int) -> list[RLStepRecord]:
        """Sample replay rows with replacement for a fixed-size critic update."""

        indices = self.rng.integers(0, len(self.records), size=batch_size)
        return [self.records[int(index)] for index in indices]


class OnlinePolicyTrainer:
    """Fine-tune the loaded critic with replayed fitted-Q targets."""

    def __init__(self, policy: StructuredCriticPolicy, config: RLConfig) -> None:
        self.policy = policy
        self.config = config

    def fit(self, records: list[RLStepRecord]) -> Path:
        """Fit the policy critic from online transitions and save a snapshot."""

        if not records:
            msg = "Online training requires at least one RL transition record"
            raise ValueError(msg)

        import torch

        torch.manual_seed(self.config.training_seed)
        replay = ReplayBuffer(REPLAY_CAPACITY, self.config.training_seed)
        replay.extend(records)
        target_model = copy.deepcopy(self.policy.model)
        target_model.eval()
        optimizer = torch.optim.Adam(
            self.policy.model.parameters(),
            lr=self.config.training_learning_rate,
        )
        total_updates = len(records) * self.config.training_updates_per_transition
        last_loss = torch.tensor(0.0, dtype=torch.float32)

        # Regress the online critic toward target-network Bellman values from replay.
        self.policy.model.train()
        for update_index in range(total_updates):
            batch = replay.sample(self.config.training_batch_size)
            target_values = self.policy.fitted_q_targets(batch, target_model)
            features, physical, target = self.policy.build_transition_tensors(
                batch, target_values
            )
            prediction = self.policy.model(features, physical)
            last_loss = torch.nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad()
            last_loss.backward()
            optimizer.step()
            if (update_index + 1) % TARGET_UPDATE_INTERVAL == 0:
                target_model.load_state_dict(self.policy.model.state_dict())

        target_model.load_state_dict(self.policy.model.state_dict())
        self.policy.model.eval()
        output_root = Path(self.config.training_output_root)
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        snapshot_dir = output_root / f"online-critic_{timestamp}"
        output_root.mkdir(parents=True, exist_ok=True)
        snapshot_dir.mkdir(exist_ok=False)

        target_values = self.policy.fitted_q_targets(records, target_model)
        features, physical, target = self.policy.build_transition_tensors(records, target_values)
        torch.set_grad_enabled(False)
        prediction = self.policy.model(features, physical).detach().cpu()
        torch.set_grad_enabled(True)
        metrics = {
            "rows": len(records),
            "replay_rows": len(replay),
            "gradient_updates": total_updates,
            "last_train_mse": float(last_loss.detach().cpu()),
            "bellman_mse": float(torch.mean((prediction - target.cpu()) ** 2)),
            **self.policy.model.lambda_values(),
        }

        # Save the fine-tuned state and enough metadata to reload the snapshot.
        torch.save(self.policy.model.state_dict(), snapshot_dir / "critic_state_dict.pt")
        config_file = (snapshot_dir / "config.json").open("w", encoding="utf-8")
        json.dump(
            {
                "snapshot-schema-version": 2,
                "action-grid-count": self.policy.action_grid_count,
                "learning-rate": self.config.training_learning_rate,
                "replay-capacity": REPLAY_CAPACITY,
                "target-update-interval": TARGET_UPDATE_INTERVAL,
                "source-critic-artifact-dir": str(self.policy.critic_artifact_dir),
                "source-time-model-artifact-dir": str(self.policy.time_model_artifact_dir),
                "rl-config": self.config.model_dump(mode="json", by_alias=True),
            },
            config_file,
            indent=2,
            sort_keys=True,
        )
        config_file.write("\n")
        config_file.close()
        lambda_file = (snapshot_dir / "lambda.json").open("w", encoding="utf-8")
        json.dump(self.policy.model.lambda_values(), lambda_file, indent=2, sort_keys=True)
        lambda_file.write("\n")
        lambda_file.close()
        metrics_file = (snapshot_dir / "metrics.json").open("w", encoding="utf-8")
        json.dump(metrics, metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")
        metrics_file.close()
        normalization_file = (snapshot_dir / "normalization.json").open("w", encoding="utf-8")
        json.dump(
            {
                "feature-order": [
                    "sin-theta",
                    "cos-theta",
                    "omega/ref",
                    "energy-error",
                    "bbar",
                    "hbar",
                ],
                "feature-mean": self.policy.feature_mean.tolist(),
                "feature-std": self.policy.feature_std.tolist(),
            },
            normalization_file,
            indent=2,
            sort_keys=True,
        )
        normalization_file.write("\n")
        normalization_file.close()
        return snapshot_dir
