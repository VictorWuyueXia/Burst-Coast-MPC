"""Online Lightning fine-tuning handle for the structured critic policy."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from inverted_pendulum.RL.policy import REPO_ROOT, StructuredCriticPolicy
from inverted_pendulum.utils.config_schema import RLConfig
from inverted_pendulum.utils.messages import RLStepRecord


class OnlinePolicyTrainer:
    """Fine-tune the loaded critic from one online simulated episode."""

    def __init__(self, policy: StructuredCriticPolicy, config: RLConfig) -> None:
        self.policy = policy
        self.config = config

    def fit(self, records: list[RLStepRecord]) -> Path:
        """Fit the policy critic on return-labeled online transitions and save a snapshot."""

        if not records:
            msg = "Online training requires at least one RL transition record"
            raise ValueError(msg)

        # Import the training stack only when train mode actually runs.
        import pytorch_lightning as pl
        import torch
        from pytorch_lightning import Trainer
        from torch.utils.data import DataLoader, TensorDataset

        pl.seed_everything(self.config.training_seed, workers=True, verbose=False)
        features, physical, target = self.policy.build_training_tensors(records)
        dataset = TensorDataset(features, physical, target)
        loader = DataLoader(
            dataset,
            batch_size=self.config.training_batch_size,
            shuffle=True,
            num_workers=0,
        )
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            precision="32-true",
            max_epochs=self.config.training_max_epochs,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
        )

        # Fit in place so the policy can continue from the updated critic weights.
        trainer.fit(self.policy.model, loader, loader)
        self.policy.model.eval()
        output_root = Path(self.config.training_output_root)
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        snapshot_dir = output_root / f"online-critic_{timestamp}"
        output_root.mkdir(parents=True, exist_ok=True)
        snapshot_dir.mkdir(exist_ok=False)

        torch.set_grad_enabled(False)
        prediction = self.policy.model(features, physical).detach().cpu()
        torch.set_grad_enabled(True)
        metrics = {
            "rows": len(records),
            "train_mse": float(torch.mean((prediction - target.cpu()) ** 2)),
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
