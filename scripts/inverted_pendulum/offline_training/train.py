"""Train the offline structured residual critic from Monte Carlo artifacts."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
SNAPSHOT_SCHEMA_VERSION = 1
TRAINER_NUM_WORKERS = 8

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from critic import StructuredResidualCritic
from dataset import (
    build_tensor_table,
    collect_run_dirs,
    load_artifact_rows,
    read_training_config,
    split_tensor_table,
)
from diagnostics import evaluate_snapshot


class EpochMSELogger(Callback):
    def __init__(self, snapshot_dir: Path) -> None:
        self.snapshot_dir = snapshot_dir
        self.train_error_sum = 0.0
        self.train_row_count = 0
        self.validation_error_sum = 0.0
        self.validation_row_count = 0

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        curve_file = (self.snapshot_dir / "training_curve.csv").open("w", encoding="utf-8")
        curve_file.write("epoch,train_mse,val_mse\n")
        curve_file.close()

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self.train_error_sum = 0.0
        self.train_row_count = 0

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        features, physical, target = batch
        torch.set_grad_enabled(False)
        prediction = pl_module(features, physical)
        torch.set_grad_enabled(True)
        self.train_error_sum += float(torch.sum((prediction - target) ** 2).detach().cpu())
        self.train_row_count += int(target.numel())

    def on_validation_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self.validation_error_sum = 0.0
        self.validation_row_count = 0

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return
        features, physical, target = batch
        torch.set_grad_enabled(False)
        prediction = pl_module(features, physical)
        torch.set_grad_enabled(True)
        self.validation_error_sum += float(torch.sum((prediction - target) ** 2).detach().cpu())
        self.validation_row_count += int(target.numel())

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if trainer.sanity_checking:
            return
        train_mse = self.train_error_sum / self.train_row_count
        val_mse = self.validation_error_sum / self.validation_row_count
        line = f"{trainer.current_epoch},{train_mse:.9f},{val_mse:.9f}\n"
        curve_file = (self.snapshot_dir / "training_curve.csv").open("a", encoding="utf-8")
        curve_file.write(line)
        curve_file.close()
        print(
            "offline-training epoch "
            f"epoch={trainer.current_epoch} train_mse={train_mse:.9f} val_mse={val_mse:.9f}"
        )


def create_snapshot_dir(config) -> Path:
    """Create one timestamped model snapshot directory."""

    # Keep all trained artifacts under the ignored model-snapshots root.
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    snapshot_dir = config.output_root / f"{config.run_name}_{timestamp}"
    config.output_root.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(exist_ok=False)
    (snapshot_dir / "checkpoints").mkdir(exist_ok=False)
    return snapshot_dir


def write_snapshot_metadata(snapshot_dir: Path, config, table, split, model) -> None:
    """Write static training metadata needed to reload the critic later."""

    # Save the exact training config and snapshot schema in a compact JSON record.
    config_file = (snapshot_dir / "config.json").open("w", encoding="utf-8")
    json.dump(
        {
            "snapshot-schema-version": SNAPSHOT_SCHEMA_VERSION,
            "data-roots": [str(path) for path in config.data_roots],
            "output-root": str(config.output_root),
            "run-name": config.run_name,
            "seed": config.seed,
            "train-fraction": config.train_fraction,
            "batch-size": config.batch_size,
            "max-epochs": config.max_epochs,
            "learning-rate": config.learning_rate,
            "action-grid-count": config.action_grid_count,
            "constants": table["constants"],
        },
        config_file,
        indent=2,
        sort_keys=True,
    )
    config_file.write("\n")
    config_file.close()

    # Store train-only normalization statistics as deployment-facing numeric lists.
    norm_file = (snapshot_dir / "normalization.json").open("w", encoding="utf-8")
    json.dump(
        {
            "feature-order": [
                "sin-theta", "cos-theta", "omega/ref", "energy-error", "bbar", "hbar",
            ],
            "feature-mean": split["feature_mean"].detach().cpu().tolist(),
            "feature-std": split["feature_std"].detach().cpu().tolist(),
        },
        norm_file,
        indent=2,
        sort_keys=True,
    )
    norm_file.write("\n")
    norm_file.close()

    # Preserve the model structure in plain text for quick architecture inspection.
    structure_file = (snapshot_dir / "model_structure.txt").open("w", encoding="utf-8")
    structure_file.write(str(model))
    structure_file.write("\n")
    structure_file.close()


def main() -> None:
    """Run one fixed-config offline training job on a single NVIDIA GPU."""

    # Load artifact rows and construct train/validation tensors before touching Lightning.
    config = read_training_config(CONFIG_PATH)
    pl.seed_everything(config.seed, workers=True, verbose=False)
    run_dirs = collect_run_dirs(config)
    rows, constants = load_artifact_rows(run_dirs)
    table = build_tensor_table(rows, constants)
    split = split_tensor_table(table, config)
    print(
        "offline-training load "
        f"runs={len(run_dirs)} rows={len(rows)} "
        f"train_rows={len(split['train'])} validation_rows={len(split['validation'])}"
    )

    # Initialize the structured critic from the data-generation cost weights.
    lambda_init = {
        "lambda_t": constants["time-weight"],
        "lambda_c": constants["compute-weight"],
        "lambda_u": constants["action-weight"],
    }
    model = StructuredResidualCritic(config.learning_rate, lambda_init)
    snapshot_dir = create_snapshot_dir(config)
    write_snapshot_metadata(snapshot_dir, config, table, split, model)

    # Bind minimal data loaders and a single checkpoint callback for the GPU training run.
    train_loader = DataLoader(
        split["train"], batch_size=config.batch_size, shuffle=True,
        num_workers=TRAINER_NUM_WORKERS,
    )
    validation_loader = DataLoader(
        split["validation"], batch_size=config.batch_size, shuffle=False,
        num_workers=TRAINER_NUM_WORKERS,
    )
    checkpoint = ModelCheckpoint(
        dirpath=snapshot_dir / "checkpoints",
        filename="critic-{epoch:03d}-{val_mse:.6f}",
        monitor="val_mse",
        mode="min",
        save_last=True,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        max_epochs=config.max_epochs,
        callbacks=[checkpoint, EpochMSELogger(snapshot_dir)],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
    )

    # Fit the full critic and then write final weights, lambdas, and offline diagnostics.
    print(
        "offline-training train "
        f"epochs={config.max_epochs} batch_size={config.batch_size}"
    )
    trainer.fit(model, train_loader, validation_loader)
    torch.save(model.state_dict(), snapshot_dir / "critic_state_dict.pt")
    lambda_file = (snapshot_dir / "lambda.json").open("w", encoding="utf-8")
    json.dump(model.lambda_values(), lambda_file, indent=2, sort_keys=True)
    lambda_file.write("\n")
    lambda_file.close()
    metrics = evaluate_snapshot(model, table, split, snapshot_dir, config.action_grid_count)

    # Keep terminal output to the run status and final paths needed by downstream scripts.
    print(
        "offline-training finish "
        f"validation_mse={metrics['validation_mse']:.9f} "
        f"physical_validation_mse={metrics['physical_validation_mse']:.9f}"
    )
    print(f"offline-training snapshot_dir={snapshot_dir}")


if __name__ == "__main__":
    main()
