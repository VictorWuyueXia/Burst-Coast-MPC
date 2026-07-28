"""Dataset construction for offline structured critic training."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

REQUIRED_COLUMNS = (
    "s-sin-theta", "s-cos-theta", "s-omega-rad-s", "bbar", "hbar",
    "burst-steps", "horizon-steps", "return-cost", "solve-time-s", "done", "plan-id",
)


@dataclass(frozen=True)
class TrainingConfig:
    data_roots: list[Path]
    output_root: Path
    run_name: str
    seed: int
    train_fraction: float
    batch_size: int
    max_epochs: int
    learning_rate: float
    action_grid_count: int


class OfflineRLDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        physical: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        # Bind contiguous tensors so the Lightning loop reads one supervised tuple per row.
        self.features = features.contiguous()
        self.physical = physical.contiguous()
        self.target = target.contiguous()

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.physical[index], self.target[index]


def read_training_config(config_path: Path) -> TrainingConfig:
    # Resolve config-relative paths once so training can be launched from any cwd.
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"Offline training config must be a mapping: {config_path}")
    repo_root = config_path.resolve().parents[3]
    data_roots = [repo_root / str(path) for path in raw["data-roots"]]
    output_root = repo_root / str(raw["output-root"])

    # Validate only semantic ranges that would corrupt the supervised split or tensors.
    train_fraction, batch_size = float(raw["train-fraction"]), int(raw["batch-size"])
    max_epochs, learning_rate = int(raw["max-epochs"]), float(raw["learning-rate"])
    action_grid_count = int(raw["action-grid-count"])
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train-fraction must be strictly inside (0, 1)")
    if batch_size <= 0 or max_epochs <= 0 or action_grid_count <= 1:
        raise ValueError("batch-size, max-epochs, and action-grid-count must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")

    return TrainingConfig(
        data_roots, output_root, str(raw["run-name"]), int(raw["seed"]), train_fraction,
        batch_size, max_epochs, learning_rate, action_grid_count,
    )


def collect_run_dirs(config: TrainingConfig) -> list[Path]:
    # Search recursively under declared roots so one config can train across epoch folders.
    run_dirs: set[Path] = set()
    for root in config.data_roots:
        if not root.exists():
            raise FileNotFoundError(f"Offline training data root does not exist: {root}")
        for rl_path in root.rglob("rl_steps.csv"):
            run_dirs.add(rl_path.parent)
    ordered = sorted(run_dirs)
    if not ordered:
        raise FileNotFoundError(f"No rl_steps.csv files found under: {config.data_roots}")
    return ordered


def load_artifact_rows(run_dirs: list[Path]) -> tuple[list[dict[str, str]], dict[str, float]]:
    # Read the first run config as the fixed physical contract for this training set.
    constant_paths = [
        ("mass-kg", ("environment", "pendulum", "mass-kg")),
        ("gravity-m-s2", ("environment", "pendulum", "gravity-m-s2")),
        ("length-m", ("environment", "pendulum", "length-m")),
        ("timestep-s", ("environment", "simulation", "timestep-s")),
        ("time-weight", ("data-generation", "time-weight")),
        ("action-weight", ("data-generation", "action-weight")),
        ("compute-weight", ("data-generation", "compute-weight")),
        ("hbar-max", ("data-generation", "hbar-max")),
    ]
    first_config_file = (run_dirs[0] / "config.json").open(encoding="utf-8")
    first_config = json.load(first_config_file)
    first_config_file.close()
    constants = {}
    for name, path in constant_paths:
        value = first_config
        for key in path:
            value = value[key]
        constants[name] = float(value)

    # Concatenate rows while requiring every run to match the same physical constants.
    rows: list[dict[str, str]] = []
    for run_index, run_dir in enumerate(run_dirs):
        config_file = (run_dir / "config.json").open(encoding="utf-8")
        run_config = json.load(config_file)
        config_file.close()
        run_constants = {}
        for name, path in constant_paths:
            value = run_config
            for key in path:
                value = value[key]
            run_constants[name] = float(value)
        if run_constants != constants:
            raise ValueError(f"Training run has different physical or cost constants: {run_dir}")

        csv_file = (run_dir / "rl_steps.csv").open(encoding="utf-8", newline="")
        reader = csv.DictReader(csv_file)
        missing_columns = reader.fieldnames is None or any(
            name not in reader.fieldnames for name in REQUIRED_COLUMNS
        )
        if missing_columns:
            raise ValueError(f"rl_steps.csv is missing required critic columns: {run_dir}")
        for row in reader:
            row["__run-index"] = str(run_index)
            row["__run-dir"] = str(run_dir)
            rows.append(row)
        csv_file.close()

    if not rows:
        raise ValueError(f"No RL transition rows loaded from: {run_dirs}")
    return rows, constants


def build_tensor_table(rows: list[dict[str, str]], constants: dict[str, float]) -> dict[str, Any]:
    # Parse CSV columns into aligned NumPy arrays before any Torch tensor conversion.
    sin_theta = np.array([float(row["s-sin-theta"]) for row in rows], dtype=np.float64)
    cos_theta = np.array([float(row["s-cos-theta"]) for row in rows], dtype=np.float64)
    omega = np.array([float(row["s-omega-rad-s"]) for row in rows], dtype=np.float64)
    bbar = np.array([float(row["bbar"]) for row in rows], dtype=np.float64)
    hbar = np.array([float(row["hbar"]) for row in rows], dtype=np.float64)
    burst_steps = np.array([float(row["burst-steps"]) for row in rows], dtype=np.float64)
    horizon_steps = np.array([float(row["horizon-steps"]) for row in rows], dtype=np.float64)
    solve_time_s = np.array([float(row["solve-time-s"]) for row in rows], dtype=np.float64)
    target = np.array([float(row["return-cost"]) for row in rows], dtype=np.float64)
    run_index = np.array([int(row["__run-index"]) for row in rows], dtype=np.int64)

    # Convert terminal text explicitly so malformed Boolean fields fail at the boundary.
    done_text = [row["done"] for row in rows]
    valid_done_text = {"True", "False", "true", "false"}
    if any(value not in valid_done_text for value in done_text):
        raise ValueError("done column must contain True or False text values")
    done = np.array([value.lower() == "true" for value in done_text], dtype=bool)

    # Derive energy and angular-velocity features from the compact logged state contract.
    mass, gravity = constants["mass-kg"], constants["gravity-m-s2"]
    length, timestep_s = constants["length-m"], constants["timestep-s"]
    inertia = mass * length**2
    upright_energy = 2.0 * mass * gravity * length
    energy = 0.5 * inertia * omega**2 + mass * gravity * length * (1.0 + cos_theta)
    omega_ref = 2.0 * math.sqrt(gravity / length)
    energy_error_normalized = (energy - upright_energy) / upright_energy

    # Package the exact critic inputs and physical terms used by the structured model.
    features_raw = np.column_stack(
        (sin_theta, cos_theta, omega / omega_ref, energy_error_normalized, bbar, hbar)
    )
    physical = np.column_stack(
        (horizon_steps * timestep_s, solve_time_s / timestep_s, burst_steps)
    )
    full_horizon_steps = math.ceil((math.tau / math.sqrt(gravity / length)) / timestep_s)

    return {
        "features_raw": torch.as_tensor(features_raw, dtype=torch.float32),
        "physical": torch.as_tensor(physical, dtype=torch.float32),
        "target": torch.as_tensor(target, dtype=torch.float32),
        "run_index": torch.as_tensor(run_index, dtype=torch.int64),
        "metadata": {"sin_theta": sin_theta, "cos_theta": cos_theta, "bbar": bbar,
                     "hbar": hbar, "done": done, "target": target,
                     "omega_normalized": omega / omega_ref,
                     "energy_error_normalized": energy_error_normalized,
                     "plan_id": [row["plan-id"] for row in rows],
                     "run_dir": [row["__run-dir"] for row in rows]},
        "constants": constants | {"full-horizon-steps": float(full_horizon_steps)},
    }


def split_tensor_table(table: dict[str, Any], config: TrainingConfig) -> dict[str, Any]:
    # Select whole artifact runs for train or validation so trajectories do not leak across splits.
    run_index = table["run_index"]
    unique_runs = torch.unique(run_index).numpy()
    rng = np.random.default_rng(config.seed)
    shuffled_runs = rng.permutation(unique_runs)
    train_run_count = round(config.train_fraction * len(shuffled_runs))
    if train_run_count <= 0 or train_run_count >= len(shuffled_runs):
        raise ValueError("train-fraction must leave at least one train run and one validation run")
    train_runs = torch.as_tensor(shuffled_runs[:train_run_count], dtype=torch.int64)
    val_runs = torch.as_tensor(shuffled_runs[train_run_count:], dtype=torch.int64)
    train_mask = torch.isin(run_index, train_runs)
    val_mask = torch.isin(run_index, val_runs)

    # Fit residual-network normalization on the training split and require all scales to vary.
    features_raw = table["features_raw"]
    feature_mean = features_raw[train_mask].mean(dim=0)
    feature_std = features_raw[train_mask].std(dim=0, unbiased=False)
    if torch.any(feature_std == 0.0):
        raise ValueError("Training split has a zero-variance critic input feature")
    features = (features_raw - feature_mean) / feature_std

    train_dataset = OfflineRLDataset(
        features[train_mask], table["physical"][train_mask], table["target"][train_mask]
    )
    val_dataset = OfflineRLDataset(
        features[val_mask], table["physical"][val_mask], table["target"][val_mask]
    )
    return {
        "train": train_dataset,
        "validation": val_dataset,
        "train_indices": torch.nonzero(train_mask, as_tuple=False).flatten(),
        "validation_indices": torch.nonzero(val_mask, as_tuple=False).flatten(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "train_runs": train_runs,
        "validation_runs": val_runs,
    }
