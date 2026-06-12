import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "inverted_pendulum" / "offline_training"
)


def _load_script_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _training_modules():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    return (
        torch,
        _load_script_module("dataset"),
        _load_script_module("critic"),
        _load_script_module("diagnostics"),
    )


def _write_run(root: Path, name: str, return_offset: float) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    config = {
        "environment": {
            "pendulum": {
                "mass-kg": 1.0,
                "gravity-m-s2": 9.80665,
                "length-m": 1.0,
            },
            "simulation": {"timestep-s": 0.1},
        },
        "data-generation": {
            "time-weight": 1.0,
            "action-weight": 1.0,
            "compute-weight": 1.0,
        },
    }
    config_file = (run_dir / "config.json").open("w", encoding="utf-8")
    json.dump(config, config_file)
    config_file.close()

    rows = [
        [0.0, 1.0, 0.0, 0.20, 0.40, 2, 8, 10.0 + return_offset, 0.02, "False", "p0"],
        [1.0, 0.0, 1.0, 0.40, 0.60, 3, 12, 12.0 + return_offset, 0.03, "False", "p1"],
        [0.0, -1.0, -1.0, 0.60, 0.80, 4, 16, 14.0 + return_offset, 0.04, "False", "p2"],
        [-1.0, 0.0, 2.0, 0.80, 1.00, 5, 20, 16.0 + return_offset, 0.05, "True", "p3"],
    ]
    fieldnames = (
        "s-sin-theta", "s-cos-theta", "s-omega-rad-s", "bbar", "hbar",
        "burst-steps", "horizon-steps", "return-cost", "solve-time-s", "done", "plan-id",
    )
    csv_file = (run_dir / "rl_steps.csv").open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(zip(fieldnames, row, strict=True)))
    csv_file.close()


def _build_split(tmp_path: Path, dataset):
    root = tmp_path / "artifacts"
    _write_run(root, "run-a", 0.0)
    _write_run(root, "run-b", 20.0)
    config = dataset.TrainingConfig(
        [root], tmp_path / "snapshots", "test", 1, 0.5, 2, 1, 0.001, 5
    )
    run_dirs = dataset.collect_run_dirs(config)
    rows, constants = dataset.load_artifact_rows(run_dirs)
    table = dataset.build_tensor_table(rows, constants)
    split = dataset.split_tensor_table(table, config)
    return config, table, split, rows


def test_offline_dataset_uses_run_split_and_train_normalization(tmp_path) -> None:
    torch, dataset, _, _ = _training_modules()
    _, table, split, rows = _build_split(tmp_path, dataset)

    assert len(rows) == 8
    assert split["train"].features.shape[1] == 6
    assert split["validation"].physical.shape[1] == 3
    assert set(split["train_runs"].tolist()).isdisjoint(split["validation_runs"].tolist())
    assert torch.allclose(split["train"].features.mean(dim=0), torch.zeros(6), atol=1.0e-6)

    upright_rows = np.flatnonzero(
        (table["metadata"]["sin_theta"] == 0.0)
        & (table["metadata"]["cos_theta"] == 1.0)
        & (table["metadata"]["omega_normalized"] == 0.0)
    )
    assert table["metadata"]["energy_error_normalized"][upright_rows[0]] == pytest.approx(0.0)


def test_structured_residual_critic_outputs_positive_lambdas() -> None:
    torch, _, critic, _ = _training_modules()
    model = critic.StructuredResidualCritic(
        0.001,
        {"lambda_t": 1.0, "lambda_c": 1.0, "lambda_u": 1.0},
    )
    features = torch.randn(4, 6)
    physical = torch.ones(4, 3)
    target = torch.arange(4, dtype=torch.float32)
    prediction = model(features, physical)
    loss = torch.nn.functional.mse_loss(prediction, target)

    assert prediction.shape == (4,)
    assert torch.isfinite(loss)
    assert all(value > 0.0 for value in model.lambda_values().values())
    assert all(
        name == "eta" or name.startswith("residual.")
        for name, _ in model.named_parameters()
    )


def test_offline_diagnostics_write_snapshot_artifacts(tmp_path) -> None:
    _, dataset, critic, diagnostics = _training_modules()
    config, table, split, _ = _build_split(tmp_path, dataset)
    model = critic.StructuredResidualCritic(
        config.learning_rate,
        {"lambda_t": 1.0, "lambda_c": 1.0, "lambda_u": 1.0},
    )
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()

    metrics = diagnostics.evaluate_snapshot(
        model, table, split, snapshot_dir, config.action_grid_count
    )

    assert metrics["validation_mse"] >= 0.0
    assert (snapshot_dir / "metrics.json").exists()
    assert (snapshot_dir / "region_metrics.csv").exists()
    assert (snapshot_dir / "observed_policy_sanity.csv").exists()
    for label in ["downward", "high_energy", "near_upright", "terminal"]:
        assert (snapshot_dir / f"q_surface_{label}.png").exists()
