"""Offline validation diagnostics for the structured residual critic."""
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def evaluate_snapshot(
    model: Any,
    table: dict[str, Any],
    split: dict[str, Any],
    snapshot_dir: Path,
    action_grid_count: int,
) -> dict[str, float]:
    # Evaluate train and validation rows on the model device without grid-policy claims.
    model.eval()
    device = model.device
    train, validation = split["train"], split["validation"]
    lambda_values = model.lambda_values()
    lambda_tensor = torch.tensor(
        [lambda_values["lambda_t"], lambda_values["lambda_c"], lambda_values["lambda_u"]],
        dtype=torch.float32,
        device=device,
    )

    torch.set_grad_enabled(False)
    train_prediction = model(train.features.to(device), train.physical.to(device)).cpu()
    validation_prediction = model(
        validation.features.to(device), validation.physical.to(device)
    ).cpu()
    validation_q_phys = (validation.physical.to(device) @ lambda_tensor).cpu()
    torch.set_grad_enabled(True)

    # Summarize the full critic and physical-only held-out regression quality.
    train_target = train.target
    validation_target = validation.target
    metrics = {
        "train_mse": float(torch.mean((train_prediction - train_target) ** 2)),
        "validation_mse": float(torch.mean((validation_prediction - validation_target) ** 2)),
        "validation_mae": float(torch.mean(torch.abs(validation_prediction - validation_target))),
        "physical_validation_mse": float(torch.mean((validation_q_phys - validation_target) ** 2)),
        "train_rows": float(len(train)),
        "validation_rows": float(len(validation)),
        **lambda_values,
    }
    metrics_file = (snapshot_dir / "metrics.json").open("w", encoding="utf-8")
    json.dump(metrics, metrics_file, indent=2, sort_keys=True)
    metrics_file.write("\n")
    metrics_file.close()

    # Write CSV and figure diagnostics from held-out observed rows.
    validation_indices = split["validation_indices"].cpu().numpy()
    validation_prediction_np = validation_prediction.detach().numpy()
    validation_q_phys_np = validation_q_phys.detach().numpy()
    validation_target_np = validation_target.detach().numpy()
    diag_args = (
        snapshot_dir, table["metadata"], validation_indices, validation_prediction_np,
        validation_q_phys_np, validation_target_np,
    )
    write_region_metrics(*diag_args)
    write_observed_policy_sanity(*diag_args)
    write_q_surfaces(model, table, split, snapshot_dir, action_grid_count)
    return metrics


def write_region_metrics(
    snapshot_dir: Path,
    metadata: dict[str, Any],
    indices: np.ndarray,
    prediction: np.ndarray,
    q_phys: np.ndarray,
    target: np.ndarray,
) -> None:
    # Build coarse, interpretable regions from the validation rows themselves.
    bbar = metadata["bbar"][indices]
    hbar = metadata["hbar"][indices]
    energy = metadata["energy_error_normalized"][indices]
    omega_abs = np.abs(metadata["omega_normalized"][indices])
    error = prediction - target
    physical_error = q_phys - target
    bbar_mid, hbar_mid = np.median(bbar), np.median(hbar)
    energy_mid, omega_mid = np.median(energy), np.median(omega_abs)
    regions = [
        ("action", "low_bbar_low_hbar", (bbar <= bbar_mid) & (hbar <= hbar_mid)),
        ("action", "low_bbar_high_hbar", (bbar <= bbar_mid) & (hbar >= hbar_mid)),
        ("action", "high_bbar_low_hbar", (bbar >= bbar_mid) & (hbar <= hbar_mid)),
        ("action", "high_bbar_high_hbar", (bbar >= bbar_mid) & (hbar >= hbar_mid)),
        ("state", "low_energy_slow", (energy <= energy_mid) & (omega_abs <= omega_mid)),
        ("state", "low_energy_fast", (energy <= energy_mid) & (omega_abs >= omega_mid)),
        ("state", "high_energy_slow", (energy >= energy_mid) & (omega_abs <= omega_mid)),
        ("state", "high_energy_fast", (energy >= energy_mid) & (omega_abs >= omega_mid)),
    ]

    # Persist region-wise full-critic and physical-only errors for direct inspection.
    csv_file = (snapshot_dir / "region_metrics.csv").open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=("region-type", "region-name", "rows", "mse", "mae", "physical-mse"),
    )
    writer.writeheader()
    for region_type, region_name, mask in regions:
        writer.writerow(
            {
                "region-type": region_type,
                "region-name": region_name,
                "rows": int(np.sum(mask)),
                "mse": float(np.mean(error[mask] ** 2)),
                "mae": float(np.mean(np.abs(error[mask]))),
                "physical-mse": float(np.mean(physical_error[mask] ** 2)),
            }
        )
    csv_file.close()


def write_observed_policy_sanity(
    snapshot_dir: Path,
    metadata: dict[str, Any],
    indices: np.ndarray,
    prediction: np.ndarray,
    q_phys: np.ndarray,
    target: np.ndarray,
) -> None:
    # Sort held-out rows by predicted cost to inspect the critic's observed action preference.
    order = np.argsort(prediction)
    csv_file = (snapshot_dir / "observed_policy_sanity.csv").open(
        "w", encoding="utf-8", newline=""
    )
    writer = csv.DictWriter(
        csv_file,
        fieldnames=(
            "rank", "run-dir", "plan-id", "bbar", "hbar", "done", "return-cost",
            "predicted-q", "physical-q", "residual-q", "error",
        ),
    )
    writer.writeheader()
    for rank, local_index in enumerate(order):
        row_index = indices[local_index]
        writer.writerow(
            {
                "rank": rank,
                "run-dir": metadata["run_dir"][row_index],
                "plan-id": metadata["plan_id"][row_index],
                "bbar": float(metadata["bbar"][row_index]),
                "hbar": float(metadata["hbar"][row_index]),
                "done": bool(metadata["done"][row_index]),
                "return-cost": float(target[local_index]),
                "predicted-q": float(prediction[local_index]),
                "physical-q": float(q_phys[local_index]),
                "residual-q": float(prediction[local_index] - q_phys[local_index]),
                "error": float(prediction[local_index] - target[local_index]),
            }
        )
    csv_file.close()


def write_q_surfaces(
    model: Any,
    table: dict[str, Any],
    split: dict[str, Any],
    snapshot_dir: Path,
    action_grid_count: int,
) -> None:
    # Select representative validation states from actual held-out rows.
    import matplotlib.pyplot as plt

    metadata = table["metadata"]
    constants = table["constants"]
    indices = split["validation_indices"].cpu().numpy()
    terminal_indices = indices[metadata["done"][indices]]
    if terminal_indices.size == 0:
        raise ValueError("Validation split has no terminal row for terminal Q-surface")
    selected = [
        ("downward", indices[np.argmin(metadata["cos_theta"][indices])]),
        ("high_energy", indices[np.argmax(metadata["energy_error_normalized"][indices])]),
        ("near_upright", indices[np.argmax(metadata["cos_theta"][indices])]),
        ("terminal", terminal_indices[0]),
    ]

    # Evaluate the direct B/H and natural-period action grid against measured compute scale.
    bbar_axis = np.linspace(0.0, 1.0, action_grid_count, dtype=np.float64)
    hbar_axis = np.linspace(0.0, constants["hbar-max"], action_grid_count, dtype=np.float64)
    bbar_grid, hbar_grid = np.meshgrid(bbar_axis, hbar_axis, indexing="xy")
    full_horizon_steps = constants["full-horizon-steps"]
    timestep_s = constants["timestep-s"]
    horizon_steps = np.maximum(1.0, np.ceil(hbar_grid * full_horizon_steps))
    burst_steps = np.maximum(1.0, np.rint(bbar_grid * horizon_steps))
    device = model.device

    for label, row_index in selected:
        state = np.array(
            [
                metadata["sin_theta"][row_index],
                metadata["cos_theta"][row_index],
                metadata["omega_normalized"][row_index],
                metadata["energy_error_normalized"][row_index],
            ],
            dtype=np.float64,
        )
        feature_raw = np.column_stack(
            (
                np.repeat(state[0], bbar_grid.size),
                np.repeat(state[1], bbar_grid.size),
                np.repeat(state[2], bbar_grid.size),
                np.repeat(state[3], bbar_grid.size),
                bbar_grid.reshape(-1),
                hbar_grid.reshape(-1),
            )
        )
        physical = np.column_stack(
            (
                (horizon_steps * timestep_s).reshape(-1),
                np.repeat(table["physical"][row_index, 1].item(), bbar_grid.size),
                burst_steps.reshape(-1),
            )
        )
        normalized = (torch.as_tensor(feature_raw, dtype=torch.float32) - split["feature_mean"])
        normalized = normalized / split["feature_std"]

        torch.set_grad_enabled(False)
        q_values = model(
            normalized.to(device),
            torch.as_tensor(physical, dtype=torch.float32, device=device),
        ).detach().cpu().numpy()
        torch.set_grad_enabled(True)

        figure, axis_handle = plt.subplots(figsize=(6, 5))
        surface = axis_handle.contourf(bbar_grid, hbar_grid, q_values.reshape(bbar_grid.shape), 24)
        axis_handle.scatter(
            [metadata["bbar"][row_index]], [metadata["hbar"][row_index]],
            color="white", edgecolor="black", s=42,
        )
        axis_handle.set_xlabel("bbar")
        axis_handle.set_ylabel("hbar")
        axis_handle.set_title(f"Observed-compute Q surface: {label}")
        figure.colorbar(surface, ax=axis_handle, label="predicted Q")
        figure.tight_layout()
        figure.savefig(snapshot_dir / f"q_surface_{label}.png", dpi=150)
        plt.close(figure)
