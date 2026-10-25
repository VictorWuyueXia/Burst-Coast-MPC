"""Reconstruct the recorded MPC+RL holding motion without rerunning the controller."""

import csv
import json
from pathlib import Path

import numpy as np


def holding_statistics(record_root: Path) -> dict:
    """Audit complete sessions and separate holding oscillation from unresolved drift."""

    # Preserve the nominal plan records and the independently executed state transitions.
    design = json.loads((record_root / "design.json").read_text())
    assert json.loads((record_root / "completed.json").read_text())["decisions"] == 400
    fields = (
        "stage_states", "plans", "executed", "objective", "terminal_costs",
        "selected", "accepted", "refinement_error", "arm_margin_rad",
    )
    blocks = {field: [] for field in fields}
    paths = sorted(record_root.glob("chunk-*.npz"))
    assert len(paths) == 40 and design["count"] == 32
    for path in paths:
        archive = np.load(path, allow_pickle=False)
        for field in fields:
            blocks[field].append(archive[field])
        archive.close()
    data = {field: np.concatenate(value) for field, value in blocks.items()}
    assert data["accepted"].shape == (400, 32) and data["accepted"].all()
    np.testing.assert_array_equal(data["stage_states"][1:, :, 0], data["executed"][:-1, :, -1])
    assert np.isfinite(data["executed"]).all() and (data["arm_margin_rad"] > 0).all()
    command = data["stage_states"][:, :, 0].copy()
    successor = data["executed"][:, :, -1].copy()
    command[..., 1] = np.arctan2(-np.sin(command[..., 1]), -np.cos(command[..., 1]))
    successor[..., 1] = np.arctan2(-np.sin(successor[..., 1]), -np.cos(successor[..., 1]))

    # Analyze the same final ten seconds as the archived report, retaining signed upright error.
    tail = data["executed"][300:].transpose(1, 0, 2, 3).reshape(32, 10000, 4).copy()
    tail[..., 1] = np.arctan2(-np.sin(tail[..., 1]), -np.cos(tail[..., 1]))
    means, deviations = tail.mean(axis=1), tail.std(axis=1)
    drift = tail[:, 5000:].mean(axis=1) - tail[:, :5000].mean(axis=1)
    actions = data["plans"][300:, :, 0].T
    spectrum = np.abs(np.fft.rfft(actions - actions.mean(axis=1, keepdims=True))) ** 2
    spectrum[:, 0] = 0
    peaks = spectrum.argmax(axis=1)
    frequencies = np.fft.rfftfreq(100, 0.1)[peaks]
    spectral_fraction = spectrum[np.arange(32), peaks] / spectrum.sum(axis=1)
    names = ("arm_rad", "upright_rad", "arm_speed_rad_s", "pendulum_speed_rad_s")
    rows = []
    for lane in range(32):
        row = dict(episode=lane, action_peak_hz=float(frequencies[lane]))
        row["action_peak_power_fraction"] = float(spectral_fraction[lane])
        for coordinate, name in enumerate(names):
            row[f"mean_{name}"] = float(means[lane, coordinate])
            row[f"std_{name}"] = float(deviations[lane, coordinate])
            row[f"half_window_drift_{name}"] = float(drift[lane, coordinate])
            row[f"min_{name}"] = float(tail[lane, :, coordinate].min())
            row[f"max_{name}"] = float(tail[lane, :, coordinate].max())
        rows.append(row)
    first_min = tail[:, :5000].min(axis=(0, 1))
    first_max = tail[:, :5000].max(axis=(0, 1))
    outside = ((tail[:, 5000:] < first_min) | (tail[:, 5000:] > first_max)).any(axis=-1)
    summary = dict(
        scope="Recorded finite-session behavior, not a regional invariance certificate",
        mean_error=means.mean(axis=0).tolist(),
        pooled_min=tail.min(axis=(0, 1)).tolist(), pooled_max=tail.max(axis=(0, 1)).tolist(),
        pooled_rms=np.sqrt(np.mean(tail**2, axis=(0, 1))).tolist(),
        lane_mean_min=means.min(axis=0).tolist(), lane_mean_max=means.max(axis=0).tolist(),
        largest_absolute_half_window_drift=np.abs(drift).max(axis=0).tolist(),
        first_half_box_later_escape_samples=int(outside.sum()),
        first_half_box_later_escape_episodes=int(outside.any(axis=1).sum()),
        action_peak_hz_min=float(frequencies.min()), action_peak_hz_max=float(frequencies.max()),
        selected_seed_counts=np.bincount(data["selected"][300:].ravel(), minlength=3).tolist(),
        action_rms=float(np.sqrt(np.mean(actions**2))),
        coordinate_order=list(names),
    )
    return dict(
        design=design, command=command, successor=successor, plans=data["plans"],
        objective=data["objective"], terminal_costs=data["terminal_costs"],
        stage_states=data["stage_states"], summary=summary, rows=rows,
    )


if __name__ == "__main__":
    # Save compact command-level records; the original archive remains the dense-data source.
    repository = Path(__file__).resolve().parents[3]
    source = repository / "artifacts/rotary_pendulum/inner-mpc/20261022-lqr-cost-fine-confirm-16"
    result = holding_statistics(source / "machine-scannables")
    output = repository / "artifacts/rotary_pendulum/inner-mpc/actual-holding-study-20261025"
    machine = output / "machine-scannables"
    machine.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        machine / "recorded_transitions.npz",
        **{key: result[key] for key in (
            "command", "successor", "plans", "objective", "terminal_costs", "stage_states"
        )},
    )
    handle = (machine / "holding_by_episode.csv").open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
    writer.writeheader()
    writer.writerows(result["rows"])
    handle.close()
    (machine / "holding_statistics.json").write_text(json.dumps(result["summary"], indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)
