"""Export the historical figure sessions or representative recovery outcomes for replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rotary_pendulum.environment.dynamics import mechanism_points
from rotary_pendulum.utils.config_schema import RotaryPendulumConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", action="store_true", help="Export typical recovery outcomes")
    recovery = parser.parse_args().recovery
    root = Path(__file__).resolve().parents[1]
    campaign_name = "recovery_confirm_20261008" if recovery else "confirm_20261008"
    trial = "recovery" if recovery else "energy_soft"
    campaign = (
        root
        / (
            "artifacts/rotary_pendulum/experiment-results/"
            "physical-energy-transfer-controller/raw-runs"
        )
        / campaign_name
    )
    source = campaign / trial / "machine-scannables"
    output = (
        root
        / "artifacts/rotary_pendulum/experiment-results"
        / (
            "braking-prediction-controller-replay"
            if recovery
            else "original-analytical-controller-replay"
        )
        / "records"
    )
    figure = root / (
        "artifacts/rotary_pendulum/experiment-results/"
        "physical-energy-transfer-controller/figures/heuristic_confirmation.png"
    )
    episodes = list(csv.DictReader((source / "episodes.csv").open()))
    if recovery:
        selection = {}
        for label in ("downward", "moving", "near", "tight"):
            eligible = [
                row
                for row in episodes
                if row["stratum"] == label
                and row["arm_violation"] == "False"
                and row["success"] == ("False" if label == "downward" else "True")
            ]
            metric = "peak_arm_rad" if label == "downward" else "duration_s"
            ordered = sorted(eligible, key=lambda row: (float(row[metric]), int(row["lane"])))
            selection[label] = int(ordered[len(ordered) // 2]["lane"])
    else:
        original = source.parent / "human-readables/representative_sessions.png"
        assert figure.read_bytes() == original.read_bytes(), "Figure does not match archived trial"
        selection = json.loads((source / "plot_selection.json").read_text())
    initial = np.load(campaign / "machine-scannables/initial_states.npz")["x"]
    traces = dict(np.load(source / "trajectories.npz"))
    initial[:, 1] = initial[:, 1] % (2 * np.pi) % (2 * np.pi)
    for key in ("start_x", "x", "physics_x"):
        traces[key][..., 1] = traces[key][..., 1] % (2 * np.pi) % (2 * np.pi)
    snapshot = campaign / "machine-scannables/source_snapshot"
    physics = snapshot / "src/rotary_pendulum/configs/physics.yaml"
    physical = RotaryPendulumConfig.model_validate(
        OmegaConf.to_container(OmegaConf.load(physics), resolve=True)["rotary-pendulum"]
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, path in {
        "physics.yaml": physics,
        "campaign.json": campaign / "machine-scannables/campaign.json",
        "provenance.json": campaign / "machine-scannables/provenance.json",
    }.items():
        (output / name).write_bytes(path.read_bytes())
    (output / "plot_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    sessions = []
    for label, lane in selection.items():
        active = traces["physics_active"][:, lane]
        decisions, substeps = np.nonzero(active)
        state = np.concatenate((initial[lane][None], traces["physics_x"][:, lane][active]))
        time = np.arange(len(state), dtype=float) * 0.02
        torque = np.r_[0.0, traces["torque_nm"][decisions, lane]]
        geometry = mechanism_points(state, physical)
        table = np.column_stack(
            (time, state, torque, np.r_[-1, decisions], geometry.reshape(-1, 12))
        )
        columns = [
            "time_s",
            "theta_rad",
            "alpha_rad",
            "omega_rad_s",
            "nu_rad_s",
            "interval_torque_nm",
            "decision_index",
        ]
        columns += [
            f"{point}_{axis}_m"
            for point in ("origin", "pivot", "center", "tip")
            for axis in ("x", "y", "z")
        ]
        np.savetxt(
            output / f"{label}.csv",
            table,
            delimiter=",",
            header=",".join(columns),
            comments="",
            fmt="%.17g",
        )
        np.savez_compressed(
            output / f"{label}.npz",
            initial_x=initial[lane],
            **{key: traces[key][:, lane] for key in traces},
        )
        np.testing.assert_allclose(
            time[1:],
            (traces["time_s"][:, lane] - traces["elapsed_s"][:, lane])[decisions]
            + 0.02 * (substeps + 1),
            atol=1e-12,
        )
        assert np.isfinite(table).all() and len(state) > 1
        np.testing.assert_array_equal(
            np.loadtxt(output / f"{label}.csv", delimiter=",", skiprows=1), table
        )
        sessions.append(
            {
                "stratum": label,
                "source_lane": lane,
                "samples_including_initial": len(state),
                "duration_s": time[-1],
                "episode": episodes[lane],
            }
        )
        print(f"Exported {label}: lane {lane}, {len(state)} samples, {time[-1]:.2f} s", flush=True)
    manifest = {
        "schema_version": 1,
        "source_trial": str(source.parent.relative_to(root)),
        "source_figure": None if recovery else str(figure.relative_to(root)),
        "selection_rule": (
            "Within seed 20261008, no-excursion successes: upper median duration per stratum; "
            "downward no-excursion failures: upper median peak arm angle. Ties: lane index."
            if recovery
            else "Original figure plot_selection.json"
        ),
        "figure_sha256": None if recovery else hashlib.sha256(figure.read_bytes()).hexdigest(),
        "source_trajectories_sha256": hashlib.sha256(
            (source / "trajectories.npz").read_bytes()
        ).hexdigest(),
        "physics_dt_s": 0.02,
        "control_dt_s": 0.1,
        "geometry_m": {
            "arm_length": physical.arm_length_m,
            "pendulum_length": physical.pendulum_length_m,
        },
        "pendulum_angle_convention": "alpha_rad in [0, 2*pi); 0 downward, pi upright",
        "state_order": ["theta_rad", "alpha_rad", "omega_rad_s", "nu_rad_s"],
        "torque_convention": (
            "Row i > 0 torque applies on (time[i-1], time[i]]; initial row zero is a sentinel"
        ),
        "sessions": sessions,
        "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.iterdir())
            if p.suffix in (".csv", ".npz", ".yaml", ".json") and p.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
