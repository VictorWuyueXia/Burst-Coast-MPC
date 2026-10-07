"""Extract the four recorded sessions in the historical heuristic confirmation figure."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rotary_pendulum.environment.dynamics import mechanism_points
from rotary_pendulum.utils.config_schema import RotaryPendulumConfig


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    campaign = root / "artifacts/rotary_pendulum/energy-heuristic/confirm_20261008"
    source = campaign / "energy_soft/machine-scannables"
    output = root / "docs/12-energy-transfer/machine-scannables/heuristic_confirmation_replay"
    figure = root / "docs/12-energy-transfer/human-readables/heuristic_confirmation.png"
    original = source.parent / "human-readables/representative_sessions.png"
    assert figure.read_bytes() == original.read_bytes(), "Figure does not match archived trial"
    selection = json.loads((source / "plot_selection.json").read_text())
    initial = np.load(campaign / "machine-scannables/initial_states.npz")["x"]
    traces = np.load(source / "trajectories.npz")
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
        "plot_selection.json": source / "plot_selection.json",
    }.items():
        (output / name).write_bytes(path.read_bytes())
    episodes = list(csv.DictReader((source / "episodes.csv").open()))
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
            **{key: traces[key][:, lane] for key in traces.files},
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
        "source_figure": str(figure.relative_to(root)),
        "figure_sha256": hashlib.sha256(figure.read_bytes()).hexdigest(),
        "source_trajectories_sha256": hashlib.sha256(
            (source / "trajectories.npz").read_bytes()
        ).hexdigest(),
        "physics_dt_s": 0.02,
        "control_dt_s": 0.1,
        "geometry_m": {
            "arm_length": physical.arm_length_m,
            "pendulum_length": physical.pendulum_length_m,
        },
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
