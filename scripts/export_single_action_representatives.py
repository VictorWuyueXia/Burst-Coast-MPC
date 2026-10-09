"""Export eight deterministic, portable animation records from the final validation runs."""

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rotary_pendulum.environment.dynamics import derive_model, energy_components, mechanism_points
from rotary_pendulum.utils.config_schema import RotaryPendulumConfig


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    study = root / (
        "artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller"
    )
    output = study / "portable-animation-records/records"
    output.mkdir(parents=True, exist_ok=True)
    events = json.loads((study / "records/boundary_events.json").read_text())
    sessions, sources = [], []
    for seed in (20261022, 20261023):
        campaign = study / "raw-runs" / f"validation_{seed}"
        source = campaign / "gain_0.004_work_0.02/machine-scannables"
        episodes = list(csv.DictReader((source / "episodes.csv").open()))
        initial = np.load(campaign / "machine-scannables/initial_states.npz")["x"]
        traces = np.load(source / "trajectories.npz")
        initial[:, 1] = (initial[:, 1] + np.pi) % (2 * np.pi) - np.pi
        archive = traces
        traces = {key: archive[key] for key in archive.files}
        archive.close()
        for key in ("start_x", "x", "physics_x"):
            traces[key][..., 1] = (traces[key][..., 1] + np.pi) % (2 * np.pi) - np.pi
        traces["pendulum_angle_deg"] = np.rad2deg(traces["physics_x"][..., 1])
        selection = []
        if seed == 20261022:
            for group, label in (
                ("downward", "downward_median_capture_time"),
                ("moving", "moving_median_capture_time_without_arm_crossing"),
                ("near", "near_upright_median_capture_time"),
                ("tight", "tight_upright_median_capture_time"),
            ):
                eligible = sorted(
                    (
                        r
                        for r in episodes
                        if r["stratum"] == group
                        and r["success"] == "True"
                        and r["arm_violation"] == "False"
                    ),
                    key=lambda r: (float(r["duration_s"]), int(r["lane"])),
                )
                selection.append((label, int(eligible[len(eligible) // 2]["lane"])))
                if group == "downward":
                    selection.append(("downward_slowest_capture", int(eligible[-1]["lane"])))
            exact = np.flatnonzero(np.all(initial == 0, axis=1))
            assert len(exact) == 1
            selection.append(("exact_downward_rest", int(exact[0])))
            eligible_events = [
                e for e in events if e["seed"] == seed and e["bounded_torque_options"] > 0
            ]
            event = max(eligible_events, key=lambda e: (e["peak_arm_deg"], e["lane"]))
            selection.append(("arm_crossing_despite_bounded_tested_torques", event["lane"]))
        else:
            eligible_events = [
                e for e in events if e["seed"] == seed and e["bounded_torque_options"] == 0
            ]
            event = max(eligible_events, key=lambda e: (e["peak_arm_deg"], e["lane"]))
            selection.append(("arm_crossing_without_bounded_tested_torque", event["lane"]))
        snapshot = campaign / "machine-scannables/source_snapshot"
        physics = snapshot / "src/rotary_pendulum/configs/physics.yaml"
        physical = RotaryPendulumConfig.model_validate(
            OmegaConf.to_container(OmegaConf.load(physics), resolve=True)["rotary-pendulum"]
        )
        model = derive_model(physical)
        provenance = output / "provenance" / f"validation_seed_{seed}"
        provenance.mkdir(parents=True, exist_ok=True)
        for name, path in {
            "physics.yaml": physics,
            "campaign.json": campaign / "machine-scannables/campaign.json",
            "source_provenance.json": campaign / "machine-scannables/provenance.json",
            "controller_settings.json": source / "settings.json",
            "full_validation_episodes.csv": source / "episodes.csv",
            "integration_audit.json": source / "integration_audit.json",
        }.items():
            shutil.copy2(path, provenance / name)
        sources.append(
            {
                "seed": seed,
                "original_directory": str(source.relative_to(root)),
                "trajectory_sha256": hashlib.sha256(
                    (source / "trajectories.npz").read_bytes()
                ).hexdigest(),
            }
        )
        for name, lane in selection:
            print(f"Exporting {name}: seed {seed}, lane {lane}", flush=True)
            active = traces["physics_active"][:, lane]
            decisions, substeps = np.nonzero(active)
            state = np.concatenate((initial[lane][None], traces["physics_x"][:, lane][active]))
            time = np.arange(len(state)) * 0.02
            torque = np.r_[0.0, traces["torque_nm"][decisions, lane]]
            geometry = mechanism_points(state, physical)
            kinetic, potential, total = energy_components(state, physical, model)
            arm = 0.5 * model.arm_inertia_kg_m2 * state[:, 2] ** 2
            table = np.column_stack(
                (
                    time,
                    state,
                    torque,
                    np.r_[-1, decisions],
                    arm,
                    kinetic - arm,
                    potential,
                    total,
                    geometry.reshape(-1, 12),
                )
            )
            columns = [
                "time_s",
                "arm_angle_rad",
                "pendulum_angle_rad",
                "arm_speed_rad_s",
                "pendulum_speed_rad_s",
                "preceding_interval_torque_nm",
                "decision_index",
                "arm_kinetic_energy_j",
                "pendulum_kinetic_energy_j",
                "potential_energy_j",
                "total_energy_j",
            ]
            columns += [
                f"{point}_{axis}_m"
                for point in ("origin", "pivot", "center", "tip")
                for axis in "xyz"
            ]
            np.savetxt(
                output / f"{name}.csv",
                table,
                delimiter=",",
                comments="",
                header=",".join(columns),
                fmt="%.17g",
            )
            np.savez_compressed(
                output / f"{name}.npz",
                initial_x=initial[lane],
                **{key: traces[key][:, lane] for key in traces},
            )
            np.testing.assert_array_equal(
                np.loadtxt(output / f"{name}.csv", delimiter=",", skiprows=1), table
            )
            np.testing.assert_allclose(
                time[1:],
                (traces["time_s"][:, lane] - traces["elapsed_s"][:, lane])[decisions]
                + 0.02 * (substeps + 1),
                atol=1e-12,
                rtol=0,
            )
            assert np.isfinite(table).all() and np.all(
                (state[:, 1] >= -np.pi) & (state[:, 1] < np.pi)
            )
            np.testing.assert_allclose(
                np.linalg.norm(geometry[:, 1] - geometry[:, 0], axis=1),
                physical.arm_length_m,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                np.linalg.norm(geometry[:, 3] - geometry[:, 1], axis=1),
                physical.pendulum_length_m,
                atol=1e-14,
            )
            inside = (np.pi - np.abs(state[1:, 1])) <= np.deg2rad(15)
            capture = np.flatnonzero(np.convolve(inside.astype(int), np.ones(5, int), "valid") == 5)
            assert len(capture) and capture[0] + 5 == len(state) - 1
            crossing = bool(np.any(np.abs(state[:, 0]) >= np.pi))
            episode = episodes[lane]
            assert int(episode["lane"]) == lane and episode["success"] == "True"
            assert crossing == (episode["arm_violation"] == "True")
            np.testing.assert_allclose(time[-1], float(episode["duration_s"]), atol=1e-12)
            np.testing.assert_allclose(
                np.max(np.abs(state[1:, 0])), float(episode["peak_arm_rad"]), atol=1e-12
            )
            np.testing.assert_allclose(
                total[-1], float(episode["final_total_energy_j"]), atol=1e-14
            )
            sessions.append(
                {
                    "name": name,
                    "seed": seed,
                    "source_lane": lane,
                    "episode": episode,
                    "samples_including_initial": len(state),
                    "duration_s": float(time[-1]),
                    "peak_arm_degrees": float(np.rad2deg(np.max(np.abs(state[:, 0])))),
                    "initial_state": initial[lane].tolist(),
                    "first_crossing_audit": [
                        e for e in events if e["seed"] == seed and e["lane"] == lane
                    ],
                }
            )
    assert len(sessions) == 8
    manifest = {
        "schema_version": 2,
        "coordinate_conversion": (
            "Original source archives retained; exported positions wrapped "
            "to [-pi, pi). No simulation rerun."
        ),
        "export_repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "export_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "physics_sample_interval_s": 0.02,
        "decision_interval_s": 0.1,
        "pendulum_angle_convention": "[-pi, pi) radians: 0 downward, ±pi upright",
        "selection_rule": "Seed 20261022: upper median duration among noncrossing captures in "
        "each of four groups (ties by lane); slowest downward (ties highest lane); exact downward "
        "rest; largest arm peak among audited crossings with bounded tested options. Seed "
        "20261023: largest arm peak among audited crossings without bounded tested options.",
        "csv_columns": columns,
        "sources": sources,
        "sessions": sessions,
        "sha256": {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified and exported {len(sessions)} complete episodes to {output}", flush=True)


if __name__ == "__main__":
    main()
