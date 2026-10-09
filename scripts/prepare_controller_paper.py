"""Prepare traceable, embedded figure data for the controller manuscript."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rotary_pendulum.environment.dynamics import derive_model, energy_components, rk4_step
from rotary_pendulum.utils.config_schema import RotaryPendulumConfig


def prepare_replay_tables(records: Path, output: Path) -> tuple[dict, dict]:
    """Recover complete validation outcomes and selected, unmodified replay samples."""
    manifest = json.loads((records / "manifest.json").read_text())
    paths = [
        records / "provenance" / f"validation_seed_{seed}" / "full_validation_episodes.csv"
        for seed in (20261022, 20261023)
    ]
    rows = [row for path in paths for row in csv.DictReader(path.read_text().splitlines())]
    tables, audit = {}, {"outcomes": {}, "source_sha256": {}}
    classes = {"downward": "Down", "moving": "Moving", "near": "Near", "tight": "Tight"}

    # Preserve every observed capture time in the empirical distributions.
    for label, prefix in classes.items():
        group = [row for row in rows if row["stratum"] == label]
        times = np.array([float(row["duration_s"]) for row in group])
        unique, counts = np.unique(times, return_counts=True)
        tables[f"{prefix}Cdf"] = np.column_stack(
            (np.r_[0.0, unique, 20.0], np.r_[0.0, np.cumsum(counts) / len(group), 1.0])
        )
        audit["outcomes"][label] = {
            "episodes": len(group),
            "captured": sum(row["success"] == "True" for row in group),
            "arm_crossings": sum(row["arm_violation"] == "True" for row in group),
            "median_capture_s": float(np.median(times)),
        }
        assert len(group) == 1024 and audit["outcomes"][label]["captured"] == 1024
    assert [audit["outcomes"][key]["arm_crossings"] for key in classes] == [0, 6, 0, 0]
    probes = [row for row in rows if row["stratum"] == "probe"]
    assert len(probes) == 8 and all(row["success"] == "True" for row in probes)
    assert all(row["arm_violation"] == "False" for row in probes)
    audit["fixed_probes"] = {"episodes": 8, "captured": 8, "arm_crossings": 0}
    selected = {
        "Median": "downward_median_capture_time",
        "Slow": "downward_slowest_capture",
        "CrossBounded": "arm_crossing_despite_bounded_tested_torques",
        "CrossUnbounded": "arm_crossing_without_bounded_tested_torque",
    }

    # Read original samples and insert gaps only at discontinuous pendulum wraps.
    for prefix, name in selected.items():
        path = records / f"{name}.csv"
        paths.append(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest["sha256"][path.name]
        data = np.genfromtxt(path, delimiter=",", names=True)
        time = data["time_s"]
        session = next(item for item in manifest["sessions"] if item["name"] == name)
        np.testing.assert_allclose(time[-1], session["duration_s"], atol=1e-12)
        if prefix.startswith("Cross"):
            crossing = np.flatnonzero(np.abs(data["arm_angle_rad"]) >= np.pi)[0]
            local_time = time - time[crossing]
            keep = np.abs(local_time) <= 1.200001
            tables[prefix] = np.column_stack(
                (local_time[keep], np.rad2deg(np.abs(data["arm_angle_rad"][keep])))
            )
            audit[prefix] = {"first_crossing_s": float(time[crossing]), "source_case": name}
            continue
        tables[f"{prefix}Arm"] = np.column_stack((time, np.rad2deg(data["arm_angle_rad"])))
        alpha = np.rad2deg(data["pendulum_angle_rad"])
        wraps = np.flatnonzero(np.abs(np.diff(alpha)) > 180.0) + 1
        tables[f"{prefix}Pend"] = np.column_stack(
            (np.insert(time, wraps, np.nan), np.insert(alpha, wraps, np.nan))
        )
        if prefix == "Median":
            columns = {
                "Torque": "preceding_interval_torque_nm",
                "Ka": "arm_kinetic_energy_j",
                "Kp": "pendulum_kinetic_energy_j",
                "V": "potential_energy_j",
                "E": "total_energy_j",
            }
            for suffix, column in columns.items():
                tables[f"Median{suffix}"] = np.column_stack((time, 1000 * data[column]))
    for path in paths:
        audit["source_sha256"][str(path.relative_to(records))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for name, values in tables.items():
        np.savetxt(output / f"{name}.csv", values, delimiter=",", header="x,y", comments="")
    return tables, audit


def prepare_decoder_curve(records: Path, output: Path) -> tuple[dict, dict]:
    """Evaluate the existing nonlinear model at one recorded decision state."""
    physical_path = records / "provenance/validation_seed_20261022/physics.yaml"
    physical = RotaryPendulumConfig.model_validate(
        OmegaConf.to_container(OmegaConf.load(physical_path))["rotary-pendulum"]
    )
    model = derive_model(physical)
    source = records / "downward_median_capture_time.npz"
    replay = np.load(source)
    state, request = replay["start_x"][20], replay["requested_work_j"][20]
    selected = float(replay["torque_nm"][20])
    target = 2 * model.gravity_torque_nm
    limit = 0.9 * physical.torque_limit_nm
    grid, dense = np.linspace(-limit, limit, 33), np.linspace(-limit, limit, 257)
    torques = np.r_[grid, dense, selected]
    endpoint = np.broadcast_to(state, (len(torques), 4)).copy()
    peak = np.abs(endpoint[:, 0])

    # Predict all torques in a single batch with the decoder's five integration steps.
    for _ in range(5):
        endpoint = rk4_step(endpoint, torques, 0.02, physical, model)
        peak = np.maximum(peak, np.abs(endpoint[:, 0]))
    work = torques * (endpoint[:, 0] - state[0])
    error = work[:33] - request
    brackets = np.flatnonzero(error[:-1] * error[1:] <= 0)
    left, right, left_error = grid[brackets], grid[brackets + 1], error[brackets]

    # Resolve precisely the sign-changing grid brackets used by the controller.
    for _ in range(10):
        middle = (left + right) / 2
        predicted = np.broadcast_to(state, (len(middle), 4)).copy()
        for _ in range(5):
            predicted = rk4_step(predicted, middle, 0.02, physical, model)
        middle_error = middle * (predicted[:, 0] - state[0]) - request
        root_on_left = left_error * middle_error <= 0
        right = np.where(root_on_left, middle, right)
        left, left_error = (
            np.where(root_on_left, left, middle),
            np.where(root_on_left, left_error, middle_error),
        )
    roots = (left + right) / 2
    root_states = np.broadcast_to(state, (len(roots), 4)).copy()
    for _ in range(5):
        root_states = rk4_step(root_states, roots, 0.02, physical, model)

    # Encode predicted endpoints by body energy and compare the selected logged endpoint.
    states = np.concatenate((endpoint, root_states), axis=0)
    kinetic, potential, total = energy_components(states, physical, model)
    arm = 0.5 * model.arm_inertia_kg_m2 * states[:, 2] ** 2
    energies = np.column_stack((arm, kinetic - arm, potential)) / target
    np.testing.assert_allclose(
        energies[len(torques) - 1] * target, replay["predicted_energy_j"][20], rtol=0, atol=1e-12
    )
    assert np.max(peak) < np.pi
    _, _, initial_total = energy_components(state, physical, model)
    budget = float((initial_total + request) / target)
    curve = energies[33:290]
    low, high = np.min(curve[:, :2], axis=0), np.max(curve[:, :2], axis=0)
    corners = np.array([low, [high[0], low[1]], high, [low[0], high[1]]])
    plane = np.column_stack((corners, budget - np.sum(corners, axis=1)))
    tables = {
        "CurveData": curve,
        "PlaneData": plane,
        "RootData": energies[len(torques) :],
        "SelectedData": energies[len(torques) - 1 : len(torques)],
    }
    np.savetxt(
        output / "decoder_curve.csv",
        np.column_stack((dense, curve, peak[33:290])),
        delimiter=",",
        header="torque_nm,Ka_over_target,Kp_over_target,V_over_target,peak_arm_rad",
        comments="",
    )
    audit = {
        "source_case": source.name,
        "decision_index": 20,
        "time_s": 2.0,
        "physical_state": state.tolist(),
        "requested_work_j": float(request),
        "selected_torque_nm": selected,
        "bracket_roots_nm": roots.tolist(),
        "plane_total_over_target": budget,
        "maximum_predicted_arm_rad": float(max(peak)),
        "selected_endpoint_match_atol_j": 1e-12,
        "curve_total_energy_balance_max_j": float(
            max(abs(total[: len(torques)] - initial_total - work))
        ),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    return tables, audit


# Save generated plot data separately from the editable manuscript.
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    paper = root / "docs/15-controller-paper"
    records = root / (
        "artifacts/rotary_pendulum/experiment-results/"
        "single-constant-torque-action-controller/portable-animation-records/records"
    )
    output = paper / "machine-scannables/figure-data"
    output.mkdir(parents=True, exist_ok=True)
    tables, audit = prepare_replay_tables(records, output)
    print(
        "Validated 4,096 randomized episodes and eight probes; prepared replay tables.", flush=True
    )
    geometry, audit["decoder_diagnostic"] = prepare_decoder_curve(records, output)
    tables.update(geometry)
    definitions = []
    for name, values in tables.items():
        coordinates = " ".join(
            "(" + ",".join(f"{value:.8g}" for value in row) + ")" for row in values
        )
        definitions.append(f"\\def\\{name}{{{coordinates}}}")
    colored = np.column_stack((geometry["CurveData"], np.linspace(-18.36, 18.36, 257)))
    definitions.append(
        "\\pgfplotstableread[row sep=\\\\]{x y z u\\\\\n"
        + "\n".join(" ".join(f"{value:.8g}" for value in row) + "\\\\" for row in colored)
        + "\n}\\CurveTable"
    )
    (paper / "figures/data.tex").write_text(
        "% Numerical plot data generated by scripts/prepare_controller_paper.py.\n"
        + "\n".join(definitions)
        + "\n"
    )
    (paper / "machine-scannables/figure_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        "Saved figure data; diagnostic endpoint agrees with the recorded prediction.",
        flush=True,
    )
