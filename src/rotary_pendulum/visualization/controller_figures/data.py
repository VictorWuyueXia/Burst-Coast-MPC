"""Recorded-state selection and physical geometry for the controller figures."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rotary_pendulum.environment.dynamics import derive_model, energy_components, rk4_step
from rotary_pendulum.utils.config_schema import PHYSICS_CONFIG_PATH, RotaryPendulumConfig

PHYSICAL = RotaryPendulumConfig.model_validate(
    OmegaConf.to_container(OmegaConf.load(PHYSICS_CONFIG_PATH))["rotary-pendulum"]
)
MODEL = derive_model(PHYSICAL)
TARGET = 2 * MODEL.gravity_torque_nm
LIMIT = 0.9 * PHYSICAL.torque_limit_nm
SETTINGS = json.loads((PHYSICS_CONFIG_PATH.parent / "heuristic.json").read_text())["trials"][0]


def predict(state: np.ndarray, torques: np.ndarray) -> dict[str, np.ndarray]:
    """Predict independent 100 ms held-torque endpoints with the existing physical model."""
    endpoint = np.broadcast_to(state, (*torques.shape, 4)).copy()
    peak = np.abs(endpoint[..., 0])

    # Only the five time steps are sequential; all torque trials advance in one batch.
    for _ in range(5):
        endpoint = rk4_step(endpoint, torques, 0.02, PHYSICAL, MODEL)
        peak = np.maximum(peak, np.abs(endpoint[..., 0]))
    kinetic, potential, _ = energy_components(endpoint, PHYSICAL, MODEL)
    arm = 0.5 * MODEL.arm_inertia_kg_m2 * endpoint[..., 2] ** 2
    energy = np.stack((arm, kinetic - arm, potential), axis=-1)
    work = torques * (endpoint[..., 0] - state[..., 0])
    assert np.isfinite(endpoint).all() and np.isfinite(energy).all()
    return {
        "state": endpoint,
        "energy": energy / TARGET,
        "work": work,
        "peak": peak,
    }


def select_examples(records: Path, output: Path) -> list[dict]:
    """Compare every complete decision in the preserved validation subset."""
    manifest = json.loads((records / "manifest.json").read_text())
    for source in manifest["sources"]:
        provenance = records / f"provenance/validation_seed_{source['seed']}"
        physical = OmegaConf.to_container(OmegaConf.load(provenance / "physics.yaml"))
        assert RotaryPendulumConfig.model_validate(physical["rotary-pendulum"]) == PHYSICAL
        settings = json.loads((provenance / "controller_settings.json").read_text())
        for key in ("work_gain", "work_weight", "arm_limit_weight"):
            assert settings[key] == SETTINGS[key]
    examples = []
    counts = {}
    for session in manifest["sessions"]:
        path = records / f"{session['name']}.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest["sha256"][path.name], path
        replay = np.load(path, allow_pickle=False)
        indices = np.flatnonzero(np.isclose(replay["elapsed_s"], 0.1))
        counts[session["name"]] = {"stored": len(replay["elapsed_s"]), "complete": len(indices)}
        for index in indices:
            examples.append({
                "case": session["name"], "seed": session["seed"],
                "lane": session["source_lane"], "decision": int(index),
                "time_s": float(replay["time_s"][index] - replay["elapsed_s"][index]),
                "state": replay["start_x"][index],
                "request": float(replay["requested_work_j"][index]),
                "selected": float(replay["torque_nm"][index]),
                "logged_energy": replay["predicted_energy_j"][index], "sha256": digest,
            })
        replay.close()

    # Compare the full bounded sweep; only explanatory clarity determines selection.
    states = np.array([row["state"] for row in examples])
    request = np.array([row["request"] for row in examples])
    grid = np.broadcast_to(np.linspace(-LIMIT, LIMIT, 33), (len(states), 33))
    prediction = predict(states[:, None, :], grid)
    kinetic, potential, total = energy_components(states, PHYSICAL, MODEL)
    arm = 0.5 * MODEL.arm_inertia_kg_m2 * states[:, 2] ** 2
    current = np.column_stack((arm, kinetic - arm, potential)) / TARGET
    np.testing.assert_allclose(request, np.clip(SETTINGS["work_gain"]*(TARGET-total),
                               -0.04*TARGET, 0.04*TARGET), rtol=0, atol=1e-14)
    error = prediction["work"] - request[:, None]
    changes = error[:, :-1] * error[:, 1:] < 0
    drift = prediction["energy"].sum(-1) - total[:, None]/TARGET - prediction["work"]/TARGET
    rows = []
    for index, example in enumerate(examples):
        brackets = np.flatnonzero(changes[index])
        eligible = (len(brackets) == 2 and 0.25 < total[index]/TARGET < 1.05
                    and np.max(prediction["peak"][index]) < np.pi
                    and np.max(np.abs(drift[index])) < 5e-6)
        separation = 0.0
        if eligible:
            fraction = -error[index, brackets] / np.diff(error[index])[brackets]
            roots = grid[index, brackets] + fraction * np.diff(grid[index])[brackets]
            points = prediction["energy"][index, brackets] + fraction[:, None] * np.diff(
                prediction["energy"][index], axis=0)[brackets]
            separation = np.linalg.norm(points[1]-points[0]) / np.linalg.norm(
                np.ptp(prediction["energy"][index], axis=0))
            eligible = bool(np.max(np.abs(roots)) < 0.85*LIMIT
                            and np.min(np.abs(roots-example["selected"])) > 0.12*LIMIT)
        example["current"] = current[index]
        example["selection_score"] = float(separation) if eligible else 0.0
        rows.append({**{key: example[key] for key in
                       ("case", "seed", "lane", "decision", "time_s")},
                     "root_brackets": len(brackets), "eligible": eligible,
                     "relative_root_separation": separation,
                     "energy_balance_max_j": float(np.max(np.abs(drift[index]))*TARGET),
                     "current_total_over_target": float(total[index]/TARGET),
                     "requested_work_over_target": float(request[index]/TARGET)})
    handle = (output / "candidate_comparison.csv").open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    handle.close()
    ranked = sorted((row for row in examples if row["selection_score"] > 0),
                    key=lambda row: -row["selection_score"])
    assert len(ranked) >= 3, "The preserved records must supply three eligible illustrations."
    (output / "selection_audit.json").write_text(json.dumps({
        "episode_counts": counts, "complete_decisions": len(examples), "eligible": len(ranked),
        "selection": "Largest root-energy separation relative to the full curve extent; "
                     "two interior work matches, distinct selection, bounded arm, low RK4 drift.",
        "criteria": {"total_energy_over_target": [0.25, 1.05], "root_abs_torque_over_limit": 0.85,
                     "selected_root_torque_gap_over_limit": 0.12, "peak_arm_rad": float(np.pi),
                     "max_energy_balance_over_target": 5e-6, "root_brackets": 2},
        "scope": "Eight preserved episodes from validation seeds 20261022 and 20261023; "
                 "full raw batch archives are not present locally.",
        "maximum_abs_recorded_request_over_target": float(np.max(np.abs(request))/TARGET),
    }, indent=2) + "\n")
    print(f"Compared {len(examples)} decisions; {len(ranked)} eligible examples.", flush=True)
    return ranked


def physical_energy_plane(example: dict) -> dict:
    """Construct the requested plane and its exact mechanically compatible subset."""
    budget = example["current"].sum() + example["request"]/TARGET
    v = np.linspace(0, min(budget, 1), 401)
    cosine = 1-2*v
    # Completing the velocity square gives the minimum compatible pendulum kinetic energy.
    eta = (MODEL.base_inertia_kg_m2-MODEL.arm_inertia_kg_m2
           + MODEL.pendulum_inertia_kg_m2*(1-cosine**2)
           - (MODEL.coupling_inertia_kg_m2*cosine)**2/MODEL.pendulum_inertia_kg_m2
           )/MODEL.arm_inertia_kg_m2
    a = (budget-v)/(1+eta)
    boundary = np.column_stack((a, budget-v-a, v))
    region = np.vstack((np.column_stack((np.zeros_like(v), budget-v, v)), boundary[::-1]))
    plane = np.array([[budget, 0, 0], [0, budget, 0],
                      [0, budget-v[-1], v[-1]], [budget-v[-1], 0, v[-1]]])

    # Both polygons lie on the same requested total-energy plane.
    np.testing.assert_allclose(region.sum(-1), budget, atol=1e-14, rtol=0)
    np.testing.assert_allclose(plane.sum(-1), budget, atol=1e-14, rtol=0)
    return {"budget": budget, "boundary": boundary, "region": region, "plane": plane}


def prepare_example(example: dict, output: Path) -> dict:
    """Reproduce decoder candidates and construct the exact energy-plane boundary."""
    state, request = example["state"], example["request"]
    grid = np.linspace(-LIMIT, LIMIT, 33)
    error = predict(state, grid)["work"] - request
    brackets = np.flatnonzero(error[:-1] * error[1:] <= 0)
    left, right, left_error = grid[brackets], grid[brackets+1], error[brackets]
    for _ in range(10):
        middle = (left+right)/2
        middle_error = predict(state, middle)["work"] - request
        on_left = left_error * middle_error <= 0
        right = np.where(on_left, middle, right)
        left = np.where(on_left, left, middle)
        left_error = np.where(on_left, left_error, middle_error)
    roots = (left+right)/2
    candidates = np.r_[roots, grid]
    tested = predict(state, candidates)
    score = ((1-tested["energy"][:, 2])**2
             + SETTINGS["work_weight"]*((tested["work"]-request)/(0.04*TARGET))**2
             + SETTINGS["arm_limit_weight"]*(np.maximum(tested["peak"]-np.pi, 0)/np.pi)**2)
    chosen = np.lexsort((-candidates, np.abs(candidates), score))[0]
    np.testing.assert_allclose(candidates[chosen], example["selected"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(tested["energy"][chosen]*TARGET, example["logged_energy"],
                               rtol=0, atol=1e-11)

    # The drawing mesh does not change the decoder's finite candidate set.
    torques = np.linspace(-LIMIT, LIMIT, 513)
    curve = predict(state, torques)
    geometry = physical_energy_plane(example)
    budget, boundary, region, plane = (geometry[key] for key in
                                       ("budget", "boundary", "region", "plane"))
    a, v = boundary[:, 0], boundary[:, 2]
    cosine = 1-2*v

    # Clip the physical polygon to the local curve view without a translucent mesh.
    low, high = curve["energy"].min(0), curve["energy"].max(0)
    extent = high-low
    low, high = low-0.12*extent, high+0.15*extent
    local_region = region.copy()
    for dimension in range(3):
        for limit, sign in ((low[dimension], 1), (high[dimension], -1)):
            distance = sign*(local_region[:, dimension]-limit)
            next_point = np.roll(local_region, -1, axis=0)
            next_distance = np.roll(distance, -1)
            inside = distance >= 0
            crossing = inside != np.roll(inside, -1)
            pieces = np.repeat(local_region[:, None, :], 2, axis=1)
            fraction = distance[crossing]/(distance[crossing]-next_distance[crossing])
            pieces[crossing, 1] += fraction[:, None]*(next_point-local_region)[crossing]
            local_region = pieces[np.column_stack((inside, crossing))]
    assert len(local_region) >= 3

    # Reconstruct the coupling-limited physical states and re-encode their energies.
    omega = np.sqrt(2*a*TARGET/MODEL.arm_inertia_kg_m2)
    nu = -MODEL.coupling_inertia_kg_m2*cosine*omega/MODEL.pendulum_inertia_kg_m2
    reconstructed = np.column_stack((np.zeros_like(v), np.arccos(cosine), omega, nu))
    kinetic, potential, _ = energy_components(reconstructed, PHYSICAL, MODEL)
    recovered = np.column_stack((a, kinetic/TARGET-a, potential/TARGET))
    np.testing.assert_allclose(recovered, boundary, atol=1e-14, rtol=0)
    np.testing.assert_allclose(region.sum(-1), budget, atol=1e-14, rtol=0)
    balance = curve["energy"].sum(-1)*TARGET - example["current"].sum()*TARGET - curve["work"]
    assert np.max(np.abs(balance)) < 5e-6*TARGET
    assert np.max(np.abs(tested["work"][:len(roots)]-request)) < 1e-4*TARGET
    data = {**example, "budget": budget, "boundary": boundary, "region": region, "plane": plane,
            "torques": torques, "curve": curve, "roots": roots, "tested": tested,
            "candidates": candidates, "score": score, "chosen": int(chosen),
            "local_region": local_region, "low": low, "high": high}
    audit = {key: example[key] for key in ("case", "seed", "lane", "decision", "time_s", "sha256")}
    audit.update({"state": state.tolist(), "current_energy_j": (example["current"]*TARGET).tolist(),
                  "requested_work_j": request, "selected_torque_nm": example["selected"],
                  "root_torques_nm": roots.tolist(), "root_work_error_j":
                  (tested["work"][:len(roots)]-request).tolist(), "plane_total_over_target": budget,
                  "energy_balance_max_j": float(np.max(np.abs(balance))),
                  "selected_endpoint_max_error_j": float(np.max(np.abs(
                      tested["energy"][chosen]*TARGET-example["logged_energy"]))),
                  "region_reconstruction_max_error_j":
                  float(np.max(np.abs(recovered-boundary))*TARGET),
                  "physical_inputs": PHYSICAL.model_dump(), "controller_settings": SETTINGS,
                  "action_s": 0.1, "step_s": 0.02})
    (output / "example.json").write_text(json.dumps(audit, indent=2)+"\n")
    np.savetxt(output / "torque_curve.csv", np.column_stack((torques, curve["energy"],
               curve["work"], curve["peak"], balance)), delimiter=",", comments="",
               header="torque_nm,Ka_over_target,Kp_over_target,V_over_target,work_j,peak_arm_rad,balance_j")
    np.savetxt(output / "candidates.csv", np.column_stack((candidates, tested["energy"],
               tested["work"]-request, tested["peak"], score)), delimiter=",", comments="",
               header="torque_nm,Ka_over_target,Kp_over_target,V_over_target,work_error_j,peak_arm_rad,score")
    np.savetxt(output / "physical_boundary.csv", boundary, delimiter=",", comments="",
               header="Ka_over_target,Kp_over_target,V_over_target")
    print(f"Verified {example['case']} at {example['time_s']:.1f} s.", flush=True)
    return data
