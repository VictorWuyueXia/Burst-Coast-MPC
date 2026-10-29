"""Collect selected saved results; run on the source workstation without simulation."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "artifacts/rotary_pendulum"
OUTPUT = SOURCE / "final-results"
manifest = {"scope": "Selected final controllers and energy/torque diagnostics", "files": []}
rename_map = {}

# Portable selected actor and critics. Training histories and unused settings stay local.
print("Collecting selected RL inference package.", flush=True)
for source in sorted((SOURCE / "1-TD3_20261016").rglob("*")):
    if not source.is_file() or source.suffix == ".pyc" or "-CN." in source.name:
        continue
    if source.name in ("training_settings.json", "heldout_comparison.csv"):
        continue
    destination = OUTPUT / "rl-policy" / source.relative_to(SOURCE / "1-TD3_20261016")
    if source.name == "interpretation_summary.md":
        destination = OUTPUT / "rl-policy/machine-scannables/operating_contract.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    rename_map[str(source.relative_to(ROOT))] = str(destination.relative_to(ROOT))

# Keep only the active reward/transition settings used by model parity checks.
source = SOURCE / "1-TD3_20261016/machine-scannables/training_settings.json"
arrays = json.loads(source.read_text())
arrays = {
    field: arrays[field]
    for field in (
        "torque_limit_nm",
        "discount",
        "torque_cost_per_s",
        "arm_cost_per_s",
        "energy_cost_per_s",
        "upright_reward_per_s",
        "upright_widths",
        "deadline_s",
    )
}
(OUTPUT / "rl-policy/machine-scannables/policy_settings.json").write_text(
    json.dumps(arrays, indent=2) + "\n"
)

# Full population metrics; exactly the sessions used by the existing figures.
source = SOURCE / "1-TD3_20261016/machine-scannables/heldout_comparison.csv"
table = list(csv.DictReader(source.open()))
table = [row for row in table if row["run"].startswith("actor_quarter_")]
destination = OUTPUT / "rl-policy/machine-scannables/population_performance.csv"
stream = destination.open("w", newline="")
writer = csv.DictWriter(stream, fieldnames=list(table[0]))
writer.writeheader()
writer.writerows(table)
stream.close()
for name in ("nominal", "disturbed"):
    print(f"Extracting RL {name} representative sessions.", flush=True)
    source = SOURCE / f"stationary-balancing-20261013/actor-heldout/actor_quarter_{name}"
    destination = OUTPUT / f"rl-policy/machine-scannables/{name}"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source / "machine-scannables/plot_selection.json", destination / "plot_selection.json"
    )
    table = list(csv.DictReader((source / "machine-scannables/episodes.csv").open()))
    table = [row for row in table if float(row["horizon_s"]) == 600]
    stream = (destination / "episodes.csv").open("w", newline="")
    writer = csv.DictWriter(stream, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
    stream.close()
    arrays = json.loads((source / "machine-scannables/settings.json").read_text())
    arrays = {
        field: value
        for field, value in arrays.items()
        if field
        in (
            "contract_id",
            "validation_seed",
            "deadline_s",
            "action_mode",
            "controller",
            "torque_limit_nm",
            "arm_limit_rad",
            "integration_dt_s",
            "hold_duration_s",
            "upright_widths",
            "perturbation_times_s",
            "perturbation_delta",
        )
    }
    (destination / "settings.json").write_text(json.dumps(arrays, indent=2) + "\n")
    indices = json.loads((destination / "plot_selection.json").read_text())
    manifest[f"rl_{name}_source_sha256"] = hashlib.sha256(
        (source / "machine-scannables/trajectories.npz").read_bytes()
    ).hexdigest()
    archive = np.load(source / "machine-scannables/trajectories.npz", allow_pickle=False)
    arrays = {}
    for field in archive.files:
        arrays[field] = archive[field][indices] if field == "labels" else archive[field][:, indices]
    np.savez_compressed(destination / "representative_sessions.npz", **arrays)
    archive.close()
    for suffix, label in (("", "sessions"), ("_last10s", "final_ten_seconds")):
        shutil.copy2(
            source.parent / f"human-readables/actor_quarter_{name}{suffix}.png",
            OUTPUT / f"rl-policy/human-readables/{name}_{label}.png",
        )
    manifest[f"rl_{name}_selection"] = indices

# Final-only MPC report and complete accepted session records, including exact source snapshots.
print("Collecting official MPC+RL sessions and figures.", flush=True)
source = SOURCE / "inner-mpc/mpc-rl-transfer"
for path in sorted(source.rglob("*")):
    if not path.is_file() or "-CN." in path.name:
        continue
    if path.name in ("manifest.json", "audit_panel_alignment.py"):
        continue
    destination = OUTPUT / "rl-mpc" / path.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    rename_map[str(path.relative_to(ROOT))] = str(destination.relative_to(ROOT))
source = SOURCE / "inner-mpc/20261022-lqr-cost-fine-confirm-16/machine-scannables"
for path in sorted(source.rglob("*")):
    if not path.is_file() or path.suffix == ".pyc":
        continue
    destination = OUTPUT / "rl-mpc/machine-scannables/session-records" / path.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    rename_map[str(path.relative_to(ROOT))] = str(destination.relative_to(ROOT))
path = OUTPUT / "rl-mpc/machine-scannables/baseline_report.py"
path.write_text(
    path.read_text().replace(
        'output.parent / "20261022-lqr-cost-fine-confirm-16/machine-scannables"',
        'output / "machine-scannables/session-records"',
    )
)

# Final energy-controller validation only: two seeds and the independent fine-plant check.
print("Collecting final energy-controller metrics and representative sessions.", flush=True)
source = SOURCE / "experiment-results/single-constant-torque-action-controller"
for name, label in (
    ("validation", "population_performance"),
    ("validation_seed_20261022_trajectories", "validation_seed_20261022_sessions"),
    ("validation_seed_20261023_trajectories", "validation_seed_20261023_sessions"),
    ("two_millisecond_plant_validation_trajectories", "fine_plant_validation_sessions"),
):
    destination = OUTPUT / f"energy-controller/human-readables/{label}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / f"figures/{name}.png", destination)
for name in ("validation_20261022", "validation_20261023", "fine_plant"):
    destination = OUTPUT / f"energy-controller/machine-scannables/{name}"
    destination.mkdir(parents=True, exist_ok=True)
    path = source / f"raw-runs/{name}/gain_0.004_work_0.02/machine-scannables"
    for field in (
        "episodes.csv",
        "summary.csv",
        "settings.json",
        "plot_selection.json",
        "integration_audit.json",
    ):
        shutil.copy2(path / field, destination / field)
    for field in ("initial_states.npz", "provenance.json"):
        shutil.copy2(path.parent.parent / "machine-scannables" / field, destination / field)
    indices = list(json.loads((destination / "plot_selection.json").read_text()).values())
    manifest[f"energy_{name}_source_sha256"] = hashlib.sha256(
        (path / "trajectories.npz").read_bytes()
    ).hexdigest()
    archive = np.load(path / "trajectories.npz", allow_pickle=False)
    arrays = {field: archive[field][:, indices] for field in archive.files}
    np.savez_compressed(destination / "representative_sessions.npz", **arrays)
    archive.close()
    manifest[f"energy_{name}_selection"] = indices
    print(f"Retained {name}: all episode metrics, four plotted sessions.", flush=True)
path = SOURCE / "0-to-show/single-constant-torque-action-controller/portable-animation-records"
for source in sorted((path / "records").rglob("*")):
    if not source.is_file() or source.suffix == ".pyc":
        continue
    destination = (
        OUTPUT
        / "energy-controller/machine-scannables/animation"
        / source.relative_to(path / "records")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
shutil.copy2(
    path / "interpretation_summary.md",
    OUTPUT / "energy-controller/human-readables/animation_guide.md",
)

# The requested sensitivity-plane study, including finite-action and coast diagnostics.
print("Collecting total-energy versus relative swing-energy torque sensitivities.", flush=True)
rename_map.update(
    {
        "controlled_0": "controlled_typical_downward",
        "controlled_1": "controlled_slow_downward",
        "controlled_2": "controlled_typical_moving",
        "coast_0": "coast_low_energy",
        "coast_1": "coast_intermediate_energy",
        "coast_2": "coast_near_upright_energy",
        "controlled_maps": "controlled_energy_sensitivity_planes",
        "coast_maps": "coast_energy_sensitivity_planes",
        "controlled_states": "controlled_physical_sessions",
        "coast_states": "coast_physical_sessions",
        "task_rates": "energy_task_preferences_and_rates",
    }
)
source = SOURCE / "actuation-opportunities"
for path in sorted(source.rglob("*")):
    if not path.is_file() or "-CN." in path.name or "qa" in path.parts:
        continue
    if path.name in ("run.log", "requirements-qa.txt"):
        continue
    destination = OUTPUT / "energy-torque-sensitivity" / path.relative_to(source)
    if "decision-comparison" in destination.parts:
        destination = Path(str(destination).replace("decision-comparison", "action-comparison"))
    for name, label in rename_map.items():
        if "/" not in name:
            destination = Path(str(destination).replace(name, label))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)

# Rename references inside packaged sensitivity reports and serialized provenance.
for path in sorted((OUTPUT / "energy-torque-sensitivity").rglob("*")):
    if not path.is_file() or path.suffix not in (".md", ".json"):
        continue
    text = path.read_text()
    for name, label in rename_map.items():
        if "/" not in name:
            text = text.replace(name, label)
    text = text.replace("decision-comparison", "action-comparison")
    text = text.replace(
        "artifacts/rotary_pendulum/actuation-opportunities/",
        "artifacts/rotary_pendulum/final-results/energy-torque-sensitivity/",
    )
    path.write_text(text)

# Manifest records content identity and selection without requiring the local full archives.
manifest["renamed_paths"] = rename_map
manifest["coordinate_contract"] = {
    "state_order": ["arm angle", "pendulum angle from downward", "arm speed", "pendulum speed"],
    "units": ["rad", "rad", "rad/s", "rad/s"],
    "representative_session_axis": 1,
    "padding": "Energy-controller physics_active masks must exclude post-capture padding.",
    "sensitivity": "Instantaneous power derivative per torque; swing energy is hinge-relative.",
}
(OUTPUT / "machine-scannables/packaging_provenance.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print(
    "Selected payload collected. Documentation and final checksums are finalized separately.",
    flush=True,
)
