"""Select archived heuristic cases and test their zero-torque continuations on CPU."""

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from quantities import cycle_metrics, observables

from rotary_pendulum.environment.dynamics import derive_model, rk4_step, state_derivative
from rotary_pendulum.utils.config_schema import RotaryPendulumConfig

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "artifacts/rotary_pendulum/actuation-opportunities/machine-scannables"
SOURCE = ROOT / (
    "artifacts/rotary_pendulum/experiment-results/"
    "single-constant-torque-action-controller/raw-runs/"
    "fine_plant/gain_0.004_work_0.02/machine-scannables"
)
OUTPUT.mkdir(parents=True, exist_ok=True)
physical = RotaryPendulumConfig.model_validate(
    OmegaConf.to_container(OmegaConf.load(ROOT / "src/rotary_pendulum/configs/physics.yaml"))[
        "rotary-pendulum"
    ]
)
model = derive_model(physical)
rows = list(csv.DictReader((SOURCE / "episodes.csv").open()))
selection = []
for stratum, statistic, label in (
    ("downward", "median", "Typical downward capture"),
    ("downward", "slowest", "Slow downward capture"),
    ("moving", "median", "Typical moving-start capture"),
):
    candidates = sorted(
        [r for r in rows if r["stratum"] == stratum],
        key=lambda r: (float(r["duration_s"]), int(r["lane"])),
    )
    chosen = candidates[len(candidates) // 2] if statistic == "median" else candidates[-1]
    selection.append(dict(label=label, rule=f"{statistic} duration in {stratum}", **chosen))

print("Loading archived 2 ms physics validation; selecting cases by recorded duration.", flush=True)
archive = np.load(SOURCE / "trajectories.npz")
fields = {
    key: archive[key]
    for key in ("physics_x", "physics_active", "time_s", "elapsed_s", "torque_nm", "start_x")
}
records, metrics, provenance = {}, {}, dict(selection=selection, releases=[], checks={})
for index, chosen in enumerate(selection):
    lane = int(chosen["lane"])
    mask = fields["physics_active"][:, lane]
    samples = fields["physics_x"][:, lane][mask]
    x = np.vstack((fields["start_x"][0, lane], samples))
    x[:, 1] = (x[:, 1] + np.pi) % (2 * np.pi) - np.pi
    time = (
        fields["time_s"][:, lane, None]
        - fields["elapsed_s"][:, lane, None]
        + 0.02 * np.arange(1, 6)
    )[mask]
    torque = np.broadcast_to(fields["torque_nm"][:, lane, None], mask.shape)[mask]
    record = dict(
        time_s=np.r_[0, time], **observables(x, np.r_[torque[0], torque], physical, model)
    )
    assert np.allclose(np.diff(record["time_s"]), 0.02, atol=1e-10)
    records[f"controlled_{index}"] = record
    chosen["retained_samples"] = len(x)
    chosen["excluded_inactive_samples"] = int(mask.size - mask.sum())
    print(f"Selected lane {lane}: {chosen['label']}, {time[-1]:.2f} s.", flush=True)

# Three measured release states span low, intermediate, and near-upright total energy.
record = records["controlled_1"]
release_indices = [
    int(np.flatnonzero(record["total_j"] >= level * 2 * model.gravity_torque_nm)[0])
    for level in (0.2, 0.6, 0.95)
]
initial = np.column_stack(
    [record[k] for k in ("theta_rad", "alpha_rad", "omega_rad_s", "nu_rad_s")]
)
initial = initial[release_indices]
for index, sample in enumerate(release_indices):
    provenance["releases"].append(
        dict(
            case=f"coast_{index}",
            source_case="controlled_1",
            source_lane=325,
            source_sample=sample,
            source_time_s=float(record["time_s"][sample]),
            initial_state=initial[index].tolist(),
            energy_fraction=float(record["total_j"][sample] / (2 * model.gravity_torque_nm)),
        )
    )

# Independent initial conditions are integrated together; time evolution remains sequential.
coasts = []
for step in (0.001, 0.0005):
    trajectory = np.empty((round(12 / step) + 1, 3, 4))
    trajectory[0] = initial
    for index in range(1, len(trajectory)):
        trajectory[index] = rk4_step(trajectory[index - 1], 0.0, step, physical, model)
        if index % round(2 / step) == 0:
            print(
                f"Batched torque-off continuations: dt={step:g} s, t={index * step:g}/12 s.",
                flush=True,
            )
    coasts.append(trajectory)
for index in range(3):
    x = coasts[0][:, index]
    records[f"coast_{index}"] = dict(
        time_s=np.arange(len(x)) * 0.001, **observables(x, np.zeros(len(x)), physical, model)
    )
error = coasts[0] - coasts[1][::2]
error[:, :, 1] = np.angle(np.exp(1j * error[:, :, 1]))
provenance["checks"]["step_halving_max_state_error"] = np.max(abs(error), axis=(0, 1)).tolist()
print(
    "Step-halving maximum state errors:",
    provenance["checks"]["step_halving_max_state_error"],
    flush=True,
)
assert np.max(abs(error)) < 1e-4

for name, record in records.items():
    x = np.column_stack([record[k] for k in ("theta_rad", "alpha_rad", "omega_rad_s", "nu_rad_s")])
    perturbation = 1e-5
    plus = state_derivative(x, perturbation, physical, model)
    minus = state_derivative(x, -perturbation, physical, model)
    finite_swing = (
        model.pendulum_inertia_kg_m2 * x[:, 3] * (plus[:, 3] - minus[:, 3]) / (2 * perturbation)
    )
    # Differentiate energy with respect to all four state coordinates independently.
    gradient = np.empty_like(x)
    for coordinate in range(4):
        offset = np.zeros_like(x)
        offset[:, coordinate] = 1e-6
        gradient[:, coordinate] = (
            observables(x + offset, np.zeros(len(x)), physical, model)["total_j"]
            - observables(x - offset, np.zeros(len(x)), physical, model)["total_j"]
        ) / 2e-6
    finite_total = np.sum(gradient * (plus - minus), axis=1) / (2 * perturbation)
    total_rate = np.sum(
        gradient * state_derivative(x, record["torque_nm"], physical, model), axis=1
    )
    metrics[name] = cycle_metrics(record)
    metrics[name]["signed_state_phase_reverse_fraction"] = float(
        np.mean(np.diff(record["psi_rad"]) <= 0)
    )
    if name.startswith("controlled"):
        replayed = x[:-1].copy()
        for _ in range(10):
            replayed = rk4_step(replayed, record["torque_nm"][1:], 0.002, physical, model)
        replay_error = replayed - x[1:]
        replay_error[:, 1] = np.angle(np.exp(1j * replay_error[:, 1]))
        metrics[name]["archive_one_interval_error"] = float(np.max(abs(replay_error)))
        assert metrics[name]["archive_one_interval_error"] < 1e-7
    metrics[name]["swing_sensitivity_error"] = float(
        np.max(abs(finite_swing - record["s_swing_rad_s"]))
    )
    metrics[name]["total_sensitivity_error"] = float(np.max(abs(finite_total - x[:, 2])))
    metrics[name]["total_power_identity_error_w"] = float(
        np.max(abs(total_rate - record["total_rate_w"]))
    )
    assert metrics[name]["swing_sensitivity_error"] < 1e-7
    assert metrics[name]["total_sensitivity_error"] < 1e-6
    assert metrics[name]["total_power_identity_error_w"] < 1e-7
    if name.startswith("coast"):
        metrics[name]["relative_energy_drift"] = float(
            np.ptp(record["total_j"]) / record["total_j"][0]
        )
        metrics[name]["momentum_drift_nms"] = float(np.ptp(record["momentum_nms"]))
        assert metrics[name]["relative_energy_drift"] < 1e-6
        assert metrics[name]["momentum_drift_nms"] < 1e-8
    # Verify that reflecting the swing angle leaves the proposed map unchanged.
    x[:, 1] *= -1
    mirrored = observables(x, record["torque_nm"], physical, model)
    metrics[name]["mirror_map_error"] = float(
        np.max(abs(mirrored["s_swing_rad_s"] - record["s_swing_rad_s"]))
    )
    assert metrics[name]["mirror_map_error"] == 0
    np.savez_compressed(OUTPUT / f"{name}.npz", **record)
    np.savetxt(
        OUTPUT / f"{name}.csv",
        np.column_stack(list(record.values())),
        delimiter=",",
        header=",".join(record),
        comments="",
    )
    print(
        f"{name}: {metrics[name]['positive_downward_crossings']} section crossings; "
        f"phase reverse fraction {metrics[name]['phase']['0.01']['reverse_fraction']:.3f}.",
        flush=True,
    )

provenance.update(
    source=str(SOURCE.relative_to(ROOT)),
    source_sha256=hashlib.sha256((SOURCE / "trajectories.npz").read_bytes()).hexdigest(),
    physical=physical.model_dump(),
    model=asdict(model),
    coast_duration_s=12,
    integration_steps_s=[0.001, 0.0005],
    source_settings=json.loads((SOURCE / "settings.json").read_text()),
)
(OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
(OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
print("Analysis and validation complete.", flush=True)
