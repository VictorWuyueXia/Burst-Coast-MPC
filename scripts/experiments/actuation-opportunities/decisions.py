"""Compare recorded heuristic actions against explicit short-horizon energy objectives."""

import json
from pathlib import Path

import numpy as np
from quantities import observables

from rotary_pendulum.environment.dynamics import derive_model, rk4_step
from rotary_pendulum.utils.config_schema import RotaryPendulumConfig

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/rotary_pendulum/actuation-opportunities"
OUT = BASE / "decision-comparison/machine-scannables"
OUT.mkdir(parents=True, exist_ok=True)
provenance = json.loads((BASE / "machine-scannables/provenance.json").read_text())
physical = RotaryPendulumConfig.model_validate(provenance["physical"])
model = derive_model(physical)
target = 2 * model.gravity_torque_nm
archive = np.load(ROOT / provenance["source"] / "trajectories.npz")
lane = 325
active = np.isclose(archive["elapsed_s"][:, lane], 0.1)
x = archive["start_x"][active, lane].copy()
x[:, 1] = (x[:, 1] + np.pi) % (2 * np.pi) - np.pi
actual = archive["torque_nm"][active, lane]
time = archive["time_s"][active, lane] - archive["elapsed_s"][active, lane]
initial = observables(x, actual, physical, model)
limit = 0.9 * physical.torque_limit_nm
# Include the exact recorded action, in addition to the uniform comparison grid.
grid = np.linspace(-limit, limit, 257)
torques = np.column_stack((np.broadcast_to(grid, (len(x), len(grid))), actual))
endpoints = []
print(
    f"Comparing {len(x)} complete recorded decisions, {torques.shape[1]} actions each.", flush=True
)
for step in (0.001, 0.0005):
    states = np.broadcast_to(x[:, None], (*torques.shape, 4)).copy()
    for index in range(round(0.1 / step)):
        states = rk4_step(states, torques, step, physical, model)
        if (index + 1) % round(0.02 / step) == 0:
            print(
                f"Counterfactual integration dt={step:g}, horizon={(index + 1) * step:.3f} s.",
                flush=True,
            )
    endpoints.append(states)
error = endpoints[0] - endpoints[1]
error[..., 1] = np.angle(np.exp(1j * error[..., 1]))
assert abs(error).max() < 1e-5
states = endpoints[1]
terminal = observables(states.reshape(-1, 4), torques.ravel(), physical, model)
energy = terminal["total_j"].reshape(torques.shape)
swing = terminal["swing_j"].reshape(torques.shape)
errors = np.stack(((energy / target - 1) ** 2, (swing / target - 1) ** 2), axis=-1)
weights = np.array([0.25, 1.0, 4.0])
scores = 0.5 * (errors[..., 0, None] + errors[..., 1, None] * weights)
best = np.argmin(scores, axis=1)
rows = np.arange(len(x))
chosen = best[:, 1]
recommended = torques[rows, chosen]
regret = scores[:, -1, 1] - scores[rows, chosen, 1]
instant = -limit * np.sign(initial["c_total"] + initial["c_swing"])
coast = 128
# Reproduce both the archived fine plant and the historical coarse prediction separately.
replay = x.copy()
predicted = x.copy()
for _ in range(50):
    replay = rk4_step(replay, actual, 0.002, physical, model)
for _ in range(5):
    predicted = rk4_step(predicted, actual, 0.02, physical, model)
replay_error = replay - archive["x"][active, lane]
replay_error[:, 1] = np.angle(np.exp(1j * replay_error[:, 1]))
assert abs(replay_error).max() < 1e-8
work = actual * (predicted[:, 0] - x[:, 0])
assert np.max(abs(work - archive["predicted_work_j"][active, lane])) < 1e-10
potential_score = ((1 + np.cos(predicted[:, 1])) / 2) ** 2
assert np.max(abs(potential_score - archive["selected_potential_cost"][active, lane])) < 1e-10
work_score = 0.02 * ((work - archive["requested_work_j"][active, lane]) / (0.04 * target)) ** 2
assert np.max(abs(work_score - archive["selected_work_cost"][active, lane])) < 1e-10
assert np.max(archive["selected_arm_limit_cost"][active, lane]) == 0
legacy = np.broadcast_to(x[:, None], (*torques.shape, 4)).copy()
for _ in range(5):
    legacy = rk4_step(legacy, torques, 0.02, physical, model)
    assert np.max(abs(legacy[..., 0])) < np.pi
legacy_work = torques * (legacy[..., 0] - x[:, 0, None])
legacy_score = ((1 + np.cos(legacy[..., 1])) / 2) ** 2 + 0.02 * (
    (legacy_work - archive["requested_work_j"][active, lane, None]) / (0.04 * target)
) ** 2
assert np.max(abs(legacy_score[:, -1] - potential_score - work_score)) < 1e-10
assert np.all(initial["s_swing_rad_s"] != 0)
# Deterministic illustrative selection: categories are chosen before inspecting point values.
conflict = initial["c_total"] * initial["c_swing"] < 0
candidates = [
    (np.flatnonzero(time >= 0.8), "First decision at or after 0.8 s"),
    (np.argsort(-np.where(conflict, regret, -np.inf)), "Largest loss gap with competing tasks"),
    (
        np.argsort(-np.where(instant * recommended < 0, regret, -np.inf)),
        "Largest loss gap where instantaneous and predicted torque signs differ",
    ),
    (np.flatnonzero(abs(actual) < 1e-10)[::-1], "Latest recorded zero-torque action"),
    (
        np.argsort(abs(initial["s_total_rad_s"]) / abs(initial["s_swing_rad_s"])),
        "Small total sensitivity relative to swing sensitivity",
    ),
    (np.flatnonzero(time >= 15.0), "First decision at or after 15 s"),
]
selection = []
for indices, reason in candidates:
    available = [int(i) for i in indices if int(i) not in [v["index"] for v in selection]]
    assert available
    selection.append(dict(index=available[0], reason=reason))
selection.sort(key=lambda item: time[item["index"]])
for letter, item in zip("ABCDEF", selection, strict=True):
    i = item["index"]
    item.update(
        label=letter,
        time_s=float(time[i]),
        actual_mnm=float(1000 * actual[i]),
        best_mnm=float(1000 * recommended[i]),
        instant_mnm=float(1000 * instant[i]),
        task_conflict=bool(conflict[i]),
        initial_E=float(initial["total_j"][i] / target),
        initial_H=float(initial["swing_j"][i] / target),
        s_E=float(initial["s_total_rad_s"][i]),
        s_H=float(initial["s_swing_rad_s"][i]),
        score_actual=float(scores[i, -1, 1]),
        score_best=float(scores[i, chosen[i], 1]),
        score_coast=float(scores[i, coast, 1]),
        legacy_actual=float(legacy_score[i, -1]),
        legacy_energy_best=float(legacy_score[i, chosen[i]]),
        delta_E_actual_mj=float(1000 * (energy[i, -1] - initial["total_j"][i])),
        delta_H_actual_mj=float(1000 * (swing[i, -1] - initial["swing_j"][i])),
        delta_E_best_mj=float(1000 * (energy[i, chosen[i]] - initial["total_j"][i])),
        delta_H_best_mj=float(1000 * (swing[i, chosen[i]] - initial["swing_j"][i])),
        best_by_weight_mnm=(1000 * torques[i, best[i]]).tolist(),
    )
    print(json.dumps(item), flush=True)
summary = dict(
    lane=lane,
    complete_decisions=len(x),
    omitted_partial_decisions=1,
    duration_s=0.1,
    weights=weights.tolist(),
    selection=selection,
    step_halving_max_error=float(abs(error).max()),
    archive_replay_max_error=float(abs(replay_error).max()),
    coarse_grid_max_loss_gap=float(
        np.max(np.min(scores[:, :257:2, 1], axis=1) - np.min(scores[:, :257, 1], axis=1))
    ),
    actual_vs_best_opposite_sign_count=int(np.sum(actual * recommended < -1e-12)),
    instant_vs_best_opposite_sign_count=int(np.sum(instant * recommended < -1e-12)),
    regret_mean=float(regret.mean()),
    regret_max=float(regret.max()),
    weight_extremes_opposite_sign_count=int(
        np.sum(torques[rows, best[:, 0]] * torques[rows, best[:, 2]] < -1e-12)
    ),
    all_candidate_peak_endpoint_arm_deg=float(np.rad2deg(abs(states[..., 0])).max()),
)
np.savez_compressed(
    OUT / "comparison.npz",
    time_s=time,
    state=x,
    torques=torques,
    scores=scores,
    legacy_score=legacy_score,
    best=best,
    energy_j=energy,
    swing_j=swing,
    actual=actual,
    recommended=recommended,
    instant=instant,
    s_E=initial["s_total_rad_s"],
    s_H=initial["s_swing_rad_s"],
    c_E=initial["c_total"],
    c_H=initial["c_swing"],
    initial_E=initial["total_j"],
    initial_H=initial["swing_j"],
    swing_drift_w=initial["swing_drift_w"],
    target_j=target,
)
np.savetxt(
    OUT / "decisions.csv",
    np.column_stack(
        (
            time,
            actual,
            recommended,
            instant,
            initial["s_total_rad_s"],
            initial["s_swing_rad_s"],
            regret,
        )
    ),
    delimiter=",",
    header="time_s,actual_nm,recommended_nm,instant_nm,s_E,s_H,loss_gap",
    comments="",
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("Validated all comparisons and exported numerical results.", flush=True)
