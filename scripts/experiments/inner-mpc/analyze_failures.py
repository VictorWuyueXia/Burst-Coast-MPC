"""Classify full saved sessions and inspect physical, plan and solver evidence without reruns."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path("artifacts/rotary_pendulum/inner-mpc")
source = root / "20261020-session-plots/machine-scannables"
output = root / "20261020-failure-analysis"
machine, human = output / "machine-scannables", output / "human-readables"
machine.mkdir(parents=True, exist_ok=True)
human.mkdir(exist_ok=True)
selection = json.loads((source / "selection.json").read_text())
archive = np.load(source / "executed_sessions.npz", allow_pickle=False)
x, torque, time = archive["states"], archive["torque_nm"], archive["time_s"]
error, hold, groups = archive["upright_distance_deg"], archive["hold"], archive["groups"]
archive.close()
assert x.shape == (9, 32, 5001, 4) and time[-1] == 5
near = error < 5
end_dwell = np.cumprod(near[..., ::-1], axis=-1).sum(axis=-1) * 0.001
late = ~hold & (end_dwell >= 0.2) & (np.abs(x[..., -1, 3]) < 0.4)
loss = ~hold & ~late & near[..., 0] & (np.abs(x[..., 0, 3]) < 0.4) & (error.max(-1) > 45)
never = ~hold & ~near.any(axis=-1)
uncaptured = ~hold & ~late & ~loss & ~never
masks = np.stack((hold, never, uncaptured, loss, late))
assert np.all(masks.sum(axis=0) == 1)
category = np.argmax(masks, axis=0)
names = [
    "final_second_hold",
    "not_reached_upright",
    "reached_without_capture",
    "lost_initial_upright",
    "late_settling_window",
]
entries = near[..., 1:] & ~near[..., :-1]
entry_count = entries.sum(axis=-1)
fast_entries = (entries & (np.abs(x[..., 1:, 3]) > 0.4)).sum(axis=-1)
first_entry = np.where(near.any(-1), np.argmax(near, axis=-1) * 0.001, np.nan)
plan_end = np.empty((9, 32, 50, 4))
plan_costs = np.empty((9, 32, 50, 4))
solver_evidence = np.full((9, 32, 50, 4), np.nan)
refinement = np.empty((9, 32, 50, 4))
physics = None

for k, controller in enumerate(selection["controllers"]):
    folder = root / controller["run"] / "machine-scannables"
    design = json.loads((folder / "design.json").read_text())
    current = design["model"]["derived_physics"]
    if physics is None:
        physics = current
    assert current == physics
    config = design["configurations"][controller["configuration"]]
    for step in range(50):
        location = folder / f"config-{controller['configuration']:02d}" / f"step-{step:04d}"
        archive = np.load(location / "plans.npz", allow_pickle=False)
        selected = archive["selected"]
        ids = np.arange(32), selected
        assert archive["accepted"][ids].all() and archive["physical_pass"][ids].all()
        refined = archive["refined"][ids]
        np.testing.assert_allclose(
            refined[:, 0], x[k, :, step * 100 + 1 : (step + 1) * 100 + 1], atol=1e-12
        )
        plan_end[k, :, step] = refined[:, -1, -1]
        plan_costs[k, :, step] = np.stack(
            (
                archive["objective"][ids],
                archive["effort"][ids],
                archive["centering"][ids],
                archive["costs"][ids].max(axis=-1),
            ),
            axis=-1,
        )
        refinement[k, :, step] = archive["refinement_error"][ids]
        archive.close()
        history = np.load(location / "history.npz", allow_pickle=False)
        if config["solver"] == "gpu_sqp":
            flat = np.arange(32) * 3 + selected
            solver_evidence[k, :, step] = np.stack(
                (
                    history["qp_error"][-1, flat],
                    history["stationarity"][-1, flat],
                    (history["step_length"][:, flat] == 0).mean(axis=0),
                    history["objective"][0, flat] - history["objective"][-1, flat],
                ),
                axis=-1,
            )
        else:
            solver_evidence[k, :, step] = np.stack(
                (
                    history["feasible_count"][-1][ids] / config["population"],
                    history["active"][-1][ids],
                    np.mean(history["spreads"][-1][ids], axis=-1),
                    history["objective"][0][ids] - history["objective"][-1][ids],
                ),
                axis=-1,
            )
        history.close()
    print(f"Read all 50 full-batch plans and solver histories: {controller['label']}", flush=True)

inertia = physics["pendulum_inertia_kg_m2"]
gravity = physics["gravity_torque_nm"]
energy = (0.5 * inertia * x[..., 3] ** 2 + gravity * (1 - np.cos(x[..., 1]))) / (2 * gravity)
turns = np.unwrap(x[..., 1], axis=-1) / (2 * np.pi)
terminal_error = np.rad2deg(
    np.abs(np.arctan2(-np.sin(plan_end[..., 1]), -np.cos(plan_end[..., 1])))
)
predicted_slow = (terminal_error[:, :, :41] < 5) & (np.abs(plan_end[:, :, :41, 3]) < 0.4)
realized = x[:, :, np.arange(10, 51) * 100]
actual_slow = (error[:, :, np.arange(10, 51) * 100] < 5) & (np.abs(realized[..., 3]) < 0.4)
rows = []
for k, controller in enumerate(selection["controllers"]):
    for episode in range(32):
        rows.append(
            dict(
                controller=k,
                label=controller["label"],
                episode=episode,
                group=int(groups[episode]),
                category=names[category[k, episode]],
                final_hold=bool(hold[k, episode]),
                first_upright_s=float(first_entry[k, episode])
                if np.isfinite(first_entry[k, episode])
                else None,
                final_dwell_s=float(end_dwell[k, episode]),
                entries=int(entry_count[k, episode]),
                fast_entries=int(fast_entries[k, episode]),
                initial_error_deg=float(error[k, episode, 0]),
                final_error_deg=float(error[k, episode, -1]),
                final_speed_rad_s=float(x[k, episode, -1, 3]),
                initial_energy_ratio=float(energy[k, episode, 0]),
                final_energy_ratio=float(energy[k, episode, -1]),
                median_last_energy_ratio=float(np.median(energy[k, episode, -1000:])),
                peak_arm_deg=float(np.max(np.abs(np.rad2deg(x[k, episode, :, 0])))),
                saturated_fraction=float(np.mean(np.abs(torque[k, episode]) > 0.99 * 0.00459)),
                predicted_slow_count=int(predicted_slow[k, episode].sum()),
                predicted_slow_not_realized_count=int(
                    (predicted_slow[k, episode] & ~actual_slow[k, episode]).sum()
                ),
                actual_slow_count=int(actual_slow[k, episode].sum()),
            )
        )
stream = (machine / "sessions.csv").open("w", newline="")
writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
writer.writeheader()
writer.writerows(rows)
stream.close()
# Readable examples chosen from the best aggregate controller, with explicit selection reasons.
representatives = [
    dict(
        controller=3,
        episode=24,
        category=names[1],
        reason="Median downward case in prior matched figure",
    ),
    dict(
        controller=3,
        episode=5,
        category=names[2],
        reason="Moving swing-up with a transient near-upright visit",
    ),
    dict(
        controller=3,
        episode=6,
        category=names[3],
        reason="Initially near upright, subsequently large departures",
    ),
    dict(
        controller=3,
        episode=29,
        category=names[4],
        reason="Late slow arrival; insufficient recorded dwell",
    ),
]
for r in representatives:
    assert names[category[r["controller"], r["episode"]]] == r["category"]
summary = dict(
    categories=names,
    counts=masks.sum(axis=(1, 2)).tolist(),
    counts_by_controller=masks.sum(axis=-1).T.tolist(),
    controllers=selection["controllers"],
    representatives=representatives,
    physical_constants=physics,
    criteria=dict(
        upright_deg=5,
        scored_final_s=1,
        diagnostic_late_dwell_s=0.2,
        diagnostic_slow_rad_s=0.4,
        diagnostic_departure_deg=45,
    ),
    accepted_replans=14400,
    recorded_duration_s=5,
    early_terminated_closed_loop_episodes=0,
    max_refinement_error=refinement.max(axis=(0, 1, 2)).tolist(),
    predicted_slow_count=int(predicted_slow.sum()),
    predicted_slow_not_realized_count=int((predicted_slow & ~actual_slow).sum()),
    note=(
        "Categories are retrospective descriptions, never stop conditions; "
        "conditional plans are not forecasts after replanning"
    ),
    source_sha256=hashlib.sha256((source / "executed_sessions.npz").read_bytes()).hexdigest(),
)
# Check threshold sensitivity without changing the original final-second hold definition.
summary["sensitivity"] = []
for dwell, speed in ((0.1, 0.4), (0.2, 0.4), (0.5, 0.4), (0.2, 1.0)):
    changed = ~hold & (end_dwell >= dwell) & (np.abs(x[..., -1, 3]) < speed)
    summary["sensitivity"].append(
        dict(dwell_s=dwell, speed_rad_s=speed, late_count=int(changed.sum()))
    )
(machine / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
np.savez_compressed(
    machine / "diagnostics.npz",
    time_s=time,
    states=x,
    torque_nm=torque,
    error_deg=error,
    energy_ratio=energy,
    unwrapped_turns=turns,
    categories=category,
    plan_end=plan_end,
    plan_costs=plan_costs,
    solver_evidence=solver_evidence,
    refinement_error=refinement,
    end_dwell_s=end_dwell,
)
(machine / "analyze_failures.py").write_bytes(Path(__file__).read_bytes())
print(
    f"All 288 sessions classified: {dict(zip(names, summary['counts'], strict=True))}", flush=True
)
print(f"Evidence saved to {machine}", flush=True)
