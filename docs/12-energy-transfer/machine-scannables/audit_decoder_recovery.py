"""Reproduce the crossing diagnosis and lossless arm-reversal probes."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rotary_pendulum.environment.dynamics import derive_model, energy_components, rk4_step
from rotary_pendulum.utils.config_schema import PHYSICS_CONFIG_PATH, RotaryPendulumConfig


def main() -> None:
    if any(
        device not in ("", "0", "1", "2", "3")
        for device in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    ):
        raise ValueError("Only physical GPUs 0–3 are authorized")
    directory = Path(__file__).resolve().parent
    root = directory.parents[2] / "artifacts/rotary_pendulum/energy-heuristic"
    physical = RotaryPendulumConfig.model_validate(
        OmegaConf.to_container(OmegaConf.load(PHYSICS_CONFIG_PATH), resolve=True)["rotary-pendulum"]
    )
    assert physical.rotary_damping_nms == physical.pendulum_damping_nms == 0
    model = derive_model(physical)
    records = []
    print("Classifying first crossings in the 192 saved downward confirmation lanes", flush=True)
    for seed in (20261008, 20261009, 20261010):
        campaign = root / f"confirm_{seed}"
        data = np.load(campaign / "energy_soft/machine-scannables/trajectories.npz")
        initial = np.load(campaign / "machine-scannables/initial_states.npz")["x"]
        before = np.concatenate((initial[None], data["x"][:-1]))
        for lane in range(64):
            decision = int(np.flatnonzero(data["arm_violation"][:, lane])[0])
            x = before[decision, lane]
            records.append(
                {
                    "seed": seed,
                    "lane": lane,
                    "decision": decision,
                    "mode": int(data["mode"][decision, lane]),
                    "roots": int(data["root_count"][decision, lane]),
                    "theta_rad": float(x[0]),
                    "omega_rad_s": float(x[2]),
                    "torque_nm": float(data["torque_nm"][decision, lane]),
                    "requested_work_j": float(data["requested_work_j"][decision, lane]),
                }
            )
    handle = (directory / "decoder_first_crossings.csv").open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    handle.close()
    counts = {mode: sum(row["mode"] == mode for row in records) for mode in range(4)}
    print(f"First crossing modes: {counts}", flush=True)

    print("Evaluating 2,049 held torques in parallel from the zero-work reversal probe", flush=True)
    torques = np.linspace(-0.00918, 0.00918, 2049)
    initial = np.array([1.4, 0.0, 1.0, 0.0])
    state = np.broadcast_to(initial, (len(torques), 4)).copy()
    peak = abs(state[:, 0]).copy()
    for _ in range(100):
        state = rk4_step(state, torques, 0.001, physical, model)
        peak = np.maximum(peak, abs(state[:, 0]))
    work = torques * (state[:, 0] - initial[0])
    nonzero = abs(torques) > 0.002
    selected = int(np.argmin(np.where(nonzero, abs(work), np.inf)))
    reversal = {
        "initial_x": initial.tolist(),
        "torque_nm": float(torques[selected]),
        "work_j": float(work[selected]),
        "endpoint_x": state[selected].tolist(),
        "peak_arm_rad": float(peak[selected]),
        "coast_endpoint_x": state[1024].tolist(),
    }
    assert state[selected, 2] < 0 and abs(work[selected]) < 1e-7
    print(f"Near-zero-work pulse: {reversal}", flush=True)

    print("Comparing early and late saturated braking, four probes in parallel", flush=True)
    initial = np.array([[1.4, 0, 1, 0], [1.4, 0, 3, 0], [1.55, 0, 3, 0], [1.2, 0, 3, 0]])
    state = initial.copy()
    peak = abs(state[:, 0]).copy()
    stop_time = np.zeros(4)
    for _ in range(1000):
        active = state[:, 2] > 0
        following = rk4_step(state, -0.00918, 0.001, physical, model)
        state = np.where(active[:, None], following, state)
        peak = np.maximum(peak, abs(state[:, 0]))
        stop_time += 0.001 * active
    assert np.all(state[:, 2] <= 0)
    audit = {
        "source": "confirm_20261008..20261010/energy_soft",
        "first_crossing_modes": counts,
        "nonzero_crossing_torque_count": sum(abs(row["torque_nm"]) > 1e-10 for row in records),
        "outward_accelerating_torque_count": sum(
            row["torque_nm"] * row["omega_rad_s"] > 0 for row in records
        ),
        "torque_cap_nm": 0.00918,
        "fine_step_s": 0.001,
        "reversal": reversal,
        "braking_initial_x": initial.tolist(),
        "braking_peak_arm_rad": peak.tolist(),
        "braking_stop_time_s": stop_time.tolist(),
        "braking_endpoint_x": state.tolist(),
        "small_angle_period_s": model.natural_period_s,
        "scope": "Local probes and saved-trace audit; no new closed-loop controller",
    }
    (directory / "decoder_recovery_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Stopping angles in degrees: {np.degrees(peak).tolist()}", flush=True)

    # Replan from each plant's own state; this is closed loop, not torque replay.
    import jax
    import jax.numpy as jnp

    from rotary_pendulum.heuristic.decoder import decode
    from rotary_pendulum.heuristic.energy import encode, policy

    campaign = json.loads((directory / "campaign_recovery_confirm_20261008.json").read_text())
    settings = campaign["trials"][0]
    indices = np.concatenate(
        [np.arange(start, start + 16) for start in (0, 64, 128, 192)] + [np.arange(256, 260)]
    )
    initial = np.load(root / "confirm_20261008/machine-scannables/initial_states.npz")["x"][indices]
    labels = [label for label in ("downward", "moving", "near", "tight") for _ in range(16)] + [
        "probe"
    ] * 4
    state = np.broadcast_to(initial, (2, *initial.shape)).copy()
    success = np.zeros(state.shape[:2], dtype=bool)
    count = np.zeros_like(success, dtype=int)
    elapsed = np.zeros_like(count)
    peak = abs(state[..., 0]).copy()
    run = jax.jit(
        lambda x: decode(
            x,
            policy(encode(x), settings["work_gain"], settings["kinetic_weight"]),
            jnp.array(settings["work_weight"]),
            campaign["recovery_steps"],
        )
    )
    histories, torques, modes, work_history, balance_history = [], [], [], [], []
    print(
        "Closed-loop refinement: 68 paired lanes at 20 ms / 2 ms, same 100 ms control clock",
        flush=True,
    )
    for decision in range(200):
        before = state.copy()
        torque, diagnostic = jax.device_get(run(jnp.asarray(state)))
        torque = np.where(success, 0.0, torque)
        samples = []
        for _ in range(5):
            active = ~success
            following = state.copy()
            following[0] = rk4_step(state[0], torque[0], 0.02, physical, model)
            for _ in range(10):
                following[1] = rk4_step(following[1], torque[1], 0.002, physical, model)
            state = np.where(active[..., None], following, state)
            beta = np.arctan2(np.sin(state[..., 1] - np.pi), np.cos(state[..., 1] - np.pi))
            inside = (
                (abs(beta) <= 0.08) & (abs(state[..., 2]) <= 0.15) & (abs(state[..., 3]) <= 0.20)
            )
            count = np.where(active, np.where(inside, count + 1, 0), count)
            success |= count >= 5
            elapsed += active
            peak = np.maximum(peak, abs(state[..., 0]))
            samples.append(state.copy())
        work = torque * (state[..., 0] - before[..., 0])
        balance = (
            energy_components(state, physical, model)[2]
            - energy_components(before, physical, model)[2]
            - work
        )
        histories.append(np.stack(samples, axis=-2))
        torques.append(torque)
        modes.append(diagnostic["mode"])
        work_history.append(work)
        balance_history.append(balance)
        if not np.isfinite(state).all() or not np.isfinite(torque).all():
            raise FloatingPointError("Nonfinite closed-loop refinement")
        if (decision + 1) % 10 == 0:
            print(
                f"t={(decision + 1) / 10:.1f}s capture={success.sum(axis=1).tolist()} "
                f"excursions={(peak > np.pi / 2).sum(axis=1).tolist()}",
                flush=True,
            )
    output = root / "recovery_fine_audit/machine-scannables"
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "closed_loop.npz",
        allow_pickle=False,
        initial_x=initial,
        physics_x=np.asarray(histories),
        torque_nm=np.asarray(torques),
        mode=np.asarray(modes),
        work_j=np.asarray(work_history),
        energy_balance_j=np.asarray(balance_history),
        success=success,
        peak_arm_rad=peak,
        elapsed_s=elapsed * 0.02,
        source_lanes=indices,
    )
    rows = [
        {
            "plant_step_s": dt,
            "stratum": label,
            "source_lane": int(indices[lane]),
            "capture": bool(success[plant, lane]),
            "arm_violation": bool(peak[plant, lane] > np.pi / 2),
            "peak_arm_rad": float(peak[plant, lane]),
            "duration_s": float(elapsed[plant, lane] * 0.02),
        }
        for plant, dt in enumerate((0.02, 0.002))
        for lane, label in enumerate(labels)
    ]
    handle = (output / "episodes.csv").open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    handle.close()
    summary = [
        {
            "plant_step_s": dt,
            "stratum": label,
            "episodes": labels.count(label),
            "captures": int(success[plant, np.asarray(labels) == label].sum()),
            "excursions": int((peak[plant, np.asarray(labels) == label] > np.pi / 2).sum()),
        }
        for plant, dt in enumerate((0.02, 0.002))
        for label in dict.fromkeys(labels)
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "settings.json").write_text(json.dumps(campaign, indent=2) + "\n")
    (output / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    (directory / "recovery_fine_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Closed-loop audit saved: {output}", flush=True)


if __name__ == "__main__":
    main()
