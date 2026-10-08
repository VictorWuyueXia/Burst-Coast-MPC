"""Compare 20 ms and 2 ms plants with the doubled-torque, extended-range controller."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.dynamics import rk4_step
from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD
from rotary_pendulum.heuristic.decoder import POLICY_TORQUE_NM, decode
from rotary_pendulum.heuristic.energy import encode, policy


def main() -> None:
    assert np.isclose(float(ARM_LIMIT_RAD), np.pi)
    assert np.isclose(float(POLICY_TORQUE_NM), 0.01836)
    root = Path(__file__).resolve().parents[3]
    source = root / (
        "artifacts/rotary_pendulum/experiment-results/"
        "torque-and-arm-angle-range-comparison/raw-runs/confirmation/machine-scannables"
    )
    output = root / (
        "artifacts/rotary_pendulum/experiment-results/"
        "torque-and-arm-angle-range-comparison/raw-runs/fine/machine-scannables"
    )
    output.mkdir(parents=True, exist_ok=False)
    initial = np.concatenate(
        [
            np.load(source / f"runs/both_{seed}/machine-scannables/initial_states.npz")["x"]
            for seed in (20261008, 20261009, 20261010)
        ]
    )
    labels = (
        [label for label in ("downward", "moving", "near", "tight") for _ in range(64)]
        + ["probe"] * 4
    ) * 3
    state = np.broadcast_to(initial, (2, *initial.shape)).copy()
    success = np.zeros(state.shape[:2], dtype=bool)
    count = np.zeros_like(success, dtype=int)
    elapsed = np.zeros_like(count)
    peak = abs(state[..., 0]).copy()
    run = jax.jit(lambda x: decode(x, policy(encode(x), 0.04, 1.0), jnp.array(1.0), 20))
    histories, torques = [], []
    print("START closed-loop audit: 780 paired lanes, 20 ms / 2 ms plants", flush=True)
    for decision in range(200):
        torque, _ = jax.device_get(run(jnp.asarray(state)))
        torque = np.where(success, 0.0, torque)
        samples = []
        for _ in range(5):
            active = ~success
            following = state.copy()
            following[0] = rk4_step(state[0], torque[0], 0.02, PHYSICAL, MODEL)
            for _ in range(10):
                following[1] = rk4_step(following[1], torque[1], 0.002, PHYSICAL, MODEL)
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
        histories.append(np.stack(samples, axis=-2))
        torques.append(torque)
        assert np.isfinite(state).all() and np.isfinite(torque).all()
        if (decision + 1) % 10 == 0:
            print(
                f"t={(decision + 1) / 10:.1f}s capture={success.sum(axis=1).tolist()} "
                f"excursions={(peak >= np.pi).sum(axis=1).tolist()}",
                flush=True,
            )
    np.savez_compressed(
        output / "closed_loop.npz",
        initial_x=initial,
        physics_x=np.asarray(histories),
        torque_nm=np.asarray(torques),
        success=success,
        peak_arm_rad=peak,
        elapsed_s=elapsed * 0.02,
    )
    rows = [
        {
            "plant_step_s": dt,
            "seed": (20261008, 20261009, 20261010)[lane // 260],
            "source_lane": lane % 260,
            "stratum": label,
            "capture": bool(success[plant, lane]),
            "arm_violation": bool(peak[plant, lane] >= np.pi),
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
            "excursions": int((peak[plant, np.asarray(labels) == label] >= np.pi).sum()),
        }
        for plant, dt in enumerate((0.02, 0.002))
        for label in dict.fromkeys(labels)
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "settings.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "controller": "both",
                "work_gain": 0.04,
                "kinetic_weight": 1.0,
                "work_weight": 1.0,
                "recovery_steps": 20,
                "control_interval_s": 0.1,
                "capture_and_excursion_sampling_s": 0.02,
                "jax_version": jax.__version__,
                "x64": jax.config.x64_enabled,
                "controller_source": str(Path(decode.__code__.co_filename)),
                "scope": "Each plant replans from its own state; decoder uses 20 ms.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"DONE {output}", flush=True)


if __name__ == "__main__":
    main()
