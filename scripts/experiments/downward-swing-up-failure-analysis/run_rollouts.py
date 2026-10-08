"""Run one paired Monte Carlo case with progress, full traces and independent audits."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.dynamics import energy_components, rk4_step
from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL, PHYSICS_DT_S
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, GOAL, EnvState
from rotary_pendulum.heuristic.decoder import POLICY_TORQUE_NM
from rotary_pendulum.heuristic.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    source = Path(__file__).resolve().parent
    case = json.loads((source / "case.json").read_text())
    initial = np.load(arguments.initial)
    root = arguments.output
    root.mkdir(parents=True, exist_ok=False)
    assert jax.config.x64_enabled and jax.default_backend() == "gpu"
    assert PHYSICS_DT_S == 0.02 and GOAL.hold_steps == 5
    assert PHYSICAL.rotary_damping_nms == PHYSICAL.pendulum_damping_nms == 0
    count = len(initial["x"])
    state = EnvState(
        jnp.asarray(initial["x"]),
        jnp.zeros(count, jnp.int32),
        jnp.zeros(count, jnp.int32),
        jnp.zeros(count, bool),
        jnp.zeros(count, bool),
        jnp.zeros(count, bool),
    )
    parameters = jnp.asarray(case["parameters"])
    run = jax.jit(lambda state: evaluate(state, parameters, case["recovery_steps"], 10, "energy"))
    provenance = {
        "case": case,
        "jax_version": jax.__version__,
        "float64": jax.config.x64_enabled,
        "devices": [str(device) for device in jax.devices()],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "initial_sha256": hashlib.sha256(arguments.initial.read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob("*"))
            if path.suffix in (".py", ".yaml", ".json")
        },
        "physics_dt_s": PHYSICS_DT_S,
        "decision_s": 0.1,
        "goal_angle_rad": GOAL.beta_tolerance_rad,
        "goal_samples": GOAL.hold_steps,
        "arm_limit_rad": float(ARM_LIMIT_RAD),
        "torque_limit_nm": POLICY_TORQUE_NM,
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    started = time.monotonic()
    print(f"START {case['name']} lanes={count}; compiling first 1s chunk", flush=True)
    chunks = []
    for index in range(20):
        state, trace = run(state)
        host = jax.device_get(trace)
        assert all(np.isfinite(value).all() for value in host.values())
        chunks.append(host)
        print(
            f"t={index + 1}s wall={time.monotonic() - started:.1f}s "
            f"captures={host['success'][-1].sum()} excursions={host['arm_violation'][-1].sum()}",
            flush=True,
        )
    traces = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
    print(
        "Auditing goal, bounds and 1,024 held actions against independent 2ms integration",
        flush=True,
    )
    assert np.max(abs(traces["torque_nm"])) <= POLICY_TORQUE_NM + 1e-12
    physical = traces["physics_x"].transpose(0, 2, 1, 3).reshape(1000, count, 4)
    active = traces["physics_active"].transpose(0, 2, 1).reshape(1000, count)
    beta = np.arctan2(np.sin(physical[..., 1] - np.pi), np.cos(physical[..., 1] - np.pi))
    goal = (abs(beta) <= GOAL.beta_tolerance_rad) & active
    holds = np.lib.stride_tricks.sliding_window_view(goal, 5, axis=0).all(axis=-1)
    np.testing.assert_array_equal(holds.any(axis=0), traces["success"][-1])
    np.testing.assert_array_equal(
        ((abs(physical[..., 0]) >= ARM_LIMIT_RAD) & active).any(axis=0), traces["arm_violation"][-1]
    )
    complete = np.flatnonzero(np.isclose(traces["elapsed_s"], 0.1))
    selected = complete[np.linspace(0, len(complete) - 1, min(1024, len(complete)), dtype=int)]
    start = traces["start_x"].reshape(-1, 4)[selected]
    replay = start.copy()
    torque = traces["torque_nm"].ravel()[selected]
    for _ in range(50):
        replay = rk4_step(replay, torque, 0.002, PHYSICAL, MODEL)
    work = torque * (replay[:, 0] - start[:, 0])
    total_change = (
        energy_components(replay, PHYSICAL, MODEL)[2] - energy_components(start, PHYSICAL, MODEL)[2]
    )
    difference = replay - traces["x"].reshape(-1, 4)[selected]
    difference[:, 1] = np.arctan2(np.sin(difference[:, 1]), np.cos(difference[:, 1]))
    audit = {
        "sample_count": len(selected),
        "fine_step_s": 0.002,
        "flat_decision_lane_indices": selected.tolist(),
        "max_state_difference": abs(difference).max(axis=0).tolist(),
        "max_work_difference_j": float(abs(work - traces["work_j"].ravel()[selected]).max()),
        "max_fine_energy_balance_j": float(abs(total_change - work).max()),
        "max_20ms_energy_balance_j": float(abs(traces["energy_balance_j"]).max()),
        "goal_recomputed": True,
        "excursions_recomputed": True,
    }
    (root / "integration_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    summary = []
    for label in dict.fromkeys(initial["labels"]):
        chosen = initial["labels"] == label
        success, excursion = traces["success"][-1, chosen], traces["arm_violation"][-1, chosen]
        summary.append(
            {
                "stratum": str(label),
                "episodes": int(chosen.sum()),
                "captures": int(success.sum()),
                "excursions": int(excursion.sum()),
                "clean_captures": int((success & ~excursion).sum()),
            }
        )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    print("Saving complete compressed trajectories", flush=True)
    np.savez_compressed(root / "trajectories.npz", **traces)
    print(f"DONE wall={time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
