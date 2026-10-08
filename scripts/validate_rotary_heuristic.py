"""Run paired energy-work controllers on one GPU per independent trial."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src/rotary_pendulum/configs/heuristic.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    campaign = json.loads(arguments.config.read_text())
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if any(device not in ("", "0", "1", "2", "3") for device in visible):
        raise ValueError("Only physical GPUs 0–3 are authorized for this project run")
    # Validate resource allocation before imports initialize a JAX device.
    import jax
    import jax.numpy as jnp

    from rotary_pendulum.environment.dynamics import energy_components, rk4_step
    from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL
    from rotary_pendulum.heuristic.artifacts import write_artifacts
    from rotary_pendulum.heuristic.evaluation import evaluate
    from rotary_pendulum.RL.jax_residual_evaluation import validation_resets

    trials = campaign["trials"]
    if PHYSICAL.rotary_damping_nms != 0 or PHYSICAL.pendulum_damping_nms != 0:
        raise ValueError("Energy-work validation requires explicitly zero damping")
    if len(trials) > jax.local_device_count():
        raise ValueError("Provide one visible JAX device per independent trial")
    if campaign["decisions"] % campaign["chunk_decisions"] or campaign["decisions"] > 200:
        raise ValueError("Use complete chunks within the existing 20 s episode deadline")
    if campaign["chunk_decisions"] > 10:
        raise ValueError("Progress chunks must be no longer than one simulated second")
    parameters = np.asarray(
        [[t["work_gain"], t["work_weight"], t["arm_limit_weight"]] for t in trials]
    )
    if not np.isfinite(parameters).all() or np.any(parameters < 0):
        raise ValueError("Controller gains must be finite and nonnegative")
    if len({trial["name"] for trial in trials}) != len(trials):
        raise ValueError("Trial names must be unique")

    root = arguments.output
    machine = root / "machine-scannables"
    machine.mkdir(parents=True, exist_ok=False)
    (machine / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    sources = [
        Path(__file__).resolve().relative_to(Path.cwd()),
        *Path("src/rotary_pendulum/heuristic").glob("*.py"),
        *Path("src/rotary_pendulum/environment").glob("jax_*.py"),
        Path("src/rotary_pendulum/environment/dynamics.py"),
        Path("src/rotary_pendulum/utils/config_schema.py"),
        Path("src/rotary_pendulum/RL/jax_residual_evaluation.py"),
        Path("src/rotary_pendulum/RL/jax_residual_control.py"),
        Path("src/rotary_pendulum/configs/physics.yaml"),
        Path("src/rotary_pendulum/configs/mission.yaml"),
        Path("pyproject.toml"),
    ]
    for source in sources:
        target = machine / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    devices = jax.local_devices()[: len(trials)]
    (machine / "provenance.json").write_text(
        json.dumps(
            {
                "jax_version": jax.__version__,
                "jax_enable_x64": jax.config.x64_enabled,
                "devices": [str(device) for device in devices],
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "damping_nms": [PHYSICAL.rotary_damping_nms, PHYSICAL.pendulum_damping_nms],
                "arm_bound_is_soft": True,
                "decoder_revision": "constant_action_100ms_v2",
                "capture_requires_arm_centering": False,
                "sha256": {
                    str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                    for source in sources
                },
            },
            indent=2,
        )
        + "\n"
    )
    initial, _, labels = validation_resets(campaign["seed"], campaign["episodes_per_stratum"], [20])
    if jax.config.x64_enabled:
        initial = initial._replace(x=initial.x.astype(jnp.float64))
    np.savez_compressed(
        machine / "initial_states.npz",
        allow_pickle=False,
        **jax.device_get(initial._asdict()),
        labels=np.asarray(labels),
    )
    state = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (len(trials), *value.shape)), initial
    )
    run = jax.pmap(
        lambda initial, parameters: evaluate(
            initial,
            parameters,
            campaign["chunk_decisions"],
            campaign["mode"],
        ),
        devices=devices,
    )
    print(
        f"START {len(trials)} parallel trials × {len(labels)} paired lanes on {devices}", flush=True
    )
    print("Compiling first rollout chunk; each subsequent chunk reports outcomes", flush=True)
    chunks = []
    started = time.monotonic()
    for index in range(campaign["decisions"] // campaign["chunk_decisions"]):
        state, trace = run(state, jnp.asarray(parameters))
        host = jax.device_get(trace)
        if not all(np.isfinite(value).all() for value in host.values()):
            raise FloatingPointError(f"Nonfinite trajectory at chunk {index}")
        chunks.append(host)
        captures = np.sum(host["success"][:, -1], axis=-1)
        excursions = np.sum(host["arm_violation"][:, -1], axis=-1)
        print(
            f"t={(index + 1) * campaign['chunk_decisions'] * 0.1:.1f}s "
            f"wall={time.monotonic() - started:.1f}s capture={captures.tolist()} "
            f"excursions={excursions.tolist()}",
            flush=True,
        )
    traces = {name: np.concatenate([chunk[name] for chunk in chunks], axis=1) for name in chunks[0]}
    summary = {}
    for index, trial in enumerate(trials):
        print(f"Writing and plotting {trial['name']}", flush=True)
        summary[trial["name"]] = write_artifacts(
            root / trial["name"],
            {name: value[index] for name, value in traces.items()},
            labels,
            {**campaign, **trial},
        )
        # Independent fine-step replay of evenly spaced complete held actions.
        complete = np.flatnonzero(np.isclose(traces["elapsed_s"][index], 0.1))
        chosen = complete[np.linspace(0, len(complete) - 1, min(256, len(complete)), dtype=int)]
        before = np.concatenate((np.asarray(initial.x)[None], traces["x"][index, :-1]))
        start = before.reshape(-1, 4)[chosen]
        replay = start.copy()
        torque = traces["torque_nm"][index].ravel()[chosen]
        for _ in range(50):
            replay = rk4_step(replay, torque, 0.002, PHYSICAL, MODEL)
        work = torque * (replay[:, 0] - start[:, 0])
        total_change = (
            energy_components(replay, PHYSICAL, MODEL)[2]
            - energy_components(start, PHYSICAL, MODEL)[2]
        )
        difference = replay - traces["x"][index].reshape(-1, 4)[chosen]
        difference[:, 1] = np.arctan2(np.sin(difference[:, 1]), np.cos(difference[:, 1]))
        audit = {
            "sample_count": len(chosen),
            "fine_step_s": 0.002,
            "max_state_difference": np.max(abs(difference), axis=0).tolist(),
            "max_work_difference_j": float(
                np.max(abs(work - traces["work_j"][index].ravel()[chosen]))
            ),
            "max_fine_energy_balance_j": float(np.max(abs(total_change - work))),
            "decision_lane_flat_indices": chosen.tolist(),
        }
        (root / trial["name"] / "machine-scannables" / "integration_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
    (machine / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE wall={time.monotonic() - started:.1f}s; {root}", flush=True)


if __name__ == "__main__":
    main()
