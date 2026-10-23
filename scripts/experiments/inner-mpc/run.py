"""Prepare or run paired inner-MPC solver/parameter experiments with full records."""

# ruff: noqa: E402

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if os.environ["JAX_PLATFORMS"] != "cpu":
    occupancy = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    active = [
        int(row.split(",")[0])
        for row in occupancy.splitlines()
        if any(int(value) > 0 for value in row.split(",")[1:])
    ]
    selected = [int(value) for value in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    if len(active) > 4 or any(device in active for device in selected):
        raise RuntimeError(f"GPU launch refused: occupied={active}, requested={selected}")
    print(occupancy, flush=True)

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_dynamics import PHYSICAL
from rotary_pendulum.mpc.inner_model import CONSTRAINT_JAC, OBJECTIVE_GRAD, evaluate, rollout
from rotary_pendulum.mpc.inner_solvers import sample_inner, solve_inner


def main() -> None:
    """Execute a prepared study; keep initialization, search and archival explicit."""
    root = Path(sys.argv[1])
    machine = root / "machine-scannables"
    if (machine / "solves.csv").exists():
        raise ValueError("This study already has solve records; prepare a new output directory")
    if not jax.config.x64_enabled:
        raise ValueError("Use JAX_ENABLE_X64=true")
    metadata = json.loads((machine / "design.json").read_text())
    configurations = metadata["configurations"]
    manifest = metadata["model"]
    plant = configurations[0]["plant"]
    if PHYSICAL.model_dump() != manifest["physical_config"]:
        raise ValueError("Current physical parameters differ from the trained model")
    bundle = Path("artifacts/rotary_pendulum/1-TD3_20261016/machine-scannables")
    if (
        hashlib.sha256((bundle / "weights.npz").read_bytes()).hexdigest()
        != manifest["weights_sha256"]
    ):
        raise ValueError("Portable checkpoint checksum mismatch")
    for source, digest in metadata["source_sha256"].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed after preparation: {source}; prepare a fresh study")
    starts = np.load(machine / "starts.npz", allow_pickle=False)
    initial, groups = starts["states"], starts["groups"]
    starts.close()
    count = len(initial)
    print(f"Loading frozen weights: {jax.devices()}; {len(configurations)} settings", flush=True)
    archive = np.load(bundle / "weights.npz", allow_pickle=False)
    networks = {}
    for name in ("actor", "q1", "q2"):
        layers = {}
        for i in range(len(manifest["networks"][name]) - 1):
            layers[f"Dense_{i}"] = {
                field: jnp.asarray(archive[f"{name}.{i}.{field}"], jnp.float64)
                for field in ("kernel", "bias")
            }
        networks[name] = {"params": layers}
    archive.close()
    networks = dict(actor=networks["actor"], critics=(networks["q1"], networks["q2"]))
    metadata["devices"] = [str(device) + " " + device.device_kind for device in jax.devices()]
    (machine / "design.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records, timings = [], []
    pool = ThreadPoolExecutor(max_workers=4)
    for config_id, settings in enumerate(configurations):
        horizon, solver = settings["horizon"], settings["solver"]
        print(
            f"Configuration {config_id}: {solver}, horizon={horizon}; warming kernels", flush=True
        )
        started = time.perf_counter()
        z, x0 = jnp.zeros(horizon + 1), jnp.asarray(initial[0])
        for output in ("policy", "score", "audit", "constraints", "objective"):
            jax.block_until_ready(evaluate(z, x0, networks, plant, output))
            print(f"  warm {output}", flush=True)
        if solver == "slsqp":
            jax.block_until_ready(OBJECTIVE_GRAD(z, x0, networks, plant, "objective"))
            jax.block_until_ready(CONSTRAINT_JAC(z, x0, networks, plant, "constraints"))
            solve_inner(x0, jnp.zeros((3, horizon)), networks, settings | dict(max_iterations=1))
        else:
            population_z = jnp.zeros((3, settings["population"], horizon + 1))
            jax.block_until_ready(evaluate(population_z, x0, networks, plant, "score"))
        if solver == "cem":
            sample_inner(
                x0,
                jnp.zeros((3, horizon)),
                networks,
                settings | dict(generations=1),
                jax.random.PRNGKey(0),
            )
        # Seed batches and batched audits have separate compilation signatures.
        jax.block_until_ready(evaluate(jnp.zeros((3, horizon + 1)), x0, networks, plant, "score"))
        jax.block_until_ready(evaluate(jnp.zeros((3, horizon + 1)), x0, networks, plant, "audit"))
        timings.append(dict(configuration=config_id, warmup_s=time.perf_counter() - started))
        states = initial.copy()
        alive = np.ones(count, bool)
        previous = np.zeros((count, horizon))
        for decision in range(metadata["decisions"]):
            print(
                f"Configuration {config_id}, decision {decision}: {alive.sum()} active starts",
                flush=True,
            )
            if not alive.any():
                break
            seed_started = time.perf_counter()
            actor_states, actor_actions = jnp.asarray(states), []
            for _ in range(horizon):
                action = evaluate(jnp.zeros(1), actor_states, networks, plant, "policy")
                actor_actions.append(action)
                actor_states = rollout(actor_states, action[:, None], plant)[:, -1, -1]
            seeds = np.stack(
                (
                    np.asarray(jnp.stack(actor_actions, axis=-1)),
                    np.full((count, horizon), 0.5),
                    np.full((count, horizon), -0.5),
                ),
                axis=1,
            )
            if decision:
                tail_states = rollout(jnp.asarray(states), jnp.asarray(previous[:, 1:]), plant)
                tail = evaluate(jnp.zeros(1), tail_states[:, -1, -1], networks, plant, "policy")
                seeds[:, 0] = np.column_stack((previous[:, 1:], tail))
            seed_batch_s = time.perf_counter() - seed_started
            pending = []
            for episode in np.flatnonzero(alive):
                key = jax.random.fold_in(
                    jax.random.PRNGKey(metadata["seed"]), int(episode) * 100000 + decision
                )
                arguments = (
                    jnp.asarray(states[episode]),
                    jnp.asarray(seeds[episode]),
                    networks,
                    settings,
                )
                pending.append(
                    (
                        episode,
                        pool.submit(sample_inner, *arguments, key)
                        if solver == "cem"
                        else pool.submit(solve_inner, *arguments),
                    )
                )
            for episode, future in pending:
                result = future.result()
                location = (
                    machine
                    / f"config-{config_id:02d}"
                    / f"episode-{episode:03d}"
                    / f"step-{decision:04d}"
                )
                location.mkdir(parents=True)
                np.savez(location / "input.npz", state=states[episode], seeds=seeds[episode])
                row = dict(
                    configuration=config_id,
                    solver=solver,
                    horizon=horizon,
                    episode=int(episode),
                    group=int(groups[episode]),
                    decision=decision,
                    accepted=result["accepted"],
                    elapsed_s=result["elapsed_s"],
                    seed_batch_s=seed_batch_s,
                    objective=None,
                    arm_margin_rad=None,
                    effort=None,
                    predicted_first_rl_cost=None,
                )
                for index, (plan, audit) in enumerate(result["plans"]):
                    np.savez_compressed(location / f"candidate-{index}.npz", plan=plan, **audit)
                history = [
                    {
                        name: None if isinstance(value, float) and not np.isfinite(value) else value
                        for name, value in record.items()
                    }
                    for record in result["records"]
                ]
                (location / "solver.json").write_text(
                    json.dumps(
                        dict(selected=result["selected"], records=history),
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                )
                if solver == "cem":
                    for generation, values in enumerate(result["populations"]):
                        np.savez_compressed(location / f"population-{generation}.npz", **values)
                if result["accepted"]:
                    plan, audit = result["plans"][result["selected"]]
                    previous[episode] = plan[:-1]
                    states[episode] = audit["refined"][0, -1]
                    row.update(
                        objective=float(audit["objective"]),
                        arm_margin_rad=float(audit["arm_margin_rad"]),
                        effort=float(audit["effort"]),
                        predicted_first_rl_cost=float(audit["rl_stage_costs"][0]),
                    )
                else:
                    alive[episode] = False
                records.append(row)
                print(
                    f"  episode {episode}: accepted={row['accepted']} "
                    f"elapsed={row['elapsed_s']:.3f}s",
                    flush=True,
                )
            stream = (machine / "solves.csv").open("w", newline="")
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
            stream.close()
        print(f"Configuration {config_id} complete", flush=True)
    pool.shutdown()
    (machine / "warmup.json").write_text(json.dumps(timings, indent=2) + "\n")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("report.py")), str(root)], check=True
    )
    print(f"Completed {len(records)} solves; records in {root}", flush=True)


if __name__ == "__main__":
    main()
