"""Execute a frozen study with fully batched GPU numerical solvers; argument: run directory."""

# ruff: noqa: E402

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# This gate must run before any JAX import can create a CUDA context.
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
if os.environ["JAX_PLATFORMS"] != "cuda" or len(selected) != 1:
    raise ValueError("Use JAX_PLATFORMS=cuda and one idle CUDA_VISIBLE_DEVICES device")
print(occupancy, flush=True)

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_dynamics import PHYSICAL
from rotary_pendulum.mpc.gpu_cem import initial_plans, sample_gpu
from rotary_pendulum.mpc.gpu_sqp import solve_gpu
from rotary_pendulum.mpc.inner_model import evaluate, rollout


def main() -> None:
    """Compile once per shape, synchronize timing, archive and execute accepted first actions."""
    root = Path(sys.argv[1])
    machine = root / "machine-scannables"
    if (machine / "solves.csv").exists():
        raise ValueError("Prepare a fresh directory; results cannot be overwritten")
    if not jax.config.x64_enabled or jax.default_backend() != "gpu":
        raise ValueError("Require GPU and JAX_ENABLE_X64=true")
    metadata = json.loads((machine / "design.json").read_text())
    for source, digest in metadata["source_sha256"].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed: {source}; prepare again")
    manifest = metadata["model"]
    if PHYSICAL.model_dump() != manifest["physical_config"]:
        raise ValueError("Physics differs from frozen checkpoint")
    bundle = Path("artifacts/rotary_pendulum/1-TD3_20261016/machine-scannables")
    if (
        hashlib.sha256((bundle / "weights.npz").read_bytes()).hexdigest()
        != manifest["weights_sha256"]
    ):
        raise ValueError("Checkpoint checksum mismatch")
    archive = np.load(bundle / "weights.npz", allow_pickle=False)
    networks = {}
    for name in ("actor", "q1", "q2"):
        networks[name] = {
            "params": {
                f"Dense_{i}": {
                    field: jnp.asarray(archive[f"{name}.{i}.{field}"], jnp.float64)
                    for field in ("kernel", "bias")
                }
                for i in range(len(manifest["networks"][name]) - 1)
            }
        }
    archive.close()
    networks = dict(actor=networks["actor"], critics=(networks["q1"], networks["q2"]))
    archive = np.load(machine / "starts.npz", allow_pickle=False)
    initial, groups = jnp.asarray(archive["states"]), archive["groups"].copy()
    archive.close()
    count = len(initial)
    metadata.update(
        devices=[str(d) + " " + d.device_kind for d in jax.devices()],
        timing="Synchronized whole batch incl. 1 ms audit; per-state throughput is batch/count",
    )
    (machine / "design.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows, timings = [], []
    for config_id, settings in enumerate(metadata["configurations"]):
        solver, horizon, plant = settings["solver"], settings["horizon"], settings["plant"]
        if plant["rl_cost"] or plant["centering_cost_per_s"] < 0:
            raise ValueError("GPU comparison requires primary objective and nonnegative centering")
        function = {"gpu_sqp": solve_gpu, "gpu_cem": sample_gpu}[solver]
        arguments = (
            {name: settings[name] for name in ("iterations", "qp_iterations", "step_limit", "rho")}
            if solver == "gpu_sqp"
            else dict(
                population=settings["population"],
                elites=settings["elite_count"],
                generations=settings["generations"],
                initial_std=settings["initial_std"],
            )
        )
        states, alive = initial, jnp.ones(count, bool)
        key = jax.random.PRNGKey(metadata["seed"])
        print(
            f"Config {config_id}/{len(metadata['configurations'])}: {settings}; compiling",
            flush=True,
        )
        started = time.perf_counter()
        seeds = jax.block_until_ready(initial_plans(states, networks, plant, horizon))
        if solver == "gpu_cem":
            arguments["key"] = key
        compiled = function.lower(states, seeds, networks, plant, **arguments).compile()
        print(f"  compiled in {time.perf_counter() - started:.2f}s; warming full batch", flush=True)
        jax.block_until_ready(function(states, seeds, networks, plant, **arguments))
        timings.append(dict(configuration=config_id, warmup_s=time.perf_counter() - started))
        previous = seeds[:, 0]
        for decision in range(metadata["decisions"]):
            if not bool(alive.any()):
                break
            print(f"  config {config_id}, decision {decision}: batch {count}", flush=True)
            started = time.perf_counter()
            seeds = initial_plans(states, networks, plant, horizon)
            if decision:
                end = rollout(states, previous[:, 1:], plant)[:, -1, -1]
                tail = evaluate(jnp.zeros(1), end, networks, plant, "policy")
                shifted = jnp.concatenate((previous[:, 1:], tail[:, None]), axis=-1)
                seeds = seeds.at[:, 0].set(shifted)
            jax.block_until_ready(seeds)
            seed_s = time.perf_counter() - started
            if solver == "gpu_cem":
                arguments["key"] = jax.random.fold_in(key, decision)
            started = time.perf_counter()
            result = jax.block_until_ready(function(states, seeds, networks, plant, **arguments))
            elapsed = time.perf_counter() - started
            print(f"  solved + audited in {elapsed:.3f}s; archiving", flush=True)
            costs = jnp.where(result["accepted"], result["audit"]["objective"], jnp.inf)
            chosen = jnp.argmin(costs, axis=-1)
            valid = alive & jnp.any(result["accepted"], axis=-1)
            chosen_plans = result["plans"][jnp.arange(count), chosen]
            audit = {
                name: value[jnp.arange(count), chosen] for name, value in result["audit"].items()
            }
            location = machine / f"config-{config_id:02d}" / f"step-{decision:04d}"
            location.mkdir(parents=True)
            np.savez_compressed(location / "input.npz", states=states, seeds=seeds, alive=alive)
            np.savez_compressed(
                location / "plans.npz",
                plans=result["plans"],
                accepted=result["accepted"],
                selected=chosen,
                **result["audit"],
            )
            np.savez_compressed(location / "history.npz", **result["history"])
            host = jax.device_get(dict(audit=audit, alive=alive, valid=valid, states=states))
            for episode in np.flatnonzero(host["alive"]):
                accepted = bool(host["valid"][episode])
                row = dict(
                    configuration=config_id,
                    solver=solver,
                    horizon=horizon,
                    centering_weight=plant["centering_cost_per_s"],
                    episode=int(episode),
                    group=int(groups[episode]),
                    decision=decision,
                    accepted=accepted,
                    elapsed_s=elapsed,
                    batch_size=count,
                    throughput_s=elapsed / count,
                    seed_batch_s=seed_s,
                )
                for name in ("objective", "effort", "centering", "arm_margin_rad"):
                    row[name] = float(host["audit"][name][episode]) if accepted else None
                if accepted:
                    executed = host["audit"]["refined"][episode, 0]
                    row.update(
                        executed_arm_rms_rad=float(np.sqrt(np.mean(executed[:, 0] ** 2))),
                        executed_upright_rms_rad=float(
                            np.sqrt(
                                np.mean(
                                    np.arctan2(-np.sin(executed[:, 1]), -np.cos(executed[:, 1]))
                                    ** 2
                                )
                            )
                        ),
                        predicted_first_rl_cost=float(host["audit"]["rl_stage_costs"][episode, 0]),
                    )
                else:
                    row.update(
                        executed_arm_rms_rad=None,
                        executed_upright_rms_rad=None,
                        predicted_first_rl_cost=None,
                    )
                rows.append(row)
            previous = jnp.where(valid[:, None], chosen_plans[:, :-1], previous)
            states = jnp.where(valid[:, None], audit["refined"][:, 0, -1], states)
            alive = valid
            stream = (machine / "solves.csv").open("w", newline="")
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.close()
            print(f"  accepted {int(valid.sum())}/{count}; decision complete", flush=True)
        del compiled
        (machine / "warmup.json").write_text(json.dumps(timings, indent=2) + "\n")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("report.py")), str(root)], check=True
    )
    print(f"Finished {len(rows)} attempted solves: {root}", flush=True)


if __name__ == "__main__":
    main()
