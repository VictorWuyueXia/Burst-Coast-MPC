"""Exact discrete LQR stage cost plus frozen RL terminal cost; stage, CEM generations."""

# ruff: noqa: E402

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

occupancy = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
occupied = [
    int(r.split(",")[0])
    for r in occupancy.splitlines()
    if any(int(v) > 0 for v in r.split(",")[1:])
]
selected = int(os.environ["CUDA_VISIBLE_DEVICES"])
if len(occupied) > 4 or selected in occupied:
    raise RuntimeError(f"GPU launch refused: occupied={occupied}, requested={selected}")
print(occupancy, flush=True)

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_dynamics import PHYSICAL
from rotary_pendulum.mpc.gpu_cem import initial_plans, sample_gpu
from rotary_pendulum.mpc.inner_model import evaluate, rollout

solver, stage, generations = "cem", sys.argv[1], int(sys.argv[2])
assert stage in ("screen", "confirm") and generations in (16, 32, 64)
assert jax.default_backend() == "gpu" and jax.config.x64_enabled
root = Path(f"artifacts/rotary_pendulum/runs/rl-mpc/{stage}-{generations}")
machine = root / "machine-scannables"
machine.mkdir(parents=True, exist_ok=False)
(root / "human-readables").mkdir()
bundle = Path("artifacts/rotary_pendulum/final-results/rl-policy/machine-scannables")
manifest = json.loads((bundle / "manifest.json").read_text())
assert PHYSICAL.model_dump() == manifest["physical_config"]
assert (
    hashlib.sha256((bundle / "weights.npz").read_bytes()).hexdigest() == manifest["weights_sha256"]
)
archive = np.load(bundle / "weights.npz", allow_pickle=False)
networks = {
    name: {
        "params": {
            f"Dense_{i}": {
                field: jnp.asarray(archive[f"{name}.{i}.{field}"], jnp.float64)
                for field in ("kernel", "bias")
            }
            for i in range(len(manifest["networks"][name]) - 1)
        }
    }
    for name in ("actor", "q1", "q2")
}
archive.close()
networks = dict(actor=networks["actor"], critics=(networks["q1"], networks["q2"]))
original = Path("artifacts/rotary_pendulum/final-results/rl-mpc/machine-scannables/session-records")
design = json.loads((original / "design.json").read_text())
plant = design["plant"].copy()
archive = np.load(original / "starts.npz", allow_pickle=False)
count, decisions = (8, 400) if stage == "screen" else (32, 400)
initial, groups = archive["states"][:count].copy(), archive["groups"][:count].copy()
archive.close()
np.savez(machine / "starts.npz", states=initial, groups=groups)
metadata = dict(
    solver=solver,
    stage=stage,
    objective="undiscounted discrete LQR stage sum plus max negative frozen twin Q",
    input_weight=1.0,
    plant=plant,
    decisions=decisions,
    count=count,
    duration_s=decisions / 10,
    horizon=10,
    prediction_dt_s=0.001,
    verification_dt_s=0.0005,
    record_dt_s=0.001,
    seed=design["seed"],
    population=1024,
    elites=32,
    generations=generations,
    initial_std=0.5,
    model=manifest,
    devices=[str(d) + " " + d.device_kind for d in jax.devices()],
    termination=(
        "No performance-based stopping; unresolved physical infeasibility "
        "is an explicit execution error"
    ),
    source_sha256={
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            Path("src/rotary_pendulum/mpc/inner_model.py"),
            Path("src/rotary_pendulum/mpc/gpu_cem.py"),
            Path("src/rotary_pendulum/environment/jax_dynamics.py"),
        ]
    },
)
(machine / "design.json").write_text(json.dumps(metadata, indent=2) + "\n")
(machine / "sources").mkdir()
for path in metadata["source_sha256"]:
    (machine / "sources" / Path(path).name).write_bytes(Path(path).read_bytes())


@jax.jit
def advance(carry: tuple, decision: jax.Array) -> tuple:
    """One complete batched MPC decision; return only selected evidence to the host."""
    states, previous, key = carry
    seeds = initial_plans(states, networks, plant, 10)
    end = rollout(states, previous[:, 1:], plant)[:, -1, -1]
    tail = evaluate(jnp.zeros(1), end, networks, plant, "policy")
    shifted = jnp.concatenate((previous[:, 1:], tail[:, None]), axis=-1)
    seeds = seeds.at[:, 0].set(jnp.where(decision > 0, shifted, seeds[:, 0]))
    result = sample_gpu(
        states,
        seeds,
        networks,
        plant,
        jax.random.fold_in(key, decision),
        1024,
        32,
        generations,
        0.5,
    )
    cost = jnp.where(result["accepted"], result["audit"]["objective"], jnp.inf)
    chosen = jnp.argmin(cost, axis=-1)
    ids = jnp.arange(count), chosen
    audit = {name: value[ids] for name, value in result["audit"].items()}
    plans = result["plans"][ids]
    records = dict(
        accepted=result["accepted"].any(axis=-1),
        selected=chosen,
        plans=plans,
        executed=audit["refined"][:, 0, 1::2],
        plan_end=audit["refined"][:, -1, -1],
        objective=audit["objective"],
        effort=audit["effort"],
        centering=audit["centering"],
        upright_tracking=audit["upright_tracking"],
        stage_states=jnp.concatenate((states[:, None], audit["trajectory"][:, :-1, -1]), axis=1),
        speed_pendulum=audit["speed_pendulum"],
        speed_arm=audit["speed_arm"],
        terminal_costs=audit["costs"],
        arm_margin_rad=audit["arm_margin_rad"],
        refinement_error=audit["refinement_error"],
    )
    records.update(
        feasible_count=result["history"]["feasible_count"][:, jnp.arange(count), chosen].T,
        sampling_spread=result["history"]["spreads"][-1, jnp.arange(count), chosen],
    )
    return (records["executed"][:, -1], plans[:, :-1], key), records


states = jnp.asarray(initial)
seeds = initial_plans(states, networks, plant, 10)
carry = (states, seeds[:, 0], jax.random.PRNGKey(design["seed"]))
print(f"{solver}, generations={generations}: compiling full {count}-state decision", flush=True)
started = time.perf_counter()
compiled = advance.lower(carry, jnp.asarray(0)).compile()
print(f"Compiled in {time.perf_counter() - started:.1f}s; warming", flush=True)
jax.block_until_ready(compiled(carry, jnp.asarray(0)))
print(
    f"Warmup done in {time.perf_counter() - started:.1f}s; starting {decisions / 10:g}s", flush=True
)
records, elapsed = [], []
for decision in range(decisions):
    started = time.perf_counter()
    carry, record = jax.block_until_ready(compiled(carry, jnp.asarray(decision)))
    elapsed.append(time.perf_counter() - started)
    host = jax.device_get(record)
    records.append(host)
    if (decision + 1) % 10 == 0 or not host["accepted"].all():
        payload = {name: np.stack([r[name] for r in records]) for name in host}
        payload["solve_s"] = np.array(elapsed)
        np.savez_compressed(machine / f"chunk-{decision // 10:04d}.npz", **payload)
        records, elapsed = [], []
        error = np.rad2deg(
            np.abs(
                np.arctan2(-np.sin(host["executed"][:, -1, 1]), -np.cos(host["executed"][:, -1, 1]))
            )
        )
        print(
            f"{solver} generations={generations}: t={(decision + 1) / 10:.1f}s, "
            f"inside 5deg={int((error < 5).sum())}/{count}, "
            f"last batch={payload['solve_s'][-1]:.3f}s, accepted={host['accepted'].sum()}/{count}",
            flush=True,
        )
    if not host["accepted"].all():
        raise RuntimeError(
            "No accepted feasible plan: archived diagnostic; no unapproved fallback executed"
        )
(machine / "completed.json").write_text(json.dumps(dict(decisions=decisions, count=count)) + "\n")
print(f"Complete: {root}", flush=True)
