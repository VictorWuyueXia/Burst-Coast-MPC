"""Paired long GPU study; arguments: stage (screen/confirm), dwell seconds (0 = CEM only)."""

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

solver, stage, dwell = "cem", sys.argv[1], float(sys.argv[2])
assert stage in ("screen", "confirm") and dwell in (0, 0.5, 1.0)
assert jax.default_backend() == "gpu" and jax.config.x64_enabled
root = Path(f"artifacts/rotary_pendulum/inner-mpc/20261022-hybrid-{stage}-{dwell:g}")
machine = root / "machine-scannables"
machine.mkdir(parents=True, exist_ok=False)
(root / "human-readables").mkdir()
bundle = Path("artifacts/rotary_pendulum/1-TD3_20261016/machine-scannables")
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
original = Path("artifacts/rotary_pendulum/inner-mpc/20261019-gpu-closed-cem/machine-scannables")
design = json.loads((original / "design.json").read_text())
plant = design["configurations"][3]["plant"] | dict(
    speed_cost_per_s=0.0, speed_width_rad=float(np.deg2rad(5))
)
archive = np.load(original / "starts.npz", allow_pickle=False)
count, decisions = (8, 400) if stage == "screen" else (32, 400)
initial, groups = archive["states"][:count].copy(), archive["groups"][:count].copy()
archive.close()
np.savez(machine / "starts.npz", states=initial, groups=groups)
teacher_path = Path(
    "artifacts/rotary_pendulum/stationary-balancing-20261013/teacher-data/cap_4.59/machine-scannables/settings.json"
)
teacher = json.loads(teacher_path.read_text())
assert teacher["torque_limit_nm"] == plant["torque_limit_nm"]
assert teacher["arm_limit_rad"] == plant["observation_scales"][0]
gain = jnp.asarray(teacher["lqr_gain"], jnp.float64)
# Differentiate the existing 100 ms map; normalized action is the LQR input.
a, b = jax.jacfwd(rollout, argnums=(0, 1))(jnp.array([0.0, -jnp.pi, 0.0, 0.0]), jnp.zeros(1), plant)
a, b = a[0, -1], b[0, -1]
poles = jnp.abs(jnp.linalg.eigvals(a - b @ gain[None]))
assert bool(jnp.all(poles < 1))
np.testing.assert_allclose(
    np.sort(np.asarray(poles)), np.sort(teacher["lqr_closed_loop_eigenvalues_abs"]), atol=1e-5
)
print(f"Verified saved LQR gain; pole magnitudes={np.asarray(poles)}", flush=True)
metadata = dict(
    solver=solver,
    stage=stage,
    dwell_s=dwell,
    plant=plant,
    decisions=decisions,
    count=count,
    duration_s=decisions / 10,
    horizon=10,
    seed=design["seed"],
    population=1024,
    elites=32,
    generations=16,
    initial_std=0.5,
    model=manifest,
    lqr_gain=teacher["lqr_gain"],
    lqr_poles=np.asarray(poles).tolist(),
    lqr_source=str(teacher_path),
    entry_limits=[float(np.deg2rad(5)), 0.3, 0.4],
    release_limits=[0.25, 2.0],
    devices=[str(d) + " " + d.device_kind for d in jax.devices()],
    termination=(
        "No performance-based stopping; unresolved physical infeasibility "
        "is an explicit execution error"
    ),
    source_sha256={
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            teacher_path,
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
    """Dwell-triggered LQR handoff, CEM recovery, and independent executed-step audit."""
    states, previous, key, was_local, quiet = carry
    beta = jnp.arctan2(-jnp.sin(states[:, 1]), -jnp.cos(states[:, 1]))
    region = (jnp.abs(beta) < 0.25) & (jnp.abs(states[:, 3]) < 2.0)
    local = (
        ((was_local | (quiet >= round(dwell * 10))) & region) if dwell else jnp.zeros(count, bool)
    )

    def optimize(unused: None) -> tuple:
        seeds = initial_plans(states, networks, plant, 10)
        end = rollout(states, previous[:, 1:], plant)[:, -1, -1]
        tail = evaluate(jnp.zeros(1), end, networks, plant, "policy")
        shifted = jnp.concatenate((previous[:, 1:], tail[:, None]), axis=-1)
        seeds = seeds.at[:, 0].set(
            jnp.where(((decision > 0) & ~was_local)[:, None], shifted, seeds[:, 0])
        )
        result = sample_gpu(
            states, seeds, networks, plant, jax.random.fold_in(key, decision), 1024, 32, 16, 0.5
        )
        chosen = jnp.argmin(
            jnp.where(result["accepted"], result["audit"]["objective"], jnp.inf), axis=-1
        )
        return result["plans"][jnp.arange(count), chosen, :-1], result["accepted"].any(axis=-1)

    plans, feasible = jax.lax.cond(
        jnp.all(local), lambda _: (previous, jnp.ones(count, bool)), optimize, None
    )
    error = states.at[:, 1].set(beta)
    command = jnp.clip(-error @ gain, -1.0, 1.0)
    action = jnp.where(local, command, plans[:, 0])
    nominal = rollout(states, action[:, None], plant)[:, 0]
    executed = rollout(states, action[:, None], plant, substeps=20)[:, 0]
    difference = executed[:, 1::2] - nominal
    difference = difference.at[..., 1].set(
        jnp.arctan2(jnp.sin(difference[..., 1]), jnp.cos(difference[..., 1]))
    )
    refinement = jnp.max(jnp.abs(difference), axis=1)
    accepted = (
        (local | feasible)
        & jnp.isfinite(executed).all(axis=(1, 2))
        & (jnp.abs(executed[..., 0]) < plant["observation_scales"][0]).all(axis=1)
        & (refinement <= jnp.array([0.001, 0.001, 0.01, 0.01])).all(axis=1)
    )
    upright = jnp.abs(jnp.arctan2(-jnp.sin(executed[..., 1]), -jnp.cos(executed[..., 1])))
    settled = (
        (upright <= jnp.deg2rad(5))
        & (jnp.abs(executed[..., 2]) <= 0.3)
        & (jnp.abs(executed[..., 3]) <= 0.4)
    ).all(axis=1)
    quiet = jnp.where(settled, quiet + 1, 0)
    records = dict(
        accepted=accepted,
        executed=executed,
        action=action,
        local=local,
        handoff=local & ~was_local,
        release=~local & was_local,
        quiet=quiet,
        solver_called=jnp.broadcast_to(~jnp.all(local), (count,)),
        refinement_error=refinement,
    )
    return (executed[:, -1], plans, key, local, quiet), records


states = jnp.asarray(initial)
seeds = initial_plans(states, networks, plant, 10)
carry = (
    states,
    seeds[:, 0],
    jax.random.PRNGKey(design["seed"]),
    jnp.zeros(count, bool),
    jnp.zeros(count, jnp.int32),
)
print(f"{solver}, dwell={dwell}: compiling full {count}-state decision", flush=True)
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
            f"{solver} dwell={dwell}: t={(decision + 1) / 10:.1f}s, "
            f"inside 5deg={int((error < 5).sum())}/{count}, "
            f"LQR={host['local'].sum()}/{count}, last batch={payload['solve_s'][-1]:.3f}s, "
            f"accepted={host['accepted'].sum()}/{count}",
            flush=True,
        )
    if not host["accepted"].all():
        raise RuntimeError(
            "No accepted feasible plan: archived diagnostic; no unapproved fallback executed"
        )
(machine / "completed.json").write_text(json.dumps(dict(decisions=decisions, count=count)) + "\n")
print(f"Complete: {root}", flush=True)
