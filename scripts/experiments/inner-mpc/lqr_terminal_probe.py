"""GPU diagnostic: frozen terminal bias along the stationary upright arm-angle family."""

# ruff: noqa: E402

import hashlib
import json
import os
import subprocess
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
from rotary_pendulum.mpc.inner_model import evaluate, rollout

assert jax.default_backend() == "gpu" and jax.config.x64_enabled
machine = Path(
    "artifacts/rotary_pendulum/inner-mpc/20261022-lqr-cost-confirm-summary/machine-scannables"
)
machine.mkdir(parents=True, exist_ok=True)
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
original = Path("artifacts/rotary_pendulum/inner-mpc/20261019-gpu-closed-cem/machine-scannables")
design = json.loads((original / "design.json").read_text())
plant = design["configurations"][3]["plant"] | dict(
    speed_cost_per_s=0.0,
    speed_width_rad=float(np.deg2rad(5)),
    discount=1.0,
    torque_cost_per_s=0.0,
    centering_cost_per_s=0.0,
    lqr_state_weights=[
        1 / design["configurations"][3]["plant"]["observation_scales"][0] ** 2,
        1 / 0.15**2,
        0.25,
        0.25,
    ],
)

print("Evaluating 241 stationary upright states on GPU", flush=True)
angles = jnp.linspace(-jnp.pi / 3, jnp.pi / 3, 241)
states = jnp.zeros((241, 4)).at[:, 0].set(angles).at[:, 1].set(-jnp.pi)
value = jax.block_until_ready(evaluate(jnp.zeros((241, 11)), states, networks, plant, "score"))
terminal = value["costs"].max(axis=-1)
stage_cost = 10 * (angles / plant["observation_scales"][0]) ** 2
index = int(jnp.argmin(value["objective"]))
record = dict(
    description=(
        "One-second zero-action cost at upright rest, swept over arm angle; not a solved MPC"
    ),
    grid_count=241,
    grid_range_deg=[-60, 60],
    best_arm_deg=float(jnp.rad2deg(angles[index])),
    best_total=float(value["objective"][index]),
    zero_arm_total=float(value["objective"][120]),
    best_terminal=float(terminal[index]),
    zero_arm_terminal=float(terminal[120]),
    nominal_quadratic_reconstruction_max_error=float(
        jnp.max(jnp.abs(value["objective"] - terminal - stage_cost))
    ),
    all_feasible=bool(value["feasible"].all()),
    model_weights_sha256=manifest["weights_sha256"],
)
np.savetxt(
    machine / "stationary_arm_value.csv",
    np.column_stack(
        (
            np.asarray(jnp.rad2deg(angles)),
            np.asarray(stage_cost),
            np.asarray(terminal),
            np.asarray(value["objective"]),
        )
    ),
    delimiter=",",
    header="arm_deg,quadratic_stage_cost,terminal_cost,total_cost",
    comments="",
)
(machine / "stationary_arm_value.json").write_text(json.dumps(record, indent=2) + "\n")
(machine / "lqr_terminal_probe.py").write_bytes(Path(__file__).read_bytes())
print(json.dumps(record, indent=2), flush=True)

print(
    "Comparing saved CEM and LQR-generated plans at eight identical final-decision states",
    flush=True,
)
archive = np.load(
    "artifacts/rotary_pendulum/inner-mpc/20261022-lqr-cost-fine-screen-16/machine-scannables/chunk-0039.npz"
)
measured = jnp.asarray(archive["stage_states"][-1, :, 0])
saved = jnp.asarray(archive["plans"][-1])
saved_objective = archive["objective"][-1].copy()
archive.close()
teacher = json.loads(
    Path(
        "artifacts/rotary_pendulum/stationary-balancing-20261013/teacher-data/cap_4.59/machine-scannables/settings.json"
    ).read_text()
)
gain = jnp.asarray(teacher["lqr_gain"])
state, commands = measured, []
for _decision in range(10):
    error = state.at[:, 1].set(jnp.arctan2(-jnp.sin(state[:, 1]), -jnp.cos(state[:, 1])))
    action = jnp.clip(-error @ gain, -1, 1)
    commands.append(action)
    state = rollout(state, action[:, None], plant, substeps=20)[:, -1, -1]
lqr_plan = jnp.pad(jnp.stack(commands, axis=1), ((0, 0), (0, 1)))
lqr_audit = jax.block_until_ready(evaluate(lqr_plan, measured, networks, plant, "audit"))
lqr_plan = lqr_plan.at[:, -1].set(lqr_audit["costs"].max(axis=-1))
lqr_audit = jax.block_until_ready(evaluate(lqr_plan, measured, networks, plant, "audit"))
assert bool(jnp.all(lqr_audit["min_constraint"] >= -1e-10))
cem_audit = jax.block_until_ready(evaluate(saved, measured, networks, plant, "audit"))
np.testing.assert_allclose(cem_audit["objective"], saved_objective, rtol=1e-10, atol=1e-10)
comparison = dict(
    state_source="screen-16, decision 399 (39.9 seconds), all eight starts",
    lqr_feasible=np.asarray(lqr_audit["physical_pass"]).tolist(),
    cem_feasible=np.asarray(cem_audit["physical_pass"]).tolist(),
    cem_objective=np.asarray(cem_audit["objective"]).tolist(),
    lqr_objective=np.asarray(lqr_audit["objective"]).tolist(),
    lqr_lower=int(jnp.sum(lqr_audit["objective"] < cem_audit["objective"])),
    mean_cem=float(jnp.mean(cem_audit["objective"])),
    mean_lqr=float(jnp.mean(lqr_audit["objective"])),
)
(machine / "lqr_plan_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
np.savez_compressed(
    machine / "lqr_plan_comparison.npz",
    states=np.asarray(measured),
    cem_plans=np.asarray(saved),
    lqr_plans=np.asarray(lqr_plan),
)
(machine / "lqr_terminal_probe.py").write_bytes(Path(__file__).read_bytes())
print(json.dumps(comparison, indent=2), flush=True)

print("Checking near-boundary executed intervals at 0.125 and 0.0625 ms", flush=True)
critical_states, critical_actions, interval_ids = [], [], []
for chunk, path in enumerate(
    sorted(
        Path(
            "artifacts/rotary_pendulum/final-results/rl-mpc/machine-scannables/session-records"
        ).glob("chunk-*.npz")
    )
):
    archive = np.load(path)
    near = np.rad2deg(np.abs(archive["executed"][..., 0]).max(axis=-1)) > 134
    critical_states.append(archive["stage_states"][..., 0, :][near])
    critical_actions.append(archive["plans"][..., 0][near])
    ids = np.argwhere(near)
    ids[:, 0] += chunk * 10
    interval_ids.append(ids)
    archive.close()
critical_states = jnp.asarray(np.concatenate(critical_states))
critical_actions = jnp.asarray(np.concatenate(critical_actions))[:, None]
interval_ids = np.concatenate(interval_ids)
checks = []
for substeps in (160, 320):
    replay = jax.block_until_ready(
        rollout(critical_states, critical_actions, plant, substeps=substeps)
    )
    peak = np.rad2deg(np.asarray(jnp.abs(replay[..., 0]).max(axis=(-2, -1))))
    checks.append(
        dict(
            dt_s=0.02 / substeps,
            count=len(peak),
            peak_arm_deg=float(peak.max()),
            crossings=int((peak >= 135).sum()),
            interval_peak_arm_deg=peak.tolist(),
        )
    )
boundary = dict(
    selection="Executed intervals with recorded arm magnitude above 134 degrees",
    decision_and_lane=interval_ids.tolist(),
    checks=checks,
)
(machine / "boundary_replay.json").write_text(json.dumps(boundary, indent=2) + "\n")
(machine / "lqr_terminal_probe.py").write_bytes(Path(__file__).read_bytes())
print(json.dumps(boundary, indent=2), flush=True)
