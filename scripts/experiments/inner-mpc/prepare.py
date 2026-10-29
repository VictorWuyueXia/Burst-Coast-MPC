"""Prepare a frozen paired solver study without starting computation."""

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--study", choices=("smoke", "parameters", "gpu"), default="smoke")
parser.add_argument("--decisions", type=int, default=1)
parser.add_argument("--seed", type=int, default=20261019)
args = parser.parse_args()
if args.decisions < 1:
    raise ValueError("Use a positive number of decisions")
root = args.output
machine, human = root / "machine-scannables", root / "human-readables"
machine.mkdir(parents=True, exist_ok=False)
human.mkdir()
bundle = Path("artifacts/rotary_pendulum/final-results/rl-policy/machine-scannables")
manifest = json.loads((bundle / "manifest.json").read_text())
base = json.loads((bundle / "policy_settings.json").read_text())
if hashlib.sha256((bundle / "weights.npz").read_bytes()).hexdigest() != manifest["weights_sha256"]:
    raise ValueError("Portable checkpoint checksum mismatch")
plant = {
    name: base[name]
    for name in (
        "torque_limit_nm",
        "discount",
        "torque_cost_per_s",
        "arm_cost_per_s",
        "energy_cost_per_s",
        "upright_reward_per_s",
        "upright_widths",
    )
} | dict(
    observation_scales=manifest["observation"]["divisors"],
    rl_cost=False,
    centering_cost_per_s=1.0,
    speed_cost_per_s=0.0,
    speed_width_rad=float(np.deg2rad(5)),
)
local = [dict(max_iterations=100, tolerance=1e-7), dict(max_iterations=200, tolerance=1e-7)]
sampling = [
    dict(population=256, elite_count=32, generations=3, initial_std=0.5),
    dict(population=1024, elite_count=64, generations=5, initial_std=0.5),
]
if args.study == "parameters":
    local.append(dict(max_iterations=100, tolerance=1e-6))
    sampling = [
        sampling[1],
        sampling[1] | dict(population=4096),
        sampling[1] | dict(generations=10),
        sampling[1] | dict(initial_std=0.25),
        sampling[1] | dict(elite_count=128),
    ]
configurations = [
    dict(solver=solver, horizon=n, plant=plant, **settings)
    for n in ([5, 10] if args.study == "smoke" else [5, 10, 20, 40])
    for solver, choices in (("slsqp", local), ("cem", sampling))
    for settings in choices
]
if args.study == "gpu":
    configurations = [
        dict(solver=solver, horizon=n, plant=plant | dict(centering_cost_per_s=weight), **budget)
        for n in (5, 10, 20)
        for weight in (0.1, 1.0, 10.0)
        for solver, budgets in (
            (
                "gpu_sqp",
                [dict(iterations=i, qp_iterations=80, step_limit=0.5, rho=10.0) for i in (20, 40)],
            ),
            (
                "gpu_cem",
                [
                    dict(population=p, elite_count=32, generations=8, initial_std=0.5)
                    for p in (256, 1024)
                ],
            ),
        )
        for budget in budgets
    ]
rng = np.random.default_rng(args.seed)
count = 4 if args.study == "smoke" else 64 if args.study == "gpu" else 32
groups = np.arange(count) % 4
initial = np.array(
    [
        [0.0, 0.05, 0.0, 0.0],
        [0.25, 1.2, -0.3, 2.0],
        [0.05, np.pi - 0.03, 0.02, -0.04],
        [1.9, np.pi - 0.1, 0.3, -0.1],
    ]
)[groups]
initial += rng.uniform(-1, 1, (count, 4)) * np.array([0.03, 0.02, 0.02, 0.04])
if args.study == "gpu":
    initial += rng.uniform(-1, 1, (count, 4)) * np.array([0.15, 0.15, 0.15, 0.3])
    initial[count // 2 :] *= -1
initial[:, 1] = (initial[:, 1] + np.pi) % (2 * np.pi) - np.pi
np.savez(machine / "starts.npz", states=initial, groups=groups)
metadata = dict(
    study=args.study,
    seed=args.seed,
    decisions=args.decisions,
    configurations=configurations,
    model=manifest,
    workers=1 if args.study == "gpu" else 4,
    state_groups=["downward", "swing_up", "near_upright", "outward_arm"],
    timing="Four-worker solve + transfer + 1 ms audit; seed time separate; excludes queue/I/O",
    versions={name: version(name) for name in ("jax", "jaxlib", "flax", "numpy", "scipy")},
    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    source_sha256={
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in (
            __file__,
            "scripts/experiments/inner-mpc/run.py",
            "scripts/experiments/inner-mpc/gpu_run.py",
            "src/rotary_pendulum/mpc/gpu_sqp.py",
            "src/rotary_pendulum/mpc/gpu_cem.py",
            "src/rotary_pendulum/mpc/inner_model.py",
            "src/rotary_pendulum/mpc/inner_solvers.py",
            "src/rotary_pendulum/environment/jax_dynamics.py",
        )
    },
)
(machine / "design.json").write_text(json.dumps(metadata, indent=2) + "\n")
(machine / "sources").mkdir()
for source in metadata["source_sha256"]:
    (machine / "sources" / Path(source).name).write_bytes(Path(source).read_bytes())
print(f"Prepared {len(configurations)} settings x {count} paired starts in {root}", flush=True)
