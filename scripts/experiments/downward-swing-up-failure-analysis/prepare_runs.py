"""Prepare paired Monte Carlo states and run isolated cases on idle GPUs 0–3."""

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    settings = json.loads(arguments.config.read_text())
    repository = Path(__file__).resolve().parents[3]
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "campaign.json").write_text(json.dumps(settings, indent=2) + "\n")
    shutil.copy2(__file__, output / "prepare_runs.py")
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert devices and all(device in ("0", "1", "2", "3") for device in devices)
    # Independent continuous draws; every case reads the exact same saved states.
    boxes = {
        "downward": ([-0.2, -0.2, -0.5, -0.5], [0.2, 0.2, 0.5, 0.5]),
        "rest": ([-0.2, -0.02, -0.05, -0.05], [0.2, 0.02, 0.05, 0.05]),
        "wide_down": ([-2.8, -0.35, -0.5, -0.5], [2.8, 0.35, 0.5, 0.5]),
        "moving": ([-0.5, 0.4, -2.0, -6.0], [0.5, 2.6, 2.0, 6.0]),
        "near": ([-0.25, np.pi - 0.25, -1.0, -1.0], [0.25, np.pi + 0.25, 1.0, 1.0]),
        "tight": ([-0.08, np.pi - 0.12, -0.15, -0.3], [0.08, np.pi + 0.12, 0.15, 0.3]),
    }
    (output / "reset_boxes.json").write_text(json.dumps(boxes, indent=2) + "\n")
    for seed in settings["seeds"]:
        rng = np.random.default_rng(seed)
        states, labels = [], []
        for label, count in settings["counts"].items():
            state = rng.uniform(*boxes[label], (count, 4))
            if label == "moving":
                state[:, 1] *= rng.choice([-1.0, 1.0], count)
            state[:, 1] = (state[:, 1] + np.pi) % (2 * np.pi) - np.pi
            states.append(state)
            labels.extend([label] * count)
        states.append(
            np.array(
                [[0, 0, 0, 0], [0, np.pi, 0, 0], [2.8, 0, 0, 0], [-2.8, 0, 0, 0], [0, 0, 0.5, 0]]
            )
        )
        labels.extend(["probe"] * 5)
        np.savez_compressed(
            output / f"initial_{seed}.npz", x=np.concatenate(states), labels=np.asarray(labels)
        )

    jobs = []
    for case in settings["cases"]:
        snapshot = output / "sources" / case["name"]
        shutil.copytree(
            repository / "src/rotary_pendulum",
            snapshot / "src/rotary_pendulum",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(Path(__file__).with_name("run_rollouts.py"), snapshot)
        decoder = snapshot / "src/rotary_pendulum/heuristic/decoder.py"
        original = decoder.read_text()
        changed = original
        if case["change"] == "grid65":
            changed = changed.replace(", 33)", ", 65)").replace(":32", ":64")
        elif case["change"] == "speed05":
            changed = changed.replace(
                "jnp.abs(endpoint[..., 2]) <= 0.15", "jnp.abs(endpoint[..., 2]) <= 0.5"
            )
        elif case["change"] in ("speed", "speed_fine"):
            changed = changed.replace(
                "jnp.abs(endpoint[..., 2]) <= 0.15",
                f"jnp.abs(endpoint[..., 2]) <= {case['recovery_speed']!r}",
            )
        elif case["change"] == "soft_work":
            changed = changed.replace(
                "has_exact = jnp.any(exact, axis=-1)",
                "has_exact = jnp.zeros_like(requested_work, dtype=bool)",
            )
        elif case["change"] not in ("none", "fine"):
            raise ValueError(f"Unknown isolated change: {case['change']}")
        if case["change"] not in ("none", "fine"):
            assert changed != original
        decoder.write_text(changed)
        (snapshot / "experiment.patch").write_text(
            "".join(
                difflib.unified_diff(
                    original.splitlines(True),
                    changed.splitlines(True),
                    fromfile="runtime/decoder.py",
                    tofile=f"{case['name']}/decoder.py",
                )
            )
        )
        if case["change"] in ("fine", "speed_fine"):
            # Only the simulated plant gets 2 ms substeps. Decisions, goal sampling,
            # decoder predictions and the 20 ms environment clock stay unchanged.
            for relative in ("jax_dynamics.py", "jax_environment.py"):
                path = snapshot / "src/rotary_pendulum/environment" / relative
                before = path.read_text()
                if relative == "jax_dynamics.py":
                    prefix, kernel = before.split("def rk4_step(", 1)
                    kernel = kernel.replace("u: ArrayLike)", "u: ArrayLike, dt: float = 0.02)")
                    kernel = kernel.replace("PHYSICS_DT_S", "dt")
                    after = prefix + "def rk4_step(" + kernel
                else:
                    after = before.replace(
                        "integrated = rk4_step(state.x, applied_torque)",
                        "integrated = jax.lax.fori_loop(\n"
                        "            0, 10, lambda _, fine: "
                        "rk4_step(fine, applied_torque, 0.002),\n"
                        "            state.x\n        )",
                    )
                assert before != after
                path.write_text(after)
                patch = snapshot / "experiment.patch"
                patch.write_text(
                    patch.read_text()
                    + "".join(
                        difflib.unified_diff(
                            before.splitlines(True),
                            after.splitlines(True),
                            fromfile="runtime/" + relative,
                            tofile=case["name"] + "/" + relative,
                        )
                    )
                )
        (snapshot / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        for seed in settings["seeds"]:
            jobs.append({"name": f"{case['name']}_{seed}", "source": snapshot, "seed": seed})

    pending, active = jobs.copy(), []
    print(f"Prepared {len(jobs)} jobs; devices={devices}; output={output}", flush=True)
    while pending or active:
        for device in devices:
            if not pending or device in {job["device"] for job in active}:
                continue
            status = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            occupancy = [tuple(map(int, row.split(","))) for row in status.strip().splitlines()]
            if sum(memory > 100 or use > 0 for memory, use in occupancy) > 4:
                raise RuntimeError("More than four GPUs occupied; no further jobs launched")
            if occupancy[int(device)][0] > 100 or occupancy[int(device)][1] > 0:
                raise RuntimeError(f"GPU {device} occupied; no job launched")
            job = pending.pop(0)
            log = output / f"{job['name']}.log"
            handle = log.open("w")
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=device,
                JAX_ENABLE_X64="true",
                JAX_PLATFORMS="cuda",
                PYTHONPATH=str(job["source"] / "src"),
                XLA_PYTHON_CLIENT_PREALLOCATE="false",
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(job["source"] / "run_rollouts.py"),
                    "--initial",
                    str(output / f"initial_{job['seed']}.npz"),
                    "--output",
                    str(output / "records" / job["name"]),
                ],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active.append(
                dict(
                    job,
                    device=device,
                    process=process,
                    handle=handle,
                    log=log,
                    offset=0,
                    started=time.monotonic(),
                )
            )
            print(f"START {job['name']} GPU={device}", flush=True)
        for job in active.copy():
            lines = job["log"].read_text().splitlines()
            for line in lines[job["offset"] :]:
                print(f"[{job['name']}] {line}", flush=True)
            job["offset"] = len(lines)
            status = job["process"].poll()
            if status is not None:
                job["handle"].close()
                active.remove(job)
                if status != 0:
                    for other in active:
                        other["process"].terminate()
                    raise RuntimeError(f"{job['name']} failed; see {job['log']}")
            else:
                print(
                    f"RUNNING {job['name']} wall={time.monotonic() - job['started']:.0f}s",
                    flush=True,
                )
        if active:
            time.sleep(5)
    print("DONE all paired jobs", flush=True)


if __name__ == "__main__":
    main()
