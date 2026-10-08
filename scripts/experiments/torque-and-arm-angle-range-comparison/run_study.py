"""Run paired torque/range experiments from explicit, isolated source snapshots."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    settings = json.loads(arguments.config.read_text())
    output = arguments.output.resolve()
    machine, human = output / "machine-scannables", output / "human-readables"
    machine.mkdir(parents=True, exist_ok=False)
    human.mkdir()
    (machine / "study.json").write_text(json.dumps(settings, indent=2) + "\n")
    shutil.copy2(__file__, machine / "run_study.py")
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert devices == [""] or all(device in ("0", "1", "2", "3") for device in devices)
    jobs = []
    for case in settings["cases"]:
        snapshot = machine / "sources" / case["name"]
        snapshot.mkdir(parents=True)
        shutil.copytree(
            repository / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__")
        )
        (snapshot / "scripts").mkdir()
        shutil.copy2(repository / "scripts/validate_rotary_heuristic.py", snapshot / "scripts")
        shutil.copy2(repository / "pyproject.toml", snapshot)
        replacements = {
            "src/rotary_pendulum/heuristic/decoder.py": [
                (
                    "POLICY_TORQUE_NM = 0.45 * TORQUE_LIMIT_NM",
                    f"POLICY_TORQUE_NM = {0.45 * case['torque_multiplier']!r} * TORQUE_LIMIT_NM",
                )
            ],
            "src/rotary_pendulum/environment/jax_environment.py": [
                (
                    "ARM_LIMIT_RAD = jnp.pi / 2.0",
                    f"ARM_LIMIT_RAD = jnp.deg2rad({case['arm_limit_deg']!r})",
                )
            ],
            "src/rotary_pendulum/heuristic/artifacts.py": [
                (
                    "from rotary_pendulum.heuristic.energy import TARGET_ENERGY_J",
                    "from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD\n"
                    "from rotary_pendulum.heuristic.energy import TARGET_ENERGY_J",
                ),
                ("np.pi / 2", "float(ARM_LIMIT_RAD)"),
            ],
        }
        patch = []
        for relative, edits in replacements.items():
            target = snapshot / relative
            original = target.read_text()
            changed = original
            for before, after in edits:
                assert before in changed, (relative, before)
                changed = changed.replace(before, after)
            target.write_text(changed)
            patch.extend(
                difflib.unified_diff(
                    original.splitlines(True),
                    changed.splitlines(True),
                    fromfile="a/" + relative,
                    tofile="b/" + relative,
                )
            )
        (snapshot / "experiment.patch").write_text("".join(patch))
        assert 0.00918 * case["torque_multiplier"] <= 0.0204
        for seed in settings["seeds"]:
            campaign = {
                "seed": seed,
                "episodes_per_stratum": settings["episodes_per_stratum"],
                "decisions": settings["decisions"],
                "chunk_decisions": settings["chunk_decisions"],
                "recovery_steps": case["recovery_steps"],
                "mode": "energy",
                "trials": [
                    {
                        "name": case["name"],
                        "work_gain": 0.04,
                        "kinetic_weight": 1.0,
                        "work_weight": case["work_weight"],
                    }
                ],
            }
            config = snapshot / f"campaign_{seed}.json"
            config.write_text(json.dumps(campaign, indent=2) + "\n")
            jobs.append(
                {
                    "name": f"{case['name']}_{seed}",
                    "case": case,
                    "seed": seed,
                    "source": snapshot,
                    "config": config,
                }
            )
    pending, active = jobs.copy(), []
    print(f"Prepared {len(jobs)} runs; devices={devices}; output={output}", flush=True)
    while pending or active:
        occupied = {job["device"] for job in active}
        for device in devices:
            if not pending or device in occupied:
                continue
            if device:
                status = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
                occupancy = [tuple(map(int, row.split(","))) for row in status.splitlines()]
                if sum(memory > 100 or use > 0 for memory, use in occupancy) > 4:
                    raise RuntimeError("More than four GPUs occupied; stopping experiment")
                if occupancy[int(device)][0] > 100 or occupancy[int(device)][1] > 0:
                    raise RuntimeError(f"GPU {device} is occupied; no run launched")
            job = pending.pop(0)
            log = machine / f"{job['name']}.log"
            handle = log.open("w")
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=device,
                JAX_ENABLE_X64="true",
                PYTHONPATH=str(job["source"] / "src"),
                MPLBACKEND="Agg",
                MPLCONFIGDIR=str(machine / "matplotlib"),
                XLA_PYTHON_CLIENT_PREALLOCATE="false",
            )
            command = [
                sys.executable,
                "scripts/validate_rotary_heuristic.py",
                "--config",
                str(job["config"]),
                "--output",
                str(machine / "runs" / job["name"]),
            ]
            process = subprocess.Popen(
                command, cwd=job["source"], env=environment, stdout=handle, stderr=subprocess.STDOUT
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
            print(f"START {job['name']} device={device or 'cpu'}", flush=True)
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

    records, initials, audits = [], {}, []
    for job in jobs:
        run = machine / "runs" / job["name"]
        initial = np.load(run / "machine-scannables/initial_states.npz")["x"]
        if job["seed"] in initials:
            np.testing.assert_array_equal(initial, initials[job["seed"]])
        initials[job["seed"]] = initial
        trial = run / job["case"]["name"]
        data = np.load(trial / "machine-scannables/trajectories.npz")
        active_samples = data["physics_active"]
        bound = np.deg2rad(job["case"]["arm_limit_deg"])
        np.testing.assert_array_equal(
            data["arm_violation"][-1],
            np.any((abs(data["physics_x"][..., 0]) >= bound) & active_samples, axis=(0, 2)),
        )
        assert np.max(abs(data["torque_nm"])) <= 0.00918 * job["case"]["torque_multiplier"] + 1e-12
        handle = (trial / "machine-scannables/episodes.csv").open()
        for row in csv.DictReader(handle):
            records.append({"case": job["case"]["name"], "seed": job["seed"], **row})
        handle.close()
        audits.append(
            {
                "case": job["case"]["name"],
                "seed": job["seed"],
                **json.loads((trial / "machine-scannables/integration_audit.json").read_text()),
            }
        )
        shutil.copy2(
            trial / "human-readables/representative_sessions.png", human / f"{job['name']}.png"
        )
    summary = []
    for case in settings["cases"]:
        for stratum in ("downward", "moving", "near", "tight", "probe"):
            rows = [
                row for row in records if row["case"] == case["name"] and row["stratum"] == stratum
            ]
            success = [row for row in rows if row["success"] == "True"]
            summary.append(
                {
                    "case": case["name"],
                    "torque_limit_nm": 0.00918 * case["torque_multiplier"],
                    **{
                        key: case[key] for key in ("arm_limit_deg", "recovery_steps", "work_weight")
                    },
                    "stratum": stratum,
                    "episodes": len(rows),
                    "captures": len(success),
                    "clean_captures": sum(row["arm_violation"] == "False" for row in success),
                    "excursions": sum(row["arm_violation"] == "True" for row in rows),
                    **{
                        "mean_" + field: float(np.mean([float(row[field]) for row in rows]))
                        for field in (
                            "maximum_potential_fraction",
                            "absolute_work_j",
                            "recovery_unresolved_fraction",
                            "work_override_fraction",
                        )
                    },
                    "max_peak_arm_rad": max(float(row["peak_arm_rad"]) for row in rows),
                    "max_work_error_j": max(float(row["max_work_error_j"]) for row in rows),
                    "max_energy_balance_j": max(float(row["max_energy_balance_j"]) for row in rows),
                }
            )
    for name, rows in (("episodes", records), ("summary", summary)):
        handle = (machine / f"{name}.csv").open("w", newline="")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.close()
    (machine / "integration_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    positions = np.arange(4)
    width = 0.8 / len(settings["cases"])
    for index, case in enumerate(settings["cases"]):
        rows = [row for row in summary if row["case"] == case["name"] and row["stratum"] != "probe"]
        for axis, field in zip(axes, ("captures", "clean_captures", "excursions"), strict=True):
            axis.bar(
                positions - 0.4 + width * (index + 0.5),
                [100 * row[field] / row["episodes"] for row in rows],
                width,
                label=case["name"],
            )
            axis.set_xticks(positions, ["Downward", "Moving", "Near", "Tight"])
            axis.set_ylabel("Episodes [%]")
            axis.set_ylim(0, 105)
            axis.grid(axis="y", alpha=0.2)
    for axis, title in zip(
        axes,
        ("Capture", "Capture without excursion", "Excursion beyond each case's limit"),
        strict=True,
    ):
        axis.set_title(title)
    axes[0].legend(fontsize=8)
    figure.savefig(human / "comparison.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2), flush=True)
    print(
        f"DONE paired initial states, torque caps, and excursion labels verified: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
