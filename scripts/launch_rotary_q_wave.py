"""Launch one resolved rotary-Q experiment wave across independent GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    """Resolve a wave plan, launch its trials, and emit bounded heartbeats."""

    if len(sys.argv) != 2:
        raise ValueError("Usage: launch_rotary_q_wave.py WAVE_PLAN.json")
    plan_path = Path(sys.argv[1])
    plan = json.loads(plan_path.read_text())
    gpu_indices = [trial["gpu"] for trial in plan["trials"]]
    if (
        not gpu_indices
        or any(type(index) is not int or index not in range(4) for index in gpu_indices)
        or len(set(gpu_indices)) != len(gpu_indices)
    ):
        raise ValueError("A wave requires one to four distinct physical GPUs from 0–3")
    wave_dir = Path(plan["wave_dir"])
    config_dir = wave_dir / "machine-scannables" / "configs"
    log_dir = wave_dir / "machine-scannables" / "logs"
    config_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    child_code = (
        "import json,sys; from pathlib import Path; "
        "from bringup.rotary_q_workflow import train; "
        "train(json.loads(Path(sys.argv[1]).read_text()))"
    )
    processes: list[subprocess.Popen[bytes]] = []
    streams = []
    names: list[str] = []
    for trial in plan["trials"]:
        experiment = {**plan["base_experiment"], **trial["experiment"]}
        name = experiment["experiment_id"]
        config_path = config_dir / f"{name}.json"
        config_path.write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n")
        log_path = log_dir / f"{name}.log"
        stream = log_path.open("wb")
        environment = os.environ.copy()
        environment["JAX_PLATFORMS"] = "cuda"
        environment["CUDA_VISIBLE_DEVICES"] = str(trial["gpu"])
        process = subprocess.Popen(
            [sys.executable, "-c", child_code, str(config_path)],
            cwd=Path.cwd(),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        processes.append(process)
        streams.append(stream)
        names.append(name)
        print(f"wave-launch trial={name} gpu={trial['gpu']} pid={process.pid}", flush=True)

    started = time.monotonic()
    while any(process.poll() is None for process in processes):
        completed = sum(process.poll() is not None for process in processes)
        elapsed = time.monotonic() - started
        print(
            f"wave-heartbeat elapsed_s={elapsed:.1f} completed={completed}/{len(processes)}",
            flush=True,
        )
        time.sleep(10.0)
    exit_codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    for name, exit_code in zip(names, exit_codes, strict=True):
        print(f"wave-complete trial={name} exit_code={exit_code}", flush=True)
    if any(exit_codes):
        raise RuntimeError("At least one rotary-Q wave trial failed; inspect its declared log")


if __name__ == "__main__":
    main()
