"""Launch paired TD3 trials after checking shared GPU occupancy."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    """Launch a declared wave, snapshot sources, and report every child's progress."""
    plan = json.loads(Path(sys.argv[1]).read_text())
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    occupied = {
        int(row.split(",")[0])
        for row in query.splitlines()
        if any(int(value.strip()) > 0 for value in row.split(",")[1:])
    }
    if len(occupied) > 4:
        raise RuntimeError(f"More than four GPUs occupied: {sorted(occupied)}")
    requested = [trial["gpu"] for trial in plan["trials"]]
    if len(set(requested)) != len(requested) or set(requested) - set(range(8)):
        raise ValueError("Use distinct physical GPUs 0–7")
    if occupied.intersection(requested):
        raise RuntimeError(f"Requested GPUs occupied: {sorted(occupied.intersection(requested))}")
    root = Path(plan["wave_dir"]) / "machine-scannables"
    root.mkdir(parents=True, exist_ok=False)
    (root / "occupancy.csv").write_text(query)
    (root / "plan.json").write_text(json.dumps(plan, indent=2))
    sources = list(Path("src/rotary_pendulum").rglob("*.py")) + [
        Path("src/bringup/rotary_residual_workflow.py"),
        Path("pyproject.toml"),
    ]
    sources += list(Path("scripts/experiments/td3-diagnosis").glob("*.py"))
    sources += list(Path("scripts/experiments/heuristic-demonstrations").glob("*.py"))
    sources += list(Path("scripts/experiments/weak-motor").glob("*.py"))
    sources += list(Path("scripts/experiments/sustained-balance").glob("*.py"))
    sources += list(Path("scripts/experiments/stationary-balancing").glob("*.py"))
    sources += list(Path("src/bringup").glob("rotary_stationary_workflow.py"))
    sources += list(Path("src/rotary_pendulum/configs").glob("*.yaml"))
    hashes = {}
    for source in sources:
        relative = source.resolve().relative_to(Path.cwd())
        output = root / "source_snapshot" / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (root / "source_hashes.json").write_text(json.dumps(hashes, indent=2))
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True
    ).stdout
    (root / "environment.txt").write_text(packages)
    if plan["operation"] == "stationary_train":
        code = (
            "import json,sys; from pathlib import Path; "
            "from bringup.rotary_stationary_workflow import train; "
            "train(json.loads(Path(sys.argv[1]).read_text()))"
        )
    elif plan["operation"] == "train":
        code = (
            "import json,sys; from pathlib import Path; "
            "from bringup.rotary_residual_workflow import train; "
            "train(json.loads(Path(sys.argv[1]).read_text()))"
        )
    elif plan["operation"] == "audit":
        code = (
            "import runpy; "
            "runpy.run_path('scripts/experiments/td3-diagnosis/audit.py', run_name='__main__')"
        )
    elif plan["operation"] == "weak_audit":
        code = (
            "import runpy; "
            "runpy.run_path('scripts/experiments/weak-motor/evaluate.py', run_name='__main__')"
        )
    elif plan["operation"] == "balance":
        code = (
            "import runpy; "
            "runpy.run_path('scripts/experiments/sustained-balance/run.py', run_name='__main__')"
        )
    else:
        raise ValueError("Declare train, audit, weak_audit or balance operation")
    processes, streams, logs = [], [], []
    for trial in plan["trials"]:
        settings = plan["base"] | trial["settings"]
        settings["run_dir"] = str(Path(plan["wave_dir"]) / trial["name"])
        config = root / f"{trial['name']}.json"
        config.write_text(json.dumps(settings, indent=2))
        log = root / f"{trial['name']}.log"
        stream = log.open("wb")
        environment = os.environ | {
            "CUDA_VISIBLE_DEVICES": str(trial["gpu"]),
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MPLCONFIGDIR": f"/tmp/td3-mpl-{trial['gpu']}",
        }
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", code, str(config)],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        processes.append(process)
        streams.append(stream)
        logs.append(log)
        print(f"launch {trial['name']} GPU={trial['gpu']} PID={process.pid}", flush=True)
    started = time.monotonic()
    while any(process.poll() is None for process in processes):
        time.sleep(20)
        print(f"wave elapsed={time.monotonic() - started:.0f}s", flush=True)
        for process, log in zip(processes, logs, strict=True):
            lines = log.read_text().splitlines()
            print(
                f"{log.stem} exit={process.poll()} " + (lines[-1] if lines else "starting"),
                flush=True,
            )
    for stream in streams:
        stream.close()
    statuses = {log.stem: process.returncode for log, process in zip(logs, processes, strict=True)}
    (root / "exit_codes.json").write_text(json.dumps(statuses, indent=2))
    if any(statuses.values()):
        raise RuntimeError(f"Failed trials: {statuses}")


if __name__ == "__main__":
    main()
