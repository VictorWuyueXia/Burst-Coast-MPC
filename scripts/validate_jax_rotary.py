"""Run JAX rotary validation with live CPU/GPU telemetry and recorded artifacts."""

from __future__ import annotations

import csv
import json
import logging
import platform
import shutil
import subprocess
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import psutil  # type: ignore[import-untyped]
from jax_rotary_artifacts import SEED, measure_accuracy_and_rollouts

from rotary_pendulum.environment.jax_dynamics import TORQUE_LIMIT_NM
from rotary_pendulum.environment.jax_environment import MAX_PHYSICS_STEPS, reset, step

BATCH_SIZES = (1024, 8192, 65536)


def sample_resources(stop: threading.Event, rows: list[dict[str, float | int | str]]) -> None:
    """Record live process, host, and visible-GPU utilization until validation ends."""

    process = psutil.Process()
    process.cpu_percent()
    nvidia_smi = shutil.which("nvidia-smi") if jax.default_backend() == "gpu" else None
    sample_index = 0
    started_at = time.perf_counter()
    while not stop.wait(0.5):
        cpu_system = psutil.cpu_percent()
        cpu_process = process.cpu_percent()
        memory = psutil.virtual_memory()
        gpu_rows = [[-1, np.nan, np.nan, np.nan]]
        if nvidia_smi is not None:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            gpu_rows = [
                [float(value.strip()) for value in line.split(",")]
                for line in result.stdout.splitlines()
            ]
        for gpu_index, gpu_percent, gpu_memory_mib, gpu_total_mib in gpu_rows:
            rows.append(
                {
                    "elapsed_s": time.perf_counter() - started_at,
                    "cpu_system_percent": cpu_system,
                    "cpu_process_percent": cpu_process,
                    "process_rss_mib": process.memory_info().rss / 2**20,
                    "system_memory_percent": memory.percent,
                    "gpu_index": int(gpu_index),
                    "gpu_percent": gpu_percent,
                    "gpu_memory_mib": gpu_memory_mib,
                    "gpu_total_mib": gpu_total_mib,
                }
            )
        sample_index += 1
        if sample_index % 10 == 0:
            if nvidia_smi is None:
                logging.info(
                    "resource cpu_system=%.1f%% process_rss=%.0fMiB",
                    cpu_system,
                    process.memory_info().rss / 2**20,
                )
            else:
                logging.info(
                    "resource cpu_system=%.1f%% process_rss=%.0fMiB gpu_max=%.1f%%",
                    cpu_system,
                    process.memory_info().rss / 2**20,
                    max(row[1] for row in gpu_rows),
                )


def benchmark_batches(machine_dir: Path) -> None:
    """Benchmark warm compiled 200-decision batches at the contract sizes."""

    rows = []
    for batch_size in BATCH_SIZES:
        logging.info("benchmark batch=%d compiling", batch_size)
        keys = jax.random.split(jax.random.key(SEED + batch_size), batch_size)
        initial = jax.vmap(reset)(keys, jnp.arange(batch_size) % 3)
        decisions = jnp.arange(200, dtype=initial.x.dtype)[:, None]
        episodes = jnp.arange(batch_size, dtype=initial.x.dtype)[None, :]
        actions = TORQUE_LIMIT_NM * jnp.sin(0.13 * decisions + 0.017 * episodes)
        rollout = jax.jit(
            lambda carry, action_sequence: jax.lax.scan(
                lambda state, action: (jax.vmap(step)(state, action), None),
                carry,
                action_sequence,
            )[0]
        )
        compile_started = time.perf_counter()
        final = rollout(initial, actions)
        final.x.block_until_ready()
        compile_seconds = time.perf_counter() - compile_started
        timing_repeats = 300 if jax.default_backend() == "gpu" else 3
        run_times = []
        for _ in range(timing_repeats):
            run_started = time.perf_counter()
            final = rollout(initial, actions)
            final.x.block_until_ready()
            run_times.append(time.perf_counter() - run_started)
        run_seconds = float(np.median(run_times))
        memory_stats = jax.devices()[0].memory_stats()
        rows.append(
            {
                "backend": jax.default_backend(),
                "batch_size": batch_size,
                "compile_seconds": compile_seconds,
                "run_seconds": run_seconds,
                "run_min_seconds": min(run_times),
                "run_max_seconds": max(run_times),
                "timing_repeats": timing_repeats,
                "rollouts_per_second": batch_size / run_seconds,
                "physics_transitions_per_second": batch_size * MAX_PHYSICS_STEPS / run_seconds,
                "peak_device_memory_mib": memory_stats["peak_bytes_in_use"] / 2**20
                if memory_stats is not None and "peak_bytes_in_use" in memory_stats
                else "",
            }
        )
        logging.info(
            "benchmark batch=%d compile=%.3fs run=%.3fs rollouts_per_s=%.1f",
            batch_size,
            compile_seconds,
            run_seconds,
            batch_size / run_seconds,
        )
    benchmark_file = (machine_dir / "benchmark.csv").open("w", newline="")
    writer = csv.DictWriter(benchmark_file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    benchmark_file.close()


def plot_resource_usage(
    rows: list[dict[str, float | int | str]], backend_dir: Path, result_root: Path
) -> None:
    """Plot one backend's telemetry and compare throughput when both backends exist."""

    human_dir = backend_dir / "human-readables"
    cpu_rows = [row for row in rows if row["gpu_index"] in (-1, 0)]
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True, sharex=True)
    axes[0].plot(
        [row["elapsed_s"] for row in cpu_rows],
        [row["cpu_system_percent"] for row in cpu_rows],
        label="host CPU",
    )
    axes[0].plot(
        [row["elapsed_s"] for row in cpu_rows],
        [row["cpu_process_percent"] for row in cpu_rows],
        label="process CPU (100%=one core)",
    )
    gpu_indices = sorted({int(row["gpu_index"]) for row in rows if row["gpu_index"] != -1})
    for gpu_index in gpu_indices:
        gpu_rows = [row for row in rows if row["gpu_index"] == gpu_index]
        axes[1].plot(
            [row["elapsed_s"] for row in gpu_rows],
            [row["gpu_percent"] for row in gpu_rows],
            label=f"GPU {gpu_index}",
        )
    axes[0].set(ylabel="Utilization (%)")
    axes[0].legend()
    axes[1].set(xlabel="Wall time (s)", ylabel="GPU utilization (%)")
    if gpu_indices:
        axes[1].legend(ncol=4)
    else:
        axes[1].text(0.5, 0.5, "GPU polling disabled for CPU backend", ha="center")
    figure.savefig(human_dir / "resource_usage.png", dpi=160)
    plt.close(figure)

    benchmark_paths = {
        backend: result_root / backend / "machine-scannables" / "benchmark.csv"
        for backend in ("cpu", "gpu")
    }
    if all(path.exists() for path in benchmark_paths.values()):
        comparison_dir = result_root / "human-readables"
        comparison_dir.mkdir(exist_ok=True)
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for backend, path in benchmark_paths.items():
            benchmark_file = path.open()
            benchmark_rows = list(csv.DictReader(benchmark_file))
            benchmark_file.close()
            axis.loglog(
                [int(row["batch_size"]) for row in benchmark_rows],
                [float(row["rollouts_per_second"]) for row in benchmark_rows],
                marker="o",
                label=backend.upper(),
            )
        axis.set(xlabel="Parallel episode batch", ylabel="200-decision rollouts/s")
        axis.legend()
        figure.savefig(comparison_dir / "backend_throughput.png", dpi=160)
        plt.close(figure)


def main(result_root: Path) -> None:
    """Run all validations with dual live/file logs and sampled resource telemetry."""

    backend_dir = result_root / jax.default_backend()
    machine_dir = backend_dir / "machine-scannables"
    human_dir = backend_dir / "human-readables"
    machine_dir.mkdir(parents=True, exist_ok=True)
    human_dir.mkdir(exist_ok=True)
    log_handler = logging.FileHandler(machine_dir / "validation.log", mode="w")
    stream_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[log_handler, stream_handler],
    )
    metadata = {
        "seed": SEED,
        "platform": platform.platform(),
        "logical_cpu_count": psutil.cpu_count(),
        "jax_version": jax.__version__,
        "jaxlib_version": version("jaxlib"),
        "backend": jax.default_backend(),
        "x64_enabled": jax.config.x64_enabled,
        "devices": [str(device) for device in jax.devices()],
    }
    metadata_file = (machine_dir / "metadata.json").open("w")
    json.dump(metadata, metadata_file, indent=2)
    metadata_file.close()
    logging.info("validation start metadata=%s", metadata)
    resource_rows: list[dict[str, float | int | str]] = []
    stop = threading.Event()
    monitor = threading.Thread(target=sample_resources, args=(stop, resource_rows), daemon=True)
    monitor.start()
    measure_accuracy_and_rollouts(backend_dir)
    benchmark_batches(machine_dir)
    stop.set()
    monitor.join()
    resource_file = (machine_dir / "resource_usage.csv").open("w", newline="")
    writer = csv.DictWriter(resource_file, fieldnames=resource_rows[0].keys())
    writer.writeheader()
    writer.writerows(resource_rows)
    resource_file.close()
    plot_resource_usage(resource_rows, backend_dir, result_root)
    logging.info("validation complete resource_samples=%d", len(resource_rows))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
