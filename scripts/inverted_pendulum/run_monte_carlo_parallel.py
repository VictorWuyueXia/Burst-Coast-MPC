"""Run Monte Carlo data generation in four low-priority local workers."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

WORKER_COUNT = 4
NICE_INCREMENT = 10
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))


def split_epochs(total_epochs: int) -> list[int]:
    """Divide Monte Carlo epochs into at most four nonempty worker batches."""

    if total_epochs <= 0:
        msg = "total_epochs must be positive"
        raise ValueError(msg)
    base_epochs = total_epochs // WORKER_COUNT
    remainder = total_epochs % WORKER_COUNT
    return [
        base_epochs + int(worker_index < remainder)
        for worker_index in range(WORKER_COUNT)
        if base_epochs + int(worker_index < remainder) > 0
    ]


def run_worker(worker_index: int, epochs: int, epoch_index_offset: int) -> None:
    """Run one worker through the current Monte Carlo mode."""

    from rich.console import Console

    from burst_coast_mpc.monte_carlo_mode import run_monte_carlo_mode

    print(f"worker={worker_index} epochs={epochs} status=started")
    run_monte_carlo_mode(
        "inverted_pendulum",
        epochs=epochs,
        console=Console(),
        epoch_index_offset=epoch_index_offset,
    )


def main() -> None:
    """Launch the bounded local Monte Carlo batch and fail on any worker error."""

    if len(sys.argv) != 2:
        msg = "usage: python scripts/inverted_pendulum/run_monte_carlo_parallel.py TOTAL_EPOCHS"
        raise SystemExit(msg)
    total_epochs = int(sys.argv[1])
    epoch_batches = split_epochs(total_epochs)

    # Anchor artifact paths in the repository and lower local compute pressure.
    os.chdir(REPO_ROOT)
    os.nice(NICE_INCREMENT)
    os.environ.update(THREAD_ENV)

    # Launch all worker processes before waiting so the four runs overlap on macOS.
    workers: list[tuple[int, mp.Process]] = []
    epoch_index_offset = 0
    for worker_index, epochs in enumerate(epoch_batches, start=1):
        worker = mp.Process(target=run_worker, args=(worker_index, epochs, epoch_index_offset))
        worker.start()
        workers.append((worker_index, worker))
        epoch_index_offset += epochs
    failures = []
    for worker_index, worker in workers:
        worker.join()
        return_code = worker.exitcode
        print(f"worker={worker_index} return_code={return_code} status=finished")
        if return_code != 0:
            failures.append((worker_index, return_code))
    if failures:
        msg = f"Monte Carlo worker failures: {failures}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
