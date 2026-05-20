"""Resource limiting helpers for local runs."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import psutil
from threadpoolctl import threadpool_limits

from wsmpc.utils.config_schema import RuntimeConfig
from wsmpc.utils.log_events import log_event


@dataclass(frozen=True)
class RuntimeResourceReport:
    """Report what resource limits were requested and applied."""

    env_threads: dict[str, str]
    threadpool_limit_applied: bool
    cpu_affinity_requested: tuple[int, ...] = field(default_factory=tuple)
    cpu_affinity_applied: tuple[int, ...] = field(default_factory=tuple)


def configure_runtime_resources(
    config: RuntimeConfig,
    logger: logging.Logger,
) -> RuntimeResourceReport:
    """Apply conservative process-level resource limits where the platform allows it."""

    # Set common native threadpool environment variables before heavy imports when possible.
    env_threads: dict[str, str] = {}
    if config.set_env:
        env_threads = _set_thread_env(config.blas_threads)

    # Limit already-loaded BLAS/OpenMP pools through the declared dependency.
    threadpool_limits(limits=config.blas_threads)

    # Apply process CPU affinity only when explicitly requested.
    applied_affinity: tuple[int, ...] = ()
    if config.cpu_affinity:
        applied_affinity = _apply_process_affinity(config.cpu_affinity)

    report = RuntimeResourceReport(
        env_threads=env_threads,
        threadpool_limit_applied=True,
        cpu_affinity_requested=tuple(config.cpu_affinity),
        cpu_affinity_applied=applied_affinity,
    )

    log_event(
        logger,
        logging.INFO,
        identity=config.node_id,
        status="ready",
        action="configure_runtime_resources",
        action_result="applied",
        max_worker_threads=config.max_worker_threads,
        blas_threads=config.blas_threads,
        threadpool_limit_applied=report.threadpool_limit_applied,
        cpu_affinity_requested=report.cpu_affinity_requested,
        cpu_affinity_applied=report.cpu_affinity_applied,
    )
    return report


def _set_thread_env(thread_count: int) -> dict[str, str]:
    """Set native math library thread counts to avoid aggressive CPU use."""

    value = str(thread_count)
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for key in keys:
        os.environ[key] = value
    return {key: value for key in keys}


def _apply_process_affinity(cpu_affinity: list[int]) -> tuple[int, ...]:
    """Apply process-level CPU affinity using the declared psutil dependency."""

    process = psutil.Process()
    if not hasattr(process, "cpu_affinity"):
        msg = "CPU affinity was requested but this platform does not expose cpu_affinity"
        raise RuntimeError(msg)
    process.cpu_affinity(cpu_affinity)
    return tuple(process.cpu_affinity())
