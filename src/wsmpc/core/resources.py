"""Resource limiting helpers for local runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os

from wsmpc.config.schema import RuntimeConfig
from wsmpc.core.logging import log_event


@dataclass(frozen=True)
class RuntimeResourceReport:
    """Report what resource limits were requested and applied."""

    env_threads: dict[str, str]
    threadpool_limit_applied: bool
    cpu_affinity_requested: tuple[int, ...] = field(default_factory=tuple)
    cpu_affinity_applied: tuple[int, ...] = field(default_factory=tuple)
    cpu_affinity_error: str | None = None


def configure_runtime_resources(
    config: RuntimeConfig,
    logger: logging.Logger | None = None,
) -> RuntimeResourceReport:
    """Apply conservative process-level resource limits where the platform allows it."""

    active_logger = logger or logging.getLogger(__name__)

    # Set common native threadpool environment variables before heavy imports when possible.
    env_threads: dict[str, str] = {}
    if config.set_env:
        env_threads = _set_thread_env(config.blas_threads)

    # Limit already-loaded BLAS/OpenMP pools when threadpoolctl is available.
    threadpool_limit_applied = _limit_threadpools(config.blas_threads, active_logger)

    # Apply process CPU affinity where supported; macOS commonly reports unsupported here.
    applied_affinity: tuple[int, ...] = ()
    affinity_error: str | None = None
    if config.cpu_affinity:
        applied_affinity, affinity_error = _apply_process_affinity(config.cpu_affinity)

    report = RuntimeResourceReport(
        env_threads=env_threads,
        threadpool_limit_applied=threadpool_limit_applied,
        cpu_affinity_requested=tuple(config.cpu_affinity),
        cpu_affinity_applied=applied_affinity,
        cpu_affinity_error=affinity_error,
    )

    log_event(
        active_logger,
        logging.INFO,
        identity=config.node_id,
        status="ready",
        action="configure_runtime_resources",
        action_result="applied",
        max_worker_threads=config.max_worker_threads,
        blas_threads=config.blas_threads,
        threadpool_limit_applied=threadpool_limit_applied,
        cpu_affinity_requested=report.cpu_affinity_requested,
        cpu_affinity_applied=report.cpu_affinity_applied,
        cpu_affinity_error=report.cpu_affinity_error,
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


def _limit_threadpools(thread_count: int, logger: logging.Logger) -> bool:
    """Limit loaded native threadpools without making threadpoolctl mandatory at import time."""

    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=thread_count)
    except ImportError:
        log_event(
            logger,
            logging.DEBUG,
            identity="Runtime",
            status="degraded",
            action="limit_threadpools",
            action_result="threadpoolctl_missing",
        )
        return False
    return True


def _apply_process_affinity(cpu_affinity: list[int]) -> tuple[tuple[int, ...], str | None]:
    """Attempt process-level CPU affinity using psutil when the OS exposes it."""

    try:
        import psutil

        process = psutil.Process()
        if not hasattr(process, "cpu_affinity"):
            return (), "cpu_affinity_unsupported"
        process.cpu_affinity(cpu_affinity)
        return tuple(process.cpu_affinity()), None
    except ImportError:
        return (), "psutil_missing"
    except Exception as exc:  # pragma: no cover - platform-specific failure path.
        return (), f"{type(exc).__name__}: {exc}"
