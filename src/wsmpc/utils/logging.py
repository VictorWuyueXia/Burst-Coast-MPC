"""Episode execution helpers and console logging setup."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.logging import RichHandler

from wsmpc.utils.messages import ExperimentSummary, StateObs, StepRecord


def configure_logging() -> None:
    """Configure console logging at INFO with Rich."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
        force=True,
    )


@dataclass(frozen=True)
class ThirdPersonObservers:
    """Optional third-person observers invoked while an episode runs."""

    at_episode_start: Callable[[StateObs], None] | None = None
    before_step: Callable[[StateObs], None] | None = None
    after_step: Callable[[StateObs, StepRecord], None] | None = None
    at_episode_finish: Callable[[ExperimentSummary], None] | None = None


def episode_output(
    summary: ExperimentSummary,
    *,
    artifact_dir: str | None,
) -> dict[str, Any]:
    """Build the CLI result dictionary printed after a run."""

    output = {
        "run_id": summary.run_id,
        "status": summary.status,
        "total_steps": summary.total_steps,
        "final_t_sec": round(summary.final_t_sec, 6),
        "goal_reached": summary.goal_reached,
    }
    if artifact_dir is not None:
        output["artifact_dir"] = artifact_dir
    return output
