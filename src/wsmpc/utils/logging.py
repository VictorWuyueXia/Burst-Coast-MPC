"""Episode execution helpers and console logging setup."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.logging import RichHandler

from wsmpc.utils.log_events import log_event
from wsmpc.utils.messages import EpisodeResult, ExperimentSummary, StateObs, StepRecord
from wsmpc.utils.time import monotonic_s


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
class EpisodeHooks:
    """Optional observers invoked while an episode runs."""

    on_episode_start: Callable[[StateObs], None] | None = None
    on_before_step: Callable[[StateObs], None] | None = None
    on_step: Callable[[StateObs, StepRecord], None] | None = None
    on_episode_finish: Callable[[ExperimentSummary], None] | None = None


def run_episode(coordinator, hooks: EpisodeHooks | None = None) -> EpisodeResult:
    """Run one episode with unified logging, hooks, and interrupt handling."""

    hooks = hooks or EpisodeHooks()
    wall_started_at = monotonic_s()
    environment = coordinator.create_environment()
    observation = environment.reset(coordinator.experiment_config.initial_state)
    records: list[StepRecord] = []
    goal_hold_count = 1 if observation.goal_reached else 0

    log_event(
        coordinator.logger,
        logging.INFO,
        identity=coordinator.identity,
        status="running",
        action="episode_start",
        action_result="initialized",
        t_index=observation.t_index,
        t_sec=observation.t_sec,
        max_steps=coordinator.experiment_config.max_steps,
    )
    if hooks.on_episode_start is not None:
        hooks.on_episode_start(observation)

    status = "max_steps_reached"
    try:
        for _ in range(coordinator.experiment_config.max_steps):
            if coordinator.should_stop_for_goal(goal_hold_count):
                status = "goal_reached"
                break

            if hooks.on_before_step is not None:
                hooks.on_before_step(observation)

            action = coordinator.mpc_controller.select_action(observation)
            coordinator.log_decision_epoch(observation)
            observation, record = environment.step(action)
            records.append(record)
            log_step_record(coordinator.logger, record)
            if hooks.on_step is not None:
                hooks.on_step(observation, record)

            goal_hold_count = goal_hold_count + 1 if observation.goal_reached else 0
            if coordinator.should_stop_for_goal(goal_hold_count):
                status = "goal_reached"
                break
    except KeyboardInterrupt:
        return finish_episode(
            coordinator,
            status="interrupted",
            observation=observation,
            records=records,
            wall_started_at=wall_started_at,
            hooks=hooks,
            log_level=logging.WARNING,
            action="episode_interrupt",
        )

    return finish_episode(
        coordinator,
        status=status,
        observation=observation,
        records=records,
        wall_started_at=wall_started_at,
        hooks=hooks,
        log_level=logging.INFO,
        action="episode_finish",
    )


def log_step_record(logger: logging.Logger, record: StepRecord) -> None:
    """Emit one concise line per simulator step."""

    log_event(
        logger,
        logging.DEBUG,
        identity="Environment",
        status="running",
        action="step_record",
        action_result=record.action_result,
        t_index=record.t_index,
        t_sec=record.t_sec,
        u_commanded_nm=f"{record.u_commanded_nm:.6f}",
        u_applied_nm=f"{record.u_applied_nm:.6f}",
        mode=record.mode,
    )


def finish_episode(
    coordinator,
    *,
    status: str,
    observation: StateObs,
    records: list[StepRecord],
    wall_started_at: float,
    hooks: EpisodeHooks,
    log_level: int,
    action: str,
) -> EpisodeResult:
    """Build the episode summary, log it, invoke finish hooks, and return."""

    summary = coordinator.build_summary(
        status=status,
        observation=observation,
        records=records,
        wall_started_at=wall_started_at,
    )
    log_event(
        coordinator.logger,
        log_level,
        identity=coordinator.identity,
        status=summary.status,
        action=action,
        action_result=summary.status,
        t_index=summary.final_t_index,
        t_sec=summary.final_t_sec,
        total_steps=summary.total_steps,
        goal_reached=summary.goal_reached,
        total_wall_time_s=f"{summary.total_wall_time_s:.6f}",
    )
    if hooks.on_episode_finish is not None:
        hooks.on_episode_finish(summary)
    return EpisodeResult(summary=summary, records=records)


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
