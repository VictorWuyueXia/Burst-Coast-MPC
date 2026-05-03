"""Synchronous experiment coordinator."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from wsmpc.config.schema import CoordinatorConfig, EnvironmentConfig, ExperimentConfig
from wsmpc.core.logging import log_event
from wsmpc.core.messages import ActionCommand, ExperimentSummary, StateObs, StepRecord
from wsmpc.core.time import monotonic_s
from wsmpc.environment import Environment


@dataclass(frozen=True)
class EpisodeResult:
    """In-memory episode result returned by the Coordinator."""

    summary: ExperimentSummary
    records: list[StepRecord]


class Coordinator:
    """Owns deterministic episode ordering and simulated experiment time."""

    identity = "Coordinator"

    def __init__(
        self,
        coordinator_config: CoordinatorConfig,
        environment_config: EnvironmentConfig,
        experiment_config: ExperimentConfig,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = coordinator_config
        self.environment_config = environment_config
        self.experiment_config = experiment_config
        self.logger = logger or logging.getLogger(__name__)

        log_event(
            self.logger,
            logging.INFO,
            identity=self.identity,
            status="initialized",
            action="create_coordinator",
            action_result="ready",
            mode=self.config.mode,
            global_seed=self.experiment_config.global_seed,
        )

    def run_episode(self) -> EpisodeResult:
        """Run one deterministic episode through the Environment."""

        wall_started_at = monotonic_s()
        environment = Environment(
            self.environment_config,
            run_id=self.experiment_config.run_id,
            episode_id=self.experiment_config.episode_id,
            logger=self.logger,
        )
        observation = environment.reset(self.experiment_config.initial_state)
        records: list[StepRecord] = []
        goal_hold_count = 1 if observation.goal_reached else 0

        log_event(
            self.logger,
            logging.INFO,
            identity=self.identity,
            status="running",
            action="episode_start",
            action_result="initialized",
            t_index=observation.t_index,
            t_sec=observation.t_sec,
            max_steps=self.experiment_config.max_steps,
        )

        # The Coordinator applies the configured default command at every simulator step.
        status = "max_steps_reached"
        for _ in range(self.experiment_config.max_steps):
            if self._should_stop_for_goal(goal_hold_count):
                status = "goal_reached"
                break

            action = self._build_default_action(observation)
            self._log_decision_epoch(observation)
            observation, record = environment.step(action)
            records.append(record)

            goal_hold_count = goal_hold_count + 1 if observation.goal_reached else 0
            if self._should_stop_for_goal(goal_hold_count):
                status = "goal_reached"
                break

        summary = self._build_summary(
            status=status,
            observation=observation,
            records=records,
            wall_started_at=wall_started_at,
        )
        log_event(
            self.logger,
            logging.INFO,
            identity=self.identity,
            status="finished",
            action="episode_finish",
            action_result=summary.status,
            t_index=summary.final_t_index,
            t_sec=summary.final_t_sec,
            total_steps=summary.total_steps,
            goal_reached=summary.goal_reached,
            total_wall_time_s=f"{summary.total_wall_time_s:.6f}",
        )
        return EpisodeResult(summary=summary, records=records)

    def _build_default_action(self, observation: StateObs) -> ActionCommand:
        """Create the configured default action for the current simulator step."""

        action = ActionCommand(
            run_id=self.experiment_config.run_id,
            episode_id=self.experiment_config.episode_id,
            t_index=observation.t_index,
            t_sec=observation.t_sec,
            u=self.experiment_config.default_action.u,
            source=self.experiment_config.default_action.source,
        )
        log_event(
            self.logger,
            logging.DEBUG,
            identity=self.identity,
            status="running",
            action="build_default_action",
            action_result="command_ready",
            t_index=observation.t_index,
            t_sec=observation.t_sec,
            u=f"{action.u:.6f}",
            source=action.source,
        )
        return action

    def _log_decision_epoch(self, observation: StateObs) -> None:
        """Log deterministic decision epochs owned by the Coordinator."""

        is_decision_epoch = observation.t_index % self.config.decision_interval_steps == 0
        if is_decision_epoch and observation.t_index % self.config.debug_log_every_n_steps == 0:
            log_event(
                self.logger,
                logging.DEBUG,
                identity=self.identity,
                status="running",
                action="decision_epoch",
                action_result="default_policy_selected",
                t_index=observation.t_index,
                t_sec=observation.t_sec,
            )

    def _should_stop_for_goal(self, goal_hold_count: int) -> bool:
        """Check the configured goal hold rule without consulting wall-clock time."""

        return (
            self.experiment_config.stop_on_goal
            and goal_hold_count >= self.environment_config.goal.hold_steps
        )

    def _build_summary(
        self,
        *,
        status: str,
        observation: StateObs,
        records: list[StepRecord],
        wall_started_at: float,
    ) -> ExperimentSummary:
        """Build the final typed summary for CLI output and tests."""

        return ExperimentSummary(
            run_id=self.experiment_config.run_id,
            episode_id=self.experiment_config.episode_id,
            status=status,
            total_steps=len(records),
            final_t_index=observation.t_index,
            final_t_sec=observation.t_sec,
            goal_reached=observation.goal_reached,
            records_emitted=len(records),
            total_wall_time_s=monotonic_s() - wall_started_at,
            final_observation=observation,
        )
