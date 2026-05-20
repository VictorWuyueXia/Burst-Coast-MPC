"""Synchronous experiment coordinator with built-in MPC."""

from __future__ import annotations

import logging

from wsmpc.environment import Environment
from wsmpc.mpc.controller import CasadiMPCController
from wsmpc.utils.config_schema import (
    CoordinatorConfig,
    EnvironmentConfig,
    ExperimentConfig,
    MPCConfig,
    RuntimeConfig,
)
from wsmpc.utils.log_events import log_event
from wsmpc.utils.messages import EpisodeResult, ExperimentSummary, StateObs, StepRecord
from wsmpc.utils.time import monotonic_s


class Coordinator:
    """Owns deterministic episode ordering and always uses CasADi MPC."""

    identity = "Coordinator"

    def __init__(
        self,
        coordinator_config: CoordinatorConfig,
        environment_config: EnvironmentConfig,
        experiment_config: ExperimentConfig,
        mpc_config: MPCConfig,
        runtime_config: RuntimeConfig,
        *,
        logger: logging.Logger,
    ) -> None:
        self.config = coordinator_config
        self.environment_config = environment_config
        self.experiment_config = experiment_config
        self.mpc_config = mpc_config
        self.runtime_config = runtime_config
        self.logger = logger
        self.mpc_controller = CasadiMPCController(
            environment_config,
            mpc_config,
            runtime_config,
            logger=logger,
        )

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
        """Run one deterministic MPC episode."""

        from wsmpc.utils.logging import run_episode

        return run_episode(self)

    def create_environment(self) -> Environment:
        """Create a fresh environment for one episode."""

        return Environment(
            self.environment_config,
            run_id=self.experiment_config.run_id,
            episode_id=self.experiment_config.episode_id,
            logger=self.logger,
        )

    def should_stop_for_goal(self, goal_hold_count: int) -> bool:
        """Check the configured goal hold rule."""

        return (
            self.experiment_config.stop_on_goal
            and goal_hold_count >= self.environment_config.goal.hold_steps
        )

    def log_decision_epoch(self, observation: StateObs) -> None:
        """Log deterministic decision epochs."""

        is_decision_epoch = observation.t_index % self.config.decision_interval_steps == 0
        if is_decision_epoch and observation.t_index % self.config.debug_log_every_n_steps == 0:
            log_event(
                self.logger,
                logging.DEBUG,
                identity=self.identity,
                status="running",
                action="decision_epoch",
                action_result="mpc_action_selected",
                t_index=observation.t_index,
                t_sec=observation.t_sec,
            )

    def build_summary(
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
