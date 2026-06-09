"""Synchronous experiment coordinator with event-triggered replanning."""

from __future__ import annotations

import logging
import math

from wsmpc.environment import Environment
from wsmpc.mpc.controller import CasadiMPCController
from wsmpc.utils.config_schema import (
    CoordinatorConfig,
    EnvironmentConfig,
    ExperimentConfig,
    MPCConfig,
)
from wsmpc.utils.log_events import log_event
from wsmpc.utils.logging import ThirdPersonObservers
from wsmpc.utils.messages import EpisodeResult, ExperimentSummary, StateObs, StepRecord
from wsmpc.utils.time import monotonic_s

# Empty observer set keeps the normal episode path explicit and allocation-free.
EMPTY_THIRD_PERSON_OBSERVERS = ThirdPersonObservers()


class Coordinator:
    """Own deterministic episode ordering, event triggers, and CasADi MPC execution."""

    identity = "Coordinator"

    def __init__(
        self,
        coordinator_config: CoordinatorConfig,
        environment_config: EnvironmentConfig,
        experiment_config: ExperimentConfig,
        mpc_config: MPCConfig,
        *,
        logger: logging.Logger,
    ) -> None:
        # Bind the three experiment contracts the coordinator advances together.
        self.config = coordinator_config
        self.environment_config = environment_config
        self.experiment_config = experiment_config
        self.mpc_config = mpc_config
        self.logger = logger
        self.mpc_controller = CasadiMPCController(environment_config, mpc_config, logger=logger)

        # Announce the synchronous coordinator mode before the first episode starts.
        log_event(
            self.logger,
            logging.INFO,
            identity=self.identity,
            status="initialized",
            action="create_coordinator",
            action_result="ready",
            mode=self.config.mode,
            event_trigger=self.config.event_trigger,
        )

    def run_episode(
        self,
        third_person_observers: ThirdPersonObservers = EMPTY_THIRD_PERSON_OBSERVERS,
    ) -> EpisodeResult:
        """Run one deterministic MPC episode with optional third-person observers."""

        # 1. Construct the environment and reset the physical state once per episode.
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
        status = "max_steps_reached"

        # 2. Publish the reset observation to logs and optional observers.
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
        if third_person_observers.at_episode_start is not None:
            third_person_observers.at_episode_start(observation)

        # 3. Advance the closed-loop system until the goal or horizon terminates the episode.
        for _ in range(self.experiment_config.max_steps):
            if (
                self.experiment_config.stop_on_goal
                and goal_hold_count >= self.environment_config.goal.hold_steps
            ):
                status = "goal_reached"
                break

            if third_person_observers.before_step is not None:
                third_person_observers.before_step(observation)

            # 4. Evaluate wake-trigger logic at the observation boundary before selecting action.
            event_triggered = self.config.event_trigger and self.event_trigger(observation)
            action = self.mpc_controller.select_action(
                observation,
                force_replan=event_triggered,
            )
            if (
                observation.t_index % self.config.decision_interval_steps == 0
                and observation.t_index % self.config.debug_log_every_n_steps == 0
            ):
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

            # 5. Apply the selected action, record the transition, and notify observers.
            observation, record = environment.step(action)
            records.append(record)
            log_event(
                self.logger,
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
            if third_person_observers.after_step is not None:
                third_person_observers.after_step(observation, record)

            # 6. Update the hold counter after each transition so goal dwell is consecutive.
            goal_hold_count = goal_hold_count + 1 if observation.goal_reached else 0
            if (
                self.experiment_config.stop_on_goal
                and goal_hold_count >= self.environment_config.goal.hold_steps
            ):
                status = "goal_reached"
                break

        # 7. Summarize the terminal observation and total wall-clock episode cost.
        summary = ExperimentSummary(
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
        # 8. Emit the finish event and expose the immutable episode result to callers.
        log_event(
            self.logger,
            logging.INFO,
            identity=self.identity,
            status=summary.status,
            action="episode_finish",
            action_result=summary.status,
            t_index=summary.final_t_index,
            t_sec=summary.final_t_sec,
            total_steps=summary.total_steps,
            goal_reached=summary.goal_reached,
            total_wall_time_s=f"{summary.total_wall_time_s:.6f}",
        )
        if third_person_observers.at_episode_finish is not None:
            third_person_observers.at_episode_finish(summary)
        return EpisodeResult(summary=summary, records=records)

    def event_trigger(self, observation: StateObs) -> bool:
        """Return true near either upright or downward angular section."""

        # The trigger fires near the two angular sections where a new plan is informative.
        wrapped_angle = abs(observation.wrapped_angle_error_rad)
        angle_tolerance = self.environment_config.goal.angle_tolerance_rad
        return wrapped_angle <= angle_tolerance or abs(math.pi - wrapped_angle) <= angle_tolerance
