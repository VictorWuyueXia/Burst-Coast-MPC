"""Synchronous epoch coordinator with optional RL replanning records."""

from __future__ import annotations

import logging
import math
from typing import Protocol

from inverted_pendulum.RL.policy import RLActionSelection
from inverted_pendulum.RL.transitions import RLSegmentState
from inverted_pendulum.environment import Environment
from inverted_pendulum.mpc.controller import CasadiMPCController
from inverted_pendulum.utils.config_schema import (
    CoordinatorConfig,
    EnvironmentConfig,
    ExperimentConfig,
    MPCConfig,
    RLConfig,
)
from inverted_pendulum.utils.log_events import log_event
from inverted_pendulum.utils.logging import ThirdPersonObservers
from inverted_pendulum.utils.messages import (
    EpisodeResult,
    ExperimentSummary,
    RLStepRecord,
    StateObs,
    StepRecord,
)
from inverted_pendulum.utils.time import monotonic_s

EMPTY_THIRD_PERSON_OBSERVERS = ThirdPersonObservers()


class EpochPolicy(Protocol):
    """Policy object that can select one normalized burst-horizon grid action."""

    def select_action(self, observation: StateObs, *, explore: bool) -> RLActionSelection:
        """Return the action selected for the current replanning observation."""

        ...


class EpochCoordinator:
    """Own deterministic epoch ordering, event triggers, and CasADi MPC execution."""

    identity = "EpochCoordinator"

    def __init__(
        self,
        coordinator_config: CoordinatorConfig,
        environment_config: EnvironmentConfig,
        experiment_config: ExperimentConfig,
        mpc_config: MPCConfig,
        *,
        logger: logging.Logger,
        rl_policy: EpochPolicy | None = None,
        rl_config: RLConfig | None = None,
        rl_explore: bool = False,
    ) -> None:
        if rl_policy is not None and rl_config is None:
            msg = "rl_config is required when rl_policy is provided"
            raise ValueError(msg)

        # Bind the experiment contracts and optional RL policy advanced together.
        self.config = coordinator_config
        self.environment_config = environment_config
        self.experiment_config = experiment_config
        self.mpc_config = mpc_config
        self.logger = logger
        self.rl_policy = rl_policy
        self.rl_config = rl_config
        self.rl_explore = rl_explore
        self.mpc_controller = CasadiMPCController(environment_config, mpc_config, logger=logger)

        log_event(
            self.logger,
            logging.INFO,
            identity=self.identity,
            status="initialized",
            action="create_epoch_coordinator",
            action_result="ready",
            mode=self.config.mode,
            event_trigger=self.config.event_trigger,
            rl_enabled=self.rl_policy is not None,
            rl_explore=self.rl_explore,
        )

    def run_episode(
        self,
        third_person_observers: ThirdPersonObservers = EMPTY_THIRD_PERSON_OBSERVERS,
    ) -> EpisodeResult:
        """Run one episode with MPC-only or RL-selected burst-horizon replanning."""

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
        rl_records: list[RLStepRecord] = []
        active_rl_segment: RLSegmentState | None = None
        rl_steps_remaining = 0
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

        # 3. Advance until the goal dwell or the configured horizon terminates the episode.
        for _ in range(self.experiment_config.max_steps):
            if (
                self.experiment_config.stop_on_goal
                and goal_hold_count >= self.environment_config.goal.hold_steps
            ):
                status = "goal_reached"
                break

            if third_person_observers.before_step is not None:
                third_person_observers.before_step(observation)

            event_triggered = self.config.event_trigger and self.event_trigger(observation)
            if self.rl_policy is None:
                action = self.mpc_controller.select_action(
                    observation,
                    force_replan=event_triggered,
                )
            else:
                assert self.rl_config is not None
                if event_triggered or active_rl_segment is None or rl_steps_remaining <= 0:
                    if active_rl_segment is not None and active_rl_segment.records:
                        rl_records.append(
                            active_rl_segment.close(
                                run_id=self.experiment_config.run_id,
                                episode_id=self.experiment_config.episode_id,
                                replan_index=len(rl_records),
                                end_observation=observation,
                                done=False,
                                terminal_status=status,
                                environment_config=self.environment_config,
                                rl_config=self.rl_config,
                            )
                        )
                    selection = self.rl_policy.select_action(
                        observation,
                        explore=self.rl_explore,
                    )
                    selected_plan = self.mpc_controller.start_burst_coast_plan(
                        observation,
                        selection.action,
                        plan_source=selection.mode,
                    )
                    active_rl_segment = RLSegmentState(
                        start_observation=observation,
                        selection=selection,
                        selected_plan=selected_plan,
                        records=[],
                    )
                    rl_steps_remaining = selected_plan.predicted_inputs_nm.size
                action = self.mpc_controller.select_action(observation, force_replan=False)
                rl_steps_remaining -= 1

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
                    action_result="action_selected",
                    t_index=observation.t_index,
                    t_sec=observation.t_sec,
                )

            # 4. Apply the selected action, record the transition, and notify observers.
            observation, record = environment.step(action)
            records.append(record)
            if active_rl_segment is not None:
                active_rl_segment.records.append(record)
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

            # 5. Update the hold counter after each transition so goal dwell is consecutive.
            goal_hold_count = goal_hold_count + 1 if observation.goal_reached else 0
            if (
                self.experiment_config.stop_on_goal
                and goal_hold_count >= self.environment_config.goal.hold_steps
            ):
                status = "goal_reached"
                break

        # 6. Close any active RL transition at the terminal observation and attach returns.
        if active_rl_segment is not None and active_rl_segment.records:
            done = (
                status == "goal_reached"
                or observation.t_index >= self.experiment_config.max_steps
            )
            rl_records.append(
                active_rl_segment.close(
                    run_id=self.experiment_config.run_id,
                    episode_id=self.experiment_config.episode_id,
                    replan_index=len(rl_records),
                    end_observation=observation,
                    done=done,
                    terminal_status=status,
                    environment_config=self.environment_config,
                    rl_config=self.rl_config,
                )
            )
        if rl_records:
            assert self.rl_config is not None
            returned_records: list[RLStepRecord] = []
            return_cost = 0.0
            for record in reversed(rl_records):
                return_cost = record.step_cost + self.rl_config.gamma * return_cost
                returned_records.append(record.model_copy(update={"return_cost": return_cost}))
            rl_records = list(reversed(returned_records))

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
            rl_steps=len(rl_records),
        )
        if third_person_observers.at_episode_finish is not None:
            third_person_observers.at_episode_finish(summary)
        return EpisodeResult(summary=summary, records=records, rl_records=rl_records)

    def event_trigger(self, observation: StateObs) -> bool:
        """Return true near either upright or downward angular section."""

        # The trigger fires near the two angular sections where a new plan is informative.
        wrapped_angle = abs(observation.wrapped_angle_error_rad)
        angle_tolerance = self.environment_config.goal.angle_tolerance_rad
        return wrapped_angle <= angle_tolerance or abs(math.pi - wrapped_angle) <= angle_tolerance
