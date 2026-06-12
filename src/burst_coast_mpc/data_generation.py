"""Sequential Monte Carlo data generation for offline RL training."""

from __future__ import annotations

import json
import logging
import math

import numpy as np

from inverted_pendulum.environment import Environment
from inverted_pendulum.mpc.controller import CasadiMPCController
from inverted_pendulum.utils.artifacts import ArtifactWriter
from inverted_pendulum.utils.config_schema import DataGenerationRootConfig
from inverted_pendulum.utils.log_events import log_event
from inverted_pendulum.utils.messages import RLStepRecord, StepRecord
from inverted_pendulum.utils.monte_carlo import (
    sample_uniform_initial_state,
    sample_uniform_monte_carlo_action,
)


class MonteCarloDataGenerator:
    """Run sampled burst-coast MPC episodes and emit replanning-level RL transitions."""

    identity = "MonteCarloDataGenerator"

    def __init__(self, config: DataGenerationRootConfig, logger: logging.Logger) -> None:
        # Bind the data-generation package and seeded sampler for the full run.
        self.config = config
        self.logger = logger
        self.rng = np.random.default_rng(config.data_generation.seed)

    def run(self, artifact_writer: ArtifactWriter) -> tuple[list[StepRecord], list[RLStepRecord]]:
        """Generate sequential Monte Carlo transitions with backward returns."""

        # 1. Accumulate dense simulator records and sparse RL transition records together.
        step_records: list[StepRecord] = []
        rl_records: list[RLStepRecord] = []
        for episode_offset in range(self.config.data_generation.episodes):
            # 2. Create one independent environment-controller pair per sampled episode.
            episode_id = self.config.experiment.episode_id + episode_offset
            environment = Environment(
                self.config.environment,
                run_id=self.config.experiment.run_id,
                episode_id=episode_id,
                logger=self.logger,
            )
            controller = CasadiMPCController(
                self.config.environment,
                self.config.mpc,
                logger=self.logger,
            )
            # 3. Sample the physical initial condition from the declared Monte Carlo domain.
            initial_state = sample_uniform_initial_state(
                self.rng,
                self.config.data_generation,
                self.config.environment,
            )
            observation = environment.reset(initial_state)
            episode_records: list[RLStepRecord] = []
            goal_hold_count = 1 if observation.goal_reached else 0
            status = "max_steps_reached"

            # 4. Log the sampled initial condition so the episode is reproducible.
            log_event(
                self.logger,
                logging.INFO,
                identity=self.identity,
                status="running",
                action="episode_start",
                action_result="initialized",
                episode_id=episode_id,
                max_steps=self.config.experiment.max_steps,
                initial_theta_rad=f"{initial_state.theta_rad:.6f}",
                initial_omega_rad_s=f"{initial_state.omega_rad_s:.6f}",
            )

            while observation.t_index < self.config.experiment.max_steps:
                # 5. Stop only when the configured goal dwell has been satisfied.
                if (
                    self.config.experiment.stop_on_goal
                    and goal_hold_count >= self.config.environment.goal.hold_steps
                ):
                    status = "goal_reached"
                    break

                # 6. Sample one normalized burst-horizon action and solve its MPC plan.
                start_observation = observation
                monte_carlo_action = sample_uniform_monte_carlo_action(
                    self.rng,
                    self.config.data_generation,
                    self.config.environment,
                )
                selected_plan = controller.start_monte_carlo_plan(
                    start_observation,
                    monte_carlo_action,
                )
                segment_records: list[StepRecord] = []

                # 7. Execute the sampled plan as one RL transition with dense step logging.
                for _ in range(selected_plan.predicted_inputs_nm.size):
                    action = controller.select_action(observation, force_replan=False)
                    observation, step_record = environment.step(action)
                    artifact_writer.write_step(step_record)
                    step_records.append(step_record)
                    segment_records.append(step_record)
                    goal_hold_count = goal_hold_count + 1 if observation.goal_reached else 0
                    if observation.t_index >= self.config.experiment.max_steps:
                        break
                    if (
                        self.config.experiment.stop_on_goal
                        and goal_hold_count >= self.config.environment.goal.hold_steps
                    ):
                        status = "goal_reached"
                        break

                # 8. Convert the executed segment into one immediate-cost transition row.
                done = status == "goal_reached" or observation.t_index >= (
                    self.config.experiment.max_steps
                )
                terminal_fail_cost = 0.0
                if done and status != "goal_reached":
                    terminal_fail_cost = self.config.data_generation.fail_penalty
                duration_s = observation.t_sec - start_observation.t_sec
                normalized_effort = sum(
                    (record.u_applied_nm / self.config.environment.pendulum.torque_limit_nm) ** 2
                    for record in segment_records
                )
                step_cost = (
                    self.config.data_generation.time_weight * duration_s
                    + self.config.data_generation.action_weight * normalized_effort
                    + self.config.data_generation.compute_weight
                    * selected_plan.solve_time_s
                    / self.config.environment.simulation.timestep_s
                    + terminal_fail_cost
                )
                episode_records.append(
                    RLStepRecord(
                        run_id=self.config.experiment.run_id,
                        episode_id=episode_id,
                        replan_index=len(episode_records),
                        start_t_index=start_observation.t_index,
                        start_t_sec=start_observation.t_sec,
                        end_t_index=observation.t_index,
                        end_t_sec=observation.t_sec,
                        s_sin_theta=math.sin(start_observation.theta_rad),
                        s_cos_theta=math.cos(start_observation.theta_rad),
                        s_omega_rad_s=start_observation.omega_rad_s,
                        bbar=monte_carlo_action.bbar,
                        hbar=monte_carlo_action.hbar,
                        burst_steps=monte_carlo_action.burst_steps,
                        horizon_steps=monte_carlo_action.horizon_steps,
                        next_s_sin_theta=math.sin(observation.theta_rad),
                        next_s_cos_theta=math.cos(observation.theta_rad),
                        next_s_omega_rad_s=observation.omega_rad_s,
                        done=done,
                        step_cost=step_cost,
                        return_cost=0.0,
                        u_nm_json=json.dumps([record.u_applied_nm for record in segment_records]),
                        solve_time_s=selected_plan.solve_time_s,
                        plan_id=selected_plan.plan_id,
                    )
                )
                if done:
                    break

            # 9. Sweep backward through the episode to attach Monte Carlo return costs.
            returned_records: list[RLStepRecord] = []
            return_cost = 0.0
            for record in reversed(episode_records):
                return_cost = record.step_cost + self.config.data_generation.gamma * return_cost
                returned_records.append(record.model_copy(update={"return_cost": return_cost}))
            rl_records.extend(reversed(returned_records))
            # 10. Report the sparse transition count emitted by this sampled episode.
            log_event(
                self.logger,
                logging.INFO,
                identity=self.identity,
                status=status,
                action="episode_finish",
                action_result=status,
                episode_id=episode_id,
                t_index=observation.t_index,
                t_sec=observation.t_sec,
                rl_steps=len(episode_records),
            )
        return step_records, rl_records
