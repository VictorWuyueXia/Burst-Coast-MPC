"""RL transition construction for executed burst-coast segments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from inverted_pendulum.mpc.controller import SelectedPlan
from inverted_pendulum.RL.policy import RLActionSelection
from inverted_pendulum.utils.config_schema import EnvironmentConfig, RLConfig
from inverted_pendulum.utils.messages import RLStepRecord, StateObs, StepRecord


@dataclass
class RLSegmentState:
    """Mutable record of one RL-selected plan while its MPC inputs execute."""

    start_observation: StateObs
    selection: RLActionSelection
    selected_plan: SelectedPlan
    records: list[StepRecord]

    def close(
        self,
        *,
        run_id: str,
        episode_id: int,
        replan_index: int,
        end_observation: StateObs,
        done: bool,
        terminal_status: str,
        environment_config: EnvironmentConfig,
        rl_config: RLConfig,
    ) -> RLStepRecord:
        """Convert the executed segment into one return-ready RL transition."""

        normalized_effort = sum(
            (record.u_applied_nm / environment_config.pendulum.torque_limit_nm) ** 2
            for record in self.records
        )
        terminal_fail_cost = 0.0
        if done and terminal_status != "goal_reached":
            terminal_fail_cost = rl_config.fail_penalty
        step_cost = (
            rl_config.time_weight * (end_observation.t_sec - self.start_observation.t_sec)
            + rl_config.action_weight * normalized_effort
            + rl_config.compute_weight
            * self.selected_plan.solve_time_s
            / environment_config.simulation.timestep_s
            + terminal_fail_cost
        )
        action = self.selection.action
        return RLStepRecord(
            run_id=run_id,
            episode_id=episode_id,
            replan_index=replan_index,
            start_t_index=self.start_observation.t_index,
            start_t_sec=self.start_observation.t_sec,
            end_t_index=end_observation.t_index,
            end_t_sec=end_observation.t_sec,
            s_sin_theta=math.sin(self.start_observation.theta_rad),
            s_cos_theta=math.cos(self.start_observation.theta_rad),
            s_omega_rad_s=self.start_observation.omega_rad_s,
            bbar=action.bbar,
            hbar=action.hbar,
            burst_steps=action.burst_steps,
            horizon_steps=action.horizon_steps,
            next_s_sin_theta=math.sin(end_observation.theta_rad),
            next_s_cos_theta=math.cos(end_observation.theta_rad),
            next_s_omega_rad_s=end_observation.omega_rad_s,
            done=done,
            step_cost=step_cost,
            return_cost=0.0,
            u_nm_json=json.dumps([record.u_applied_nm for record in self.records]),
            solve_time_s=self.selected_plan.solve_time_s,
            plan_id=self.selected_plan.plan_id,
        )
