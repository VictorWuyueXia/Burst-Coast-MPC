"""Closed-loop natural-period MPC controller."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import count

import numpy as np

from wsmpc.mpc.ip_dynamics_natural_period.problem import (
    solve_candidate,
    split_candidates,
)
from wsmpc.mpc.types import CandidateSolution, SelectedPlan, SplitCandidate
from wsmpc.utils.config_schema import EnvironmentConfig, MPCConfig
from wsmpc.utils.log_events import log_event
from wsmpc.utils.messages import ActionCommand, StateObs
from wsmpc.utils.monte_carlo import MonteCarloAction

MPC_WORKER_COUNT = 1


@dataclass
class _ActivePlan:
    """Mutable execution cursor for the selected burst-coast input sequence."""

    plan: SelectedPlan
    next_input_index: int = 0


class NaturalPeriodMPCController:
    """Compute burst-coast actions by enumerating natural-period split NLPs."""

    identity = "IP-dynamics-naturalPeriod"

    def __init__(
        self,
        environment: EnvironmentConfig,
        mpc: MPCConfig,
        *,
        logger: logging.Logger,
    ) -> None:
        self.environment = environment
        self.mpc = mpc
        self.logger = logger
        self._active_plan: _ActivePlan | None = None
        self._previous_input_nm = 0.0
        self._plan_counter = count()

    def select_action(self, observation: StateObs, *, force_replan: bool) -> ActionCommand:
        """Return the next executable action for the current observation."""

        if force_replan or self._active_plan is None or self._active_plan.next_input_index >= (
            self._active_plan.plan.predicted_inputs_nm.size
        ):
            selected_plan = self._solve_new_plan(observation)
            self._active_plan = _ActivePlan(plan=selected_plan)

        active_plan = self._active_plan
        assert active_plan is not None
        input_index = active_plan.next_input_index
        torque_nm = float(active_plan.plan.predicted_inputs_nm[input_index])
        source = (
            "mpc_burst"
            if input_index < active_plan.plan.candidate.burst_steps
            else "mpc_coast"
        )
        active_plan.next_input_index += 1
        self._previous_input_nm = torque_nm
        return ActionCommand(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            t_index=observation.t_index,
            t_sec=observation.t_sec,
            u_nm=torque_nm,
            source=source,
            early_wake_flag=force_replan,
            plan_id=active_plan.plan.plan_id,
        )

    def start_monte_carlo_plan(
        self,
        observation: StateObs,
        monte_carlo_action: MonteCarloAction,
    ) -> SelectedPlan:
        """Solve one sampled burst-coast candidate and make it the active plan."""

        state = np.asarray([observation.theta_rad, observation.omega_rad_s], dtype=np.float64)
        candidate = SplitCandidate(
            lambda_value=monte_carlo_action.bbar,
            total_steps=monte_carlo_action.horizon_steps,
            burst_steps=monte_carlo_action.burst_steps,
            coast_steps=monte_carlo_action.coast_steps,
        )
        selected = solve_candidate(
            state,
            self._previous_input_nm,
            candidate,
            self.environment,
        )
        plan = self._selected_plan(observation, selected, [selected])
        self._active_plan = _ActivePlan(plan=plan)
        log_event(
            self.logger,
            logging.INFO,
            identity=self.identity,
            status="ready",
            action="solve_monte_carlo_candidate",
            action_result="plan_selected",
            t_index=observation.t_index,
            t_sec=observation.t_sec,
            plan_id=plan.plan_id,
            bbar=f"{monte_carlo_action.bbar:.6f}",
            hbar=f"{monte_carlo_action.hbar:.6f}",
            burst_steps=monte_carlo_action.burst_steps,
            horizon_steps=monte_carlo_action.horizon_steps,
            solve_time_s=f"{plan.solve_time_s:.6f}",
            objective_value=f"{plan.objective_value:.9f}",
        )
        return plan

    def _solve_new_plan(self, observation: StateObs) -> SelectedPlan:
        """Solve all split candidates in parallel and choose the minimum objective."""

        state = np.asarray([observation.theta_rad, observation.omega_rad_s], dtype=np.float64)
        candidates = split_candidates(self.environment, self.mpc)
        # Serial candidate solves keep runtime policy fixed and inspection straightforward.
        solutions = [
            solve_candidate(state, self._previous_input_nm, candidate, self.environment)
            for candidate in candidates
        ]
        selected = min(solutions, key=lambda solution: solution.objective_value)
        plan = self._selected_plan(observation, selected, solutions)
        self._log_solver_result(observation, plan)
        return plan

    def _selected_plan(
        self,
        observation: StateObs,
        selected: CandidateSolution,
        solutions: list[CandidateSolution],
    ) -> SelectedPlan:
        """Convert the best candidate solve into an executable plan."""

        return SelectedPlan(
            plan_id=self._plan_id(observation, selected),
            candidate=selected.candidate,
            objective_value=selected.objective_value,
            burst_inputs_nm=selected.burst_inputs_nm,
            predicted_states=selected.predicted_states,
            predicted_inputs_nm=selected.predicted_inputs_nm,
            solve_time_s=sum(solution.solve_time_s for solution in solutions),
            message=selected.message,
        )

    def _plan_id(self, observation: StateObs, selected: CandidateSolution) -> str:
        """Build a compact deterministic plan identifier for logs and records."""

        plan_number = next(self._plan_counter)
        lambda_text = f"{selected.candidate.lambda_value:.3f}".rstrip("0").rstrip(".")
        return f"mpc-{observation.t_index}-{plan_number}-lambda-{lambda_text}"

    def _log_solver_result(self, observation: StateObs, selected: SelectedPlan) -> None:
        """Emit one structured log event per MPC decision."""

        log_event(
            self.logger,
            logging.INFO,
            identity=self.identity,
            status="ready",
            action="solve_split_candidates",
            action_result="plan_selected",
            t_index=observation.t_index,
            t_sec=observation.t_sec,
            plan_id=selected.plan_id,
            lambda_value=f"{selected.candidate.lambda_value:.6f}",
            burst_steps=selected.candidate.burst_steps,
            coast_steps=selected.candidate.coast_steps,
            worker_count=MPC_WORKER_COUNT,
            objective_value=f"{selected.objective_value:.9f}",
        )
