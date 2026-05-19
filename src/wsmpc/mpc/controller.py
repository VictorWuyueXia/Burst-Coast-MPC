"""Closed-loop CasADi split-ratio MPC action provider."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import count

import numpy as np

from wsmpc.mpc.prediction import split_candidates
from wsmpc.mpc.solver import solve_candidate_task
from wsmpc.mpc.types import (
    CandidateSolution,
    CandidateSolveTask,
    MPCControllerContext,
    SelectedPlan,
)
from wsmpc.utils.logging import log_event
from wsmpc.utils.messages import ActionCommand, StateObs
from wsmpc.utils.parallel import ordered_process_map


@dataclass
class _ActivePlan:
    """Mutable execution cursor for the selected burst-coast input sequence."""

    plan: SelectedPlan
    next_input_index: int = 0


class CasadiMPCController:
    """Compute burst-coast actions by enumerating configured split-ratio NLPs."""

    identity = "MPC"

    def __init__(
        self,
        context: MPCControllerContext,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.context = context
        self.logger = logger or logging.getLogger(__name__)
        self._active_plan: _ActivePlan | None = None
        self._last_burst_inputs_nm: np.ndarray | None = None
        self._previous_input_nm = 0.0
        self._plan_counter = count()
        self.solver_failure_count = 0

    def select_action(self, observation: StateObs) -> ActionCommand:
        """Return the next executable action for the current observation."""

        if self._active_plan is None or self._active_plan.next_input_index >= (
            self._active_plan.plan.predicted_inputs_nm.size
        ):
            selected_plan = self._solve_new_plan(observation)
            if selected_plan is None:
                return self._fallback_action(observation)
            self._active_plan = _ActivePlan(plan=selected_plan)

        active_plan = self._active_plan
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
            plan_id=active_plan.plan.plan_id,
        )

    def _solve_new_plan(self, observation: StateObs) -> SelectedPlan | None:
        """Solve all configured split candidates and choose the minimum objective."""

        state = np.asarray([observation.theta_rad, observation.omega_rad_s], dtype=np.float64)
        candidates = split_candidates(self.context.environment, self.context.mpc)
        tasks = [
            CandidateSolveTask(
                state=state,
                previous_input_nm=self._previous_input_nm,
                candidate=candidate,
                environment_config=self.context.environment,
                mpc_config=self.context.mpc,
                warm_start_nm=self._last_burst_inputs_nm,
            )
            for candidate in candidates
        ]
        solutions = self._solve_tasks(tasks)
        successful = [
            solution
            for solution in solutions
            if solution.success and math.isfinite(solution.objective_value)
        ]
        if not successful:
            self.solver_failure_count += 1
            self._log_solver_result(observation, solutions, selected=None)
            return None

        selected = min(successful, key=lambda solution: solution.objective_value)
        self._last_burst_inputs_nm = selected.burst_inputs_nm.copy()
        plan = SelectedPlan(
            plan_id=self._plan_id(observation, selected),
            candidate=selected.candidate,
            objective_value=selected.objective_value,
            burst_inputs_nm=selected.burst_inputs_nm,
            predicted_states=selected.predicted_states,
            predicted_inputs_nm=selected.predicted_inputs_nm,
            solve_time_s=sum(solution.solve_time_s for solution in solutions),
            solver_success_count=len(successful),
            solver_failure_count=len(solutions) - len(successful),
            message=selected.message,
        )
        self._log_solver_result(observation, solutions, selected=plan)
        return plan

    def _solve_tasks(self, tasks: list[CandidateSolveTask]) -> list[CandidateSolution]:
        """Solve split candidates sequentially or with configured process parallelism."""

        use_parallel = self.context.mpc.solve_candidates_in_parallel and len(tasks) > 1
        if not use_parallel:
            return [solve_candidate_task(task) for task in tasks]

        max_workers = (
            self.context.mpc.max_parallel_workers
            or self.context.runtime.max_worker_threads
        )
        return ordered_process_map(
            solve_candidate_task,
            tasks,
            max_workers=max_workers,
        )

    def _fallback_action(self, observation: StateObs) -> ActionCommand:
        """Use the configured fallback command when all candidate solves fail."""

        fallback = self.context.experiment.default_action
        self._previous_input_nm = fallback.u_nm
        return ActionCommand(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            t_index=observation.t_index,
            t_sec=observation.t_sec,
            u_nm=fallback.u_nm,
            source="mpc_failure_fallback",
        )

    def _plan_id(self, observation: StateObs, selected: CandidateSolution) -> str:
        """Build a compact deterministic plan identifier for logs and records."""

        plan_number = next(self._plan_counter)
        lambda_text = f"{selected.candidate.lambda_value:.3f}".rstrip("0").rstrip(".")
        return f"mpc-{observation.t_index}-{plan_number}-lambda-{lambda_text}"

    def _log_solver_result(
        self,
        observation: StateObs,
        solutions: list[CandidateSolution],
        *,
        selected: SelectedPlan | None,
    ) -> None:
        """Emit one structured log event per MPC decision."""

        if selected is None:
            log_event(
                self.logger,
                logging.WARNING,
                identity=self.identity,
                status="fallback",
                action="solve_split_candidates",
                action_result="all_candidates_failed",
                t_index=observation.t_index,
                t_sec=observation.t_sec,
                candidate_count=len(solutions),
            )
            return

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
            objective_value=f"{selected.objective_value:.9f}",
            solver_success_count=selected.solver_success_count,
            solver_failure_count=selected.solver_failure_count,
        )
