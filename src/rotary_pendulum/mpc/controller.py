"""Plan-level controller for strict rotary burst-coast MPC execution."""

from __future__ import annotations

import numpy as np

from rotary_pendulum.environment.dynamics import derive_model
from rotary_pendulum.mpc.discrete_model import split_candidates
from rotary_pendulum.mpc.solver import build_single_shooting_problem, solve_candidate
from rotary_pendulum.utils.config_schema import EpisodeConfig, MPCConfig
from rotary_pendulum.utils.messages import ActionPlan, CandidateRecord, StateObservation


class RotaryMPCController:
    """Solve every configured split and select the first minimum-cost plan."""

    def __init__(self, physics: EpisodeConfig, mpc: MPCConfig) -> None:
        self.physics = physics
        self.mpc = mpc
        self._problems = [
            build_single_shooting_problem(candidate, physics, mpc)
            for candidate in split_candidates(physics, mpc)
        ]

    def solve_plan(
        self,
        observation: StateObservation,
        previous_applied_torque_nm: float,
        replan_index: int,
    ) -> ActionPlan:
        """Solve all candidates from one measured state and return the accepted plan."""

        # Give every split the identical measured state and prior physical actuator value.
        state = np.asarray(
            [
                observation.theta_rad,
                observation.alpha_rad,
                observation.omega_rad_s,
                observation.nu_rad_s,
            ],
            dtype=np.float64,
        )
        results = [
            solve_candidate(
                state,
                previous_applied_torque_nm,
                problem,
                self.physics,
            )
            for problem in self._problems
        ]

        # Commit every candidate-local primal solution only after every solve succeeds.
        for problem, result in zip(self._problems, results, strict=True):
            problem.initial_guess = result.burst_inputs_nm.copy()

        # Python's stable minimum preserves configuration order for exact objective ties.
        selected_index = min(range(len(results)), key=lambda index: results[index].objective_value)
        selected = results[selected_index]
        split_text = f"{selected.candidate.split_ratio:.6f}".rstrip("0").rstrip(".")
        total_solve_time_s = sum(result.solve_time_s for result in results)
        natural_period_s = derive_model(self.physics.rotary_pendulum).natural_period_s
        return ActionPlan(
            plan_id=f"rotary-mpc-{replan_index:06d}-lambda-{split_text}",
            replan_index=replan_index,
            hbar=(
                selected.candidate.total_steps
                * self.physics.simulation.timestep_s
                / natural_period_s
            ),
            bbar=selected.candidate.burst_steps / selected.candidate.total_steps,
            horizon_steps=selected.candidate.total_steps,
            burst_steps=selected.candidate.burst_steps,
            coast_steps=selected.candidate.coast_steps,
            objective_value=selected.objective_value,
            solve_time_s=total_solve_time_s,
            solver_status=selected.solver_status,
            torques_nm=selected.predicted_inputs_nm,
            predicted_states=selected.predicted_states,
            candidates=tuple(
                CandidateRecord(
                    split_ratio=result.candidate.split_ratio,
                    horizon_steps=result.candidate.total_steps,
                    burst_steps=result.candidate.burst_steps,
                    coast_steps=result.candidate.coast_steps,
                    objective_value=result.objective_value,
                    solve_time_s=result.solve_time_s,
                    solver_status=result.solver_status,
                    selected=index == selected_index,
                )
                for index, result in enumerate(results)
            ),
        )
