"""Deterministic four-state rotary-pendulum simulation environment."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from rotary_pendulum.environment.dynamics import derive_model, rk4_step
from rotary_pendulum.utils.config_schema import EpisodeConfig
from rotary_pendulum.utils.messages import ActionPlan, StateObservation, StepRecord


class RotaryPendulumEnvironment:
    """Own true state, RK4 integration, goal evaluation, and realtime pacing."""

    def __init__(self, config: EpisodeConfig) -> None:
        # Bind one validated physical source of truth and its derived model constants.
        self.config = config
        self.model = derive_model(config.rotary_pendulum)
        self._state = np.zeros(4, dtype=np.float64)
        self._t_index = 0
        self._goal_hold_count = 0

    @property
    def state(self) -> NDArray[np.float64]:
        """Return a defensive copy of the current true state."""

        return self._state.copy()

    @property
    def t_index(self) -> int:
        """Return the current discrete simulation index."""

        return self._t_index

    @property
    def t_sec(self) -> float:
        """Return simulated physical time independently of wall-clock jitter."""

        return self._t_index * self.config.simulation.timestep_s

    def reset(self) -> StateObservation:
        """Reset to the configured unwrapped initial state and emit its observation."""

        # Store the complete minimal state in governing-equation coordinate order.
        initial = self.config.experiment.initial_state
        self._state = np.array(
            [initial.theta_rad, initial.alpha_rad, initial.omega_rad_s, initial.nu_rad_s],
            dtype=np.float64,
        )
        self._t_index = 0
        self._goal_hold_count = 0
        return self._make_observation()

    def step(
        self,
        commanded_torque_nm: float,
        plan: ActionPlan,
        *,
        replan_flag: bool,
    ) -> tuple[StateObservation, StepRecord]:
        """Apply one bounded torque, advance RK4 dynamics, and record the transition."""

        # Reject invalid controller diagnostics instead of contaminating the physical rollout.
        if not np.isfinite(commanded_torque_nm):
            raise ValueError(f"Commanded torque must be finite, got {commanded_torque_nm}")
        # Saturate only at the physical plant boundary while retaining the original command.
        step_started_at = time.perf_counter()
        torque_limit_nm = self.config.rotary_pendulum.torque_limit_nm
        applied_torque_nm = float(np.clip(commanded_torque_nm, -torque_limit_nm, torque_limit_nm))
        self._state = rk4_step(
            self._state,
            applied_torque_nm,
            self.config.simulation.timestep_s,
            self.config.rotary_pendulum,
            self.model,
        )
        self._t_index += 1
        observation = self._make_observation()

        # Pace completed physics steps without coupling simulated time to rendering latency.
        elapsed_wall_s = time.perf_counter() - step_started_at
        time.sleep(max(0.0, self.config.simulation.pace_s - elapsed_wall_s))

        # Preserve the physical observation and sparse plan diagnostics in one flat record.
        record = StepRecord(
            t_index=observation.t_index,
            t_sec=observation.t_sec,
            theta_rad=observation.theta_rad,
            alpha_rad=observation.alpha_rad,
            omega_rad_s=observation.omega_rad_s,
            nu_rad_s=observation.nu_rad_s,
            kinetic_energy_j=observation.kinetic_energy_j,
            potential_energy_j=observation.potential_energy_j,
            energy_j=observation.energy_j,
            energy_error_j=observation.energy_error_j,
            normalized_energy_error=observation.normalized_energy_error,
            beta_rad=observation.beta_rad,
            goal_reached=observation.goal_reached,
            u_commanded_nm=float(commanded_torque_nm),
            u_applied_nm=applied_torque_nm,
            plan_id=plan.plan_id,
            replan_index=plan.replan_index,
            hbar=plan.hbar,
            bbar=plan.bbar,
            horizon_steps=plan.horizon_steps,
            burst_steps=plan.burst_steps,
            coast_steps=plan.coast_steps,
            objective_value=plan.objective_value,
            solver_status=plan.solver_status,
            solve_time_s=plan.solve_time_s if replan_flag else 0.0,
            replan_flag=replan_flag,
        )
        return observation, record

    def _make_observation(self) -> StateObservation:
        """Build one typed state, energy, phase, and held-goal observation."""

        # Compute objective-aligned swing diagnostics from the integrator's unwrapped state.
        theta_rad, alpha_rad, omega_rad_s, nu_rad_s = self._state
        kinetic_energy_j = 0.5 * self.model.pendulum_inertia_kg_m2 * nu_rad_s**2
        potential_energy_j = self.model.gravity_torque_nm * (1.0 - np.cos(alpha_rad))
        swing_energy_j = kinetic_energy_j + potential_energy_j
        beta_rad = float(np.arctan2(np.sin(alpha_rad - np.pi), np.cos(alpha_rad - np.pi)))

        # Require consecutive complete-state membership before declaring episode success.
        goal = self.config.goal
        inside_goal = (
            abs(theta_rad) <= goal.theta_tolerance_rad
            and abs(beta_rad) <= goal.beta_tolerance_rad
            and abs(omega_rad_s) <= goal.omega_tolerance_rad_s
            and abs(nu_rad_s) <= goal.nu_tolerance_rad_s
        )
        self._goal_hold_count = self._goal_hold_count + 1 if inside_goal else 0
        energy_j = float(swing_energy_j)
        target_energy_j = 2.0 * self.model.gravity_torque_nm
        energy_error_j = energy_j - target_energy_j
        return StateObservation(
            t_index=self._t_index,
            t_sec=self.t_sec,
            theta_rad=float(theta_rad),
            alpha_rad=float(alpha_rad),
            omega_rad_s=float(omega_rad_s),
            nu_rad_s=float(nu_rad_s),
            kinetic_energy_j=float(kinetic_energy_j),
            potential_energy_j=float(potential_energy_j),
            energy_j=energy_j,
            energy_error_j=energy_error_j,
            normalized_energy_error=energy_error_j / target_energy_j,
            beta_rad=beta_rad,
            goal_reached=self._goal_hold_count >= goal.hold_steps,
        )
