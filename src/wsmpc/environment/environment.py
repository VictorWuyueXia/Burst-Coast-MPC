"""Deterministic pendulum environment."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from wsmpc.environment.dynamics import (
    clip_torque,
    pendulum_energy,
    rk4_step,
    state_features,
    upright_energy,
)
from wsmpc.utils.config_schema import EnvironmentConfig, InitialStateConfig
from wsmpc.utils.log_events import log_event
from wsmpc.utils.messages import ActionCommand, StateObs, StepRecord
from wsmpc.utils.time import monotonic_s, sleep_s


class Environment:
    """Owns true state, deterministic integration, diagnostics, and optional pacing."""

    identity = "Environment"

    def __init__(
        self,
        config: EnvironmentConfig,
        *,
        run_id: str,
        episode_id: int,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.episode_id = episode_id
        self.logger = logger
        self._state = np.zeros(2, dtype=np.float64)
        self._t_index = 0

        log_event(
            self.logger,
            logging.INFO,
            identity=self.identity,
            status="initialized",
            action="create_environment",
            action_result="ready",
            timestep_s=self.config.simulation.timestep_s,
            pace_s=self.config.simulation.pace_s,
        )

    @property
    def state(self) -> NDArray[np.float64]:
        """Return a defensive copy of the current true state."""

        return self._state.copy()

    @property
    def t_index(self) -> int:
        """Return the current simulator step index."""

        return self._t_index

    @property
    def t_sec(self) -> float:
        """Return simulated physical time, independent of wall-clock time."""

        return self._t_index * self.config.simulation.timestep_s

    def reset(self, initial_state: InitialStateConfig) -> StateObs:
        """Reset true state and publish the initial observation."""

        # Store the initial condition as a compact NumPy vector for fast simulation math.
        self._state = np.asarray(
            [initial_state.theta_rad, initial_state.omega_rad_s],
            dtype=np.float64,
        )
        self._t_index = 0
        observation = self._make_observation()

        log_event(
            self.logger,
            logging.INFO,
            identity=self.identity,
            status="reset",
            action="reset",
            action_result="observation_ready",
            t_index=observation.t_index,
            t_sec=observation.t_sec,
            theta_rad=f"{observation.theta_rad:.6f}",
            omega_rad_s=f"{observation.omega_rad_s:.6f}",
        )
        return observation

    def step(self, action: ActionCommand) -> tuple[StateObs, StepRecord]:
        """Apply one action, advance true state, pace wall-clock time, and record diagnostics."""

        step_started_at = monotonic_s()
        log_event(
            self.logger,
            logging.DEBUG,
            identity=self.identity,
            status="running",
            action="step_start",
            action_result="accepted_action",
            t_index=self._t_index,
            t_sec=self.t_sec,
            action_source=action.source,
            u_commanded_nm=f"{action.u_nm:.6f}",
        )

        # Clip the command before integration so logs keep both commanded and applied action.
        applied_torque_nm = float(clip_torque(action.u_nm, self.config.pendulum))

        # Integrate one fixed computation deltaT and advance only simulated time.
        self._state = rk4_step(
            self._state,
            applied_torque_nm,
            self.config.simulation.timestep_s,
            self.config.pendulum,
        )
        self._t_index += 1
        observation = self._make_observation()

        # Enforce optional real-world pacing after computation without changing simulation results.
        compute_wall_s = monotonic_s() - step_started_at
        pace_sleep_s = max(0.0, self.config.simulation.pace_s - compute_wall_s)
        sleep_s(pace_sleep_s)

        record = StepRecord(
            run_id=self.run_id,
            episode_id=self.episode_id,
            t_index=observation.t_index,
            t_sec=observation.t_sec,
            theta_rad=observation.theta_rad,
            omega_rad_s=observation.omega_rad_s,
            energy_j=observation.energy_j,
            energy_error_j=observation.energy_error_j,
            u_commanded_nm=action.u_nm,
            u_applied_nm=applied_torque_nm,
            mode=action.source,
            plan_id=action.plan_id,
            constraint_margin=observation.constraint_margin,
            goal_flag=observation.goal_reached,
            early_wake_flag=False,
            step_compute_wall_s=compute_wall_s,
            pace_sleep_s=pace_sleep_s,
            action_result="applied",
        )

        log_event(
            self.logger,
            logging.DEBUG,
            identity=self.identity,
            status="running",
            action="step_finish",
            action_result=record.action_result,
            t_index=observation.t_index,
            t_sec=observation.t_sec,
            u_applied_nm=f"{record.u_applied_nm:.6f}",
            theta_rad=f"{observation.theta_rad:.6f}",
            omega_rad_s=f"{observation.omega_rad_s:.6f}",
            goal_reached=observation.goal_reached,
            constraint_margin=f"{observation.constraint_margin:.6f}",
        )
        return observation, record

    def _make_observation(self) -> StateObs:
        """Build a typed observation from current true state and diagnostics."""

        # Centralize all diagnostic feature calculations for consistent message semantics.
        theta_rad = float(self._state[0])
        omega_rad_s = float(self._state[1])
        energy_j = float(pendulum_energy(theta_rad, omega_rad_s, self.config.pendulum))
        feature = state_features(self._state, self.config.pendulum)
        constraint_margin = min(
            self.config.pendulum.theta_limit_abs_rad - abs(theta_rad),
            self.config.pendulum.omega_limit_abs_rad_s - abs(omega_rad_s),
        )
        wrapped_angle_error_rad = float(feature[1])
        goal_reached = (
            abs(wrapped_angle_error_rad) <= self.config.goal.angle_tolerance_rad
            and abs(omega_rad_s) <= self.config.goal.omega_tolerance_rad_s
        )

        return StateObs(
            run_id=self.run_id,
            episode_id=self.episode_id,
            t_index=self._t_index,
            t_sec=self.t_sec,
            theta_rad=theta_rad,
            omega_rad_s=omega_rad_s,
            energy_j=energy_j,
            energy_error_j=energy_j - upright_energy(self.config.pendulum),
            wrapped_angle_error_rad=wrapped_angle_error_rad,
            constraint_margin=float(constraint_margin),
            goal_reached=goal_reached,
        )
