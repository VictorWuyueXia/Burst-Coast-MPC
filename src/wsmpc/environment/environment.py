"""Deterministic pendulum environment."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from wsmpc.config.schema import EnvironmentConfig, InitialStateConfig
from wsmpc.core.logging import log_event
from wsmpc.core.messages import ActionCommand, StateObs, StepRecord
from wsmpc.core.time import monotonic_s, sleep_s
from wsmpc.environment.dynamics import (
    clip_torque,
    pendulum_energy,
    rk4_step,
    state_features,
    upright_energy,
)


class Environment:
    """Owns true state, deterministic integration, diagnostics, and optional pacing."""

    identity = "Environment"

    def __init__(
        self,
        config: EnvironmentConfig,
        *,
        run_id: str,
        episode_id: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.episode_id = episode_id
        self.logger = logger or logging.getLogger(__name__)
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
        self._state = np.asarray([initial_state.theta, initial_state.omega], dtype=np.float64)
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
            theta=f"{observation.theta:.6f}",
            omega=f"{observation.omega:.6f}",
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
            u_commanded=f"{action.u:.6f}",
        )

        # Clip the command before integration so logs keep both commanded and applied action.
        applied_torque = float(clip_torque(action.u, self.config.pendulum))

        # Integrate one fixed computation deltaT and advance only simulated time.
        self._state = rk4_step(
            self._state,
            applied_torque,
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
            theta=observation.theta,
            omega=observation.omega,
            energy=observation.energy,
            energy_error=observation.energy_error,
            u_commanded=action.u,
            u_applied=applied_torque,
            mode=action.source,
            plan_id=action.plan_id,
            constraint_margin=observation.constraint_margin,
            goal_flag=observation.goal_reached,
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
            u_applied=f"{record.u_applied:.6f}",
            theta=f"{observation.theta:.6f}",
            omega=f"{observation.omega:.6f}",
            goal_reached=observation.goal_reached,
            constraint_margin=f"{observation.constraint_margin:.6f}",
        )
        return observation, record

    def rollout(self, x0: NDArray[np.float64], actions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Roll out a deterministic action sequence with preallocated trajectory storage."""

        # Validate rollout size before allocation to avoid accidental large memory requests.
        action_array = np.asarray(actions, dtype=np.float64)
        if action_array.ndim != 1:
            msg = "actions must be a one-dimensional array"
            raise ValueError(msg)
        if action_array.size > self.config.simulation.max_rollout_steps:
            msg = "actions exceed max-rollout-steps"
            raise ValueError(msg)

        # The recurrence is sequential, while each dynamics evaluation uses NumPy math internally.
        trajectory = np.empty((action_array.size + 1, 2), dtype=np.float64)
        trajectory[0] = np.asarray(x0, dtype=np.float64)
        for index, torque in enumerate(action_array):
            trajectory[index + 1] = rk4_step(
                trajectory[index],
                float(torque),
                self.config.simulation.timestep_s,
                self.config.pendulum,
            )
        return trajectory

    def _make_observation(self) -> StateObs:
        """Build a typed observation from current true state and diagnostics."""

        # Centralize all diagnostic feature calculations for consistent message semantics.
        theta = float(self._state[0])
        omega = float(self._state[1])
        energy = float(pendulum_energy(theta, omega, self.config.pendulum))
        feature = state_features(self._state, self.config.pendulum)
        constraint_margin = min(
            self.config.pendulum.theta_limit_abs_rad - abs(theta),
            self.config.pendulum.omega_limit_abs_rad_s - abs(omega),
        )
        wrapped_angle_error = float(feature[1])
        goal_reached = (
            abs(wrapped_angle_error) <= self.config.goal.angle_tolerance_rad
            and abs(omega) <= self.config.goal.omega_tolerance_rad_s
        )

        return StateObs(
            run_id=self.run_id,
            episode_id=self.episode_id,
            t_index=self._t_index,
            t_sec=self.t_sec,
            theta=theta,
            omega=omega,
            energy=energy,
            energy_error=energy - upright_energy(self.config.pendulum),
            wrapped_angle_error=wrapped_angle_error,
            constraint_margin=float(constraint_margin),
            goal_reached=goal_reached,
        )
