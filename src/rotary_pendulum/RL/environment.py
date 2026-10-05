"""Vectorized direct-torque PPO environment around the validated rotary dynamics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rotary_pendulum.environment.dynamics import derive_model, rk4_step, state_derivative
from rotary_pendulum.utils.config_schema import EpisodeConfig, PPOConfig

THETA_LIMIT_RAD = 0.5 * np.pi
ENERGY_TRANSITION_WIDTH = 0.25
PHASE_ORIGIN_ENERGY_RATIO = 0.05
PHASE_ORIGIN_SPEED_RATIO = 0.01
REWARD_COMPONENT_NAMES = (
    "progress",
    "phase",
    "torque_slew",
    "torque_effort",
    "arm_boundary",
    "terminal",
)


class BatchedRotaryPPOEnvironment:
    """Own batched physical state, curriculum resets, reward, and termination."""

    def __init__(
        self,
        episode: EpisodeConfig,
        ppo: PPOConfig,
        environment_count: int,
        goal_hold_steps: int,
        seed: int,
    ) -> None:
        self.episode = episode
        self.ppo = ppo
        self.environment_count = environment_count
        self.goal_hold_steps = goal_hold_steps
        self.rng = np.random.default_rng(seed)
        self.model = derive_model(episode.rotary_pendulum)
        self.target_energy_j = 2.0 * self.model.gravity_torque_nm
        self.pendulum_speed_scale = 2.0 * np.sqrt(
            self.model.gravity_torque_nm / self.model.pendulum_inertia_kg_m2
        )
        self.arm_speed_scale = THETA_LIMIT_RAD * self.model.natural_frequency_rad_s
        self.state = np.zeros((environment_count, 4), dtype=np.float64)
        self.previous_action = np.zeros(environment_count, dtype=np.float64)
        self.physics_steps = np.zeros(environment_count, dtype=np.int64)
        self.goal_hold_count = np.zeros(environment_count, dtype=np.int64)
        self.live = np.zeros(environment_count, dtype=bool)
        self.reset(np.ones(environment_count, dtype=bool), stage=0, exact=False)

    def reset(self, mask: ArrayLike, *, stage: int, exact: bool) -> NDArray[np.float32]:
        """Reset selected environments from one documented curriculum distribution."""

        reset_mask = np.asarray(mask, dtype=bool).reshape(self.environment_count)
        indices = np.flatnonzero(reset_mask)
        if stage not in (0, 1, 2):
            raise ValueError(f"PPO curriculum stage must be 0, 1, or 2, got {stage}")
        if exact:
            initial = self.episode.experiment.initial_state
            self.state[indices] = np.array(
                [initial.theta_rad, initial.alpha_rad, initial.omega_rad_s, initial.nu_rad_s]
            )
        else:
            if stage == 0:
                groups = [(indices, "near-upright")]
            elif stage == 1:
                downward = self.rng.random(indices.size) < self.ppo.stage_b_downward_fraction
                groups = [
                    (indices[downward], "downward-noise"),
                    (indices[~downward], "phase"),
                ]
            else:
                exact_group = self.rng.random(indices.size) < self.ppo.stage_c_exact_fraction
                initial = self.episode.experiment.initial_state
                self.state[indices[exact_group]] = np.array(
                    [initial.theta_rad, initial.alpha_rad, initial.omega_rad_s, initial.nu_rad_s]
                )
                groups = [(indices[~exact_group], "near-upright")]
            for group_indices, name in groups:
                bounds = self.ppo.initialization_bounds[name]
                samples = self.rng.uniform(bounds[::2], bounds[1::2], size=(group_indices.size, 4))
                if name == "near-upright":
                    samples[:, 1] += np.pi
                self.state[group_indices] = samples
        self.previous_action[indices] = 0.0
        self.physics_steps[indices] = 0
        self.goal_hold_count[indices] = 0
        self.live[indices] = True
        return self.observe()

    def observe(self) -> NDArray[np.float32]:
        """Encode every physical state into the fixed eight-feature policy input."""

        theta, alpha, omega, nu = np.moveaxis(self.state, -1, 0)
        swing_energy = (
            0.5 * self.model.pendulum_inertia_kg_m2 * nu** 2
            + self.model.gravity_torque_nm * (1.0 - np.cos(alpha))
        )
        energy_error = (swing_energy - self.target_energy_j) / self.target_energy_j
        energy_ratio = swing_energy / self.target_energy_j
        origin_speed = PHASE_ORIGIN_SPEED_RATIO * self.pendulum_speed_scale
        phase_velocity = nu + origin_speed * np.exp(
            -((energy_ratio / PHASE_ORIGIN_ENERGY_RATIO) ** 2)
        )
        coupled_velocity = np.cos(alpha) * phase_velocity
        phase = coupled_velocity / np.sqrt(coupled_velocity**2 + origin_speed**2)
        return np.column_stack(
            (
                theta / THETA_LIMIT_RAD,
                np.sin(alpha),
                np.cos(alpha),
                omega / self.arm_speed_scale,
                nu / self.pendulum_speed_scale,
                np.tanh(energy_error),
                phase,
                self.previous_action,
            )
        ).astype(np.float32)

    def task_potential(self, state: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the bounded energy-gated complete-state task potential."""

        state_array = np.asarray(state, dtype=np.float64)
        theta, alpha, omega, nu = np.moveaxis(state_array, -1, 0)
        swing_energy = (
            0.5 * self.model.pendulum_inertia_kg_m2 * nu** 2
            + self.model.gravity_torque_nm * (1.0 - np.cos(alpha))
        )
        energy_error = (swing_energy - self.target_energy_j) / self.target_energy_j
        beta = np.arctan2(np.sin(alpha - np.pi), np.cos(alpha - np.pi))
        local_error = (
            (beta / np.pi) ** 2
            + (nu / self.pendulum_speed_scale) ** 2
            + 0.1 * ((theta / THETA_LIMIT_RAD) ** 2 + (omega / self.arm_speed_scale) ** 2)
        )
        local_gate = np.exp(-((energy_error / ENERGY_TRANSITION_WIDTH) ** 2))
        return np.asarray(
            2.0 * energy_error**2 / (1.0 + energy_error**2)
            + local_gate * local_error / (1.0 + local_error),
            dtype=np.float64,
        )

    def step(
        self, action: ArrayLike
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.int8],
    ]:
        """Hold bounded torque for one RL interval and return decomposed rewards."""

        normalized_action = np.asarray(action, dtype=np.float64).reshape(self.environment_count)
        if not np.isfinite(normalized_action).all() or np.any(np.abs(normalized_action) >= 1.0):
            raise ValueError("PPO normalized actions must be finite and strictly inside (-1, 1)")
        starting_potential = self.task_potential(self.state)
        torque_nm = self.episode.rotary_pendulum.torque_limit_nm * normalized_action
        starting_live = self.live.copy()
        active = starting_live.copy()
        terminal_reason = np.zeros(self.environment_count, dtype=np.int8)
        phase_cost_sum = np.zeros(self.environment_count, dtype=np.float64)

        for _ in range(self.ppo.action_repeat_steps):
            alpha = self.state[:, 1]
            nu = self.state[:, 3]
            swing_energy = (
                0.5 * self.model.pendulum_inertia_kg_m2 * nu** 2
                + self.model.gravity_torque_nm * (1.0 - np.cos(alpha))
            )
            energy_error = (swing_energy - self.target_energy_j) / self.target_energy_j
            energy_ratio = swing_energy / self.target_energy_j
            energy_command = -np.tanh(energy_error / ENERGY_TRANSITION_WIDTH)
            origin_speed = PHASE_ORIGIN_SPEED_RATIO * self.pendulum_speed_scale
            phase_velocity = nu + origin_speed * np.exp(
                -((energy_ratio / PHASE_ORIGIN_ENERGY_RATIO) ** 2)
            )
            coupled_velocity = np.cos(alpha) * phase_velocity
            phase = coupled_velocity / np.sqrt(coupled_velocity**2 + origin_speed**2)
            determinant = (
                self.model.pendulum_inertia_kg_m2
                * (
                    self.model.base_inertia_kg_m2
                    + self.model.pendulum_inertia_kg_m2 * np.sin(alpha) ** 2
                )
                - self.model.coupling_inertia_kg_m2**2 * np.cos(alpha) ** 2
            )
            acceleration_authority = (
                self.model.pendulum_inertia_kg_m2
                * self.episode.rotary_pendulum.torque_limit_nm
                / determinant
            )
            arm_acceleration = state_derivative(
                self.state,
                torque_nm,
                self.episode.rotary_pendulum,
                self.model,
            )[:, 2]
            phase_error = arm_acceleration / acceleration_authority + energy_command * phase
            phase_cost_sum += active * (
                energy_command**2 * np.cos(alpha) ** 2 * phase_error**2 / (1.0 + phase_error**2)
            )

            advanced_state = rk4_step(
                self.state,
                torque_nm,
                self.episode.simulation.timestep_s,
                self.episode.rotary_pendulum,
                self.model,
            )
            self.state[active] = advanced_state[active]
            self.physics_steps[active] += 1
            theta, alpha, omega, nu = np.moveaxis(self.state, -1, 0)
            beta = np.arctan2(np.sin(alpha - np.pi), np.cos(alpha - np.pi))
            goal = self.episode.goal
            inside_goal = (
                (np.abs(theta) <= goal.theta_tolerance_rad)
                & (np.abs(beta) <= goal.beta_tolerance_rad)
                & (np.abs(omega) <= goal.omega_tolerance_rad_s)
                & (np.abs(nu) <= goal.nu_tolerance_rad_s)
            )
            self.goal_hold_count[active] = np.where(
                inside_goal[active], self.goal_hold_count[active] + 1, 0
            )
            success = active & (self.goal_hold_count >= self.goal_hold_steps)
            terminal_reason[success] = 1
            active[success] = False
            arm_failure = active & (np.abs(theta) >= THETA_LIMIT_RAD)
            terminal_reason[arm_failure] = 2
            active[arm_failure] = False
            timeout = active & (self.physics_steps >= self.episode.experiment.max_steps)
            terminal_reason[timeout] = 3
            active[timeout] = False

        ending_potential = self.task_potential(self.state)
        components = np.column_stack(
            (
                starting_live * (starting_potential - ending_potential),
                -(
                    starting_live
                    * self.ppo.phase_weight
                    * phase_cost_sum
                    / self.ppo.action_repeat_steps
                ),
                -(
                    starting_live
                    * self.ppo.torque_slew_weight
                    * (normalized_action - self.previous_action) ** 2
                ),
                -(starting_live * self.ppo.torque_effort_weight * normalized_action**2),
                -(
                    starting_live
                    * self.ppo.arm_boundary_weight
                    * (self.state[:, 0] / THETA_LIMIT_RAD) ** 8
                ),
                self.ppo.success_bonus * (terminal_reason == 1)
                - self.ppo.arm_failure_penalty * (terminal_reason == 2)
                - self.ppo.timeout_penalty * (terminal_reason == 3),
            )
        )
        done = terminal_reason != 0
        self.previous_action[starting_live] = normalized_action[starting_live]
        self.live[done] = False
        return (
            self.observe(),
            components.sum(axis=1).astype(np.float32),
            components.astype(np.float32),
            done,
            terminal_reason,
        )
