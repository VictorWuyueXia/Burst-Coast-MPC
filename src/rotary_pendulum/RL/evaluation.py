"""Deterministic strict-dwell evaluation for rotary PPO checkpoints."""

from __future__ import annotations

import math

import numpy as np
import torch

from rotary_pendulum.RL.environment import BatchedRotaryPPOEnvironment
from rotary_pendulum.RL.model import RotaryActorCritic
from rotary_pendulum.utils.config_schema import EpisodeConfig, PPOConfig


def evaluate_policy(
    episode: EpisodeConfig,
    ppo: PPOConfig,
    model: RotaryActorCritic,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    """Run deterministic exact-downward episodes with the strict dwell diagnostic."""

    environment = BatchedRotaryPPOEnvironment(
        episode,
        ppo,
        ppo.evaluation_episodes,
        ppo.evaluation_hold_steps,
        ppo.random_seed + 1,
    )
    environment.reset(np.ones(ppo.evaluation_episodes, dtype=bool), stage=2, exact=True)
    if ppo.evaluation_episodes > 1:
        bounds = ppo.initialization_bounds["downward-noise"]
        evaluation_rng = np.random.default_rng(ppo.random_seed + 2)
        environment.state[1:] = evaluation_rng.uniform(
            bounds[::2], bounds[1::2], size=(ppo.evaluation_episodes - 1, 4)
        )
    observation = environment.observe()
    initial_observation = torch.as_tensor(observation, dtype=torch.float32, device=device)
    torch.set_grad_enabled(False)
    _, initial_value = model(initial_observation)
    torch.set_grad_enabled(True)
    live = np.ones(ppo.evaluation_episodes, dtype=bool)
    total_return = np.zeros(ppo.evaluation_episodes, dtype=np.float64)
    discounted_return = np.zeros(ppo.evaluation_episodes, dtype=np.float64)
    discount = np.ones(ppo.evaluation_episodes, dtype=np.float64)
    effort = np.zeros(ppo.evaluation_episodes, dtype=np.float64)
    reason = np.zeros(ppo.evaluation_episodes, dtype=np.int8)
    completion_time = np.full(
        ppo.evaluation_episodes,
        episode.experiment.max_steps * episode.simulation.timestep_s,
        dtype=np.float64,
    )
    trajectory: list[dict[str, float | int]] = []
    gamma = math.exp(
        -ppo.action_repeat_steps * episode.simulation.timestep_s / ppo.discount_time_constant_s
    )
    maximum_decisions = math.ceil(episode.experiment.max_steps / ppo.action_repeat_steps)
    for decision in range(maximum_decisions):
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        torch.set_grad_enabled(False)
        _, action, _, value = model.sample(observation_tensor, deterministic=True)
        torch.set_grad_enabled(True)
        action_array = action.detach().cpu().numpy()
        next_observation, reward, _, done, terminal_reason = environment.step(action_array)
        total_return[live] += reward[live]
        discounted_return[live] += discount[live] * reward[live]
        discount[live] *= gamma
        effort[live] += action_array[live] ** 2
        newly_done = live & done
        reason[newly_done] = terminal_reason[newly_done]
        completion_time[newly_done] = (
            environment.physics_steps[newly_done] * episode.simulation.timestep_s
        )
        state = environment.state[0]
        trajectory.append(
            {
                "decision": decision,
                "time_s": float(environment.physics_steps[0] * episode.simulation.timestep_s),
                "theta_rad": float(state[0]),
                "beta_rad": float(np.arctan2(np.sin(state[1] - np.pi), np.cos(state[1] - np.pi))),
                "omega_rad_s": float(state[2]),
                "nu_rad_s": float(state[3]),
                "normalized_action": float(action_array[0]),
                "reward": float(reward[0]),
                "value": float(value[0].detach().cpu()),
            }
        )
        live[newly_done] = False
        observation = next_observation
        if not np.any(live):
            break
    success = reason == 1
    median_success_time_s = (
        float(np.median(completion_time[success]))
        if np.any(success)
        else episode.experiment.max_steps * episode.simulation.timestep_s
    )
    return (
        {
            "exact_success": float(success[0]),
            "success_rate": float(np.mean(success)),
            "arm_violation_rate": float(np.mean(reason == 2)),
            "mean_return": float(np.mean(total_return)),
            "median_success_time_s": median_success_time_s,
            "mean_normalized_effort": float(np.mean(effort)),
            "critic_return_mse": float(
                np.mean((initial_value.detach().cpu().numpy() - discounted_return) ** 2)
            ),
        },
        trajectory,
    )
