"""Vectorized on-policy PPO training for direct rotary-pendulum torque control."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rotary_pendulum.RL.artifacts import PPOArtifactWriter
from rotary_pendulum.RL.environment import REWARD_COMPONENT_NAMES, BatchedRotaryPPOEnvironment
from rotary_pendulum.RL.evaluation import evaluate_policy
from rotary_pendulum.RL.model import RotaryActorCritic
from rotary_pendulum.utils.config_schema import (
    EPISODE_CONFIG_PATHS,
    EpisodeConfig,
    PPOConfig,
    load_episode_config,
    load_ppo_config,
)


class RolloutBuffer:
    """Own fixed-shape tensors for one complete on-policy PPO rollout."""

    def __init__(self, steps: int, environments: int, device: torch.device) -> None:
        shape = (steps, environments)
        self.observation = torch.empty((*shape, 8), dtype=torch.float32, device=device)
        self.latent = torch.empty((*shape, 1), dtype=torch.float32, device=device)
        self.log_probability = torch.empty(shape, dtype=torch.float32, device=device)
        self.reward = torch.empty(shape, dtype=torch.float32, device=device)
        self.reward_components = torch.empty((*shape, 6), dtype=torch.float32, device=device)
        self.done = torch.empty(shape, dtype=torch.float32, device=device)
        self.value = torch.empty(shape, dtype=torch.float32, device=device)
        self.advantage = torch.empty(shape, dtype=torch.float32, device=device)
        self.return_value = torch.empty(shape, dtype=torch.float32, device=device)


def collect_rollout(
    environment: BatchedRotaryPPOEnvironment,
    model: RotaryActorCritic,
    buffer: RolloutBuffer,
    stage: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Collect one vectorized on-policy rollout and reset each terminal environment."""

    observation = environment.observe()
    terminal_counts = np.zeros(3, dtype=np.int64)
    component_sum = np.zeros(6, dtype=np.float64)
    for step in range(buffer.reward.shape[0]):
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        torch.set_grad_enabled(False)
        latent, action, log_probability, value = model.sample(
            observation_tensor, deterministic=False
        )
        torch.set_grad_enabled(True)
        next_observation, reward, components, done, reason = environment.step(
            action.detach().cpu().numpy()
        )
        buffer.observation[step] = observation_tensor
        buffer.latent[step] = latent.detach()
        buffer.log_probability[step] = log_probability.detach()
        buffer.reward[step] = torch.as_tensor(reward, device=device)
        buffer.reward_components[step] = torch.as_tensor(components, device=device)
        buffer.done[step] = torch.as_tensor(done, dtype=torch.float32, device=device)
        buffer.value[step] = value.detach()
        component_sum += components.sum(axis=0)
        for terminal_code in (1, 2, 3):
            terminal_counts[terminal_code - 1] += int(np.count_nonzero(reason == terminal_code))
        if np.any(done):
            next_observation = environment.reset(done, stage=stage, exact=False)
        observation = next_observation
    torch.set_grad_enabled(False)
    _, last_value = model(torch.as_tensor(observation, dtype=torch.float32, device=device))
    torch.set_grad_enabled(True)
    sample_count = int(buffer.reward.numel())
    metrics: dict[str, float | int] = {
        "mean_reward": float(buffer.reward.mean().cpu()),
        **{
            f"{name}_reward": float(component_sum[index] / sample_count)
            for index, name in enumerate(REWARD_COMPONENT_NAMES)
        },
        "successes": int(terminal_counts[0]),
        "arm_failures": int(terminal_counts[1]),
        "timeouts": int(terminal_counts[2]),
    }
    return last_value.detach(), metrics


def compute_gae(
    buffer: RolloutBuffer,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> None:
    """Compute terminal-aware generalized advantages and value returns in reverse."""

    next_advantage = torch.zeros_like(last_value)
    next_value = last_value
    for step in range(buffer.reward.shape[0] - 1, -1, -1):
        continuation = 1.0 - buffer.done[step]
        delta = buffer.reward[step] + gamma * continuation * next_value - buffer.value[step]
        next_advantage = delta + gamma * gae_lambda * continuation * next_advantage
        buffer.advantage[step] = next_advantage
        next_value = buffer.value[step]
    buffer.return_value.copy_(buffer.advantage + buffer.value)


def update_ppo(
    model: RotaryActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    ppo: PPOConfig,
    generator: torch.Generator,
) -> dict[str, float]:
    """Optimize clipped policy and value losses over shuffled rollout minibatches."""

    observation = buffer.observation.flatten(0, 1)
    latent = buffer.latent.flatten(0, 1)
    old_log_probability = buffer.log_probability.flatten()
    advantage = buffer.advantage.flatten()
    return_value = buffer.return_value.flatten()
    old_value = buffer.value.flatten()
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1.0e-8)
    batch_size = observation.shape[0]
    totals = np.zeros(5, dtype=np.float64)
    minibatch_count = 0
    stop = False
    for _ in range(ppo.optimization_epochs):
        indices = torch.randperm(batch_size, generator=generator, device=observation.device)
        for start in range(0, batch_size, ppo.minibatch_size):
            selected = indices[start : start + ppo.minibatch_size]
            log_probability, entropy, value = model.evaluate(
                observation[selected], latent[selected]
            )
            log_ratio = log_probability - old_log_probability[selected]
            ratio = log_ratio.exp()
            unclipped = -advantage[selected] * ratio
            clipped = -advantage[selected] * torch.clamp(
                ratio, 1.0 - ppo.policy_clip, 1.0 + ppo.policy_clip
            )
            policy_loss = torch.maximum(unclipped, clipped).mean()
            value_loss = F.mse_loss(value, return_value[selected])
            entropy_mean = entropy.mean()
            loss = (
                policy_loss + ppo.value_loss_weight * value_loss - ppo.entropy_weight * entropy_mean
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ppo.maximum_gradient_norm)
            optimizer.step()
            approximate_kl = float((ratio - 1.0 - log_ratio).mean().detach().cpu())
            clip_fraction = float(
                (torch.abs(ratio - 1.0) > ppo.policy_clip).float().mean().detach().cpu()
            )
            totals += (
                float(policy_loss.detach().cpu()),
                float(value_loss.detach().cpu()),
                float(entropy_mean.detach().cpu()),
                approximate_kl,
                clip_fraction,
            )
            minibatch_count += 1
            if approximate_kl > ppo.target_kl:
                stop = True
                break
        if stop:
            break
    residual = return_value - old_value
    explained_variance = 1.0 - float(residual.var(unbiased=False).cpu()) / (
        float(return_value.var(unbiased=False).cpu()) + 1.0e-8
    )
    means = totals / minibatch_count
    return {
        "policy_loss": float(means[0]),
        "value_loss": float(means[1]),
        "entropy": float(means[2]),
        "approximate_kl": float(means[3]),
        "clip_fraction": float(means[4]),
        "explained_variance": explained_variance,
    }


def train_ppo(episode: EpisodeConfig, ppo: PPOConfig) -> Path:
    """Run the complete curriculum, evaluation, checkpoint, and artifact workflow."""

    if ppo.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PPO device is cuda but PyTorch cannot access a CUDA device")
    torch.manual_seed(ppo.random_seed)
    np.random.seed(ppo.random_seed)
    device = torch.device(ppo.device)
    generator = torch.Generator(device=device).manual_seed(ppo.random_seed)
    model = RotaryActorCritic(ppo.initial_log_standard_deviation).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=ppo.initial_learning_rate, eps=ppo.adam_epsilon
    )
    environment = BatchedRotaryPPOEnvironment(
        episode,
        ppo,
        ppo.parallel_environments,
        episode.goal.hold_steps,
        ppo.random_seed,
    )
    buffer = RolloutBuffer(ppo.rollout_steps, ppo.parallel_environments, device)
    writer = PPOArtifactWriter(episode, ppo)
    decisions_per_update = ppo.rollout_steps * ppo.parallel_environments
    total_decisions = sum(ppo.curriculum_decisions)
    update_count = math.ceil(total_decisions / decisions_per_update)
    gamma = math.exp(
        -ppo.action_repeat_steps * episode.simulation.timestep_s / ppo.discount_time_constant_s
    )
    final_trajectory: list[dict[str, float | int]] = []
    final_evaluation: dict[str, float] = {}

    for update in range(1, update_count + 1):
        completed_decisions = (update - 1) * decisions_per_update
        if completed_decisions < ppo.curriculum_decisions[0]:
            stage = 0
        elif completed_decisions < sum(ppo.curriculum_decisions[:2]):
            stage = 1
        else:
            stage = 2
        progress = (update - 1) / max(update_count - 1, 1)
        learning_rate = ppo.initial_learning_rate + progress * (
            ppo.final_learning_rate - ppo.initial_learning_rate
        )
        optimizer.param_groups[0]["lr"] = learning_rate
        last_value, rollout_metrics = collect_rollout(environment, model, buffer, stage, device)
        compute_gae(buffer, last_value, gamma, ppo.gae_lambda)
        update_metrics = update_ppo(model, optimizer, buffer, ppo, generator)
        metrics: dict[str, float | int] = {
            "update": update,
            "decisions": update * decisions_per_update,
            "stage": stage,
            "learning_rate": learning_rate,
            **update_metrics,
            **rollout_metrics,
        }
        writer.record_update(metrics)
        print(
            f"rotary-ppo update={update}/{update_count} stage={stage} "
            f"reward={float(metrics['mean_reward']):.6f} "
            f"policy_loss={float(metrics['policy_loss']):.6f} "
            f"value_loss={float(metrics['value_loss']):.6f}"
        )
        if update % ppo.evaluation_every_updates == 0 or update == update_count:
            final_evaluation, final_trajectory = evaluate_policy(episode, ppo, model, device)
            writer.record_evaluation({"update": update, **final_evaluation}, model)
            print(
                f"rotary-ppo evaluation update={update} "
                f"success={final_evaluation['success_rate']:.3f} "
                f"arm_violation={final_evaluation['arm_violation_rate']:.3f}"
            )
    writer.finalize(
        model,
        {
            "updates": update_count,
            "decisions": update_count * decisions_per_update,
            "final_evaluation": final_evaluation,
        },
        final_trajectory,
    )
    print(f"rotary-ppo artifact_dir={writer.run_dir}")
    return writer.run_dir


if __name__ == "__main__":
    train_ppo(
        load_episode_config(EPISODE_CONFIG_PATHS, runtime_mode="headless"),
        load_ppo_config(),
    )
