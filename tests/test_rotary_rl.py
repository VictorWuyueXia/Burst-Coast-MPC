"""Direct-torque rotary PPO environment, update, and artifact checks."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
import torch

from rotary_pendulum.environment.dynamics import rk4_step
from rotary_pendulum.RL.environment import BatchedRotaryPPOEnvironment
from rotary_pendulum.RL.model import RotaryActorCritic
from rotary_pendulum.RL.training import RolloutBuffer, compute_gae, train_ppo, update_ppo
from rotary_pendulum.utils.config_schema import load_episode_config, load_ppo_config


def test_ppo_config_and_network_match_the_documented_small_design() -> None:
    ppo = load_ppo_config()
    model = RotaryActorCritic(ppo.initial_log_standard_deviation)

    assert ppo.action_repeat_steps == 5
    assert ppo.parallel_environments == 64
    assert ppo.curriculum_decisions == [500000, 1500000, 1000000]
    assert sum(parameter.numel() for parameter in model.parameters()) == 8963

    observation = torch.zeros((3, 8), dtype=torch.float32)
    latent, action, log_probability, value = model.sample(observation, deterministic=False)
    reevaluated_log_probability, entropy, reevaluated_value = model.evaluate(observation, latent)
    assert latent.shape == (3, 1)
    assert action.shape == log_probability.shape == value.shape == (3,)
    assert torch.all(torch.abs(action) < 1.0)
    torch.testing.assert_close(reevaluated_log_probability, log_probability)
    torch.testing.assert_close(reevaluated_value, value)
    assert torch.all(entropy > 0.0)


def test_fixed_features_are_periodic_only_for_the_pendulum_angle() -> None:
    episode = load_episode_config(runtime_mode="headless")
    ppo = load_ppo_config()
    environment = BatchedRotaryPPOEnvironment(episode, ppo, 2, episode.goal.hold_steps, 7)
    environment.state[:] = np.array([[0.2, 0.7, 1.0, -0.5], [0.2, 0.7 + 2.0 * np.pi, 1.0, -0.5]])
    encoded = environment.observe()
    np.testing.assert_allclose(encoded[0], encoded[1], atol=1.0e-7)

    environment.state[1, 0] += 2.0 * np.pi
    unwrapped = environment.observe()
    assert unwrapped[0, 0] != pytest.approx(unwrapped[1, 0])


def test_task_potential_phase_direction_and_vectorized_rk4_are_physical() -> None:
    episode = load_episode_config(runtime_mode="headless")
    ppo = load_ppo_config()
    environment = BatchedRotaryPPOEnvironment(episode, ppo, 2, 999, 11)
    downward = np.zeros(4, dtype=np.float64)
    upright = np.array([0.0, np.pi, 0.0, 0.0], dtype=np.float64)
    assert float(environment.task_potential(downward)) == pytest.approx(1.0, rel=1.0e-6)
    assert float(environment.task_potential(upright)) == pytest.approx(0.0, abs=1.0e-14)

    environment.state[:] = downward
    _, _, components, _, _ = environment.step(np.array([-0.7, 0.7]))
    assert components[0, 1] > components[1, 1]

    environment = BatchedRotaryPPOEnvironment(episode, ppo, 1, 999, 13)
    initial = np.array([0.1, 0.6, -0.2, 0.3], dtype=np.float64)
    environment.state[0] = initial
    action = np.array([0.25])
    environment.step(action)
    expected = initial.copy()
    torque_nm = action[0] * episode.rotary_pendulum.torque_limit_nm
    for _ in range(ppo.action_repeat_steps):
        expected = rk4_step(
            expected,
            torque_nm,
            episode.simulation.timestep_s,
            episode.rotary_pendulum,
            environment.model,
        )
    np.testing.assert_allclose(environment.state[0], expected, rtol=1.0e-13, atol=1.0e-13)


def test_terminal_gae_does_not_bootstrap_and_ppo_update_is_finite() -> None:
    ppo = load_ppo_config().model_copy(
        update={"optimization_epochs": 2, "minibatch_size": 2, "target_kl": 1.0}
    )
    device = torch.device("cpu")
    buffer = RolloutBuffer(2, 2, device)
    buffer.reward[:] = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    buffer.done[:] = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    buffer.value.zero_()
    compute_gae(buffer, torch.tensor([100.0, 3.0]), gamma=0.9, gae_lambda=1.0)
    assert buffer.advantage[1, 0] == pytest.approx(2.0)
    assert buffer.advantage[1, 1] == pytest.approx(4.7)

    model = RotaryActorCritic(ppo.initial_log_standard_deviation)
    torch.manual_seed(17)
    observation = torch.randn((4, 8))
    latent, _, log_probability, value = model.sample(observation, deterministic=False)
    buffer.observation[:] = observation.reshape(2, 2, 8)
    buffer.latent[:] = latent.detach().reshape(2, 2, 1)
    buffer.log_probability[:] = log_probability.detach().reshape(2, 2)
    buffer.value[:] = value.detach().reshape(2, 2)
    buffer.advantage[:] = torch.tensor([[1.0, -0.5], [0.25, -0.75]])
    buffer.return_value[:] = buffer.value + buffer.advantage
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4)
    metrics = update_ppo(
        model,
        optimizer,
        buffer,
        ppo,
        torch.Generator().manual_seed(19),
    )
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before[name], parameter) for name, parameter in model.named_parameters()
    )


def test_short_training_writes_separated_complete_artifacts(tmp_path) -> None:
    episode = load_episode_config(runtime_mode="headless")
    episode.experiment.max_steps = 10
    ppo = load_ppo_config().model_copy(
        update={
            "parallel_environments": 2,
            "rollout_steps": 4,
            "minibatch_size": 4,
            "optimization_epochs": 1,
            "curriculum_decisions": [8, 0, 0],
            "evaluation_episodes": 2,
            "evaluation_hold_steps": 5,
            "evaluation_every_updates": 1,
            "artifact_root": str(tmp_path),
        }
    )

    artifact_dir = train_ppo(episode, ppo)

    assert (artifact_dir / "machine" / "updates.csv").is_file()
    assert (artifact_dir / "machine" / "evaluations.csv").is_file()
    assert (artifact_dir / "machine" / "evaluation_trajectory.csv").is_file()
    assert (artifact_dir / "machine" / "best_actor_critic.pt").is_file()
    assert (artifact_dir / "machine" / "final_actor_critic.pt").is_file()
    assert (artifact_dir / "human" / "training_curves.png").is_file()
    assert (artifact_dir / "human" / "evaluation_trace.png").is_file()
    assert (artifact_dir / "human" / "interpretation_summary.md").is_file()
    update_file = (artifact_dir / "machine" / "updates.csv").open(encoding="utf-8")
    update_rows = list(csv.DictReader(update_file))
    update_file.close()
    assert len(update_rows) == 1
