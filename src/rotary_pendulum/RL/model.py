"""Small shared-encoder PPO actor-critic for bounded rotary torque."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal


class RotaryActorCritic(nn.Module):
    """Encode physical features and decode a tanh-Gaussian actor and state value."""

    def __init__(self, initial_log_standard_deviation: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor_decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.critic_decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.log_standard_deviation = nn.Parameter(
            torch.tensor([initial_log_standard_deviation], dtype=torch.float32)
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_decoder[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_decoder[-1].weight, gain=1.0)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the latent-action mean and scalar state value for one batch."""

        encoded = self.encoder(observation)
        mean = self.actor_decoder(encoded)
        value = self.critic_decoder(encoded).squeeze(-1)
        return mean, value

    def sample(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or select one bounded action and return its exact policy statistics."""

        mean, value = self(observation)
        standard_deviation = self.log_standard_deviation.exp().expand_as(mean)
        distribution = Normal(mean, standard_deviation)
        latent = mean if deterministic else distribution.rsample()
        action = torch.tanh(latent)
        log_probability = (
            distribution.log_prob(latent) - torch.log(1.0 - action.square() + 1.0e-6)
        ).sum(dim=-1)
        return latent, action.squeeze(-1), log_probability, value

    def evaluate(
        self,
        observation: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored latent actions under the current PPO policy."""

        mean, value = self(observation)
        standard_deviation = self.log_standard_deviation.exp().expand_as(mean)
        distribution = Normal(mean, standard_deviation)
        action = torch.tanh(latent)
        log_probability = (
            distribution.log_prob(latent) - torch.log(1.0 - action.square() + 1.0e-6)
        ).sum(dim=-1)
        return log_probability, distribution.entropy().sum(dim=-1), value
