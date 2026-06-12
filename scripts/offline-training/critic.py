"""Structured residual Q critic for offline burst-coast action scoring."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn


class StructuredResidualCritic(LightningModule):
    """Fit nonnegative physical cost weights and a small residual MLP."""

    def __init__(self, learning_rate: float, lambda_init: dict[str, float]) -> None:
        super().__init__()

        # Store only optimizer scale and make the physical coefficients trainable.
        self.learning_rate = float(learning_rate)
        initial_lambdas = torch.tensor(
            [
                float(lambda_init["lambda_t"]),
                float(lambda_init["lambda_c"]),
                float(lambda_init["lambda_u"]),
            ],
            dtype=torch.float32,
        )
        if torch.any(initial_lambdas <= 0.0):
            raise ValueError("Initial lambda weights must be strictly positive")
        self.eta = nn.Parameter(torch.log(torch.expm1(initial_lambdas)))

        # The residual critic keeps the affine-ReLU-affine structure from the design note.
        self.residual = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        """Evaluate the full structured critic on one batch."""

        # Combine measured physical terms with learned nonnegative weights.
        lambdas = F.softplus(self.eta)
        q_phys = physical @ lambdas

        # Add the residual value learned from normalized state-action features.
        residual = self.residual(features).squeeze(-1)
        return q_phys + residual

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Optimize full return-cost regression on one training batch."""

        # Fit the complete structured critic directly to Monte Carlo return cost.
        features, physical, target = batch
        prediction = self(features, physical)
        loss = F.mse_loss(prediction, target)
        self.log("train_mse", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Measure held-out return-cost regression error."""

        # Report validation MSE without changing the training objective.
        features, physical, target = batch
        prediction = self(features, physical)
        loss = F.mse_loss(prediction, target)
        self.log("val_mse", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self) -> Any:
        """Create the single optimizer used for lambdas and residual weights."""

        # Adam is sufficient for the small supervised regression model.
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def lambda_values(self) -> dict[str, float]:
        """Return learned physical cost coefficients in deployment order."""

        # Expose positive coefficients for snapshots and human inspection.
        lambdas = F.softplus(self.eta).detach().cpu()
        return {
            "lambda_t": float(lambdas[0]),
            "lambda_c": float(lambdas[1]),
            "lambda_u": float(lambdas[2]),
        }
