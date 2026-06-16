"""Frozen structured critic policy for burst-horizon action selection."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from inverted_pendulum.mpc.controller import prediction_horizon_steps
from inverted_pendulum.utils.config_schema import EnvironmentConfig, RLConfig
from inverted_pendulum.utils.messages import RLStepRecord, StateObs
from inverted_pendulum.utils.monte_carlo import MonteCarloAction

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_FEATURE_ORDER = [
    "sin-theta",
    "cos-theta",
    "omega/ref",
    "energy-error",
    "bbar",
    "hbar",
]


@dataclass(frozen=True)
class RLActionSelection:
    """One policy-selected normalized action and its critic metadata."""

    action: MonteCarloAction
    q_value: float
    probability: float
    mode: str


class StructuredCriticPolicy:
    """Load frozen critic artifacts and select burst-horizon actions from a grid."""

    def __init__(self, config: RLConfig, environment: EnvironmentConfig) -> None:
        # Resolve artifact paths relative to the repository root used by the runtime package.
        critic_artifact_dir = Path(config.critic_artifact_dir)
        if not critic_artifact_dir.is_absolute():
            critic_artifact_dir = REPO_ROOT / critic_artifact_dir
        time_model_artifact_dir = Path(config.time_model_artifact_dir)
        if not time_model_artifact_dir.is_absolute():
            time_model_artifact_dir = REPO_ROOT / time_model_artifact_dir

        # Read the structured critic metadata and normalization contract.
        critic_config_file = (critic_artifact_dir / "config.json").open(encoding="utf-8")
        critic_config = json.load(critic_config_file)
        critic_config_file.close()
        normalization_file = (critic_artifact_dir / "normalization.json").open(
            encoding="utf-8"
        )
        normalization = json.load(normalization_file)
        normalization_file.close()
        lambda_file = (critic_artifact_dir / "lambda.json").open(encoding="utf-8")
        lambda_values = json.load(lambda_file)
        lambda_file.close()
        if normalization["feature-order"] != EXPECTED_FEATURE_ORDER:
            msg = "Frozen critic feature order does not match runtime policy features"
            raise ValueError(msg)

        # Read the computation-time fit used for deployment grid scoring.
        time_model_file = (time_model_artifact_dir / "final_model.json").open(encoding="utf-8")
        time_model = json.load(time_model_file)
        time_model_file.close()

        # Import torch only when the RL mode is constructed.
        import torch

        from inverted_pendulum.RL.critic import StructuredResidualCritic

        self.config = config
        self.environment = environment
        self.critic_artifact_dir = critic_artifact_dir
        self.time_model_artifact_dir = time_model_artifact_dir
        self.action_grid_count = int(critic_config["action-grid-count"])
        self.feature_mean = np.asarray(normalization["feature-mean"], dtype=np.float32)
        self.feature_std = np.asarray(normalization["feature-std"], dtype=np.float32)
        self.time_intercept_s = float(time_model["intercept_s"])
        self.time_terms = tuple(
            (str(term["name"]), float(term["coefficient_s"]))
            for term in time_model["all_terms"]
        )
        self.rng = np.random.default_rng(config.training_seed)
        self.torch: Any = torch
        self.model = StructuredResidualCritic(float(critic_config["learning-rate"]), lambda_values)
        state_dict = torch.load(critic_artifact_dir / "critic_state_dict.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def score_actions(
        self,
        observation: StateObs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Score the full normalized action grid for one replanning observation."""

        # Build the normalized action grid and its integer MPC dimensions.
        axis = np.linspace(0.0, 1.0, self.action_grid_count, dtype=np.float64)
        bbar_grid, hbar_grid = np.meshgrid(axis, axis, indexing="xy")
        bbar = bbar_grid.reshape(-1)
        hbar = hbar_grid.reshape(-1)
        full_horizon_steps = prediction_horizon_steps(self.environment)
        horizon_steps = np.maximum(1, np.rint(hbar * full_horizon_steps)).astype(np.int64)
        burst_steps = np.maximum(1, np.rint(bbar * 0.5 * horizon_steps)).astype(np.int64)

        # Derive deployment features from the live observation and frozen normalization.
        pendulum = self.environment.pendulum
        upright_energy = 2.0 * pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m
        omega_ref = 2.0 * math.sqrt(pendulum.gravity_m_s2 / pendulum.length_m)
        energy_error = observation.energy_error_j / upright_energy
        feature_raw = np.column_stack(
            (
                np.repeat(math.sin(observation.theta_rad), bbar.size),
                np.repeat(math.cos(observation.theta_rad), bbar.size),
                np.repeat(observation.omega_rad_s / omega_ref, bbar.size),
                np.repeat(energy_error, bbar.size),
                bbar,
                hbar,
            )
        ).astype(np.float32)
        features = (feature_raw - self.feature_mean) / self.feature_std

        # Estimate candidate physical terms without solving every MPC candidate.
        compute_time_s = np.full(bbar.shape, self.time_intercept_s, dtype=np.float64)
        for name, coefficient_s in self.time_terms:
            if name == "bbar":
                compute_time_s += coefficient_s * bbar
            elif name == "hbar":
                compute_time_s += coefficient_s * hbar
            elif name == "bbar_hbar":
                compute_time_s += coefficient_s * bbar * hbar
            else:
                msg = f"Unsupported time-fit term in frozen artifact: {name}"
                raise ValueError(msg)
        physical = np.column_stack(
            (
                horizon_steps * self.environment.simulation.timestep_s,
                compute_time_s / self.environment.simulation.timestep_s,
                burst_steps,
            )
        ).astype(np.float32)

        # Evaluate the frozen critic on CPU and return NumPy scores for policy selection.
        torch = self.torch
        torch.set_grad_enabled(False)
        q_values = self.model(
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(physical, dtype=torch.float32),
        ).detach().cpu().numpy()
        torch.set_grad_enabled(True)
        return q_values, bbar, hbar, horizon_steps, burst_steps

    def select_action(self, observation: StateObs, *, explore: bool) -> RLActionSelection:
        """Select the minimum-cost or cost-softmax action for one observation."""

        q_values, bbar, hbar, horizon_steps, burst_steps = self.score_actions(observation)
        if explore:
            scaled = -(q_values - np.min(q_values)) / self.config.exploration_temperature
            weights = np.exp(scaled)
            probabilities = (1.0 - self.config.exploration_epsilon) * weights / np.sum(weights)
            probabilities += self.config.exploration_epsilon / q_values.size
            selected_index = int(self.rng.choice(q_values.size, p=probabilities))
            probability = float(probabilities[selected_index])
            mode = "rl_train"
        else:
            selected_index = int(np.argmin(q_values))
            probability = 1.0
            mode = "rl_intelligent"

        action = MonteCarloAction(
            bbar=float(bbar[selected_index]),
            hbar=float(hbar[selected_index]),
            horizon_steps=int(horizon_steps[selected_index]),
            burst_steps=int(burst_steps[selected_index]),
            coast_steps=int(horizon_steps[selected_index] - burst_steps[selected_index]),
        )
        return RLActionSelection(
            action=action,
            q_value=float(q_values[selected_index]),
            probability=probability,
            mode=mode,
        )

    def build_training_tensors(
        self,
        records: list[RLStepRecord],
    ) -> tuple[Any, Any, Any]:
        """Convert online RL transition rows into Lightning training tensors."""

        # Rebuild the same supervised tuple used by the offline critic trainer.
        sin_theta = np.array([record.s_sin_theta for record in records], dtype=np.float64)
        cos_theta = np.array([record.s_cos_theta for record in records], dtype=np.float64)
        omega = np.array([record.s_omega_rad_s for record in records], dtype=np.float64)
        bbar = np.array([record.bbar for record in records], dtype=np.float64)
        hbar = np.array([record.hbar for record in records], dtype=np.float64)
        burst_steps = np.array([record.burst_steps for record in records], dtype=np.float64)
        horizon_steps = np.array([record.horizon_steps for record in records], dtype=np.float64)
        solve_time_s = np.array([record.solve_time_s for record in records], dtype=np.float64)
        target = np.array([record.return_cost for record in records], dtype=np.float64)

        pendulum = self.environment.pendulum
        inertia = pendulum.mass_kg * pendulum.length_m**2
        upright_energy = 2.0 * pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m
        energy = (
            0.5 * inertia * omega**2
            + pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m * (1.0 + cos_theta)
        )
        omega_ref = 2.0 * math.sqrt(pendulum.gravity_m_s2 / pendulum.length_m)
        feature_raw = np.column_stack(
            (
                sin_theta,
                cos_theta,
                omega / omega_ref,
                (energy - upright_energy) / upright_energy,
                bbar,
                hbar,
            )
        ).astype(np.float32)
        features = (feature_raw - self.feature_mean) / self.feature_std
        physical = np.column_stack(
            (
                horizon_steps * self.environment.simulation.timestep_s,
                solve_time_s / self.environment.simulation.timestep_s,
                burst_steps,
            )
        ).astype(np.float32)

        torch = self.torch
        return (
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(physical, dtype=torch.float32),
            torch.as_tensor(target, dtype=torch.float32),
        )
