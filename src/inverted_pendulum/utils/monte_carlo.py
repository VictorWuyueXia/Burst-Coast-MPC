"""Monte Carlo action sampling for offline RL data generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from inverted_pendulum.mpc.controller import prediction_horizon_steps
from inverted_pendulum.utils.config_schema import (
    DataGenerationConfig,
    EnvironmentConfig,
    InitialStateConfig,
)


@dataclass(frozen=True)
class MonteCarloAction:
    """One normalized RL action mapped to integer MPC burst-coast dimensions."""

    bbar: float
    hbar: float
    horizon_steps: int
    burst_steps: int
    coast_steps: int


def sample_uniform_monte_carlo_action(
    rng: np.random.Generator,
    config: DataGenerationConfig,
    environment: EnvironmentConfig,
) -> MonteCarloAction:
    """Sample one normalized action and map it to a fixed-dimension MPC candidate."""

    bbar = float(rng.uniform(config.bbar_min, config.bbar_max))
    hbar = float(rng.uniform(config.hbar_min, config.hbar_max))
    # The normalized action is realized as the smallest valid split candidate when it rounds down.
    horizon_steps = max(1, round(hbar * prediction_horizon_steps(environment)))
    burst_steps = max(1, round(bbar * 0.5 * horizon_steps))
    if horizon_steps <= 0 or burst_steps <= 0 or burst_steps > horizon_steps:
        msg = (
            "Monte Carlo action produced invalid MPC dimensions: "
            f"bbar={bbar}, hbar={hbar}, horizon_steps={horizon_steps}, "
            f"burst_steps={burst_steps}"
        )
        raise ValueError(msg)
    return MonteCarloAction(
        bbar=bbar,
        hbar=hbar,
        horizon_steps=horizon_steps,
        burst_steps=burst_steps,
        coast_steps=horizon_steps - burst_steps,
    )


def sample_uniform_initial_state(
    rng: np.random.Generator,
    config: DataGenerationConfig,
    environment: EnvironmentConfig,
) -> InitialStateConfig:
    """Sample one physical initial state over the prescribed Monte Carlo domain."""

    theta_rad = float(rng.uniform(*config.theta_rad_sample_range))
    omega_eq = 2.0 * math.sqrt(
        environment.pendulum.gravity_m_s2 / environment.pendulum.length_m
    )
    omega_rad_s = float(
        rng.uniform(
            config.omega_eq_scale_sample_range[0] * omega_eq,
            config.omega_eq_scale_sample_range[1] * omega_eq,
        )
    )
    return InitialStateConfig(theta_rad=theta_rad, omega_rad_s=omega_rad_s)
