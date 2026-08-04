"""Uniform Monte Carlo action-plan sampling for the rotary pendulum."""

from __future__ import annotations

import numpy as np

from rotary_pendulum.environment.dynamics import ModelConstants
from rotary_pendulum.utils.config_schema import RootConfig
from rotary_pendulum.utils.messages import ActionPlan


def sample_action_plan(
    rng: np.random.Generator,
    config: RootConfig,
    model: ModelConstants,
    replan_index: int,
) -> ActionPlan:
    """Sample one signed constant-torque plan over zero to three natural periods."""

    # Sample the discrete duration uniformly over every realizable positive horizon.
    timestep_s = config.simulation.timestep_s
    max_horizon_steps = int(np.floor(3.0 * model.natural_period_s / timestep_s))
    horizon_steps = int(rng.integers(1, max_horizon_steps + 1))

    # Sample magnitude independently of an unbiased binary torque direction.
    positive_zero = np.nextafter(0.0, 1.0)
    magnitude_nm = float(rng.uniform(positive_zero, config.rotary_pendulum.torque_limit_nm))
    direction = 1.0 if int(rng.integers(0, 2)) else -1.0
    torque_nm = direction * magnitude_nm
    torques_nm = np.full(horizon_steps, torque_nm, dtype=np.float64)

    # Report normalized coordinates from the exact sampled discrete plan.
    return ActionPlan(
        plan_id=f"monte-carlo-{replan_index:06d}",
        replan_index=replan_index,
        hbar=horizon_steps * timestep_s / model.natural_period_s,
        bbar=abs(torque_nm) / config.rotary_pendulum.torque_limit_nm,
        horizon_steps=horizon_steps,
        torques_nm=torques_nm,
    )
