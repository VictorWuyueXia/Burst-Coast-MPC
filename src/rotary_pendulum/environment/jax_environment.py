"""Functional JAX episode contract for the 50 Hz rotary pendulum."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from omegaconf import OmegaConf

from rotary_pendulum.environment.jax_dynamics import TORQUE_LIMIT_NM, rk4_step
from rotary_pendulum.utils.config_schema import MISSION_CONFIG_PATH, GoalConfig

_mission_domains = OmegaConf.to_container(OmegaConf.load(MISSION_CONFIG_PATH), resolve=True)
assert isinstance(_mission_domains, dict)
GOAL = GoalConfig.model_validate(_mission_domains["goal"])
HOLD_PHYSICS_STEPS = 5
MAX_PHYSICS_STEPS = 1000
ARM_LIMIT_RAD = jnp.pi / 2.0
RESET_LOW = jnp.array(
    [[-0.20, -0.20, -0.5, -0.5], [-0.50, 0.40, -2.0, -6.0], [-0.25, jnp.pi - 0.25, -1.0, -1.0]]
)
RESET_HIGH = jnp.array(
    [[0.20, 0.20, 0.5, 0.5], [0.50, 2.60, 2.0, 6.0], [0.25, jnp.pi + 0.25, 1.0, 1.0]]
)


class EnvState(NamedTuple):
    """Carry the physical state and minimal episode memory as one JAX pytree."""

    x: Array
    physics_steps: Array
    goal_count: Array
    success: Array
    arm_violation: Array
    timeout: Array


def reset(key: Array, stratum: ArrayLike) -> EnvState:
    """Sample one continuous state from a caller-selected reset stratum."""

    uniform_key, sign_key = jax.random.split(key)
    stratum_index = jnp.asarray(stratum, dtype=jnp.int32)
    sample = jax.random.uniform(uniform_key, (4,))
    state = RESET_LOW[stratum_index] + sample * (
        RESET_HIGH[stratum_index] - RESET_LOW[stratum_index]
    )
    moving_sign = jnp.where(jax.random.bernoulli(sign_key), 1.0, -1.0)
    state = state.at[1].set(jnp.where(stratum_index == 1, moving_sign * state[1], state[1]))
    return EnvState(
        x=state,
        physics_steps=jnp.array(0, dtype=jnp.int32),
        goal_count=jnp.array(0, dtype=jnp.int32),
        success=jnp.array(False),
        arm_violation=jnp.array(False),
        timeout=jnp.array(False),
    )


def step(
    env_state: EnvState,
    u: ArrayLike,
    *,
    physics_steps: int = HOLD_PHYSICS_STEPS,
    max_physics_steps: int | Array = MAX_PHYSICS_STEPS,
) -> EnvState:
    """Hold bounded torque; only successful 100 ms dwell or timeout stops physics."""

    applied_torque = jnp.clip(jnp.asarray(u), -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM)

    def advance_one(state: EnvState, _: None) -> tuple[EnvState, None]:
        active = ~(state.success | state.timeout)
        integrated = rk4_step(state.x, applied_torque)
        next_x = jnp.where(active[..., None], integrated, state.x)
        next_steps = state.physics_steps + active.astype(jnp.int32)
        upright_error = jnp.arctan2(
            jnp.sin(next_x[..., 1] - jnp.pi), jnp.cos(next_x[..., 1] - jnp.pi)
        )
        inside_goal = (
            (jnp.abs(next_x[..., 0]) <= GOAL.theta_tolerance_rad)
            & (jnp.abs(upright_error) <= GOAL.beta_tolerance_rad)
            & (jnp.abs(next_x[..., 2]) <= GOAL.omega_tolerance_rad_s)
            & (jnp.abs(next_x[..., 3]) <= GOAL.nu_tolerance_rad_s)
        )
        next_goal_count = jnp.where(
            active, jnp.where(inside_goal, state.goal_count + 1, 0), state.goal_count
        )
        arm_violation = state.arm_violation | (active & (jnp.abs(next_x[..., 0]) >= ARM_LIMIT_RAD))
        success = state.success | (active & (next_goal_count >= GOAL.hold_steps))
        timeout = state.timeout | (active & (next_steps >= max_physics_steps) & ~success)
        return EnvState(next_x, next_steps, next_goal_count, success, arm_violation, timeout), None

    return jax.lax.scan(advance_one, env_state, None, length=physics_steps)[0]
