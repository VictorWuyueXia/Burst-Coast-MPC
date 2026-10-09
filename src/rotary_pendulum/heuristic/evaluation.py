"""Physics-resolution rollout chunks for paired analytical-controller studies."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from rotary_pendulum.environment.jax_dynamics import PHYSICS_DT_S
from rotary_pendulum.environment.jax_environment import EnvState, step
from rotary_pendulum.heuristic.decoder import decode
from rotary_pendulum.heuristic.energy import encode, policy


def evaluate(
    initial: EnvState, parameters: Array, decisions: int, mode: str
) -> tuple[EnvState, dict[str, Array]]:
    """Advance a fixed chunk with parameters [work gain, work penalty, arm-limit penalty]."""

    if mode not in ("energy", "zero"):
        raise ValueError(f"Unknown heuristic evaluation mode: {mode}")
    initial = initial._replace(
        x=initial.x.at[..., 1].set((initial.x[..., 1] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
    )

    def advance(state: EnvState, unused: None) -> tuple[EnvState, dict[str, Array]]:
        requested = policy(encode(state.x), parameters[0])
        if mode == "energy":
            torque, diagnostic = decode(state.x, requested, parameters[1], parameters[2])
        else:
            torque = jnp.zeros_like(requested)
            diagnostic = {
                "mode": jnp.full(requested.shape, -1, jnp.int32),
                "requested_work_j": jnp.zeros_like(requested),
                "predicted_work_j": jnp.zeros_like(requested),
                "predicted_peak_arm_rad": jnp.abs(state.x[..., 0]),
                "root_count": jnp.zeros_like(requested, dtype=jnp.int32),
                "predicted_terminal_arm_speed": state.x[..., 2],
            }

        def physics(current: EnvState, unused: None) -> tuple[EnvState, tuple[Array, Array]]:
            active = ~(current.success | current.timeout)
            following = step(current, torque, physics_steps=1)
            return following, (following.x, active)

        following, (samples, active) = jax.lax.scan(physics, state, None, length=5)
        active_duration = PHYSICS_DT_S * jnp.sum(active, axis=0)
        work = torque * (following.x[..., 0] - state.x[..., 0])
        return following, {
            **diagnostic,
            "start_x": state.x,
            "x": following.x,
            "physics_x": jnp.moveaxis(samples, 0, -2),
            "pendulum_angle_deg": jnp.rad2deg(jnp.moveaxis(samples[..., 1], 0, -1)),
            "physics_active": jnp.moveaxis(active, 0, -1),
            "energy_j": encode(following.x),
            "torque_nm": jnp.where(active_duration > 0.0, torque, 0.0),
            "work_j": work,
            "work_mismatch_j": jnp.where(
                active_duration > 0, work - diagnostic["requested_work_j"], 0.0
            ),
            "energy_balance_j": jnp.sum(encode(following.x) - encode(state.x), axis=-1) - work,
            "elapsed_s": active_duration,
            "time_s": following.physics_steps * PHYSICS_DT_S,
            "goal_count": following.goal_count,
            "success": following.success,
            "arm_violation": following.arm_violation,
            "timeout": following.timeout,
        }

    return jax.lax.scan(advance, initial, None, length=decisions)
