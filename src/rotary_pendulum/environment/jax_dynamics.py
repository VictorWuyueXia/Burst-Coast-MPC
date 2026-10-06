"""Pure JAX dynamics for batched rotary-pendulum prediction."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from omegaconf import OmegaConf

from rotary_pendulum.environment.dynamics import derive_model
from rotary_pendulum.utils.config_schema import PHYSICS_CONFIG_PATH, RotaryPendulumConfig

_physics_domains = OmegaConf.to_container(OmegaConf.load(PHYSICS_CONFIG_PATH), resolve=True)
assert isinstance(_physics_domains, dict)
PHYSICAL = RotaryPendulumConfig.model_validate(_physics_domains["rotary-pendulum"])
MODEL = derive_model(PHYSICAL)
PHYSICS_DT_S = 0.02
TORQUE_LIMIT_NM = PHYSICAL.torque_limit_nm


def state_derivative(x: ArrayLike, u: ArrayLike) -> Array:
    """Evaluate the nominal four-state ODE over arbitrary leading batch axes."""

    # Expand the analytic symmetric inertia solve directly over every leading batch axis.
    state = jnp.asarray(x)
    torque = jnp.broadcast_to(jnp.asarray(u), state.shape[:-1])
    _, alpha_rad, omega_rad_s, nu_rad_s = jnp.moveaxis(state, -1, 0)
    sin_alpha = jnp.sin(alpha_rad)
    cos_alpha = jnp.cos(alpha_rad)
    mass_11 = MODEL.base_inertia_kg_m2 + MODEL.pendulum_inertia_kg_m2 * sin_alpha**2
    mass_12 = MODEL.coupling_inertia_kg_m2 * cos_alpha
    mass_22 = MODEL.pendulum_inertia_kg_m2
    generalized_theta = (
        torque
        - PHYSICAL.rotary_damping_nms * omega_rad_s
        - 2.0 * MODEL.pendulum_inertia_kg_m2 * sin_alpha * cos_alpha * omega_rad_s * nu_rad_s
        + MODEL.coupling_inertia_kg_m2 * sin_alpha * nu_rad_s**2
    )
    generalized_alpha = (
        -PHYSICAL.pendulum_damping_nms * nu_rad_s
        + MODEL.pendulum_inertia_kg_m2 * sin_alpha * cos_alpha * omega_rad_s**2
        - MODEL.gravity_torque_nm * sin_alpha
    )
    determinant = mass_11 * mass_22 - mass_12**2
    theta_acceleration = (mass_22 * generalized_theta - mass_12 * generalized_alpha) / determinant
    alpha_acceleration = (-mass_12 * generalized_theta + mass_11 * generalized_alpha) / determinant
    return jnp.stack((omega_rad_s, nu_rad_s, theta_acceleration, alpha_acceleration), axis=-1)


def rk4_step(x: ArrayLike, u: ArrayLike) -> Array:
    """Advance exactly one 20 ms physics interval under constant shaft torque."""

    state = jnp.asarray(x)
    k1 = state_derivative(state, u)
    k2 = state_derivative(state + 0.5 * PHYSICS_DT_S * k1, u)
    k3 = state_derivative(state + 0.5 * PHYSICS_DT_S * k2, u)
    k4 = state_derivative(state + PHYSICS_DT_S * k3, u)
    return state + PHYSICS_DT_S * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
