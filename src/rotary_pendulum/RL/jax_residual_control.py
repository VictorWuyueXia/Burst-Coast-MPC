"""Analytical energy control and a predictive arm filter for residual policies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from rotary_pendulum.environment.jax_dynamics import (
    MODEL,
    PHYSICS_DT_S,
    TORQUE_LIMIT_NM,
    rk4_step,
    state_derivative,
)
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD

POLICY_TORQUE_NM = 0.45 * TORQUE_LIMIT_NM
TARGET_ENERGY_J = 2.0 * MODEL.gravity_torque_nm
ARM_SPEED_SCALE = float(ARM_LIMIT_RAD) * MODEL.natural_frequency_rad_s
PENDULUM_SPEED_SCALE = 2.0 * jnp.sqrt(MODEL.gravity_torque_nm / MODEL.pendulum_inertia_kg_m2)


def heuristic_torque(x: Array, energy_time_s: float, sensitivity_floor_per_s: float) -> Array:
    """Regulate relative hinge-swing energy, not the pendulum body's total energy."""

    theta, alpha, omega, nu = jnp.moveaxis(x, -1, 0)
    inertia = MODEL.pendulum_inertia_kg_m2
    coupling = MODEL.coupling_inertia_kg_m2 * jnp.cos(alpha)
    mass = MODEL.base_inertia_kg_m2 + inertia * jnp.sin(alpha) ** 2
    determinant = mass * inertia - coupling**2
    energy = 0.5 * inertia * nu**2 + MODEL.gravity_torque_nm * (1.0 - jnp.cos(alpha))
    drift = nu * (
        inertia * state_derivative(x, 0.0)[..., 3] + MODEL.gravity_torque_nm * jnp.sin(alpha)
    )
    sensitivity = -inertia * coupling * nu / determinant
    requested_power = (TARGET_ENERGY_J - energy) / energy_time_s
    torque = sensitivity * (requested_power - drift) / (sensitivity**2 + sensitivity_floor_per_s**2)
    # A memoryless, deterministic symmetry-breaking pulse at downward near-rest.
    startup = (energy < 0.05 * TARGET_ENERGY_J) & (jnp.abs(nu) < 0.2) & (jnp.abs(omega) < 0.15)
    kick = 0.02 * TORQUE_LIMIT_NM * jnp.where(theta > 0.0, -1.0, 1.0)
    return jnp.clip(jnp.where(startup, kick, torque), -POLICY_TORQUE_NM, POLICY_TORQUE_NM)


def filter_torque(x: Array, proposed: Array, filter_steps: int) -> tuple[Array, Array]:
    """Predict nominal/coast/brake paths; modes 0/1/2/3 mean accept/coast/brake/best effort."""

    horizon = filter_steps * PHYSICS_DT_S
    alpha = x[..., 1]
    inertia = MODEL.pendulum_inertia_kg_m2
    mass = MODEL.base_inertia_kg_m2 + inertia * jnp.sin(alpha) ** 2
    coupling = MODEL.coupling_inertia_kg_m2 * jnp.cos(alpha)
    arm_authority = inertia / (mass * inertia - coupling**2)
    desired_acceleration = -2.0 * x[..., 2] / horizon - x[..., 0] / horizon**2
    brake = jnp.clip(
        (desired_acceleration - state_derivative(x, 0.0)[..., 2]) / arm_authority,
        -POLICY_TORQUE_NM,
        POLICY_TORQUE_NM,
    )
    candidates = jnp.stack((proposed, jnp.zeros_like(proposed), brake), axis=-1)
    initial = jnp.broadcast_to(x[..., None, :], (*x.shape[:-1], 3, 4))

    def predict(state: Array, index: Array) -> tuple[Array, Array]:
        # Hold each candidate for the actual decision interval, then predict
        # bounded centering/braking feedback, refreshed at the same 100 ms clock.
        position, angle, velocity, _ = jnp.moveaxis(state[..., :4], -1, 0)
        m11 = MODEL.base_inertia_kg_m2 + inertia * jnp.sin(angle) ** 2
        m12 = MODEL.coupling_inertia_kg_m2 * jnp.cos(angle)
        authority = inertia / (m11 * inertia - m12**2)
        acceleration = -2.0 * velocity / horizon - position / horizon**2
        recovery = jnp.clip(
            (acceleration - state_derivative(state[..., :4], 0.0)[..., 2]) / authority,
            -POLICY_TORQUE_NM,
            POLICY_TORQUE_NM,
        )
        held = jnp.where(index == 0, candidates, state[..., 4])
        held = jnp.where((index >= 5) & (index % 5 == 0), recovery, held)
        following = rk4_step(state[..., :4], held)
        return jnp.concatenate((following, held[..., None]), axis=-1), jnp.abs(following[..., 0])

    final, excursion = jax.lax.scan(
        predict,
        jnp.concatenate((initial, candidates[..., None]), axis=-1),
        jnp.arange(filter_steps),
    )
    peak = jnp.max(excursion, axis=0)
    feasible = peak < ARM_LIMIT_RAD
    # Accept nominal first, then coast, then braking. Outside the feasible set,
    # minimize excursion with a small terminal-position tie-breaker.
    best_effort = jnp.argmin(peak + 1e-3 * jnp.abs(final[..., 0]), axis=-1)
    selected = jnp.where(jnp.any(feasible, axis=-1), jnp.argmax(feasible, axis=-1), best_effort)
    applied = jnp.take_along_axis(candidates, selected[..., None], axis=-1)[..., 0]
    mode = jnp.where(jnp.any(feasible, axis=-1), selected, 3).astype(jnp.int32)
    return applied, mode


def residual_action(
    observation: Array, residual: Array, noise: Array, settings: Mapping[str, Any]
) -> tuple[Array, Array, Array, Array]:
    """Compose physical torque before filtering; residual/noise are in cap units."""

    x = jnp.stack(
        (
            observation[..., 0] * ARM_LIMIT_RAD,
            jnp.arctan2(observation[..., 1], observation[..., 2]) % (2 * jnp.pi) % (2 * jnp.pi),
            observation[..., 3] * ARM_SPEED_SCALE,
            observation[..., 4] * PENDULUM_SPEED_SCALE,
        ),
        axis=-1,
    )
    heuristic = heuristic_torque(x, settings["energy_time_s"], settings["sensitivity_floor_per_s"])
    proposed = jnp.clip(
        heuristic + POLICY_TORQUE_NM * (2.0 * residual + noise),
        -POLICY_TORQUE_NM,
        POLICY_TORQUE_NM,
    )
    applied, mode = filter_torque(x, proposed, settings["filter_steps"])
    return applied, heuristic, proposed, mode
