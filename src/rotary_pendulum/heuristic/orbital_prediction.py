"""Batched nonlinear prediction of pulse, coast, brake and local commands."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array

from rotary_pendulum.environment.jax_dynamics import (
    MODEL,
    PHYSICS_DT_S,
    rk4_step,
    state_derivative,
)
from rotary_pendulum.heuristic.energy import encode

HORIZON_TICKS = math.ceil(2 * MODEL.natural_period_s / PHYSICS_DT_S)
PREDICTION_SUBSTEPS = 20
AUDIT_SUBSTEPS = 40


@jax.jit
def predict_candidates(
    x: Array, schedules: Array, horizons: Array, plant: tuple[float, Array, float]
) -> tuple[Array, Array, Array, Array, Array]:
    """Return tick states, energy, half/full arm peaks, first torque and first-tick peak.

    Schedule rows contain held torque, remaining ticks and feedback law (0 fixed, 1 brake,
    2 LQR). The plant tuple contains the torque cap, normalized LQR gain and brake gain.
    """
    cap, gain, brake_gain = plant
    held, ticks, law = schedules.T
    initial = jnp.broadcast_to(x, (len(schedules), 4))

    def advance(
        carry: tuple[Array, Array], index: Array
    ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
        state, previous_torque = carry
        # Brake cancels nominal arm drift and requests velocity decay at every tick.
        alpha = state[:, 1]
        inertia = MODEL.base_inertia_kg_m2 + MODEL.pendulum_inertia_kg_m2 * jnp.sin(alpha) ** 2
        cross = MODEL.coupling_inertia_kg_m2 * jnp.cos(alpha)
        inverse_input = inertia - cross**2 / MODEL.pendulum_inertia_kg_m2
        brake = (-brake_gain * state[:, 2] - state_derivative(state, 0.0)[:, 2]) * inverse_input

        # LQR shares the historical 100 ms hold, including any residual hold at entry.
        beta = jnp.arctan2(-jnp.sin(alpha), -jnp.cos(alpha))
        error = state.at[:, 1].set(beta)
        local = jnp.where((index - ticks) % 5 == 0, -cap * (error @ gain), previous_torque)
        torque = jnp.clip(
            jnp.where(
                index < ticks, held, jnp.where(law == 1, brake, jnp.where(law == 2, local, 0))
            ),
            -cap,
            cap,
        )
        samples = jax.lax.cond(
            index < horizons[1],
            lambda: rk4_step(state, torque, record_substeps=True, substeps=PREDICTION_SUBSTEPS),
            lambda: jnp.broadcast_to(state, (PREDICTION_SUBSTEPS, *state.shape)),
        )

        # Interpolate arm position at each within-substep zero of the linearized speed.
        path = jnp.concatenate((state[None], samples), axis=0)
        arm, speed = path[..., 0], path[..., 2]
        before, after = speed[:-1], speed[1:]
        crossing = before * after < 0
        s = jnp.where(crossing, before / jnp.where(crossing, before - after, 1), 0)
        dt = PHYSICS_DT_S / PREDICTION_SUBSTEPS
        extrema = (
            (1 + 2 * s) * (1 - s) ** 2 * arm[:-1]
            + s**2 * (3 - 2 * s) * arm[1:]
            + dt * s * (1 - s) * ((1 - s) * before - s * after)
        )
        peak = jnp.maximum(jnp.max(jnp.abs(arm), axis=0), jnp.max(jnp.abs(extrema), axis=0))
        return (samples[-1], torque), (samples[-1], peak, torque)

    _, (states, peaks, torques) = jax.lax.scan(advance, (initial, held), jnp.arange(HORIZON_TICKS))
    within_horizon = jnp.arange(HORIZON_TICKS) < horizons[:, None]
    arm_peaks = jnp.max(jnp.where(within_horizon[..., None], peaks, 0), axis=1).T
    return states, jnp.sum(encode(states), axis=-1), arm_peaks, torques[0], peaks[0]
