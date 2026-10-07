"""Work inversion with predictive arm recovery and phase-aware branch selection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from rotary_pendulum.environment.jax_dynamics import (
    MODEL,
    TORQUE_LIMIT_NM,
    rk4_step,
    state_derivative,
)
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD
from rotary_pendulum.heuristic.energy import TARGET_ENERGY_J, WORK_LIMIT_J, encode

POLICY_TORQUE_NM = 0.45 * TORQUE_LIMIT_NM
WORK_TOLERANCE_J = 1e-4 * TARGET_ENERGY_J
TORQUE_GRID_NM = jnp.linspace(-POLICY_TORQUE_NM, POLICY_TORQUE_NM, 33)
RECOVERY_LIMIT_RAD = ARM_LIMIT_RAD - 0.05


def decode(
    x: Array, requested_work: Array, work_weight: Array, recovery_steps: int
) -> tuple[Array, dict[str, Array]]:
    """Prioritize predicted recovery, then accurate work and useful energy transfer."""

    if recovery_steps <= 5 or recovery_steps % 5:
        raise ValueError("Recovery horizon must be a multiple of five, greater than five")

    def predict(torques: Array, steps: int) -> tuple[Array, Array, Array]:
        initial = jnp.broadcast_to(x[..., None, :], (*torques.shape, 4))

        def advance(
            carry: tuple[Array, Array, Array, Array], index: Array
        ) -> tuple[tuple[Array, Array, Array, Array], None]:
            state, peak, held_endpoint, held_torque = carry
            alpha, omega = state[..., 1], state[..., 2]
            inertia = MODEL.pendulum_inertia_kg_m2
            mass = MODEL.base_inertia_kg_m2 + inertia * jnp.sin(alpha) ** 2
            coupling = MODEL.coupling_inertia_kg_m2 * jnp.cos(alpha)
            authority = inertia / (mass * inertia - coupling**2)
            brake = jnp.clip(
                (-omega / 0.1 - state_derivative(state, 0.0)[..., 2]) / authority,
                -POLICY_TORQUE_NM,
                POLICY_TORQUE_NM,
            )
            held_torque = jnp.where((index >= 5) & (index % 5 == 0), brake, held_torque)
            following = rk4_step(state, held_torque)
            peak = jnp.maximum(peak, jnp.abs(following[..., 0]))
            held_endpoint = jnp.where(index == 4, following[..., 0], held_endpoint)
            return (following, peak, held_endpoint, held_torque), None

        (endpoint, peak, held_angle, _), _ = jax.lax.scan(
            advance,
            (initial, jnp.abs(initial[..., 0]), initial[..., 0], torques),
            jnp.arange(steps),
        )
        work = torques * (held_angle - x[..., 0, None])
        return endpoint, peak, work

    # Retain all grid brackets; no assumed number of physical work roots.
    grid = jnp.broadcast_to(TORQUE_GRID_NM.astype(x.dtype), (*x.shape[:-1], 33))
    _, _, grid_work = predict(grid, 5)
    error = grid_work - requested_work[..., None]
    bracketed = (error[..., :-1] * error[..., 1:] <= 0.0) & jnp.isfinite(
        error[..., :-1] + error[..., 1:]
    )

    def bisect(index: int, brackets: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        left, right, left_error = brackets
        middle = 0.5 * (left + right)
        _, _, middle_work = predict(middle, 5)
        middle_error = middle_work - requested_work[..., None]
        root_on_left = left_error * middle_error <= 0.0
        return (
            jnp.where(root_on_left, left, middle),
            jnp.where(root_on_left, middle, right),
            jnp.where(root_on_left, left_error, middle_error),
        )

    left, right, _ = jax.lax.fori_loop(
        0, 10, bisect, (grid[..., :-1], grid[..., 1:], error[..., :-1])
    )
    roots = 0.5 * (left + right)
    candidates = jnp.concatenate((roots, grid), axis=-1)
    endpoint, peak, predicted_work = predict(candidates, recovery_steps)
    candidate_mask = jnp.concatenate((bracketed, jnp.ones_like(grid, dtype=bool)), axis=-1)
    work_error = predicted_work - requested_work[..., None]
    matched = jnp.abs(work_error) <= WORK_TOLERANCE_J
    energy_error = encode(endpoint) / TARGET_ENERGY_J - jnp.array([0.0, 0.0, 1.0])
    energy_cost = 0.5 * jnp.sum(energy_error**2, axis=-1)
    finite = candidate_mask & jnp.isfinite(
        energy_cost + peak + predicted_work + requested_work[..., None] + work_weight
    )
    recoverable = finite & (peak <= RECOVERY_LIMIT_RAD) & (jnp.abs(endpoint[..., 2]) <= 0.15)
    exact = recoverable & matched
    has_exact = jnp.any(exact, axis=-1)
    has_recovery = jnp.any(recoverable, axis=-1)
    eligible = jnp.where(
        has_exact[..., None], exact, jnp.where(has_recovery[..., None], recoverable, finite)
    )
    # Explicit best recovery when the bounded candidate search finds no admissible path.
    terminal_excess = jnp.maximum(jnp.abs(endpoint[..., 0]) - RECOVERY_LIMIT_RAD, 0.0)
    outward_speed = jnp.maximum(jnp.sign(endpoint[..., 0]) * endpoint[..., 2], 0.0)
    for violation in (jnp.maximum(peak - RECOVERY_LIMIT_RAD, 0.0), terminal_excess, outward_speed):
        minimum = jnp.min(jnp.where(eligible, violation, jnp.inf), axis=-1, keepdims=True)
        eligible = eligible & (has_recovery[..., None] | (violation == minimum))
    scores = energy_cost + jnp.where(
        (has_recovery & ~has_exact)[..., None],
        0.5 * work_weight * (work_error / WORK_LIMIT_J) ** 2,
        0.0,
    )
    minimum = jnp.min(jnp.where(eligible, scores, jnp.inf), axis=-1, keepdims=True)
    tied = eligible & (scores == minimum)
    smallest = jnp.min(jnp.where(tied, jnp.abs(candidates), jnp.inf), axis=-1, keepdims=True)
    tied = tied & (jnp.abs(candidates) == smallest)
    positive = tied & (candidates >= 0.0)
    selected = jnp.argmax(jnp.where(jnp.any(positive, axis=-1, keepdims=True), positive, tied), -1)
    torque = jnp.take_along_axis(candidates, selected[..., None], axis=-1)[..., 0]
    # Modes: 0 exact work, 1 work override, 2 recovery unresolved, 3 numerical failure.
    mode = jnp.where(has_exact, 0, jnp.where(has_recovery, 1, jnp.where(jnp.any(finite, -1), 2, 3)))
    applied = jnp.where(mode == 3, jnp.nan, torque)
    selected_work = jnp.take_along_axis(predicted_work, selected[..., None], -1)[..., 0]
    selected_peak = jnp.take_along_axis(peak, selected[..., None], -1)[..., 0]
    return applied, {
        "mode": mode,
        "requested_work_j": requested_work,
        "predicted_work_j": selected_work,
        "predicted_peak_arm_rad": selected_peak,
        "predicted_terminal_arm_speed": jnp.take_along_axis(
            endpoint[..., 2], selected[..., None], -1
        )[..., 0],
        "root_count": jnp.sum(finite[..., :32] & matched[..., :32], axis=-1),
        "recoverable_count": jnp.sum(recoverable, axis=-1),
    }
