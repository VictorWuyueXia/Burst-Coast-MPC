"""Choose one constant 100 ms torque from physical work and pendulum elevation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from rotary_pendulum.environment.jax_dynamics import TORQUE_LIMIT_NM, rk4_step
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD
from rotary_pendulum.heuristic.energy import TARGET_ENERGY_J, WORK_LIMIT_J, encode

POLICY_TORQUE_NM = 0.90 * TORQUE_LIMIT_NM
WORK_TOLERANCE_J = 1e-4 * TARGET_ENERGY_J
TORQUE_GRID_NM = jnp.linspace(-POLICY_TORQUE_NM, POLICY_TORQUE_NM, 33)


def decode(
    x: Array, requested_work: Array, work_weight: Array, arm_limit_weight: Array
) -> tuple[Array, dict[str, Array]]:
    """Score constant-torque actions; no future control sequence or speed constraint."""

    def predict(torques: Array) -> tuple[Array, Array, Array]:
        initial = jnp.broadcast_to(x[..., None, :], (*torques.shape, 4))

        def advance(carry: tuple[Array, Array], unused: None) -> tuple[tuple[Array, Array], None]:
            state, peak = carry
            following = rk4_step(state, torques)
            return (following, jnp.maximum(peak, jnp.abs(following[..., 0]))), None

        (endpoint, peak), _ = jax.lax.scan(
            advance, (initial, jnp.abs(initial[..., 0])), None, length=5
        )
        work = torques * (endpoint[..., 0] - x[..., 0, None])
        return endpoint, peak, work

    # Refine all torque intervals that bracket the requested motor work.
    grid = jnp.broadcast_to(TORQUE_GRID_NM.astype(x.dtype), (*x.shape[:-1], 33))
    _, _, grid_work = predict(grid)
    error = grid_work - requested_work[..., None]
    bracketed = (error[..., :-1] * error[..., 1:] <= 0.0) & jnp.isfinite(
        error[..., :-1] + error[..., 1:]
    )

    def bisect(index: int, brackets: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        left, right, left_error = brackets
        middle = 0.5 * (left + right)
        _, _, middle_work = predict(middle)
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
    torques = jnp.concatenate((0.5 * (left + right), grid), axis=-1)
    endpoint, peak, predicted_work = predict(torques)
    valid = jnp.concatenate((bracketed, jnp.ones_like(grid, dtype=bool)), axis=-1)
    energy = encode(endpoint)
    work_error = predicted_work - requested_work[..., None]
    potential_cost = ((TARGET_ENERGY_J - energy[..., 2]) / TARGET_ENERGY_J) ** 2
    work_cost = work_weight * (work_error / WORK_LIMIT_J) ** 2
    arm_cost = arm_limit_weight * (jnp.maximum(peak - ARM_LIMIT_RAD, 0.0) / ARM_LIMIT_RAD) ** 2
    # These dimensionless control penalties are not physical energy measurements.
    score = potential_cost + work_cost + arm_cost
    finite = valid & jnp.isfinite(score + jnp.sum(energy, axis=-1))
    minimum = jnp.min(jnp.where(finite, score, jnp.inf), axis=-1, keepdims=True)
    tied = finite & (score == minimum)
    smallest = jnp.min(jnp.where(tied, jnp.abs(torques), jnp.inf), axis=-1, keepdims=True)
    tied = tied & (jnp.abs(torques) == smallest)
    positive = tied & (torques >= 0.0)
    selected = jnp.argmax(jnp.where(jnp.any(positive, axis=-1, keepdims=True), positive, tied), -1)
    torque = jnp.take_along_axis(torques, selected[..., None], -1)[..., 0]
    selected_work = jnp.take_along_axis(predicted_work, selected[..., None], -1)[..., 0]
    selected_peak = jnp.take_along_axis(peak, selected[..., None], -1)[..., 0]
    matched = jnp.abs(selected_work - requested_work) <= WORK_TOLERANCE_J
    # Modes describe the chosen action, not admissibility or a fallback controller.
    mode = jnp.where(jnp.any(finite, -1), jnp.where(matched, 0, 1), 3)
    return jnp.where(mode == 3, jnp.nan, torque), {
        "mode": mode,
        "requested_work_j": requested_work,
        "predicted_work_j": selected_work,
        "predicted_peak_arm_rad": selected_peak,
        "predicted_terminal_arm_speed": jnp.take_along_axis(
            endpoint[..., 2], selected[..., None], -1
        )[..., 0],
        "predicted_energy_j": jnp.take_along_axis(energy, selected[..., None, None], -2)[..., 0, :],
        "root_count": jnp.sum(
            finite[..., :32] & (jnp.abs(work_error[..., :32]) <= WORK_TOLERANCE_J), axis=-1
        ),
        "bounded_count": jnp.sum(finite & (peak <= ARM_LIMIT_RAD), axis=-1),
        "selected_potential_cost": jnp.take_along_axis(potential_cost, selected[..., None], -1)[
            ..., 0
        ],
        "selected_work_cost": jnp.take_along_axis(work_cost, selected[..., None], -1)[..., 0],
        "selected_arm_limit_cost": jnp.take_along_axis(arm_cost, selected[..., None], -1)[..., 0],
    }
