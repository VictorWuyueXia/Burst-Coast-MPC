"""Measure five-action local capture reachability at the 100 ms control cadence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from rotary_pendulum.environment.jax_dynamics import TORQUE_LIMIT_NM, rk4_step
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, reset

OUTPUT = Path("artifacts/rotary_pendulum/action-resolution-study")
FINE_FRACTIONS = (0.01, 0.02, 0.03, 0.05, 0.10)
PUMP_FRACTION = 0.45
GOAL_NAMES = ("phase9", "phase10", "wide")
GOAL_LIMITS = jnp.array(
    (
        (0.05, 0.05, 0.05, 0.05),
        (0.08, 0.08, 0.15, 0.20),
        (0.10, 0.12, 0.20, 0.30),
    ),
    dtype=jnp.float32,
)
SEQUENCE_DECISIONS = 5
PHYSICS_STEPS_PER_DECISION = 5
INITIAL_STATE_COUNT = 128
HOLD_PHYSICS_STEPS = 5


def main() -> None:
    """Enumerate every five-decision action sequence and write evidence artifacts."""

    machine_dir = OUTPUT / "machine-scannables"
    human_dir = OUTPUT / "human-readables"
    machine_dir.mkdir(parents=True, exist_ok=True)
    human_dir.mkdir(parents=True, exist_ok=True)

    action_indices = jnp.arange(5, dtype=jnp.int32)
    sequences = jnp.stack(
        jnp.meshgrid(*(action_indices for _ in range(SEQUENCE_DECISIONS)), indexing="ij"),
        axis=-1,
    ).reshape((-1, SEQUENCE_DECISIONS))
    root_key = jax.random.key(20261006)
    tight_key, near_key = jax.random.split(root_key)
    tight_extent = jnp.array((0.08, 0.12, 0.15, 0.30), dtype=jnp.float32)
    tight = (
        jax.random.uniform(
            tight_key, (INITIAL_STATE_COUNT, 4), minval=-tight_extent, maxval=tight_extent
        )
        .at[:, 1]
        .add(jnp.pi)
    )
    near_keys = jax.random.split(near_key, INITIAL_STATE_COUNT)
    near = jax.vmap(reset)(near_keys, jnp.full((INITIAL_STATE_COUNT,), 2)).x

    def enumerate_capture(initial: jax.Array, fine_fraction: jax.Array) -> tuple[jax.Array, ...]:
        torques = TORQUE_LIMIT_NM * jnp.array(
            (0.0, -PUMP_FRACTION, PUMP_FRACTION, -fine_fraction, fine_fraction),
            dtype=jnp.float32,
        )
        state = jnp.repeat(initial[:, None, :], sequences.shape[0], axis=1)
        counts = jnp.zeros(state.shape[:2] + (len(GOAL_NAMES),), dtype=jnp.int32)
        success = jnp.zeros_like(counts, dtype=jnp.bool_)
        failed = jnp.zeros(state.shape[:2], dtype=jnp.bool_)

        def advance(carry: tuple[jax.Array, ...], physics_index: jax.Array):
            x, goal_counts, captured, arm_failed = carry
            decision = physics_index // PHYSICS_STEPS_PER_DECISION
            torque = torques[sequences[:, decision]][None, :]
            active = ~arm_failed
            integrated = rk4_step(x, torque)
            x = jnp.where(active[..., None], integrated, x)
            arm_failed = arm_failed | (active & (jnp.abs(x[..., 0]) >= ARM_LIMIT_RAD))
            beta = jnp.arctan2(jnp.sin(x[..., 1] - jnp.pi), jnp.cos(x[..., 1] - jnp.pi))
            errors = jnp.stack(
                (jnp.abs(x[..., 0]), jnp.abs(beta), jnp.abs(x[..., 2]), jnp.abs(x[..., 3])),
                axis=-1,
            )
            inside = jnp.all(errors[..., None, :] <= GOAL_LIMITS, axis=-1)
            goal_counts = jnp.where(inside & ~arm_failed[..., None], goal_counts + 1, 0)
            captured = captured | (goal_counts >= HOLD_PHYSICS_STEPS)
            return (x, goal_counts, captured, arm_failed), None

        final, _ = jax.lax.scan(
            advance,
            (state, counts, success, failed),
            jnp.arange(SEQUENCE_DECISIONS * PHYSICS_STEPS_PER_DECISION),
        )
        captured, arm_failed = final[2], final[3]
        reachable = jnp.mean(jnp.any(captured, axis=1), axis=0)
        sequence_capture = jnp.mean(captured, axis=(0, 1))
        safe_sequence = jnp.mean(~arm_failed)
        return reachable, sequence_capture, safe_sequence

    compiled = jax.jit(enumerate_capture)
    rows: list[dict[str, float | str]] = []
    for fine_fraction in FINE_FRACTIONS:
        print(f"action-resolution fine_fraction={fine_fraction:.3f}", flush=True)
        for stratum, initial in (("tight", tight), ("near", near)):
            reachable, sequence_capture, safe_sequence = compiled(
                initial, jnp.asarray(fine_fraction, dtype=jnp.float32)
            )
            for index, goal_name in enumerate(GOAL_NAMES):
                rows.append(
                    {
                        "fine_fraction": fine_fraction,
                        "fine_torque_nm": fine_fraction * float(TORQUE_LIMIT_NM),
                        "stratum": stratum,
                        "goal": goal_name,
                        "reachable_initial_fraction": float(reachable[index]),
                        "capturing_sequence_fraction": float(sequence_capture[index]),
                        "safe_sequence_fraction": float(safe_sequence),
                    }
                )

    impulse_rows: list[dict[str, float]] = []
    upright = jnp.array((0.0, jnp.pi, 0.0, 0.0), dtype=jnp.float32)
    for fraction in (0.0, *FINE_FRACTIONS, 1.0):
        state = upright
        for _ in range(PHYSICS_STEPS_PER_DECISION):
            state = rk4_step(state, fraction * TORQUE_LIMIT_NM)
        delta = state - upright
        impulse_rows.append(
            {
                "torque_fraction": fraction,
                "torque_nm": fraction * float(TORQUE_LIMIT_NM),
                "theta_change_rad": float(delta[0]),
                "beta_change_rad": float(delta[1]),
                "omega_rad_s": float(delta[2]),
                "nu_rad_s": float(delta[3]),
            }
        )

    with (machine_dir / "reachability.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (machine_dir / "upright_impulse.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=impulse_rows[0].keys())
        writer.writeheader()
        writer.writerows(impulse_rows)
    (machine_dir / "contract.json").write_text(
        json.dumps(
            {
                "action_order": [
                    "off",
                    "negative_pump",
                    "positive_pump",
                    "negative_fine",
                    "positive_fine",
                ],
                "fine_fractions": FINE_FRACTIONS,
                "pump_fraction": PUMP_FRACTION,
                "goal_names": GOAL_NAMES,
                "goal_limits_theta_beta_omega_nu": np.asarray(GOAL_LIMITS).tolist(),
                "sequence_decisions": SEQUENCE_DECISIONS,
                "decision_period_s": 0.1,
                "initial_states_per_stratum": INITIAL_STATE_COUNT,
                "random_seed": 20261006,
            },
            indent=2,
        )
        + "\n"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for goal_name in GOAL_NAMES:
        selected = [row for row in rows if row["stratum"] == "tight" and row["goal"] == goal_name]
        axes[0].plot(
            [100.0 * float(row["fine_fraction"]) for row in selected],
            [100.0 * float(row["reachable_initial_fraction"]) for row in selected],
            marker="o",
            label=goal_name,
        )
    axes[0].set(
        title="Tight-start local reachability",
        xlabel="Fine torque [% of limit]",
        ylabel="Reachable starts [%]",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(title="Success box")
    impulse = impulse_rows[1:-1]
    impulse_fractions = [100.0 * row["torque_fraction"] for row in impulse]
    axes[1].plot(
        impulse_fractions,
        [row["omega_rad_s"] for row in impulse],
        marker="o",
        label="arm speed",
    )
    axes[1].plot(
        impulse_fractions,
        [row["nu_rad_s"] for row in impulse],
        marker="o",
        label="pendulum speed",
    )
    axes[1].axhline(0.15, color="black", linestyle="--", linewidth=1, label="arm speed bound")
    axes[1].axhline(0.20, color="black", linestyle=":", linewidth=1, label="pendulum speed bound")
    axes[1].set(
        title="One 100 ms positive pulse from upright",
        xlabel="Torque [% of limit]",
        ylabel="Final speed [rad/s]",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(human_dir / "action_resolution.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"action-resolution artifact_dir={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
