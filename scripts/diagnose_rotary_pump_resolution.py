"""Measure pump-action energy and rotation behavior at the 100 ms cadence."""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICS_DT_S, TORQUE_LIMIT_NM
from rotary_pendulum.environment.jax_environment import GOAL, EnvState, reset, step

OUTPUT = Path("artifacts/rotary_pendulum/action-resolution-study")
PUMP_FRACTIONS = (0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 1.00)
FINE_FRACTION = 0.02
SELECTED_PUMP_FRACTION = 0.45
STATE_COUNT = 256
DECISIONS = 200
ACTION_COUNT = 5


def main() -> None:
    """Compare candidate pump magnitudes under a fixed potential-MPC controller."""

    machine_dir = OUTPUT / "machine-scannables"
    human_dir = OUTPUT / "human-readables"
    machine_dir.mkdir(parents=True, exist_ok=True)
    human_dir.mkdir(parents=True, exist_ok=True)
    keys = jax.random.split(jax.random.key(20261007), STATE_COUNT)
    initial = jax.vmap(reset)(keys, jnp.zeros((STATE_COUNT,), dtype=jnp.int32))
    action_indices = jnp.arange(ACTION_COUNT, dtype=jnp.int32)
    sequences = jnp.stack(
        jnp.meshgrid(action_indices, action_indices, action_indices, indexing="ij"), axis=-1
    ).reshape((ACTION_COUNT**3, 3))
    target_energy = 2.0 * MODEL.gravity_torque_nm

    def potential(state: EnvState) -> jax.Array:
        theta, alpha, omega, nu = jnp.moveaxis(state.x, -1, 0)
        energy = 0.5 * MODEL.pendulum_inertia_kg_m2 * nu**2 + MODEL.gravity_torque_nm * (
            1.0 - jnp.cos(alpha)
        )
        energy_error = (energy - target_energy) / target_energy
        beta = jnp.arctan2(jnp.sin(alpha - jnp.pi), jnp.cos(alpha - jnp.pi))
        capture_error = 0.25 * (
            (theta / GOAL.theta_tolerance_rad) ** 2
            + (beta / GOAL.beta_tolerance_rad) ** 2
            + (omega / GOAL.omega_tolerance_rad_s) ** 2
            + (nu / GOAL.nu_tolerance_rad_s) ** 2
        )
        active = (~(state.success | state.arm_violation | state.timeout)).astype(state.x.dtype)
        return -active * (
            energy_error**2 / (1.0 + energy_error**2) + capture_error / (1.0 + capture_error)
        )

    def rollout(pump_fraction: jax.Array) -> jax.Array:
        torques = TORQUE_LIMIT_NM * jnp.array(
            (0.0, -pump_fraction, pump_fraction, -FINE_FRACTION, FINE_FRACTION),
            dtype=jnp.float32,
        )
        initial_totals = jnp.zeros((STATE_COUNT, 8), dtype=jnp.float32)

        def advance(carry: tuple[EnvState, jax.Array], _unused: None):
            state, totals = carry
            candidates = jax.tree.map(
                lambda value: jnp.repeat(value[:, None, ...], ACTION_COUNT**3, axis=1), state
            )
            score = jnp.zeros((STATE_COUNT, ACTION_COUNT**3), dtype=jnp.float32)
            for depth in range(3):
                action = jnp.broadcast_to(sequences[None, :, depth], score.shape)
                before_done = candidates.success | candidates.arm_violation | candidates.timeout
                before_steps = candidates.physics_steps
                candidates = step(candidates, torques[action])
                elapsed = PHYSICS_DT_S * (candidates.physics_steps - before_steps)
                score += (
                    5.0 * (candidates.success & ~before_done)
                    - 5.0 * (candidates.arm_violation & ~before_done)
                    - 2.0 * (candidates.timeout & ~before_done)
                    - 0.05 * elapsed * (action != 0)
                    - 0.005 * elapsed
                )
            score += potential(candidates)
            selected = jnp.argmax(score, axis=-1)
            action = sequences[selected, 0]
            state = step(state, torques[action])
            theta, alpha, _, nu = jnp.moveaxis(state.x, -1, 0)
            energy_ratio = (
                0.5 * MODEL.pendulum_inertia_kg_m2 * nu**2
                + MODEL.gravity_torque_nm * (1.0 - jnp.cos(alpha))
            ) / target_energy
            totals = totals.at[:, 0].max(
                jnp.rad2deg(jnp.abs(jnp.arctan2(jnp.sin(alpha), jnp.cos(alpha))))
            )
            totals = totals.at[:, 1].max(jnp.abs(theta))
            totals = totals.at[:, 2].max(energy_ratio)
            totals = totals.at[:, 3].add(action == 0)
            totals = totals.at[:, 4].add((action == 1) | (action == 2))
            totals = totals.at[:, 5].add((action == 3) | (action == 4))
            totals = totals.at[:, 6].set(state.success)
            totals = totals.at[:, 7].set(state.arm_violation)
            return (state, totals), None

        return jax.lax.scan(advance, (initial, initial_totals), None, length=DECISIONS)[0][1]

    compiled = jax.jit(rollout)
    rows: list[dict[str, float]] = []
    for fraction in PUMP_FRACTIONS:
        print(f"pump-resolution pump_fraction={fraction:.2f}", flush=True)
        values = np.asarray(jax.device_get(compiled(jnp.asarray(fraction, dtype=jnp.float32))))
        rows.append(
            {
                "pump_fraction": fraction,
                "pump_torque_nm": fraction * float(TORQUE_LIMIT_NM),
                "success_rate": float(values[:, 6].mean()),
                "arm_violation_rate": float(values[:, 7].mean()),
                "mean_max_departure_from_downward_deg": float(values[:, 0].mean()),
                "p95_max_departure_from_downward_deg": float(np.quantile(values[:, 0], 0.95)),
                "mean_peak_energy_ratio": float(values[:, 2].mean()),
                "mean_pump_actions": float(values[:, 4].mean()),
                "mean_fine_actions": float(values[:, 5].mean()),
            }
        )

    with (machine_dir / "pump_sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fractions = [100.0 * row["pump_fraction"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(
        fractions,
        [row["mean_peak_energy_ratio"] for row in rows],
        marker="o",
        label="mean peak energy",
    )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1, label="upright energy")
    axes[0].axvline(100.0 * SELECTED_PUMP_FRACTION, color="tab:green", linestyle=":")
    axes[0].set(
        title="Energy reached from downward starts",
        xlabel="Pump torque [% of physical limit]",
        ylabel="Peak swing-energy ratio",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        fractions,
        [row["mean_max_departure_from_downward_deg"] for row in rows],
        marker="o",
        label="mean",
    )
    axes[1].plot(
        fractions,
        [row["p95_max_departure_from_downward_deg"] for row in rows],
        marker="o",
        label="95th percentile",
    )
    axes[1].axvline(
        100.0 * SELECTED_PUMP_FRACTION,
        color="tab:green",
        linestyle=":",
        label="selected 45%",
    )
    axes[1].set(
        title="Maximum departure from downward",
        xlabel="Pump torque [% of physical limit]",
        ylabel="Maximum angular distance from downward [deg]",
        yscale="log",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.subplots_adjust(wspace=0.28)
    figure.savefig(human_dir / "pump_resolution.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"pump-resolution artifact_dir={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
