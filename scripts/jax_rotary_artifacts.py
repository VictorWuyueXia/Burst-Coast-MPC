"""Generate machine measurements and human visuals for JAX rotary validation."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from rotary_pendulum.environment.dynamics import rk4_step as numpy_rk4_step
from rotary_pendulum.environment.dynamics import state_derivative as numpy_state_derivative
from rotary_pendulum.environment.jax_dynamics import (
    MODEL,
    PHYSICAL,
    TORQUE_LIMIT_NM,
    rk4_step,
    state_derivative,
)
from rotary_pendulum.environment.jax_environment import reset, step

SEED = 20261005


def generate_validation_cases() -> tuple[np.ndarray, np.ndarray]:
    """Build the fixed parity population and its three torque cases."""

    rng = np.random.default_rng(SEED)
    strata = np.arange(4096) % 3
    states = np.empty((4096, 4), dtype=np.float64)
    downward, moving, upright = strata == 0, strata == 1, strata == 2
    states[downward] = rng.uniform(
        [-0.20, -0.20, -0.5, -0.5], [0.20, 0.20, 0.5, 0.5], (downward.sum(), 4)
    )
    states[moving] = rng.uniform(
        [-0.50, 0.40, -2.0, -6.0], [0.50, 2.60, 2.0, 6.0], (moving.sum(), 4)
    )
    states[moving, 1] *= rng.choice([-1.0, 1.0], moving.sum())
    states[upright] = rng.uniform(
        [-0.25, np.pi - 0.25, -1.0, -1.0],
        [0.25, np.pi + 0.25, 1.0, 1.0],
        (upright.sum(), 4),
    )
    states[:8] = np.array(
        [
            [np.pi / 2 - 1e-6, 0.0, 2.0, 6.0],
            [-np.pi / 2 + 1e-6, 0.0, -2.0, -6.0],
            [0.5, 2.6, 2.0, 6.0],
            [-0.5, -2.6, -2.0, -6.0],
            [0.0, np.pi, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.4, np.pi - 0.25, 2.0, -6.0],
            [-1.4, np.pi + 0.25, -2.0, 6.0],
        ]
    )
    return np.repeat(states, 3, axis=0), np.tile(
        [-TORQUE_LIMIT_NM, 0.0, TORQUE_LIMIT_NM], len(states)
    )


def measure_accuracy_and_rollouts(backend_dir: Path) -> None:
    """Measure numerical gates, reset coverage, and seeded random trajectories."""

    machine_dir = backend_dir / "machine-scannables"
    human_dir = backend_dir / "human-readables"
    states, torques = generate_validation_cases()
    logging.info("accuracy cases=%d dtype=%s", len(states), jnp.asarray(states).dtype)
    expected_derivative = numpy_state_derivative(states, torques, PHYSICAL, MODEL)
    expected_20ms = numpy_rk4_step(states, torques, 0.02, PHYSICAL, MODEL)
    actual_derivative = np.asarray(jax.jit(state_derivative)(states, torques))
    actual_20ms = np.asarray(jax.jit(rk4_step)(states, torques))
    candidate_100ms = jnp.asarray(states)
    reference_100ms = states.copy()
    interval_excursion = np.zeros(len(states), dtype=bool)
    sampled_inside = np.abs(reference_100ms[:, 0]) < np.pi / 2
    between_sample_miss = np.zeros(len(states), dtype=bool)
    for reference_step in range(50):
        reference_100ms = numpy_rk4_step(reference_100ms, torques, 0.002, PHYSICAL, MODEL)
        interval_excursion |= np.abs(reference_100ms[:, 0]) >= np.pi / 2
        if (reference_step + 1) % 10 == 0:
            endpoint_inside = np.abs(reference_100ms[:, 0]) < np.pi / 2
            between_sample_miss |= interval_excursion & sampled_inside & endpoint_inside
            sampled_inside = endpoint_inside
            interval_excursion.fill(False)
    for _ in range(5):
        candidate_100ms = rk4_step(candidate_100ms, torques)
    errors = {
        "derivative": np.abs(actual_derivative - expected_derivative),
        "rk4_20ms": np.abs(actual_20ms - expected_20ms),
        "resolution_100ms": np.abs(np.asarray(candidate_100ms) - reference_100ms),
    }
    component_names = ("theta", "alpha", "omega", "nu")
    metric_rows = []
    for metric_name, values in errors.items():
        for component, component_name in enumerate(component_names):
            gate: float | str = ""
            if jax.config.x64_enabled and metric_name == "derivative":
                gate = 1e-9
            elif jax.config.x64_enabled and metric_name == "rk4_20ms":
                gate = 1e-9 if component < 2 else 1e-8
            elif metric_name == "resolution_100ms":
                gate = 0.02 if component < 2 else 0.5
            maximum = values[:, component].max()
            metric_rows.append(
                {
                    "metric": metric_name,
                    "component": component_name,
                    "maximum": maximum,
                    "p99": np.quantile(values[:, component], 0.99),
                    "gate": gate,
                    "passed": maximum <= gate if gate != "" else "",
                }
            )
    metric_rows.append(
        {
            "metric": "between_sample_arm_miss",
            "component": "theta",
            "maximum": int(between_sample_miss.sum()),
            "p99": between_sample_miss.mean(),
            "gate": "",
            "passed": "",
        }
    )
    metric_file = (machine_dir / "accuracy_metrics.csv").open("w", newline="")
    writer = csv.DictWriter(metric_file, fieldnames=metric_rows[0].keys())
    writer.writeheader()
    writer.writerows(metric_rows)
    metric_file.close()
    np.savez_compressed(
        machine_dir / "accuracy_cases.npz",
        states=states,
        torques=torques,
        derivative_error=errors["derivative"],
        rk4_20ms_error=errors["rk4_20ms"],
        resolution_100ms_error=errors["resolution_100ms"],
        between_sample_arm_miss=between_sample_miss,
    )
    logging.info(
        "resolution max_angle=%.6g max_rate=%.6g between_sample_misses=%d",
        errors["resolution_100ms"][:, :2].max(),
        errors["resolution_100ms"][:, 2:].max(),
        between_sample_miss.sum(),
    )

    keys = jax.random.split(jax.random.key(SEED), 30_000)
    reset_states = jax.jit(jax.vmap(reset))(keys, jnp.repeat(jnp.arange(3), 10_000)).x
    reset_array = np.asarray(reset_states).reshape(3, 10_000, 4)
    beta = np.arctan2(np.sin(reset_array[:, :, 1] - np.pi), np.cos(reset_array[:, :, 1] - np.pi))
    reset_rows = []
    for stratum in range(3):
        values = reset_array[stratum]
        goal = (
            (np.abs(values[:, 0]) <= 0.05)
            & (np.abs(beta[stratum]) <= 0.05)
            & (np.abs(values[:, 2]) <= 0.05)
            & (np.abs(values[:, 3]) <= 0.05)
        )
        reset_rows.append(
            {
                "stratum": stratum,
                "count": len(values),
                "arm_accepted_fraction": np.mean(np.abs(values[:, 0]) < np.pi / 2),
                "goal_at_reset_fraction": goal.mean(),
                "minimum_variance": np.var(values, axis=0).min(),
            }
        )
    reset_file = (machine_dir / "reset_statistics.csv").open("w", newline="")
    writer = csv.DictWriter(reset_file, fieldnames=reset_rows[0].keys())
    writer.writeheader()
    writer.writerows(reset_rows)
    reset_file.close()
    np.savez_compressed(machine_dir / "reset_samples.npz", states=reset_array)

    rollout_keys = jax.random.split(jax.random.key(SEED + 1), 300)
    strata = jnp.arange(300) % 3
    initial = jax.vmap(reset)(rollout_keys, strata)
    actions = (
        jax.random.uniform(
            jax.random.key(SEED + 2),
            (200, 300),
            minval=-TORQUE_LIMIT_NM,
            maxval=TORQUE_LIMIT_NM,
        )
        .at[:, :30]
        .set(0.0)
    )
    final, history = jax.jit(
        lambda carry, action_sequence: jax.lax.scan(
            lambda state, action: ((next_state := jax.vmap(step)(state, action)), next_state),
            carry,
            action_sequence,
        )
    )(initial, actions)
    trajectory = np.concatenate((np.asarray(initial.x)[None], np.asarray(history.x)), axis=0)
    np.savez_compressed(
        machine_dir / "random_rollouts.npz",
        states=trajectory,
        actions=np.asarray(actions),
        strata=np.asarray(strata),
        success=np.asarray(final.success),
        arm_violation=np.asarray(final.arm_violation),
        timeout=np.asarray(final.timeout),
        physics_steps=np.asarray(final.physics_steps),
    )
    rollout_rows = []
    for episode in range(300):
        rollout_rows.append(
            {
                "episode": episode,
                "mode": "zero" if episode < 30 else "random",
                "stratum": int(strata[episode]),
                "physics_steps": int(final.physics_steps[episode]),
                "success": bool(final.success[episode]),
                "arm_violation": bool(final.arm_violation[episode]),
                "timeout": bool(final.timeout[episode]),
                "max_abs_theta_rad": np.abs(trajectory[:, episode, 0]).max(),
            }
        )
    rollout_file = (machine_dir / "rollout_summary.csv").open("w", newline="")
    writer = csv.DictWriter(rollout_file, fieldnames=rollout_rows[0].keys())
    writer.writeheader()
    writer.writerows(rollout_rows)
    rollout_file.close()
    logging.info(
        "rollouts success=%d violation=%d timeout=%d",
        np.asarray(final.success).sum(),
        np.asarray(final.arm_violation).sum(),
        np.asarray(final.timeout).sum(),
    )

    figure, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
    for metric_name, values in errors.items():
        maximum = np.maximum(values.max(axis=0), 1e-16)
        axes[0].semilogy(component_names, maximum, marker="o", label=metric_name)
    axes[0].set_ylabel("Maximum absolute error (plot floor 1e-16)")
    axes[0].legend()
    for stratum in range(3):
        axes[1].hist(reset_array[stratum, :, 1], bins=50, alpha=0.5, label=f"stratum {stratum}")
    axes[1].set_xlabel("Initial pendulum angle (rad)")
    axes[1].set_ylabel("Samples")
    axes[1].legend()
    representatives = (0, 1, 2, 30, 31, 32)
    state_time = np.arange(201) * 0.1
    for episode in representatives:
        axes[2].plot(
            state_time,
            trajectory[:, episode, 0],
            label=f"{'zero' if episode < 30 else 'random'} S{int(strata[episode])}",
        )
    axes[2].axhline(np.pi / 2, color="black", linestyle="--", linewidth=0.8)
    axes[2].axhline(-np.pi / 2, color="black", linestyle="--", linewidth=0.8)
    axes[2].set(xlabel="Simulated time (s)", ylabel="Arm angle (rad)")
    axes[2].legend(ncol=2)
    figure.savefig(human_dir / "accuracy_resets_rollouts.png", dpi=160)
    plt.close(figure)
