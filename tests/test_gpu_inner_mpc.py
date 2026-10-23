"""GPU solver tests: exact costs, constrained quadratic steps and batched acceptance."""

import jax
import jax.numpy as jnp
import numpy as np
from test_inner_mpc import NETWORKS, PLANT

from rotary_pendulum.mpc.gpu_cem import initial_plans, sample_gpu
from rotary_pendulum.mpc.gpu_sqp import quadratic_step, solve_gpu
from rotary_pendulum.mpc.inner_model import CONSTRAINT_JAC, evaluate


def test_centering_and_compression() -> None:
    """Centering integrates the declared grid; extrema preserve every arm inequality."""
    assert jax.default_backend() == "gpu"
    x = jnp.array([0.7, 2.8, 0.2, -0.3])
    z = jnp.array([0.2, -0.1, 0.1, 30.0])
    audit = evaluate(z, x, NETWORKS, PLANT, "audit")
    theta = np.asarray(audit["trajectory"][..., 0])
    expected = (
        0.002
        * PLANT["centering_cost_per_s"]
        * np.sum(
            (theta / PLANT["observation_scales"][0]) ** 2
            * PLANT["discount"] ** np.arange(3)[:, None]
        )
    )
    np.testing.assert_allclose(audit["centering"], expected, rtol=1e-12)
    uncentered = evaluate(z, x, NETWORKS, PLANT | dict(centering_cost_per_s=0.0), "audit")
    np.testing.assert_allclose(audit["objective"] - uncentered["objective"], expected, atol=1e-12)
    full = evaluate(z, x, NETWORKS, PLANT, "constraints")
    compact = evaluate(z, x, NETWORKS, PLANT, "sqp")
    np.testing.assert_allclose(full.min(), compact[1:-6].min(), atol=1e-12)
    direction = jnp.array([0.1, -0.3, 0.2, 0.4])
    derivative = CONSTRAINT_JAC(z, x, NETWORKS, PLANT, "sqp") @ direction
    finite = (
        evaluate(z + 1e-5 * direction, x, NETWORKS, PLANT, "sqp")
        - evaluate(z - 1e-5 * direction, x, NETWORKS, PLANT, "sqp")
    ) / 2e-5
    np.testing.assert_allclose(derivative, finite, rtol=1e-4, atol=1e-5)


def test_quadratic_program() -> None:
    """A known active-bound QP verifies the step, multiplier sign and residual."""
    step, dual, error = jax.jit(quadratic_step, static_argnames=("iterations",))(
        jnp.eye(2),
        jnp.array([-2.0, 1.0]),
        jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
        jnp.array([0.0, 1.0, 0.0]),
        2.0,
        200,
    )
    np.testing.assert_allclose(step, [1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(dual, [0.0, 1.0, 1.0], atol=1e-8)
    assert error < 1e-8


def test_gpu_batch_acceptance() -> None:
    """Batched methods keep feasible evaluated incumbents; invalid starts stay rejected."""
    states = jnp.array([[0.0, jnp.pi, 0.0, 0.0], [0.1, 2.9, 0.05, 0.1]])
    seeds = initial_plans(states, NETWORKS, PLANT, 3)
    initial = evaluate(
        jnp.pad(seeds, ((0, 0), (0, 0), (0, 1))), states[:, None, :], NETWORKS, PLANT, "score"
    )
    baseline = jnp.where(initial["feasible"], initial["objective"], jnp.inf).min(axis=-1)
    sqp = solve_gpu(states, seeds, NETWORKS, PLANT, 8, 80, 0.5, 10.0)
    for result in (
        sqp,
        sample_gpu(states, seeds, NETWORKS, PLANT, jax.random.PRNGKey(7), 32, 8, 3, 0.5),
    ):
        best = jnp.where(result["accepted"], result["audit"]["objective"], jnp.inf).min(axis=-1)
        assert bool(result["accepted"].any(axis=-1).all())
        assert bool(jnp.all(best <= baseline + 1e-9))
        assert bool(jnp.all(jnp.abs(result["plans"][..., :-1]) <= 1))
    repeated = sample_gpu(states, seeds, NETWORKS, PLANT, jax.random.PRNGKey(7), 32, 8, 3, 0.5)
    np.testing.assert_array_equal(result["plans"], repeated["plans"])
    bad = states.at[:, 0].set(PLANT["observation_scales"][0] + 0.1)
    for result in (
        solve_gpu(bad, seeds, NETWORKS, PLANT, 2, 20, 0.5, 10.0),
        sample_gpu(bad, seeds, NETWORKS, PLANT, jax.random.PRNGKey(9), 32, 8, 2, 0.5),
    ):
        assert not bool(result["accepted"].any())


def test_local_speed() -> None:
    """Reconstruct bounded speed cost and check CEM acceptance entirely on GPU."""
    assert jax.default_backend() == "gpu"
    plant = PLANT | dict(speed_cost_per_s=0.1)
    z = jnp.array([0.02, -0.01, 0.01, 30.0])
    x = jnp.array([0.1, jnp.pi - 0.02, 0.01, -0.01])
    audit = evaluate(z, x, NETWORKS, plant, "audit")
    trajectory = np.asarray(audit["trajectory"])
    gate = np.exp(-(1 + np.cos(trajectory[..., 1])) / plant["speed_width_rad"] ** 2)
    weights = plant["discount"] ** np.arange(3)[:, None, None]
    expected = (
        0.001
        * 0.1
        * np.sum(
            gate[..., None]
            * (
                1
                - np.exp(
                    -0.5 * (trajectory[..., 2:] / np.array(plant["upright_widths"])[[2, 1]]) ** 2
                )
            )
            * weights,
            axis=(0, 1),
        )
    )
    np.testing.assert_allclose([audit["speed_arm"], audit["speed_pendulum"]], expected, atol=1e-12)
    baseline = evaluate(z, x, NETWORKS, PLANT, "audit")
    np.testing.assert_allclose(
        audit["objective"] - baseline["objective"], expected.sum(), atol=1e-12
    )
    np.testing.assert_allclose(
        baseline["objective"],
        baseline["effort"]
        + baseline["centering"]
        + PLANT["discount"] ** 3 * baseline["costs"].max(),
        atol=1e-12,
    )
    assert 0 < expected.sum() < 0.03
    for state in (jnp.array([0.0, 0.0, 0.0, 0.0]), jnp.array([0.0, jnp.pi, 0.0, 0.0])):
        value = evaluate(jnp.array([0.0, 0.0]), state, NETWORKS, plant, "audit")
        np.testing.assert_allclose(value["speed_arm"] + value["speed_pendulum"], 0, atol=1e-12)
    states = jnp.stack((x, x.at[0].set(0.2)))
    seeds = initial_plans(states, NETWORKS, plant, 3)
    result = sample_gpu(states, seeds, NETWORKS, plant, jax.random.PRNGKey(7), 32, 8, 3, 0.5)
    assert bool(result["accepted"].any(axis=-1).all())


def test_lqr_stage_objective() -> None:
    """Exact LQR time/input convention, periodic error, terminal sum and CEM feasibility."""
    assert jax.default_backend() == "gpu" and jax.config.x64_enabled
    weights = [1 / PLANT["observation_scales"][0] ** 2, 1 / 0.15**2, 0.25, 0.25]
    plant = PLANT | dict(discount=1.0, lqr_state_weights=weights)
    x = jnp.array([0.1, jnp.pi - 0.02, 0.01, -0.01])
    z = jnp.array([0.02, -0.01, 0.01, 30.0])
    audit = evaluate(z, x, NETWORKS, plant, "audit")
    nodes = np.vstack((np.asarray(x), np.asarray(audit["trajectory"])[:-1, -1]))
    nodes[:, 1] = np.arctan2(-np.sin(nodes[:, 1]), -np.cos(nodes[:, 1]))
    stage = np.sum(nodes**2 * weights) + np.sum(np.asarray(z[:-1]) ** 2)
    np.testing.assert_allclose(audit["objective"], stage + audit["costs"].max(), atol=1e-12)
    np.testing.assert_allclose(evaluate(z, x, NETWORKS, plant, "objective"), stage + 30, atol=1e-12)
    # One-step cost is measured before action; catches endpoint shifting and dt scaling.
    one = evaluate(jnp.array([0.2, 0.0]), x, NETWORKS, plant, "audit")
    np.testing.assert_allclose(
        one["objective"] - one["costs"].max(), (nodes[0] ** 2 @ weights) + 0.2**2, atol=1e-12
    )
    periodic = evaluate(z, x.at[1].add(2 * jnp.pi), NETWORKS, plant, "audit")
    np.testing.assert_allclose(periodic["objective"], audit["objective"], atol=1e-10)
    # Legacy cost rates must not leak into the replacement objective.
    changed = plant | dict(centering_cost_per_s=99.0, speed_cost_per_s=99.0, torque_cost_per_s=99.0)
    np.testing.assert_allclose(
        evaluate(z, x, NETWORKS, changed, "score")["objective"], audit["objective"]
    )
    states = jnp.stack((x, x.at[0].set(0.2)))
    seeds = initial_plans(states, NETWORKS, plant, 3)
    result = sample_gpu(states, seeds, NETWORKS, plant, jax.random.PRNGKey(17), 32, 8, 3, 0.5)
    assert bool(result["accepted"].any(axis=-1).all())
