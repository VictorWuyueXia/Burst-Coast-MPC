"""Cross-implementation, derivative and physical acceptance tests for both inner solvers."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.dynamics import rk4_step as numpy_rk4
from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL, rk4_step
from rotary_pendulum.environment.jax_environment import EnvState
from rotary_pendulum.mpc.inner_model import CONSTRAINT_JAC, OBJECTIVE_GRAD, evaluate, rollout
from rotary_pendulum.mpc.inner_solvers import sample_inner, solve_inner
from rotary_pendulum.RL.jax_stationary_task import transition

ROOT = Path("artifacts/rotary_pendulum/1-TD3_20261016/machine-scannables")
BASE = json.loads((ROOT / "training_settings.json").read_text())
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
ARCHIVE = np.load(ROOT / "weights.npz", allow_pickle=False)
NETWORKS = {
    name: {
        "params": {
            f"Dense_{i}": {
                field: jnp.asarray(ARCHIVE[f"{source}.{i}.{field}"], jnp.float64)
                for field in ("kernel", "bias")
            }
            for i in range(4)
        }
    }
    for name, source in (("actor", "actor"), ("q1", "q1"), ("q2", "q2"))
}
ARCHIVE.close()
NETWORKS = dict(actor=NETWORKS["actor"], critics=(NETWORKS["q1"], NETWORKS["q2"]))
PLANT = {
    name: BASE[name]
    for name in (
        "torque_limit_nm",
        "discount",
        "torque_cost_per_s",
        "arm_cost_per_s",
        "energy_cost_per_s",
        "upright_reward_per_s",
        "upright_widths",
    )
} | dict(
    observation_scales=MANIFEST["observation"]["divisors"],
    rl_cost=False,
    centering_cost_per_s=1.0,
    speed_cost_per_s=0.0,
    speed_width_rad=float(np.deg2rad(5)),
)
SETTINGS = dict(
    plant=PLANT,
    max_iterations=60,
    tolerance=1e-7,
    population=64,
    elite_count=8,
    generations=3,
    initial_std=0.5,
)


def test_contract_parity() -> None:
    """Saved network references, independent NumPy mechanics and original reward agree."""
    assert jax.config.x64_enabled
    reference = np.load(ROOT / "reference_vectors.npz", allow_pickle=False)
    states = jnp.asarray(reference["state"][:128], jnp.float64)
    values = evaluate(jnp.zeros(1), states, NETWORKS, PLANT, "value")
    np.testing.assert_allclose(
        values["policy"] * PLANT["torque_limit_nm"],
        reference["actor_torque_nm"][:128],
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        -values["costs"].max(axis=-1), reference["v_actor"][:128], rtol=1e-5, atol=1e-5
    )
    reference.close()
    x0 = jnp.array([0.1, 0.4, 0.3, -0.2])
    actions = jnp.array([0.2, -0.1, 0.4])
    predicted = rollout(x0, actions, PLANT)
    current = np.asarray(x0)
    recorded = []
    for action in np.repeat(np.asarray(actions), 50):
        current = numpy_rk4(
            current, float(action) * PLANT["torque_limit_nm"], 0.002, PHYSICAL, MODEL
        )
        recorded.append(current)
    np.testing.assert_allclose(predicted.reshape(-1, 4), recorded, atol=1e-11, rtol=1e-11)
    history = rk4_step(x0, 0.001, record_substeps=True)
    np.testing.assert_array_equal(history[-1], rk4_step(x0, 0.001))
    initial = EnvState(
        x0[None],
        jnp.zeros(1, jnp.int32),
        jnp.zeros(1, jnp.int32),
        jnp.zeros(1, bool),
        jnp.zeros(1, bool),
        jnp.zeros(1, bool),
    )
    following, reward, _, _ = transition(initial, actions[:1], BASE)
    audit = evaluate(jnp.array([0.2, 0.0]), x0, NETWORKS, PLANT, "audit")
    np.testing.assert_allclose(audit["rl_stage_costs"], -reward, atol=1e-10)
    np.testing.assert_allclose(audit["trajectory"][-1, -1], following.x[0], atol=1e-11)
    assert bool(audit["physical_pass"])
    assert np.max(audit["refinement_error"]) < 1e-5


def test_derivatives() -> None:
    """Check actor-through-critic and trajectory derivatives, including the angle seam."""
    x0 = jnp.array([0.1, 2.8, 0.1, -0.4])
    z = jnp.array([0.1, -0.2, 0.05, 25.0])
    direction = jnp.array([0.2, -0.5, 0.3, 0.1])
    for output, derivative in (("objective", OBJECTIVE_GRAD), ("constraints", CONSTRAINT_JAC)):
        exact = derivative(z, x0, NETWORKS, PLANT, output) @ direction
        for step in (1e-3, 1e-4, 1e-5):
            difference = (
                evaluate(z + step * direction, x0, NETWORKS, PLANT, output)
                - evaluate(z - step * direction, x0, NETWORKS, PLANT, output)
            ) / (2 * step)
            np.testing.assert_allclose(exact, difference, rtol=1e-4, atol=1e-5)
    scored = evaluate(z, x0, NETWORKS, PLANT, "score")
    tight = z.at[-1].set(scored["costs"].max())
    np.testing.assert_allclose(
        evaluate(tight, x0, NETWORKS, PLANT, "objective"), scored["objective"], atol=1e-12
    )
    left = evaluate(z, x0.at[1].set(jnp.pi - 1e-8), NETWORKS, PLANT, "value")
    right = evaluate(z, x0.at[1].set(-jnp.pi + 1e-8), NETWORKS, PLANT, "value")
    np.testing.assert_allclose(left["costs"], right["costs"], atol=1e-5)
    consistent = PLANT | dict(rl_cost=True)
    audit = evaluate(tight, x0, NETWORKS, consistent, "audit")
    np.testing.assert_allclose(
        evaluate(tight, x0, NETWORKS, consistent, "rl_objective"), audit["objective"], atol=1e-12
    )


def test_feasibility_acceptance() -> None:
    """Both solvers find executable plans and reject hidden crossings and invalid states."""
    limit = PLANT["observation_scales"][0]
    crossing = jnp.array([limit - 0.002, 0.0, 0.4, 0.0])
    crossed = evaluate(jnp.array([-1.0, 0.0]), crossing, NETWORKS, PLANT, "audit")
    assert float(crossed["trajectory"][-1, -1, 0]) < limit
    assert np.max(crossed["trajectory"][..., 0]) > limit
    assert not bool(crossed["physical_pass"])
    x0 = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    seeds = jnp.array([[0.0], [0.5], [-0.5]])
    for solver in (solve_inner, sample_inner):
        arguments = (x0, seeds, NETWORKS, SETTINGS)
        result = (
            solver(*arguments, jax.random.PRNGKey(7))
            if solver is sample_inner
            else solver(*arguments)
        )
        assert result["accepted"]
        plan, audit = result["plans"][result["selected"]]
        assert np.max(np.abs(plan[:-1])) <= 1
        assert bool(audit["physical_pass"]) and float(audit["arm_margin_rad"]) > 0
        initial = evaluate(jnp.array([0.0, 0.0]), x0, NETWORKS, PLANT, "score")
        assert float(audit["objective"]) <= float(initial["objective"]) + 1e-5
    repeated = sample_inner(x0, seeds, NETWORKS, SETTINGS, jax.random.PRNGKey(7))
    np.testing.assert_array_equal(
        result["plans"][result["selected"]][0], repeated["plans"][repeated["selected"]][0]
    )
    for state in (x0.at[0].set(limit + 0.1), x0.at[2].set(jnp.nan)):
        audit = evaluate(jnp.zeros(2), state, NETWORKS, PLANT, "audit")
        assert not bool(audit["physical_pass"])
        result = sample_inner(state, seeds, NETWORKS, SETTINGS, jax.random.PRNGKey(4))
        assert not result["accepted"] and result["selected"] == -1
    failed = solve_inner(x0, seeds[1:], NETWORKS, SETTINGS | dict(max_iterations=1))
    assert not failed["accepted"] and failed["selected"] == -1
