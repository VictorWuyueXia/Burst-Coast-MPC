"""Full-turn pendulum coordinates preserve physics, control, and recorded direction."""

# ruff: noqa: E402, I001
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from rotary_pendulum.environment.dynamics import rk4_step as numpy_step
from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL, rk4_step
from rotary_pendulum.environment.jax_environment import reset, step
from rotary_pendulum.heuristic.artifacts import write_artifacts
from rotary_pendulum.heuristic.decoder import decode
from rotary_pendulum.heuristic.energy import encode, policy
from rotary_pendulum.heuristic.evaluation import evaluate
from rotary_pendulum.mpc.discrete_model import rk4_step_symbolic
from rotary_pendulum.RL.jax_residual_evaluation import validation_resets
from rotary_pendulum.visualization.phase import oscillator_phase_points


def test_complete_steps_cross_zero_in_both_directions_and_match_models() -> None:
    states = np.array([[0, 0.01, 0, -2], [0, 2 * np.pi - 0.01, 0, 2]], dtype=float)
    symbolic = ca.MX.sym("state", 4)
    physics = SimpleNamespace(simulation=SimpleNamespace(timestep_s=0.02), rotary_pendulum=PHYSICAL)
    model = ca.Function("full_turn", [symbolic], [rk4_step_symbolic(symbolic, 0, physics)])
    expected = numpy_step(states, 0, 0.02, PHYSICAL, MODEL)
    actual = np.asarray(jax.jit(rk4_step)(states, 0))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(np.array(model.map(2)(states.T)).T, expected, atol=1e-12)
    assert 350 < np.rad2deg(actual[0, 1]) < 360
    assert 0 < np.rad2deg(actual[1, 1]) < 10
    assert actual[0, 3] < 0 < actual[1, 3]
    for turns in (-3, 2):
        equivalent = states.copy()
        equivalent[:, 1] += turns * 2 * np.pi
        np.testing.assert_allclose(rk4_step(equivalent, 0), actual, atol=1e-12)
    # Repeated rotation must not accumulate turns in the position coordinate.
    rotating = jnp.array([0.0, 0.01, 0.0, 20.0])
    for _ in range(100):
        rotating = rk4_step(rotating, 0)
        assert 0 <= rotating[1] < 2 * np.pi


def test_equivalent_angles_preserve_controller_goal_and_phase_coordinates() -> None:
    states = jnp.array([[0.1, 0.2 + turns * 2 * np.pi, 0.3, -0.4] for turns in (-1, 0, 1)])
    energy = encode(states)
    requested = policy(energy, 0.04)
    torque, diagnostic = jax.jit(lambda x, w: decode(x, w, 0.01, 10000.0))(states, requested)
    np.testing.assert_allclose(energy, jnp.broadcast_to(energy[0], energy.shape), atol=1e-12)
    np.testing.assert_allclose(torque, torque[0], atol=1e-10)
    np.testing.assert_array_equal(diagnostic["mode"], diagnostic["mode"][0])
    _, phase = oscillator_phase_points(states, PHYSICAL, MODEL)
    np.testing.assert_allclose(phase, np.broadcast_to(phase[0], phase.shape), atol=1e-12)
    initial = reset(jax.random.PRNGKey(12), 0)
    for turns in (-1, 0, 1):
        captured = step(initial._replace(x=jnp.array([0.0, np.pi + turns * 2 * np.pi, 0, 0])), 0)
        assert captured.success and 0 <= captured.x[1] < 2 * np.pi
        downward = step(initial._replace(x=jnp.array([0.0, turns * 2 * np.pi, 0, 0])), 0)
        assert not downward.success and 0 <= downward.x[1] < 2 * np.pi


def test_recorded_positions_cover_both_sides_of_downward(tmp_path: Path) -> None:
    initial, _, labels = validation_resets(123, 1, [20])
    # Accept old input coordinates, but expose only canonical positions in new traces.
    initial = initial._replace(x=initial.x.at[0].set(jnp.array([0.0, -0.1, 0, -2.0])))
    _, traces = jax.jit(lambda state: evaluate(state, jnp.array([0.04, 0.01, 10000]), 2, "zero"))(
        initial
    )
    data = jax.device_get(traces)
    assert data["start_x"][0, 0, 1] > np.pi
    for field in ("start_x", "x", "physics_x"):
        assert np.all((data[field][..., 1] >= 0) & (data[field][..., 1] < 2 * np.pi))
    np.testing.assert_allclose(data["pendulum_angle_deg"], np.rad2deg(data["physics_x"][..., 1]))
    assert data["physics_x"][0, 0, 0, 3] < 0
    write_artifacts(tmp_path, data, labels, {"name": "angle-convention"})
    recorded = np.load(tmp_path / "machine-scannables/trajectories.npz")
    np.testing.assert_array_equal(recorded["pendulum_angle_deg"], data["pendulum_angle_deg"])
    assert "[0, 360)" in (tmp_path / "machine-scannables/settings.json").read_text()
    assert (tmp_path / "human-readables/representative_sessions.png").is_file()
