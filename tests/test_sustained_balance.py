"""Long-hold control must evolve after capture and respect the declared soft range."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_dynamics import rk4_step
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, reset, step
from rotary_pendulum.RL.jax_residual_task import transition

SETTINGS = json.loads(
    Path("artifacts/rotary_pendulum/reference-data/9-RL/11-rl-continuous-residual/machine-scannables/training_template.json").read_text()
)


def test_live_post_capture_motion_timeout_and_longer_hold() -> None:
    initial = reset(jax.random.PRNGKey(0), 0)._replace(x=jnp.array([0.0, -jnp.pi, 0.0, 0.0]))
    captured = step(initial, 0.0, physics_steps=50)
    assert captured.success
    continued = step(captured, 0.001, stop_on_success=False, max_physics_steps=55)
    assert continued.timeout and continued.physics_steps == 55
    assert abs(continued.x[2]) > 0.01
    long = step(initial, 0.0, physics_steps=50, goal_hold_steps=500)
    assert not long.success and long.goal_count == 50
    completed = step(
        long._replace(x=initial.x, goal_count=jnp.array(499), physics_steps=jnp.array(499)),
        0.0,
        physics_steps=1,
        goal_hold_steps=500,
    )
    assert completed.success and completed.physics_steps == 500


def test_margin_cost_precedes_soft_bound_and_never_clips_position() -> None:
    np.testing.assert_allclose(float(ARM_LIMIT_RAD), np.deg2rad(135))
    initial = reset(jax.random.PRNGKey(1), 0)._replace(
        x=jnp.array([0.9 * ARM_LIMIT_RAD, 0.0, 0.0, 0.0])
    )
    following, _, components, *_ = transition(initial, jnp.array(0.0), jnp.array(6000), SETTINGS)
    expected = -0.1 * SETTINGS["arm_cost_per_s"] * (0.9**2 + 0.5**2)
    np.testing.assert_allclose(components[3], expected, rtol=1e-5)
    assert not following.arm_violation
    crossed = step(initial._replace(x=initial.x.at[0].set(ARM_LIMIT_RAD + 0.1)), 0.0)
    assert crossed.arm_violation and crossed.x[0] > ARM_LIMIT_RAD


def test_previous_runaway_state_is_finite_with_fine_internal_integration() -> None:
    initial = jnp.array([-2623.25, -1.91138, -149.3916, -48.7931])
    following = jax.lax.fori_loop(0, 5, lambda _, x: rk4_step(x, -0.002295), initial)
    assert np.isfinite(following).all()
    assert float(jnp.max(abs(following[2:]))) < 300
