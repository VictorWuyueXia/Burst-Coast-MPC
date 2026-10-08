"""Check the angle-only goal, physical clocks, and held-torque work in each snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_dynamics import PHYSICS_DT_S, rk4_step
from rotary_pendulum.environment.jax_environment import (
    GOAL,
    HOLD_PHYSICS_STEPS,
    MAX_PHYSICS_STEPS,
    reset,
    step,
)
from rotary_pendulum.heuristic.decoder import decode
from rotary_pendulum.heuristic.energy import encode, policy
from rotary_pendulum.heuristic.evaluation import evaluate


def main() -> None:
    assert np.isclose(GOAL.beta_tolerance_rad, np.deg2rad(15))
    assert np.isclose(GOAL.hold_steps * PHYSICS_DT_S, 0.1)
    assert np.isclose(HOLD_PHYSICS_STEPS * PHYSICS_DT_S, 0.1)
    assert np.isclose(MAX_PHYSICS_STEPS * PHYSICS_DT_S, 20.0)
    print(f"Checking angle-only goal at {PHYSICS_DT_S}s integration", flush=True)
    state = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(0), 4), jnp.zeros(4, jnp.int32))
    state = state._replace(
        x=jnp.array(
            [
                [0, jnp.pi, 0.3, 0.4],
                [0, jnp.pi + np.deg2rad(16), 0, 0],
                [4, jnp.pi, 0, 0],
                [0, jnp.pi - np.deg2rad(16), 0, 0],
            ]
        ),
        goal_count=jnp.full(4, GOAL.hold_steps - 1, dtype=jnp.int32),
    )
    following = jax.jit(lambda s: step(s, 0.0, physics_steps=1))(state)
    np.testing.assert_array_equal(following.success, [True, False, True, False])
    assert abs(following.x[0, 2]) > 0.15 and abs(following.x[0, 3]) > 0.20
    assert following.arm_violation[2]
    rest = reset(jax.random.PRNGKey(1), 0)._replace(x=jnp.array([0.0, jnp.pi, 0.0, 0.0]))
    before = jax.jit(lambda s: step(s, 0.0, physics_steps=GOAL.hold_steps - 1))(rest)
    assert not before.success
    assert step(before, 0.0, physics_steps=1).success
    torque, diagnostic = jax.jit(
        lambda x: decode(x, policy(encode(x), 0.04, 1.0), jnp.array(1.0), 4 * HOLD_PHYSICS_STEPS)
    )(state.x)
    replay = state.x
    for _ in range(HOLD_PHYSICS_STEPS):
        replay = rk4_step(replay, torque)
    np.testing.assert_allclose(
        torque * (replay[..., 0] - state.x[..., 0]), diagnostic["predicted_work_j"], atol=1e-12
    )
    final, trace = jax.jit(
        lambda s: evaluate(s, jnp.array([0.04, 1.0, 1.0]), 4 * HOLD_PHYSICS_STEPS, 1, "energy")
    )(state)
    np.testing.assert_allclose(trace["elapsed_s"][0], final.physics_steps * PHYSICS_DT_S)
    assert trace["physics_x"].shape[-2] == HOLD_PHYSICS_STEPS
    result = {
        "physics_step_s": PHYSICS_DT_S,
        "goal_angle_deg": 15,
        "goal_hold_s": 0.1,
        "control_hold_s": 0.1,
        "deadline_s": 20,
        "velocity_ignored": True,
        "outside_strip_rejected": True,
        "arm_position_ignored": True,
        "predicted_work_clock_verified": True,
        "partial_action_clock_verified": True,
    }
    output = Path(__file__).resolve().parents[3] / (
        "artifacts/rotary_pendulum/experiment-results/"
        "angle-only-goal-and-integration-step-comparison/raw-runs"
    )
    (output / f"checks_dt{round(1000 * PHYSICS_DT_S)}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(f"PASS {result}", flush=True)


if __name__ == "__main__":
    main()
