"""Prepare and verify 40 ms decisions with unchanged physics and a 100 ms goal hold."""

from __future__ import annotations

import difflib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    directory = Path(__file__).resolve().parent
    repository = directory.parents[2]
    snapshot = repository / (
        "artifacts/rotary_pendulum/experiment-results/"
        "forty-millisecond-decision-comparison/raw-runs/prepared"
    )
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        repository / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    (snapshot / "scripts").mkdir()
    shutil.copy2(repository / "scripts/validate_rotary_heuristic.py", snapshot / "scripts")
    shutil.copy2(repository / "pyproject.toml", snapshot)
    runner = snapshot / "scripts/experiments/torque-and-arm-angle-range-comparison"
    runner.mkdir(parents=True)
    shutil.copy2(directory.parent / "torque-and-arm-angle-range-comparison/run_study.py", runner)
    replacements = {
        "src/rotary_pendulum/configs/mission.yaml": [
            ("beta-tolerance-rad: 0.08", f"beta-tolerance-rad: {float(np.deg2rad(15))!r}"),
        ],
        "src/rotary_pendulum/environment/jax_environment.py": [
            ("HOLD_PHYSICS_STEPS = 5", "HOLD_PHYSICS_STEPS = 2"),
            (
                """        inside_goal = (
            (jnp.abs(upright_error) <= GOAL.beta_tolerance_rad)
            & (jnp.abs(next_x[..., 2]) <= GOAL.omega_tolerance_rad_s)
            & (jnp.abs(next_x[..., 3]) <= GOAL.nu_tolerance_rad_s)
        )""",
                "        inside_goal = jnp.abs(upright_error) <= GOAL.beta_tolerance_rad",
            ),
        ],
        "src/rotary_pendulum/heuristic/decoder.py": [
            (
                "recovery_steps <= 5 or recovery_steps % 5",
                "recovery_steps <= 2 or recovery_steps % 2",
            ),
            ("multiple of five, greater than five", "multiple of two, greater than two"),
            ("(index >= 5) & (index % 5 == 0)", "(index >= 2) & (index % 2 == 0)"),
            ("index == 4", "index == 1"),
            ("predict(grid, 5)", "predict(grid, 2)"),
            ("predict(middle, 5)", "predict(middle, 2)"),
        ],
        "src/rotary_pendulum/heuristic/evaluation.py": [("length=5)", "length=2)")],
        "src/rotary_pendulum/heuristic/artifacts.py": [
            (
                'np.isclose(traces["elapsed_s"][:, lane], 0.1)',
                'np.isclose(traces["elapsed_s"][:, lane], 0.04)',
            ),
            ("np.arange(1, 6)", "np.arange(1, 3)"),
            ("axhline(0.08,", "axhline(np.deg2rad(15),"),
        ],
        "scripts/validate_rotary_heuristic.py": [
            ('campaign["decisions"] > 200', 'campaign["decisions"] > 500'),
            ('campaign["chunk_decisions"] > 10', 'campaign["chunk_decisions"] > 25'),
            ('campaign["recovery_steps"] <= 5', 'campaign["recovery_steps"] <= 2'),
            ('campaign["recovery_steps"] % 5', 'campaign["recovery_steps"] % 2'),
            ("multiple of five, greater than five", "multiple of two, greater than two"),
            ("campaign['chunk_decisions'] * 0.1", "campaign['chunk_decisions'] * 0.04"),
            (
                'np.isclose(traces["elapsed_s"][index], 0.1)',
                'np.isclose(traces["elapsed_s"][index], 0.04)',
            ),
            ("range(50)", "range(20)"),
            (
                '"capture_requires_arm_centering": False,',
                '"capture_requires_arm_centering": False,\n                "physics_step_s": 0.02,'
                '\n                "decision_step_s": 0.04,\n                "goal_hold_s": 0.1,'
                '\n                "goal_angle_deg": 15,'
                '\n                "goal_requires_velocity": False,',
            ),
        ],
    }
    patch = []
    for relative, edits in replacements.items():
        target = snapshot / relative
        before = target.read_text()
        after = before
        for old, new in edits:
            assert old in after, (relative, old)
            after = after.replace(old, new)
        target.write_text(after)
        patch.extend(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + relative,
                tofile="b/" + relative,
            )
        )
    (
        repository
        / (
            "artifacts/rotary_pendulum/experiment-results/"
            "forty-millisecond-decision-comparison/records"
        )
        / "decision40.patch"
    ).write_text("".join(patch))
    (snapshot / "decision40.patch").write_text("".join(patch))
    settings = {
        "episodes_per_stratum": 64,
        "seeds": [20261008, 20261009, 20261010],
        "decisions": 500,
        "chunk_decisions": 25,
        "cases": [
            {
                "name": f"d40_t{torque:g}_a{arm}",
                "torque_multiplier": torque,
                "arm_limit_deg": float(arm),
                "recovery_steps": 20,
                "work_weight": 1.0,
            }
            for torque in (0.5, 1.0, 2.0)
            for arm in (90, 180)
        ],
    }
    (
        repository
        / (
            "artifacts/rotary_pendulum/experiment-results/"
            "forty-millisecond-decision-comparison/records"
        )
        / "confirmation.json"
    ).write_text(json.dumps(settings, indent=2) + "\n")
    settings["episodes_per_stratum"], settings["seeds"] = 16, [20261011]
    (
        repository
        / (
            "artifacts/rotary_pendulum/experiment-results/"
            "forty-millisecond-decision-comparison/records"
        )
        / "initial_parameter_screening.json"
    ).write_text(json.dumps(settings, indent=2) + "\n")
    os.environ["JAX_PLATFORMS"], os.environ["CUDA_VISIBLE_DEVICES"] = "cpu", ""
    os.environ["JAX_ENABLE_X64"] = "true"
    sys.path.insert(0, str(snapshot / "src"))
    import jax
    import jax.numpy as jnp

    from rotary_pendulum.environment.jax_dynamics import PHYSICS_DT_S, rk4_step
    from rotary_pendulum.environment.jax_environment import GOAL, HOLD_PHYSICS_STEPS, reset
    from rotary_pendulum.heuristic.decoder import decode
    from rotary_pendulum.heuristic.energy import encode, policy
    from rotary_pendulum.heuristic.evaluation import evaluate

    assert PHYSICS_DT_S == 0.02 and HOLD_PHYSICS_STEPS == 2 and GOAL.hold_steps == 5
    assert np.isclose(GOAL.beta_tolerance_rad, np.deg2rad(15))
    state = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(0), 3), jnp.zeros(3, jnp.int32))
    state = state._replace(x=jnp.array([[0, jnp.pi, 0, 0], [0, jnp.pi, 0.3, 0.4], [0, 0, 0, 0]]))
    print("Checking 40 ms actions and goal hold across decision boundaries", flush=True)
    final, trace = jax.jit(lambda s: evaluate(s, jnp.array([0.04, 1.0, 1.0]), 20, 3, "zero"))(state)
    np.testing.assert_allclose(trace["time_s"][:, 0], [0.04, 0.08, 0.1])
    np.testing.assert_array_equal(trace["goal_count"][:, 0], [2, 4, 5])
    np.testing.assert_array_equal(trace["success"][:, 0], [False, False, True])
    assert final.success[1] and (abs(final.x[1, 2]) > 0.15 or abs(final.x[1, 3]) > 0.20)
    assert not final.success[2]
    torque, diagnostic = jax.jit(
        lambda x: decode(x, policy(encode(x), 0.04, 1.0), jnp.array(1.0), 20)
    )(state.x)
    replay = rk4_step(rk4_step(state.x, torque), torque)
    np.testing.assert_allclose(
        torque * (replay[:, 0] - state.x[:, 0]), diagnostic["predicted_work_j"], atol=1e-12
    )
    final, trace = jax.jit(lambda s: evaluate(s, jnp.array([0.04, 1.0, 1.0]), 20, 500, "zero"))(
        state
    )
    assert final.timeout[2] and np.isclose(trace["time_s"][-1, 2], 20.0)
    assert trace["physics_x"].shape[-2] == 2
    result = {
        "physics_step_s": 0.02,
        "decision_step_s": 0.04,
        "goal_hold_s": 0.1,
        "recovery_horizon_s": 0.4,
        "deadline_s": 20,
        "goal_crosses_decision_boundaries": True,
        "partial_terminal_action_verified": True,
        "velocity_not_required": True,
        "predicted_work_interval_verified": True,
        "timeout_verified": True,
    }
    (
        repository
        / (
            "artifacts/rotary_pendulum/experiment-results/"
            "forty-millisecond-decision-comparison/records"
        )
        / "goal_clock_checks.json"
    ).write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS {result}", flush=True)


if __name__ == "__main__":
    main()
