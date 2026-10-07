"""Architecture, Double DQN, evaluation, artifact, and import checks for the Q prior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rotary_pendulum.environment.jax_environment import EnvState
from rotary_pendulum.RL.jax_artifacts import write_artifacts
from rotary_pendulum.RL.jax_evaluation import evaluate
from rotary_pendulum.RL.jax_q import QNetwork, update
from rotary_pendulum.RL.jax_task import observe

SETTINGS = {
    "hidden_widths": (128, 128),
    "activation_name": "tanh",
    "loss_name": "huber",
    "huber_delta": 1.0,
    "learning_rate": 3e-4,
    "adam_betas": (0.9, 0.999),
    "adam_epsilon": 1e-8,
    "gradient_norm_limit": 10.0,
    "target_update": "hard",
    "target_copy_updates": 2,
    "target_polyak": 0.005,
    "minibatch_size": 1,
    "updates_per_collection": 1,
    "capture_weight": 1.0,
    "success_reward": 5.0,
    "arm_failure_cost": 5.0,
    "timeout_cost": 2.0,
    "on_cost_per_s": 0.05,
    "time_cost_per_s": 0.005,
}


def test_network_parameter_budget_shapes_seed_replay_and_activations() -> None:
    inputs = jnp.arange(42, dtype=jnp.float32).reshape((2, 3, 7)) / 20.0
    for activation in ("tanh", "silu", "relu"):
        model = QNetwork((128, 128), activation)
        first = model.init(jax.random.key(11), inputs)
        second = model.init(jax.random.key(11), inputs)
        leaves = jax.tree.leaves(first["params"])
        assert len(leaves) == 6
        assert sum(leaf.size for leaf in leaves) == 18_181
        assert all(leaf.dtype == jnp.float32 for leaf in leaves)
        np.testing.assert_array_equal(model.apply(first, inputs), model.apply(second, inputs))
        assert model.apply(first, inputs).shape == (2, 3, 5)
        assert model.apply(first, inputs[0, 0]).shape == (5,)


def test_double_dqn_target_gradient_and_hard_copy_cadence() -> None:
    model = QNetwork((128, 128), "tanh")
    initialized = model.init(jax.random.key(4), jnp.zeros((1, 7), dtype=jnp.float32))
    online = jax.tree.map(jnp.zeros_like, initialized)
    target = flax.core.unfreeze(online)
    target["params"]["q_values"]["bias"] = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float32)
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(3e-4, b1=0.9, b2=0.999, eps=1e-8),
    )
    learner = {
        "params": online,
        "target_params": target,
        "opt_state": optimizer.init(online),
        "key": jax.random.key(5),
        "updates": jnp.array(0, dtype=jnp.int32),
    }
    replay = {
        "observation": jnp.zeros((1, 7), dtype=jnp.float32),
        "action": jnp.array([0], dtype=jnp.int32),
        "reward": jnp.array([1.0], dtype=jnp.float32),
        "next_observation": jnp.ones((1, 7), dtype=jnp.float32),
        "done": jnp.array([False]),
        "size": jnp.array(1, dtype=jnp.int32),
    }
    learner, metrics = jax.jit(lambda learner_: update(learner_, replay, SETTINGS))(learner)
    np.testing.assert_allclose(metrics["target_mean"], 3.0, atol=1e-6)
    np.testing.assert_allclose(metrics["loss"], 2.5, atol=1e-6)
    assert any(np.any(np.asarray(leaf) != 0.0) for leaf in jax.tree.leaves(learner["params"]))
    for actual, expected in zip(
        jax.tree.leaves(learner["target_params"]), jax.tree.leaves(target), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)

    learner, _ = jax.jit(lambda learner_: update(learner_, replay, SETTINGS))(learner)
    for actual, expected in zip(
        jax.tree.leaves(learner["target_params"]), jax.tree.leaves(learner["params"]), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)

    terminal_replay = {**replay, "reward": jnp.array([-2.0]), "done": jnp.array([True])}
    terminal_learner = {**learner, "updates": jnp.array(0, dtype=jnp.int32)}
    _, terminal_metrics = jax.jit(lambda learner_: update(learner_, terminal_replay, SETTINGS))(
        terminal_learner
    )
    np.testing.assert_allclose(terminal_metrics["target_mean"], -2.0, atol=1e-6)


def test_evaluation_masks_terminal_values_and_enumerates_all_planners() -> None:
    model = QNetwork((128, 128), "tanh")
    params = jax.tree.map(
        jnp.zeros_like,
        model.init(jax.random.key(8), jnp.zeros((1, 7), dtype=jnp.float32)),
    )
    learner = {"params": params}
    states = EnvState(
        x=jnp.zeros((1, 4), dtype=jnp.float32),
        physics_steps=jnp.array([995], dtype=jnp.int32),
        goal_count=jnp.zeros((1,), dtype=jnp.int32),
        success=jnp.zeros((1,), dtype=jnp.bool_),
        arm_violation=jnp.zeros((1,), dtype=jnp.bool_),
        timeout=jnp.zeros((1,), dtype=jnp.bool_),
    )
    for mode in ("greedy", "lookahead_zero", "lookahead_potential", "lookahead_q"):
        metrics, trajectories = jax.jit(
            lambda learner_, mode_=mode: evaluate(learner_, states, mode_, SETTINGS)
        )(learner)
        assert int(trajectories["action"][0, 0]) == 0
        assert float(metrics["timeout_rate"]) == 1.0
        assert np.isnan(float(metrics["success_powered_s"]))

    audit_metrics, audit = jax.jit(
        lambda learner_: evaluate(learner_, states, "value_audit", SETTINGS)
    )(learner)
    _, potential = observe(states, SETTINGS)
    expected_base = jnp.broadcast_to(potential.sum(axis=-1)[:, None], (1, 5))
    np.testing.assert_allclose(audit["predicted_base"], expected_base)
    assert audit["realized_base"].shape == (1, 5)
    assert all(np.isfinite(float(value)) for value in audit_metrics.values())

    validation_metrics, validation_trajectories = evaluate(
        learner,
        {name: states for name in ("downward", "moving", "near", "tight")},
        "validation_suite",
        SETTINGS,
    )
    assert "greedy_tight_success_rate" in validation_metrics
    assert "audit_rmse" not in validation_metrics
    assert validation_trajectories["state"].shape[1] == 3
    assert validation_trajectories["tight_state"].shape == (200, 1, 4)
    assert validation_trajectories["near_action"].shape == (200, 1)


def test_artifact_checkpoint_round_trip_and_clean_import(tmp_path: Path) -> None:
    run_dir = tmp_path / "trial"
    run_dir.mkdir()
    model = QNetwork((128, 128), "tanh")
    params = model.init(jax.random.key(12), jnp.zeros((1, 7), dtype=jnp.float32))
    checkpoint = {
        "params": params,
        "target_params": params,
        "opt_state": (),
        "key": jax.random.key(13),
        "updates": jnp.array(0, dtype=jnp.int32),
    }
    experiment = {
        **SETTINGS,
        "network_id": "mlp-7-128-128-5-tanh-v2",
        "reward_revision": 1,
        "experiment_id": "unit-test",
        "seed": 12,
    }
    trajectories = {
        "state": jnp.zeros((2, 1, 4)),
        "action": jnp.zeros((2, 1), dtype=jnp.int32),
        "audit_predicted_base": jnp.zeros((1, 5)),
        "audit_realized_base": jnp.zeros((1, 5)),
    }
    for stratum in ("tight", "near"):
        trajectories.update(
            {
                f"{stratum}_initial_state": jnp.zeros((2, 4)),
                f"{stratum}_state": jnp.zeros((2, 2, 4)),
                f"{stratum}_action": jnp.zeros((2, 2), dtype=jnp.int32),
                f"{stratum}_valid": jnp.ones((2, 2), dtype=jnp.bool_),
                f"{stratum}_success": jnp.array([True, False]),
                f"{stratum}_arm_violation": jnp.array([False, False]),
                f"{stratum}_timeout": jnp.array([False, True]),
            }
        )
    metrics = {
        "success_rate": 0.0,
        "greedy_overall_success_rate": 0.0,
        "greedy_overall_arm_violation_rate": 0.0,
        "history": [
            {
                "transitions": 1,
                "loss": 1.0,
                "minimum_success": 0.0,
                "overall_success": 0.0,
                "tight_success": 0.5,
                "near_success": 0.0,
                "tight_arm_violation": 0.0,
                "near_arm_violation": 0.0,
            }
        ],
    }
    write_artifacts(
        run_dir,
        metrics,
        trajectories,
        {"selected": checkpoint, "latest": checkpoint},
        experiment,
    )
    encoded = (run_dir / "machine-scannables/checkpoints/selected.msgpack").read_bytes()
    checkpoint_template = {**checkpoint, "key": jax.random.key_data(checkpoint["key"])}
    restored = flax.serialization.from_bytes(checkpoint_template, encoded)
    sample = jnp.arange(7, dtype=jnp.float32)[None, :]
    np.testing.assert_allclose(
        model.apply(restored["params"], sample), model.apply(params, sample), atol=1e-6
    )
    assert (run_dir / "machine-scannables/metadata.json").is_file()
    assert (run_dir / "machine-scannables/metrics.json").is_file()
    assert (run_dir / "machine-scannables/history.csv").is_file()
    assert (run_dir / "machine-scannables/evaluation.csv").is_file()
    assert (run_dir / "machine-scannables/initial_states.npz").is_file()
    assert (run_dir / "machine-scannables/trajectories.npz").is_file()
    assert (run_dir / "machine-scannables/value_audit.npz").is_file()
    assert (run_dir / "machine-scannables/checkpoints/latest.msgpack").is_file()
    assert (run_dir / "human-readables/q_calibration.png").is_file()
    assert (run_dir / "human-readables/validation_trajectories.png").is_file()
    metadata = json.loads((run_dir / "machine-scannables/metadata.json").read_text())
    assert metadata["contract_id"] == "rotary-q-prior-v2"
    assert metadata["action_order"] == [
        "off",
        "negative_pump",
        "positive_pump",
        "negative_fine",
        "positive_fine",
    ]

    code = (
        "import sys; import rotary_pendulum.RL.jax_q; "
        "import bringup.rotary_q_workflow; "
        "assert 'torch' not in sys.modules and 'pytorch_lightning' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
