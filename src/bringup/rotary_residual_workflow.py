"""Batched TD3 training with explicit task settings and diagnostic checkpoints."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, reset
from rotary_pendulum.RL.jax_demonstrations import pretrain
from rotary_pendulum.RL.jax_residual_contract import validate_settings
from rotary_pendulum.RL.jax_residual_evaluation import evaluate, validation_resets, write_validation
from rotary_pendulum.RL.jax_residual_task import collect
from rotary_pendulum.RL.jax_td3 import initialize, update


def train(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Run an explicitly requested bounded campaign with replay, checkpoints and validation."""

    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (*map(str, range(8)), ""):
        raise ValueError("Select one checked physical GPU or an empty CPU device list")
    validate_settings(settings)
    count, capacity = settings["parallel_environments"], settings["replay_capacity"]
    root = Path(settings["run_dir"])
    machine = root / "machine-scannables"
    machine.mkdir(parents=True, exist_ok=False)
    (machine / "settings.json").write_text(json.dumps(dict(settings), indent=2) + "\n")
    learner_key, reset_key, rollout_key = jax.random.split(jax.random.PRNGKey(settings["seed"]), 3)
    learner = initialize(learner_key, settings)
    demonstrations = None
    if settings["contract_id"] == "rotary-demonstration-td3-v2":
        if (
            settings["action_mode"] not in ("direct_unfiltered", "direct")
            or settings["goal_mode"] != "angle"
        ):
            raise ValueError("Teacher actions require the angle task and direct torque")
        if not 0 <= settings["demonstration_fraction"] <= 1 or settings["cloning_weight"] < 0:
            raise ValueError("Invalid demonstration fraction or cloning weight")
        if settings["pretrain_steps"] < 0 or settings["pretrain_steps"] % 100:
            raise ValueError("Pretraining must use complete 100-update progress chunks")
        manifest = json.loads(
            Path(settings["demonstration_path"]).with_name("manifest.json").read_text()
        )
        expected = dict(
            torque_limit_nm=settings["torque_limit_nm"],
            arm_limit_rad=float(ARM_LIMIT_RAD),
            integration_dt_s=0.002,
            hold_duration_s=settings["hold_duration_s"],
            arm_cost_per_s=settings["arm_cost_per_s"],
        )
        if not expected.items() <= manifest["goal"].items():
            raise ValueError(
                "Demonstrations must match motor, arm range/cost, integration and hold"
            )
        data = np.load(settings["demonstration_path"], allow_pickle=False)
        demonstrations = {name: jnp.asarray(data[name]) for name in data.files}
        data.close()
    rollout = {
        "state": jax.vmap(reset)(jax.random.split(reset_key, count), jnp.arange(count) % 3),
        "key": rollout_key,
    }
    if settings["reset_mode"] == "local":
        widths = jnp.where(
            (jnp.arange(count) % 2 == 0)[:, None],
            jnp.array([0.08, 0.12, 0.15, 0.30]),
            jnp.array([0.25, 0.25, 1.0, 1.0]),
        )
        local = jax.random.uniform(reset_key, (count, 4), minval=-1.0, maxval=1.0) * widths
        local = local.at[:, 1].set((local[:, 1] + 2 * jnp.pi) % (2 * jnp.pi) - jnp.pi)
        rollout["state"] = rollout["state"]._replace(x=local)
    replay = {
        "observation": jnp.zeros((capacity, 8)),
        "next_observation": jnp.zeros((capacity, 8)),
        "action": jnp.zeros(capacity),
        "reward": jnp.zeros(capacity),
        "done": jnp.zeros(capacity, jnp.bool_),
        "position": jnp.array(0, jnp.int32),
        "size": jnp.array(0, jnp.int32),
    }
    initial, deadlines, labels = validation_resets(
        settings["validation_seed"],
        settings["episodes_per_stratum"],
        [settings["deadline_s"]],
    )
    collect_jit = jax.jit(lambda model, run, memory: collect(model, run, memory, settings))
    update_jit = jax.jit(
        lambda model, memory: jax.lax.scan(
            lambda carry, _: update(carry, memory, settings, demonstrations),
            model,
            None,
            length=settings["updates_per_collection"],
        )
    )
    evaluate_jit = jax.jit(lambda actor: evaluate(initial, deadlines, settings, "policy", actor))
    best_score = (-1.0, -1.0, -float("inf"))
    if demonstrations is not None:
        validation_path = Path(settings["demonstration_path"]).with_name(
            f"validation-{settings['deadline_s']}s.npz"
        )
        validation = np.load(validation_path, allow_pickle=False)
        validation_observation = jnp.asarray(validation["observation"])
        validation_action = jnp.asarray(validation["action"])
        validation.close()
        from rotary_pendulum.RL.jax_td3 import Actor

        supervised_history = (machine / "pretrain_history.jsonl").open("w")
        pretrain_jit = jax.jit(
            lambda model: jax.lax.scan(
                lambda carry, _: pretrain(carry, demonstrations, settings), model, None, length=100
            )
        )
        print(f"Compiling behavior cloning: {settings['pretrain_steps']} updates", flush=True)
        for step in range(100, settings["pretrain_steps"] + 1, 100):
            learner, losses = pretrain_jit(learner)
            measured = {k: float(v.mean()) for k, v in losses.items()}
            if step % 500 == 0 or step == settings["pretrain_steps"]:
                prediction = cast(
                    jax.Array,
                    Actor(tuple(settings["hidden_widths"])).apply(
                        learner["actor"], validation_observation
                    ),
                )
                measured["validation_mse"] = float(jnp.mean((prediction - validation_action) ** 2))
                print(f"BC {step}: {measured}", flush=True)
            if not all(np.isfinite(value) for value in measured.values()):
                raise FloatingPointError("Nonfinite demonstration pretraining")
            supervised_history.write(json.dumps({"step": step, **measured}) + "\n")
            supervised_history.flush()
        supervised_history.close()
        (machine / "pretrained.msgpack").write_bytes(flax.serialization.to_bytes(learner))
        print("Validating behavior-cloned policy before TD3", flush=True)
        rows = write_validation(
            root / "pretrained", jax.device_get(evaluate_jit(learner["actor"])), labels, settings
        )
        ordinary = [row for row in rows if row["stratum"] != "probe"]
        best_score = (
            float(np.mean([r["success"] and not r["arm_violation"] for r in ordinary])),
            float(np.mean([r["success"] for r in ordinary])),
            float(np.mean([r["return"] for r in ordinary])),
        )
        (machine / "best.msgpack").write_bytes(flax.serialization.to_bytes(learner))
        (machine / "best.json").write_text(
            json.dumps(
                {
                    "transitions": 0,
                    "bounded_success_rate": best_score[0],
                    "success_rate": best_score[1],
                    "mean_return": best_score[2],
                }
            )
        )
        print(f"Before TD3: hold={best_score[0]:.2%}", flush=True)
    baseline_jit = jax.jit(lambda: evaluate(initial, deadlines, settings, "heuristic"))
    print(f"TD3: {count} lanes, deadline={settings['deadline_s']}s; compiling baseline", flush=True)
    write_validation(root / "heuristic-baseline", jax.device_get(baseline_jit()), labels, settings)
    history = (machine / "history.jsonl").open("w")
    started = last_progress = time.monotonic()
    print("TD3: compiling collection and update kernels", flush=True)
    for transitions in range(count, settings["total_transitions"] + 1, count):
        rollout, replay, collected = collect_jit(learner, rollout, replay)
        metrics: dict[str, Any] = {
            "transitions": transitions,
            "successes": int(collected["successes"]),
            "completed": int(collected["completed"]),
            "filter_fraction": float(collected["filter_fraction"]),
            "reward_mean": float(collected["reward_mean"]),
            **{
                name: float(collected[name])
                for name in (
                    "saturation_fraction",
                    "controllable_fraction",
                    "replay_terminal_fraction",
                )
            },
        }
        if transitions >= settings["warmup_transitions"]:
            learner, updated = update_jit(learner, replay)
            metrics.update({key: float(jnp.mean(value)) for key, value in updated.items()})
        if not all(np.isfinite(value) for value in metrics.values()):
            (machine / "numerical_failure.json").write_text(json.dumps(metrics, indent=2))
            np.savez_compressed(
                machine / "failure_states.npz", **jax.device_get(rollout["state"]._asdict())
            )
            indices = (replay["position"] - count + jnp.arange(count)) % capacity
            np.savez_compressed(
                machine / "failure_transitions.npz",
                allow_pickle=False,
                **{
                    k: np.asarray(replay[k][indices])
                    for k in ("observation", "action", "next_observation", "reward", "done")
                },
            )
            raise FloatingPointError(f"Nonfinite TD3 metrics: {metrics}")
        history.write(json.dumps(metrics) + "\n")
        if time.monotonic() - last_progress >= 20.0:
            history.flush()
            print(
                f"TD3 transitions={transitions}/{settings['total_transitions']}, "
                f"updates={int(learner['updates'])}, elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
            last_progress = time.monotonic()
        if (
            transitions % settings["evaluation_every_transitions"] == 0
            or transitions == settings["total_transitions"]
        ):
            print(f"TD3 validation and plots at {transitions} transitions", flush=True)
            traces = jax.device_get(evaluate_jit(learner["actor"]))
            rows = write_validation(root / f"validation-{transitions}", traces, labels, settings)
            summary = {
                label: {
                    name: float(np.mean([row[name] for row in rows if row["stratum"] == label]))
                    for name in ("success", "return", "arm_violation", "duration_s")
                }
                for label in ("downward", "moving", "near", "tight")
            }
            (machine / f"evaluation-{transitions}.json").write_text(json.dumps(summary, indent=2))
            ordinary = [row for row in rows if row["stratum"] != "probe"]
            score = (
                float(np.mean([row["success"] and not row["arm_violation"] for row in ordinary])),
                float(np.mean([row["success"] for row in ordinary])),
                float(np.mean([row["return"] for row in ordinary])),
            )
            encoded = flax.serialization.to_bytes(jax.device_get(learner))
            (machine / "latest.msgpack").write_bytes(encoded)
            if score > best_score:
                best_score = score
                (machine / "best.msgpack").write_bytes(encoded)
                (machine / "best.json").write_text(
                    json.dumps(
                        {
                            "transitions": transitions,
                            "bounded_success_rate": score[0],
                            "success_rate": score[1],
                            "mean_return": score[2],
                        },
                        indent=2,
                    )
                    + "\n"
                )
            print(
                f"TD3 validation bounded_hold={score[0]:.2%}, "
                f"hold={score[1]:.2%}, return={score[2]:.3f}",
                flush=True,
            )
            print(f"TD3 strata={json.dumps(summary)}", flush=True)
    history.close()
    indices = np.linspace(0, int(replay["size"]) - 1, min(8192, int(replay["size"])), dtype=int)
    np.savez_compressed(
        machine / "replay_sample.npz",
        allow_pickle=False,
        **{
            name: np.asarray(value[indices])
            for name, value in replay.items()
            if name not in ("size", "position")
        },
    )
    return {
        "learner": learner,
        "best_score": best_score,
        "transitions": settings["total_transitions"],
    }
