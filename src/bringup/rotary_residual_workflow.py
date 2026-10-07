"""Explicit future TD3 training workflow; importing this module never starts training."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_environment import reset
from rotary_pendulum.RL.jax_residual_evaluation import evaluate, validation_resets, write_validation
from rotary_pendulum.RL.jax_residual_task import collect
from rotary_pendulum.RL.jax_td3 import initialize, update


def train(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Run an explicitly requested bounded campaign with replay, checkpoints and validation."""

    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1", "2", "3", "0,1,2,3", ""):
        raise ValueError("Explicitly select GPUs 0–3 only, or an empty CPU device list")
    if settings["contract_id"] != "rotary-residual-td3-v1":
        raise ValueError("Unsupported residual training contract")
    if (
        settings["deadline_s"] not in (20, 40, 60)
        or settings["evaluation_max_s"] != settings["deadline_s"]
    ):
        raise ValueError("Training and evaluation deadlines must match and be 20, 40 or 60 s")
    count, capacity = settings["parallel_environments"], settings["replay_capacity"]
    if not (
        0 < count <= capacity
        and 0 < settings["minibatch_size"] <= settings["warmup_transitions"] <= capacity
    ):
        raise ValueError("Invalid environment, minibatch, warmup or replay sizes")
    if settings["total_transitions"] < count or settings["total_transitions"] % count:
        raise ValueError("Transition budget must be a positive number of parallel collections")
    if (
        settings["evaluation_every_transitions"] < count
        or settings["evaluation_every_transitions"] % count
    ):
        raise ValueError("Evaluation interval must align with parallel collections")
    if (
        settings["energy_time_s"] <= 0
        or settings["sensitivity_floor_per_s"] <= 0
        or settings["filter_steps"] < 5
        or settings["learning_rate"] <= 0
        or settings["exploration_noise"] < 0
    ):
        raise ValueError("Invalid controller or optimizer settings")
    if not (settings["time_cost_per_s"] > settings["upright_reward_per_s"] >= 0):
        raise ValueError(
            "Time cost must exceed upright credit to preserve early completion preference"
        )
    if (
        min(settings[key] for key in ("energy_cost_per_s", "torque_cost_per_s", "arm_cost_per_s"))
        < 0
    ):
        raise ValueError("Reward costs must be nonnegative")
    if min(settings["upright_widths"]) <= 0 or len(settings["upright_widths"]) != 3:
        raise ValueError("Three strictly positive upright widths are required")

    root = Path(settings["run_dir"])
    machine = root / "machine-scannables"
    machine.mkdir(parents=True, exist_ok=False)
    (machine / "settings.json").write_text(json.dumps(dict(settings), indent=2) + "\n")
    learner_key, reset_key, rollout_key = jax.random.split(jax.random.PRNGKey(settings["seed"]), 3)
    learner = initialize(learner_key, settings)
    rollout = {
        "state": jax.vmap(reset)(jax.random.split(reset_key, count), jnp.arange(count) % 3),
        "key": rollout_key,
    }
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
    update_jit = jax.jit(lambda model, memory: update(model, memory, settings))
    evaluate_jit = jax.jit(lambda actor: evaluate(initial, deadlines, settings, "policy", actor))
    baseline_jit = jax.jit(lambda: evaluate(initial, deadlines, settings, "heuristic"))
    print(f"TD3: {count} lanes, deadline={settings['deadline_s']}s; compiling baseline", flush=True)
    write_validation(root / "heuristic-baseline", jax.device_get(baseline_jit()), labels, settings)
    history = (machine / "history.jsonl").open("w")
    started = last_progress = time.monotonic()
    print("TD3: compiling collection and update kernels", flush=True)
    best_score = (-1.0, -float("inf"))
    for transitions in range(count, settings["total_transitions"] + 1, count):
        rollout, replay, collected = collect_jit(learner, rollout, replay)
        metrics: dict[str, Any] = {
            "transitions": transitions,
            "successes": int(collected["successes"]),
            "completed": int(collected["completed"]),
            "filter_fraction": float(collected["filter_fraction"]),
            "reward_mean": float(collected["reward_mean"]),
        }
        if transitions >= settings["warmup_transitions"]:
            learner, updated = update_jit(learner, replay)
            metrics.update({key: float(value) for key, value in updated.items()})
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite TD3 metrics; inspect history before continuing")
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
            ordinary = [row for row in rows if row["stratum"] != "probe"]
            score = (
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
                            "success_rate": score[0],
                            "mean_return": score[1],
                        },
                        indent=2,
                    )
                    + "\n"
                )
            print(f"TD3 validation hold={score[0]:.2%}, return={score[1]:.3f}", flush=True)
    history.close()
    return {
        "learner": learner,
        "best_score": best_score,
        "transitions": settings["total_transitions"],
    }
