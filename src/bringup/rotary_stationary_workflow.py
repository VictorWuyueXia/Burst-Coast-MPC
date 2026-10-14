"""Stationary discounted TD3 experiment with explicit continuing-task checkpoints."""

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD
from rotary_pendulum.RL.jax_demonstrations import pretrain
from rotary_pendulum.RL.jax_stationary_evaluation import evaluate, write_evaluation
from rotary_pendulum.RL.jax_stationary_task import collect, targeted_resets
from rotary_pendulum.RL.jax_td3 import Actor, initialize, update


def train(settings: Mapping[str, Any]) -> None:
    """Fit one declared stationary formulation, preserving pretrained and online evidence."""
    if settings["contract_id"] != "rotary-stationary-td3-v3":
        raise ValueError("Stationary training requires explicit v3 contract")
    if settings["action_mode"] != "direct_unfiltered" or not 0 < settings["discount"] < 1:
        raise ValueError("Use raw actor torque and a strictly discounted continuing objective")
    count, capacity = settings["parallel_environments"], settings["replay_capacity"]
    if count % 4 or settings["total_transitions"] % count or capacity < count:
        raise ValueError("Use complete four-stratum batches and replay larger than one batch")
    if (
        settings["pretrain_steps"] % 100
        or settings["updates_per_collection"] < 1
        or not 0 <= settings["demonstration_fraction"] <= 1
        or settings["cloning_weight"] < 0
        or settings["deadline_s"] <= 10
    ):
        raise ValueError("Invalid stationary training schedule")
    root = Path(settings["run_dir"])
    machine = root / "machine-scannables"
    machine.mkdir(parents=True, exist_ok=False)
    (machine / "settings.json").write_text(json.dumps(dict(settings), indent=2) + "\n")
    source = Path(settings["demonstration_path"])
    manifest = json.loads(source.with_name("manifest.json").read_text())
    expected = {
        key: settings[key]
        for key in [
            "contract_id",
            "discount",
            "torque_limit_nm",
            "torque_cost_per_s",
            "arm_cost_per_s",
            "energy_cost_per_s",
            "upright_reward_per_s",
            "upright_widths",
        ]
    }
    expected.update(arm_limit_rad=float(ARM_LIMIT_RAD), integration_dt_s=0.002)
    if not expected.items() <= manifest["contract"].items():
        raise ValueError("Demonstrations do not match stationary physics and reward contract")
    data = np.load(source)
    demonstrations = {
        key: jnp.asarray(data[key])
        for key in ["observation", "next_observation", "action", "reward", "done", "return"]
    }
    data.close()
    data = np.load(source.with_name("validation.npz"))
    validation_observation, validation_action = (
        jnp.asarray(data["observation"]),
        jnp.asarray(data["action"]),
    )
    data.close()
    keys = jax.random.split(jax.random.PRNGKey(settings["seed"]), 3)
    learner = initialize(keys[0], settings)
    state, _ = targeted_resets(keys[1], count)
    rollout = dict(state=state, key=keys[2])
    replay = dict(
        observation=jnp.zeros((capacity, 5)),
        next_observation=jnp.zeros((capacity, 5)),
        action=jnp.zeros(capacity),
        reward=jnp.zeros(capacity),
        done=jnp.zeros(capacity, jnp.bool_),
        position=jnp.array(0),
        size=jnp.array(0),
    )
    initial, _ = targeted_resets(
        jax.random.PRNGKey(settings["validation_seed"]), 4 * settings["episodes_per_stratum"]
    )
    pretrain_jit = jax.jit(
        lambda model: jax.lax.scan(
            lambda carry, _: pretrain(carry, demonstrations, settings), model, None, length=100
        )
    )
    validation_mse = jax.jit(
        lambda actor: jnp.mean(
            (
                cast(
                    jax.Array,
                    Actor(tuple(settings["hidden_widths"])).apply(actor, validation_observation),
                )
                - validation_action
            )
            ** 2
        )
    )
    evaluate_jit = jax.jit(lambda actor: evaluate(initial, actor, settings))
    collect_jit = jax.jit(lambda model, run, memory: collect(model, run, memory, settings))
    update_jit = jax.jit(
        lambda model, memory: jax.lax.scan(
            lambda carry, _: update(carry, memory, settings, demonstrations),
            model,
            None,
            length=settings["updates_per_collection"],
        )
    )
    log = (machine / "pretrain_history.jsonl").open("w")
    print("Stationary BC: compiling five-feature actor/critic", flush=True)
    for iteration in range(100, settings["pretrain_steps"] + 1, 100):
        learner, losses = pretrain_jit(learner)
        metrics = {key: float(value.mean()) for key, value in losses.items()}
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite stationary pretraining")
        if iteration % 1000 == 0:
            metrics["validation_mse"] = float(validation_mse(learner["actor"]))
            print(f"BC {iteration}: {metrics}", flush=True)
        log.write(json.dumps(dict(iteration=iteration, **metrics)) + "\n")
        log.flush()
    log.close()
    (machine / "pretrained.msgpack").write_bytes(flax.serialization.to_bytes(learner))
    print("Stationary BC: evaluating full-duration continued balance", flush=True)
    score, summary = write_evaluation(
        root / "pretrained", jax.device_get(evaluate_jit(learner["actor"])), settings
    )
    best = score
    (machine / "best.msgpack").write_bytes(flax.serialization.to_bytes(learner))
    (machine / "best.json").write_text(
        json.dumps(dict(transitions=0, score=score, summary=summary), indent=2)
    )
    print(f"Pretrained score={score}, strata={summary}", flush=True)
    started = last_progress = time.monotonic()
    log = (machine / "history.jsonl").open("w")
    for transitions in range(count, settings["total_transitions"] + 1, count):
        rollout, replay, collected = collect_jit(learner, rollout, replay)
        metrics = {key: float(value) for key, value in collected.items()}
        if transitions >= settings["warmup_transitions"]:
            learner, updated = update_jit(learner, replay)
            metrics.update({key: float(value.mean()) for key, value in updated.items()})
        if not all(np.isfinite(value) for value in metrics.values()):
            (machine / "numerical_failure.json").write_text(json.dumps(metrics, indent=2))
            raise FloatingPointError(f"Nonfinite stationary learning at {transitions}")
        log.write(json.dumps(dict(transitions=transitions, **metrics)) + "\n")
        if time.monotonic() - last_progress >= 20:
            log.flush()
            print(
                f"TD3 {transitions}/{settings['total_transitions']}; "
                f"elapsed={time.monotonic() - started:.1f}s; {metrics}",
                flush=True,
            )
            last_progress = time.monotonic()
        if (
            transitions % settings["evaluation_every_transitions"] == 0
            or transitions == settings["total_transitions"]
        ):
            print(f"Continuing validation at {transitions}", flush=True)
            score, summary = write_evaluation(
                root / f"validation-{transitions}",
                jax.device_get(evaluate_jit(learner["actor"])),
                settings,
            )
            encoded = flax.serialization.to_bytes(jax.device_get(learner))
            (machine / "latest.msgpack").write_bytes(encoded)
            (machine / f"evaluation-{transitions}.json").write_text(json.dumps(summary, indent=2))
            if score > best:
                best = score
                (machine / "best.msgpack").write_bytes(encoded)
                (machine / "best.json").write_text(
                    json.dumps(
                        dict(transitions=transitions, score=score, summary=summary), indent=2
                    )
                )
            print(f"Validation {transitions}: score={score}; all={summary['all']}", flush=True)
    log.close()
    indices = np.linspace(0, int(replay["size"]) - 1, min(8192, int(replay["size"])), dtype=int)
    np.savez_compressed(
        machine / "replay_sample.npz",
        allow_pickle=False,
        **{
            key: np.asarray(value[indices])
            for key, value in replay.items()
            if key not in ("position", "size")
        },
    )
    print(f"DONE stationary TD3: {root}; best={best}", flush=True)
