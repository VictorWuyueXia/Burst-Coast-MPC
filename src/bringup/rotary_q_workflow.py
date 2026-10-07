"""Host-side curriculum, evaluation, and checkpoint workflow for rotary Double DQN."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax

from rotary_pendulum.environment.jax_environment import EnvState, reset
from rotary_pendulum.RL.jax_artifacts import write_artifacts
from rotary_pendulum.RL.jax_evaluation import evaluate
from rotary_pendulum.RL.jax_q import QNetwork, update
from rotary_pendulum.RL.jax_task import collect


def train(experiment: Mapping[str, Any]) -> Path:
    """Run one resolved three-stage Q-learning trial and save its complete evidence."""

    environment_count = int(experiment["parallel_environments"])
    collection_decisions = int(experiment["collection_decisions"])
    block_transitions = environment_count * collection_decisions
    replay_capacity = int(experiment["replay_capacity"])
    minibatch_size = int(experiment["minibatch_size"])
    updates_per_collection = float(
        experiment["sample_reuse_ratio"] * block_transitions / minibatch_size
    )
    if replay_capacity < block_transitions:
        raise ValueError("Replay capacity must contain one complete collection block")
    if not updates_per_collection.is_integer() or updates_per_collection <= 0:
        raise ValueError("Sample reuse must produce a positive integer update count")
    if any(int(budget) % block_transitions for budget in experiment["stage_transition_budgets"]):
        raise ValueError("Every stage budget must contain complete collection blocks")
    if int(experiment["evaluation_every_transitions"]) % block_transitions:
        raise ValueError("Evaluation cadence must contain complete collection blocks")
    if len(experiment["stage_transition_budgets"]) != 3:
        raise ValueError("The Q curriculum requires exactly three stage budgets")
    if int(experiment["lookahead_decisions"]) != 3:
        raise ValueError("The deployment contract requires a three-decision lookahead")
    run_dir = Path(experiment["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=False)

    root_key = jax.random.fold_in(
        jax.random.key(int(experiment["seed"])), int(experiment["family_id"])
    )
    model_key, behavior_key, reset_key, replay_key = (
        jax.random.fold_in(root_key, domain) for domain in range(4)
    )
    network = QNetwork(tuple(experiment["hidden_widths"]), experiment["activation_name"])
    params = network.init(model_key, jnp.zeros((1, 7), dtype=jnp.float32))
    adam_b1, adam_b2 = experiment["adam_betas"]
    optimizer = optax.chain(
        optax.clip_by_global_norm(experiment["gradient_norm_limit"]),
        optax.adam(experiment["learning_rate"], adam_b1, adam_b2, experiment["adam_epsilon"]),
    )
    learner: dict[str, Any] = {
        "params": params,
        "target_params": params,
        "opt_state": optimizer.init(params),
        "key": replay_key,
        "updates": jnp.array(0, dtype=jnp.int32),
    }
    initial_keys = jax.random.split(jax.random.fold_in(reset_key, 0), environment_count)
    initial_state = jax.vmap(reset)(initial_keys, jnp.full((environment_count,), 2))
    rollout: dict[str, Any] = {
        "env_state": initial_state,
        "key": jnp.stack((behavior_key, reset_key)),
        "episode_totals": jnp.zeros((environment_count, 9), dtype=jnp.float32),
        "completed_totals": jnp.zeros((12,), dtype=jnp.float32),
    }
    replay: dict[str, jax.Array] = {
        "observation": jnp.zeros((replay_capacity, 7), dtype=jnp.float32),
        "action": jnp.zeros((replay_capacity,), dtype=jnp.int32),
        "reward": jnp.zeros((replay_capacity,), dtype=jnp.float32),
        "next_observation": jnp.zeros((replay_capacity, 7), dtype=jnp.float32),
        "done": jnp.zeros((replay_capacity,), dtype=jnp.bool_),
        "position": jnp.array(0, dtype=jnp.int32),
        "size": jnp.array(0, dtype=jnp.int32),
    }

    evaluation_sets: dict[str, dict[str, EnvState]] = {}
    count = int(experiment["evaluation_episodes_per_stratum"])
    tight_count = count // 2
    for split, seed in (
        ("validation", int(experiment["validation_seed"])),
        ("final", int(experiment["final_seed"])),
    ):
        evaluation_key = jax.random.fold_in(jax.random.key(seed), 4)
        split_key, tight_key = jax.random.split(evaluation_key)
        all_keys = jax.random.split(split_key, 3 * count).reshape((3, count))
        strata_sets = {
            name: jax.vmap(reset)(all_keys[index], jnp.full((count,), index))
            for index, name in enumerate(("downward", "moving", "near"))
        }
        tight_extent = jnp.array([0.08, 0.12, 0.15, 0.30], dtype=jnp.float32)
        tight_x = jax.random.uniform(tight_key, (tight_count, 4), dtype=jnp.float32)
        tight_x = -tight_extent + 2.0 * tight_extent * tight_x
        tight_x = tight_x.at[:, 1].add(jnp.asarray(jnp.pi, dtype=jnp.float32))
        zero_i = jnp.zeros((tight_count,), dtype=jnp.int32)
        zero_b = jnp.zeros((tight_count,), dtype=jnp.bool_)
        strata_sets["tight"] = EnvState(tight_x, zero_i, zero_i, zero_b, zero_b, zero_b)
        evaluation_sets[split] = strata_sets

    history: list[dict[str, float | int]] = []
    total_transitions = 0
    best_score = (-math.inf, -math.inf, -math.inf, -math.inf, -math.inf)
    best_learner = learner
    best_eligible_score = best_score
    best_eligible_learner = learner
    eligible_found = False
    last_progress = time.monotonic()
    stages_passed = 0
    runtime_experiment = dict(experiment)
    runtime_experiment["updates_per_collection"] = int(updates_per_collection)
    update_compiled = jax.jit(lambda q, r: update(q, r, runtime_experiment))
    greedy_compiled = jax.jit(lambda q, s: evaluate(q, s, "greedy", runtime_experiment))
    for stage, stage_budget in enumerate(experiment["stage_transition_budgets"]):
        stage_transitions = 0
        consecutive_passes = 0
        next_evaluation = int(experiment["evaluation_every_transitions"])
        interval_collection_metrics: dict[str, Any] = {}
        collect_compiled = jax.jit(
            lambda learner_, rollout_, replay_, epsilon_, stage_=stage: collect(
                learner_,
                rollout_,
                replay_,
                {**runtime_experiment, "epsilon": epsilon_, "stage": stage_},
            )
        )
        while stage_transitions < int(stage_budget):
            if total_transitions < int(experiment["warmup_transitions"]):
                epsilon = 1.0
            else:
                fraction = max(
                    0.0, 1.0 - stage_transitions / float(experiment["epsilon_decay_transitions"])
                )
                epsilon = (
                    experiment["epsilon_end"]
                    + (experiment["epsilon_start"] - experiment["epsilon_end"]) * fraction
                )
            rollout, replay, collection_metrics = collect_compiled(
                learner, rollout, replay, jnp.asarray(epsilon, dtype=jnp.float32)
            )
            interval_collection_metrics = {
                key: interval_collection_metrics.get(key, 0.0) + value
                for key, value in collection_metrics.items()
            }
            total_transitions += block_transitions
            stage_transitions += block_transitions
            update_metrics: Mapping[str, Any] = {}
            if total_transitions >= int(experiment["warmup_transitions"]):
                learner, update_metrics = update_compiled(learner, replay)
            now = time.monotonic()
            if now - last_progress >= float(experiment["progress_interval_s"]):
                print(
                    f"rotary-q experiment={experiment['experiment_id']} stage={stage} "
                    f"transitions={total_transitions} updates={int(learner['updates'])} "
                    f"epsilon={epsilon:.4f}",
                    flush=True,
                )
                last_progress = now
            if stage_transitions < next_evaluation and stage_transitions < int(stage_budget):
                continue
            validation_metrics: dict[str, Mapping[str, Any]] = {}
            for name, states in evaluation_sets["validation"].items():
                metrics, _ = greedy_compiled(learner, states)
                validation_metrics[name] = metrics
            rates = {
                name: float(jax.device_get(metrics["success_rate"]))
                for name, metrics in validation_metrics.items()
            }
            violations = {
                name: float(jax.device_get(metrics["arm_violation_rate"]))
                for name, metrics in validation_metrics.items()
            }
            collection_values = {
                key: float(jax.device_get(value))
                for key, value in interval_collection_metrics.items()
            }
            update_values = {
                key: float(jax.device_get(value)) for key, value in update_metrics.items()
            }
            finite_values = (
                *rates.values(),
                *violations.values(),
                *collection_values.values(),
                *update_values.values(),
            )
            arrays_finite = jnp.all(jnp.isfinite(rollout["env_state"].x)) & jnp.all(
                jnp.isfinite(replay["reward"])
            )
            if not bool(jax.device_get(arrays_finite)) or not all(
                map(math.isfinite, finite_values)
            ):
                raise FloatingPointError(
                    "Nonfinite Q-training state, reward, value, loss, or gradient"
                )
            scored_names = ("tight", "near") if stage == 0 else ("downward", "moving", "near")
            score = (
                stage,
                min(rates[name] for name in scored_names),
                sum(rates[name] for name in scored_names) / len(scored_names),
                -max(violations.values()),
                -float(jax.device_get(validation_metrics["near"]["mean_powered_s"])),
            )
            eligible = rates["tight"] >= 0.90 and rates["near"] >= 0.80
            if score > best_score:
                best_score = score
                best_learner = learner
            if eligible and score > best_eligible_score:
                best_eligible_score = score
                best_eligible_learner = learner
            eligible_found = eligible_found or eligible
            if stage == 0:
                gate = (
                    rates["tight"] >= 0.90
                    and rates["near"] >= 0.80
                    and violations["tight"] == 0.0
                    and violations["near"] <= 0.01
                )
            elif stage == 1:
                gate = (
                    rates["tight"] >= 0.90
                    and rates["near"] >= 0.80
                    and rates["downward"] >= 0.30
                    and rates["moving"] >= 0.30
                    and violations["downward"] <= 0.10
                    and violations["moving"] <= 0.10
                )
            else:
                gate = eligible
            consecutive_passes = consecutive_passes + 1 if gate else 0
            row = {
                "stage": stage,
                "transitions": total_transitions,
                "updates": int(learner["updates"]),
                "epsilon": float(epsilon),
                "minimum_success": score[1],
                "overall_success": score[2],
                "maximum_violation": -score[3],
                "loss": update_values.get("loss", math.nan),
            }
            row.update({f"update_{key}": value for key, value in update_values.items()})
            row.update({f"collection_{key}": value for key, value in collection_values.items()})
            history.append(row)
            rollout = {**rollout, "completed_totals": jnp.zeros((12,), dtype=jnp.float32)}
            interval_collection_metrics = {}
            print(
                f"rotary-q evaluation stage={stage} transitions={total_transitions} "
                f"tight={rates['tight']:.3f} near={rates['near']:.3f} "
                f"moving={rates['moving']:.3f} downward={rates['downward']:.3f} "
                f"train_successes={collection_values['successes']:.0f} "
                f"train_arm_failures={collection_values['arm_failures']:.0f}",
                flush=True,
            )
            next_evaluation += int(experiment["evaluation_every_transitions"])
            if stage < 2 and consecutive_passes >= 2:
                stages_passed = stage + 1
                break
        if stage < 2 and consecutive_passes < 2:
            break
        if stage == 2:
            stages_passed = 3

    selected = best_eligible_learner if eligible_found else best_learner
    evaluation_split = "final" if stages_passed == 3 else "validation"
    evaluation_metrics, artifact_trajectories = evaluate(
        selected, evaluation_sets[evaluation_split], f"{evaluation_split}_suite", runtime_experiment
    )
    artifact_metrics = {
        **evaluation_metrics,
        "stages_passed": stages_passed,
        "total_transitions": total_transitions,
        "checkpoint_eligible": eligible_found,
        "evaluation_split": evaluation_split,
        "history": history,
    }
    artifact_trajectories.update(
        {
            f"initial_{evaluation_split}_{name}_{field}": value
            for name, state in evaluation_sets[evaluation_split].items()
            for field, value in zip(EnvState._fields, state, strict=True)
        }
    )
    checkpoints = {"latest": learner}
    checkpoints["selected" if eligible_found else "diagnostic_best"] = selected
    write_artifacts(run_dir, artifact_metrics, artifact_trajectories, checkpoints, experiment)
    print(f"rotary-q artifact_dir={run_dir}", flush=True)
    return run_dir
