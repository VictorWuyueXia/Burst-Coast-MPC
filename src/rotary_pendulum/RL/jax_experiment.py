"""Validate a resolved dense-reward experiment before allocating or writing a run."""

import math
from collections.abc import Mapping
from typing import Any


def validate_experiment(experiment: Mapping[str, Any]) -> int:
    """Check reward and collection contracts and return the exact update count."""

    obsolete = {
        "capture_weight",
        "success_reward",
        "arm_failure_cost",
        "timeout_cost",
        "on_cost_per_s",
    }
    if obsolete.intersection(experiment) or experiment["contract_id"] != "rotary-q-prior-v3":
        raise ValueError("Dense reward requires a fresh v3 experiment without legacy reward fields")
    rates = [
        float(experiment[name])
        for name in (
            "energy_cost_per_s",
            "torque_cost_per_s",
            "time_cost_per_s",
            "arm_cost_per_s",
            "upright_reward_per_s",
        )
    ]
    widths = experiment["upright_widths"]
    if (
        not all(math.isfinite(value) and value >= 0 for value in rates)
        or rates[2] <= rates[4]
        or len(widths) != 3
        or not all(math.isfinite(value) and value > 0 for value in widths)
        or int(experiment["reward_revision"]) < 1
    ):
        raise ValueError("Reward rates/widths must be finite; time rate must exceed upright rate")
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
    return int(updates_per_collection)
