"""Validate the explicit finite-horizon TD3 training contract."""

from collections.abc import Mapping
from typing import Any


def validate_settings(settings: Mapping[str, Any]) -> None:
    """Reject incompatible task clocks, rewards, model sizes and replay budgets."""
    if settings["contract_id"] not in ("rotary-residual-td3-v1", "rotary-demonstration-td3-v2"):
        raise ValueError("Unsupported residual training contract")
    if settings["deadline_s"] <= 0 or settings["evaluation_max_s"] != settings["deadline_s"]:
        raise ValueError("Training and evaluation deadlines must match and be positive")
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
        or settings["target_noise"] < 0
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
    if settings["goal_mode"] not in ("strict", "angle"):
        raise ValueError("Declare strict or angle capture")
    if settings["action_mode"] not in (
        "residual",
        "direct",
        "residual_unfiltered",
        "direct_unfiltered",
    ) or settings["reset_mode"] not in ("mixed", "local"):
        raise ValueError("Unsupported action or reset mode")
    if settings["updates_per_collection"] < 1 or min(settings["hidden_widths"]) < 1:
        raise ValueError("Update count and network widths must be positive")

    if not 0 < settings["torque_limit_nm"] <= 0.0204:
        raise ValueError("Policy torque cap must be within the physical motor limit")
    if not 0 < settings["hold_duration_s"] < settings["deadline_s"]:
        raise ValueError("Hold duration must fit within the episode")
