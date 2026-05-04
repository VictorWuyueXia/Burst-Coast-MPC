"""Typed messages exchanged between runtime components."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MessageBase(BaseModel):
    """Common Pydantic settings for portable message serialization."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", protected_namespaces=())


class ActionCommand(MessageBase):
    """One action sent from the Coordinator to the Environment."""

    run_id: str
    episode_id: int
    t_index: int
    t_sec: float
    u: float
    source: str
    plan_id: str | None = None


class StateObs(MessageBase):
    """Observation emitted by the Environment after reset or step."""

    run_id: str
    episode_id: int
    t_index: int
    t_sec: float
    theta: float
    omega: float
    energy: float
    energy_error: float
    wrapped_angle_error: float
    constraint_margin: float
    goal_reached: bool


class StepRecord(MessageBase):
    """Dense per-step record for logging and recording."""

    run_id: str
    episode_id: int
    t_index: int
    t_sec: float
    theta: float
    omega: float
    energy: float
    energy_error: float
    u_commanded: float
    u_applied: float
    mode: str
    plan_id: str | None = None
    constraint_margin: float
    goal_flag: bool
    early_wake_flag: bool = False
    step_compute_wall_s: float = 0.0
    pace_sleep_s: float = 0.0
    action_result: str = "accepted"


class NodeStatus(MessageBase):
    """Lifecycle and health update emitted by a runtime component."""

    node_id: str = Field(alias="node-id")
    identity: str
    status: str
    action: str
    action_result: str
    t_index: int | None = None
    t_sec: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    updated_wall_time_s: float


class ExperimentSummary(MessageBase):
    """End-of-episode summary returned by the Coordinator."""

    run_id: str
    episode_id: int
    status: str
    total_steps: int
    final_t_index: int
    final_t_sec: float
    goal_reached: bool
    records_emitted: int
    solver_failure_count: int = 0
    total_wall_time_s: float
    final_observation: StateObs | None = None
