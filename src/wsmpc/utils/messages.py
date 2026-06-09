"""Typed messages exchanged between runtime components."""

from __future__ import annotations

from dataclasses import dataclass

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
    u_nm: float = Field(alias="u-nm")
    source: str
    early_wake_flag: bool = Field(alias="early-wake-flag")
    plan_id: str | None = None


class StateObs(MessageBase):
    """Observation emitted by the Environment after reset or step."""

    run_id: str
    episode_id: int
    t_index: int
    t_sec: float
    theta_rad: float = Field(alias="theta-rad")
    omega_rad_s: float = Field(alias="omega-rad-s")
    energy_j: float = Field(alias="energy-j")
    energy_error_j: float = Field(alias="energy-error-j")
    wrapped_angle_error_rad: float = Field(alias="wrapped-angle-error-rad")
    constraint_margin: float
    goal_reached: bool


class StepRecord(MessageBase):
    """Dense per-step record for logging and recording."""

    run_id: str
    episode_id: int
    t_index: int
    t_sec: float
    theta_rad: float = Field(alias="theta-rad")
    omega_rad_s: float = Field(alias="omega-rad-s")
    energy_j: float = Field(alias="energy-j")
    energy_error_j: float = Field(alias="energy-error-j")
    u_commanded_nm: float = Field(alias="u-commanded-nm")
    u_applied_nm: float = Field(alias="u-applied-nm")
    mode: str
    plan_id: str | None = None
    constraint_margin: float
    goal_flag: bool
    early_wake_flag: bool
    step_compute_wall_s: float
    pace_sleep_s: float
    action_result: str


class RLStepRecord(MessageBase):
    """One replanning transition row for Monte Carlo RL training."""

    run_id: str = Field(alias="run-id")
    episode_id: int = Field(alias="episode-id")
    replan_index: int = Field(alias="replan-index")
    start_t_index: int = Field(alias="start-t-index")
    start_t_sec: float = Field(alias="start-t-sec")
    end_t_index: int = Field(alias="end-t-index")
    end_t_sec: float = Field(alias="end-t-sec")
    s_sin_theta: float = Field(alias="s-sin-theta")
    s_cos_theta: float = Field(alias="s-cos-theta")
    s_omega_rad_s: float = Field(alias="s-omega-rad-s")
    bbar: float
    hbar: float
    burst_steps: int = Field(alias="burst-steps")
    horizon_steps: int = Field(alias="horizon-steps")
    next_s_sin_theta: float = Field(alias="next-s-sin-theta")
    next_s_cos_theta: float = Field(alias="next-s-cos-theta")
    next_s_omega_rad_s: float = Field(alias="next-s-omega-rad-s")
    done: bool
    step_cost: float = Field(alias="step-cost")
    return_cost: float = Field(alias="return-cost")
    u_nm_json: str = Field(alias="u-nm-json")
    solve_time_s: float = Field(alias="solve-time-s")
    plan_id: str = Field(alias="plan-id")


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
    total_wall_time_s: float
    final_observation: StateObs | None = None


@dataclass(frozen=True)
class EpisodeResult:
    """In-memory episode result returned after one run."""

    summary: ExperimentSummary
    records: list[StepRecord]
