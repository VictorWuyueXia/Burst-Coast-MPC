"""Typed runtime messages for rotary-pendulum MPC execution and recording."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CandidateRecord:
    """Record one split-ratio NLP result at a rotary MPC decision epoch."""

    split_ratio: float
    horizon_steps: int
    burst_steps: int
    coast_steps: int
    objective_value: float
    solve_time_s: float
    solver_status: str
    selected: bool


@dataclass(frozen=True)
class ActionPlan:
    """Carry the selected burst-coast sequence through its complete rollout."""

    plan_id: str
    replan_index: int
    hbar: float
    bbar: float
    horizon_steps: int
    burst_steps: int
    coast_steps: int
    objective_value: float
    solve_time_s: float
    solver_status: str
    torques_nm: NDArray[np.float64]
    predicted_states: NDArray[np.float64]
    candidates: tuple[CandidateRecord, ...]


@dataclass(frozen=True)
class StateObservation:
    """One complete physical state and pendulum-relative swing-energy observation."""

    t_index: int
    t_sec: float
    theta_rad: float
    alpha_rad: float
    omega_rad_s: float
    nu_rad_s: float
    kinetic_energy_j: float
    potential_energy_j: float
    energy_j: float
    energy_error_j: float
    normalized_energy_error: float
    beta_rad: float
    goal_reached: bool


@dataclass(frozen=True)
class StepRecord(StateObservation):
    """Add applied action, plan, and replanning diagnostics to an observation."""

    u_commanded_nm: float
    u_applied_nm: float
    plan_id: str
    replan_index: int
    hbar: float
    bbar: float
    horizon_steps: int
    burst_steps: int
    coast_steps: int
    objective_value: float
    solver_status: str
    solve_time_s: float
    replan_flag: bool


@dataclass(frozen=True)
class EpisodeSummary:
    """Summarize one completed physical and MPC episode for artifacts and CLI output."""

    status: str
    steps: int
    replans: int
    simulated_time_s: float
    natural_period_s: float
    final_state: tuple[float, float, float, float]
    final_energy_error_j: float
    final_normalized_energy_error: float
    goal_reached: bool
