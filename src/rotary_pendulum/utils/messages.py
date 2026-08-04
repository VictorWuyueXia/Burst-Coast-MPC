"""Typed runtime messages for rotary-pendulum simulation and monitoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ActionPlan:
    """One constant-torque Monte Carlo plan executed until the next replan."""

    plan_id: str
    replan_index: int
    hbar: float
    bbar: float
    horizon_steps: int
    torques_nm: NDArray[np.float64]


@dataclass(frozen=True)
class StateObservation:
    """One complete physical state and energy observation."""

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
    solve_time_s: float
    replan_flag: bool
