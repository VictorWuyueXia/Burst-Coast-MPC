"""Typed containers that make the MPC execution trace explicit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SplitCandidate:
    """One fixed burst-coast partition generated from a configured split ratio."""

    lambda_value: float
    total_steps: int
    burst_steps: int
    coast_steps: int


@dataclass(frozen=True)
class CandidateSolution:
    """Result of solving one fixed-dimension split-ratio nonlinear program."""

    candidate: SplitCandidate
    objective_value: float
    burst_inputs_nm: NDArray[np.float64]
    predicted_states: NDArray[np.float64]
    predicted_inputs_nm: NDArray[np.float64]
    solve_time_s: float
    message: str


@dataclass(frozen=True)
class SelectedPlan:
    """Executable burst-coast plan selected from all split candidates."""

    plan_id: str
    candidate: SplitCandidate
    objective_value: float
    burst_inputs_nm: NDArray[np.float64]
    predicted_states: NDArray[np.float64]
    predicted_inputs_nm: NDArray[np.float64]
    solve_time_s: float
    message: str
