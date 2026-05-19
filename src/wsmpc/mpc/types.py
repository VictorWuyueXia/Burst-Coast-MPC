"""Typed containers exchanged within the MPC package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wsmpc.utils.config_schema import EnvironmentConfig, ExperimentConfig, MPCConfig, RuntimeConfig


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
    success: bool
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
    solver_success_count: int
    solver_failure_count: int
    message: str


@dataclass(frozen=True)
class CandidateSolveTask:
    """Serializable task payload for optional process-level candidate solving."""

    state: NDArray[np.float64]
    previous_input_nm: float
    candidate: SplitCandidate
    environment_config: EnvironmentConfig
    mpc_config: MPCConfig
    warm_start_nm: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class MPCControllerContext:
    """Runtime configuration needed by the closed-loop MPC action provider."""

    environment: EnvironmentConfig
    experiment: ExperimentConfig
    mpc: MPCConfig
    runtime: RuntimeConfig
