"""Typed configuration schema."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator

from wsmpc.utils.time import realtime

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
STANDARD_PACKAGE = "standard"


def load_config(package_name: str) -> RootConfig:
    """Load one complete config package and validate every required field."""

    # Register the YAML-only realtime helper at the exact point where YAML is resolved.
    OmegaConf.register_new_resolver("realtime", realtime, replace=True)
    package_path = CONFIG_ROOT / package_name / "config.yaml"
    if not package_path.exists():
        msg = f"Config package not found: {package_path}"
        raise FileNotFoundError(msg)
    package_cfg = OmegaConf.load(package_path)
    resolved = OmegaConf.to_container(package_cfg, resolve=True)
    return RootConfig.model_validate(resolved)


class ConfigBase(BaseModel):
    """Base model that accepts human-readable YAML aliases."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", protected_namespaces=())


class ArtifactConfig(ConfigBase):
    """Artifact recording controls for local experiment runs."""

    root_dir: str = Field(alias="root-dir")
    alias: str | None
    enabled: bool


class InitialStateConfig(ConfigBase):
    """Initial physical state for the pendulum environment."""

    theta_rad: float = Field(alias="theta-rad")
    omega_rad_s: float = Field(alias="omega-rad-s")


class ExperimentConfig(ConfigBase):
    """Top-level experiment controls owned by the Coordinator."""

    run_id: str = Field(alias="run-id")
    global_seed: int = Field(alias="global-seed")
    episode_id: int = Field(alias="episode-id")
    max_steps: int = Field(alias="max-steps")
    stop_on_goal: bool = Field(alias="stop-on-goal")
    initial_state: InitialStateConfig = Field(alias="initial-state")

    @field_validator("max_steps")
    @classmethod
    def _max_steps_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            msg = "max-steps must be positive"
            raise ValueError(msg)
        return value


class CoordinatorConfig(ConfigBase):
    """Coordinator timing and synchronous execution policy."""

    node_id: str = Field(alias="node-id")
    mode: Literal["synchronous"]
    decision_interval_steps: int = Field(alias="decision-interval-steps")
    debug_log_every_n_steps: int = Field(alias="debug-log-every-n-steps")

    @field_validator("decision_interval_steps", "debug_log_every_n_steps")
    @classmethod
    def _step_counts_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            msg = "step counts must be positive"
            raise ValueError(msg)
        return value


class SimulationConfig(ConfigBase):
    """Simulation clock controls kept separate from wall-clock pacing."""

    timestep_s: float = Field(alias="timestep-s")
    pace_s: float = Field(alias="pace-s")
    max_rollout_steps: int = Field(alias="max-rollout-steps")

    @field_validator("timestep_s")
    @classmethod
    def _timestep_must_be_positive(cls, value: float) -> float:
        if value <= 0.0:
            msg = "timestep-s must be positive"
            raise ValueError(msg)
        return value

    @field_validator("pace_s")
    @classmethod
    def _pace_must_be_nonnegative(cls, value: float) -> float:
        if value < 0.0:
            msg = "pace-s must be nonnegative"
            raise ValueError(msg)
        return value

    @field_validator("max_rollout_steps")
    @classmethod
    def _max_rollout_steps_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            msg = "max-rollout-steps must be positive"
            raise ValueError(msg)
        return value


class PendulumConfig(ConfigBase):
    """Physical constants and conservative state bounds for the pendulum."""

    mass_kg: float = Field(alias="mass-kg")
    gravity_m_s2: float = Field(alias="gravity-m-s2")
    length_m: float = Field(alias="length-m")
    damping_nms: float = Field(alias="damping-nms")
    torque_limit_nm: float = Field(alias="torque-limit-nm")
    theta_limit_abs_rad: float = Field(alias="theta-limit-abs-rad")
    omega_limit_abs_rad_s: float = Field(alias="omega-limit-abs-rad-s")

    @field_validator(
        "mass_kg",
        "gravity_m_s2",
        "length_m",
        "torque_limit_nm",
        "theta_limit_abs_rad",
        "omega_limit_abs_rad_s",
    )
    @classmethod
    def _positive_physical_values(cls, value: float) -> float:
        if value <= 0.0:
            msg = "physical constants and limits must be positive"
            raise ValueError(msg)
        return value


class GoalConfig(ConfigBase):
    """Goal-set thresholds used by the Environment and Coordinator."""

    angle_tolerance_rad: float = Field(alias="angle-tolerance-rad")
    omega_tolerance_rad_s: float = Field(alias="omega-tolerance-rad-s")
    hold_steps: int = Field(alias="hold-steps")

    @field_validator("angle_tolerance_rad", "omega_tolerance_rad_s")
    @classmethod
    def _goal_tolerances_must_be_positive(cls, value: float) -> float:
        if value <= 0.0:
            msg = "goal tolerances must be positive"
            raise ValueError(msg)
        return value

    @field_validator("hold_steps")
    @classmethod
    def _hold_steps_must_be_nonnegative(cls, value: int) -> int:
        if value < 0:
            msg = "hold-steps must be nonnegative"
            raise ValueError(msg)
        return value


class EnvironmentConfig(ConfigBase):
    """Environment identity and simulation model configuration."""

    node_id: str = Field(alias="node-id")
    simulation: SimulationConfig
    pendulum: PendulumConfig
    goal: GoalConfig


class MPCCostConfig(ConfigBase):
    """Energy-phase objective weights used by the nonlinear MPC problem."""

    epsilon_phi: float = Field(alias="epsilon-phi")
    sigma_energy: float = Field(alias="sigma-energy")
    q_energy: float = Field(alias="q-energy")
    q_phase: float = Field(alias="q-phase")
    q_local: float = Field(alias="q-local")
    q_terminal: float = Field(alias="q-terminal")
    rho_saturation: float = Field(alias="rho-saturation")
    rho_delta_u: float = Field(alias="rho-delta-u")
    q_phase_diag: list[float] = Field(alias="q-phase-diag")
    q_local_diag: list[float] = Field(alias="q-local-diag")

    @field_validator(
        "epsilon_phi",
        "sigma_energy",
        "q_energy",
        "q_terminal",
    )
    @classmethod
    def _positive_cost_values(cls, value: float) -> float:
        if value <= 0.0 or not math.isfinite(value):
            msg = "positive MPC cost values must be finite and greater than zero"
            raise ValueError(msg)
        return value

    @field_validator("q_phase", "q_local", "rho_saturation", "rho_delta_u")
    @classmethod
    def _nonnegative_cost_values(cls, value: float) -> float:
        if value < 0.0 or not math.isfinite(value):
            msg = "nonnegative MPC cost values must be finite"
            raise ValueError(msg)
        return value

    @field_validator("q_phase_diag", "q_local_diag")
    @classmethod
    def _diagonal_weights_must_be_positive_pairs(cls, value: list[float]) -> list[float]:
        if len(value) != 2 or any(entry <= 0.0 or not math.isfinite(entry) for entry in value):
            msg = "MPC diagonal weight lists must contain two positive finite values"
            raise ValueError(msg)
        return value


class MPCConfig(ConfigBase):
    """Configuration for the CasADi split-ratio MPC controller."""

    split_ratios: list[float] = Field(alias="split-ratios")
    cost: MPCCostConfig

    @field_validator("split_ratios")
    @classmethod
    def _split_ratios_must_be_valid(cls, value: list[float]) -> list[float]:
        if not value:
            msg = "split-ratios must contain at least one value"
            raise ValueError(msg)
        if any(ratio <= 0.0 or ratio > 1.0 or not math.isfinite(ratio) for ratio in value):
            msg = "split-ratios entries must be finite values in (0, 1]"
            raise ValueError(msg)
        return value


class RuntimeConfig(ConfigBase):
    """Resource limits for local runs."""

    node_id: str = Field(alias="node-id")
    max_worker_threads: int = Field(alias="max-worker-threads")
    blas_threads: int = Field(alias="blas-threads")
    torch_threads: int = Field(alias="torch-threads")
    cpu_affinity: list[int] = Field(alias="cpu-affinity")
    set_env: bool = Field(alias="set-env")

    @field_validator("max_worker_threads", "blas_threads", "torch_threads")
    @classmethod
    def _thread_counts_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            msg = "thread counts must be positive"
            raise ValueError(msg)
        return value

    @field_validator("cpu_affinity")
    @classmethod
    def _cpu_affinity_must_be_nonnegative(cls, value: list[int]) -> list[int]:
        if any(cpu < 0 for cpu in value):
            msg = "cpu-affinity entries must be nonnegative"
            raise ValueError(msg)
        return value


class RootConfig(ConfigBase):
    """Fully composed runtime configuration."""

    experiment: ExperimentConfig
    coordinator: CoordinatorConfig
    environment: EnvironmentConfig
    mpc: MPCConfig
    runtime: RuntimeConfig
    artifacts: ArtifactConfig
