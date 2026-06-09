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
DATA_GENERATION_PACKAGE = "data-generation"


def load_config(package_name: str) -> RootConfig:
    """Load one complete config package and validate every required field."""

    resolved = _load_resolved_config(package_name)
    return RootConfig.model_validate(resolved)


def load_data_generation_config(
    package_name: str = DATA_GENERATION_PACKAGE,
) -> DataGenerationRootConfig:
    """Load the Monte Carlo data-generation config package."""

    resolved = _load_resolved_config(package_name)
    return DataGenerationRootConfig.model_validate(resolved)


def _load_resolved_config(package_name: str) -> dict:
    """Resolve one YAML config package before Pydantic validation."""

    # Register the YAML-only realtime helper at the exact point where YAML is resolved.
    OmegaConf.register_new_resolver("realtime", realtime, replace=True)
    package_path = CONFIG_ROOT / package_name / "config.yaml"
    if not package_path.exists():
        msg = f"Config package not found: {package_path}"
        raise FileNotFoundError(msg)
    return OmegaConf.to_container(OmegaConf.load(package_path), resolve=True)


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
    event_trigger: bool = Field(alias="event-trigger")
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


class MPCConfig(ConfigBase):
    """Configuration for the CasADi split-ratio MPC controller."""

    controller: Literal["IP-dynamics-naturalPeriod"]
    split_ratios: list[float] = Field(alias="split-ratios")

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
    cpu_affinity: list[int] = Field(alias="cpu-affinity")
    set_env: bool = Field(alias="set-env")

    @field_validator("max_worker_threads", "blas_threads")
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


class DataGenerationConfig(ConfigBase):
    """Monte Carlo data-generation controls for offline RL."""

    episodes: int
    seed: int | None
    visual_artifacts: bool = Field(alias="visual-artifacts")
    theta_rad_sample_range: list[float] = Field(alias="theta-rad-sample-range")
    omega_eq_scale_sample_range: list[float] = Field(alias="omega-eq-scale-sample-range")
    gamma: float
    bbar_min: float = Field(alias="bbar-min")
    bbar_max: float = Field(alias="bbar-max")
    hbar_min: float = Field(alias="hbar-min")
    hbar_max: float = Field(alias="hbar-max")
    time_weight: float = Field(alias="time-weight")
    action_weight: float = Field(alias="action-weight")
    compute_weight: float = Field(alias="compute-weight")
    fail_penalty: float = Field(alias="fail-penalty")

    @field_validator("episodes")
    @classmethod
    def _episodes_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            msg = "episodes must be positive"
            raise ValueError(msg)
        return value

    @field_validator("gamma", "bbar_min", "bbar_max", "hbar_min", "hbar_max")
    @classmethod
    def _normalized_values_must_be_unit_interval(cls, value: float) -> float:
        if value < 0.0 or value > 1.0 or not math.isfinite(value):
            msg = "normalized data-generation values must be finite values in [0, 1]"
            raise ValueError(msg)
        return value

    @field_validator("time_weight", "action_weight", "compute_weight", "fail_penalty")
    @classmethod
    def _cost_weights_must_be_nonnegative(cls, value: float) -> float:
        if value < 0.0 or not math.isfinite(value):
            msg = "cost weights must be finite nonnegative values"
            raise ValueError(msg)
        return value

    @field_validator("theta_rad_sample_range", "omega_eq_scale_sample_range")
    @classmethod
    def _sample_ranges_must_be_two_ordered_finite_values(
        cls,
        value: list[float],
    ) -> list[float]:
        if len(value) != 2:
            msg = "sample ranges must contain exactly two values"
            raise ValueError(msg)
        if any(not math.isfinite(bound) for bound in value):
            msg = "sample range bounds must be finite"
            raise ValueError(msg)
        if value[1] < value[0]:
            msg = "sample range upper bound must be greater than or equal to lower bound"
            raise ValueError(msg)
        return value

    @field_validator("bbar_max")
    @classmethod
    def _bbar_range_must_be_ordered(cls, value: float, info) -> float:
        if "bbar_min" in info.data and value < info.data["bbar_min"]:
            msg = "bbar-max must be greater than or equal to bbar-min"
            raise ValueError(msg)
        return value

    @field_validator("hbar_max")
    @classmethod
    def _hbar_range_must_be_ordered(cls, value: float, info) -> float:
        if "hbar_min" in info.data and value < info.data["hbar_min"]:
            msg = "hbar-max must be greater than or equal to hbar-min"
            raise ValueError(msg)
        return value


class DataGenerationRootConfig(RootConfig):
    """Complete runtime config plus required Monte Carlo generation controls."""

    data_generation: DataGenerationConfig = Field(alias="data-generation")
