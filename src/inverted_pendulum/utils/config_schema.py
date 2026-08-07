"""Strict typed configuration composed from task-domain YAML fragments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from inverted_pendulum.utils.time import realtime

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
ConfigMode = Literal["mpc-only", "intelligent", "online-training", "data-generation"]


def _sources(*names: str) -> tuple[Path, ...]:
    """Resolve declared package config fragments without cwd dependence."""

    return tuple(CONFIG_ROOT / name for name in names)


EPISODE_CONFIG_SOURCES = _sources(
    "physics-config.yaml",
    "mission-config.yaml",
    "runtime-config.yaml",
    "mpc-config.yaml",
    "artifacts-config.yaml",
)
MPC_ONLY_CONFIG_SOURCES = EPISODE_CONFIG_SOURCES
INTELLIGENT_CONFIG_SOURCES = EPISODE_CONFIG_SOURCES + _sources("rl-config.yaml")
ONLINE_TRAINING_CONFIG_SOURCES = INTELLIGENT_CONFIG_SOURCES
DATA_GENERATION_CONFIG_SOURCES = EPISODE_CONFIG_SOURCES + _sources("data-generation-config.yaml")
VISUALIZATION_CONFIG_SOURCE = CONFIG_ROOT / "visual-config.yaml"


def load_mpc_only_config(
    config_sources: tuple[Path, ...] = MPC_ONLY_CONFIG_SOURCES,
) -> RootConfig:
    """Compose and validate the physical MPC-only episode domains."""

    return RootConfig.model_validate(_compose_config(config_sources, "mpc-only"))


def load_intelligent_config(
    config_sources: tuple[Path, ...] = INTELLIGENT_CONFIG_SOURCES,
) -> RLRootConfig:
    """Compose and validate deterministic critic-deployment domains."""

    return RLRootConfig.model_validate(_compose_config(config_sources, "intelligent"))


def load_online_training_config(
    config_sources: tuple[Path, ...] = ONLINE_TRAINING_CONFIG_SOURCES,
) -> RLRootConfig:
    """Compose and validate exploratory online-training domains."""

    return RLRootConfig.model_validate(_compose_config(config_sources, "online-training"))


def load_data_generation_config(
    config_sources: tuple[Path, ...] = DATA_GENERATION_CONFIG_SOURCES,
) -> DataGenerationRootConfig:
    """Compose and validate Monte Carlo data-generation domains."""

    return DataGenerationRootConfig.model_validate(
        _compose_config(config_sources, "data-generation")
    )


def load_visualization_config(
    config_source: Path = VISUALIZATION_CONFIG_SOURCE,
) -> VisualizationConfig:
    """Load display controls independently from headless runtime domains."""

    resolved = _compose_config((config_source,))
    return VisualizationRootConfig.model_validate(resolved).visualization


def _compose_config(
    config_sources: tuple[Path, ...], mode: ConfigMode | None = None
) -> dict[str, object]:
    """Merge domains, apply one mode profile, and resolve interpolation once."""

    if not config_sources:
        raise ValueError("At least one config source is required")
    OmegaConf.register_new_resolver("realtime", realtime, replace=True)
    for source in config_sources:
        if not source.exists():
            raise FileNotFoundError(f"Config file not found: {source}")
    loaded_domains = tuple(OmegaConf.load(source) for source in config_sources)
    merged = cast(DictConfig, OmegaConf.merge(*loaded_domains))
    if mode is not None:
        profile = OmegaConf.select(merged, f"modes.{mode}")
        if profile is None and OmegaConf.select(merged, "modes") is not None:
            raise ValueError(f"Config mode profile not found: {mode}")
        if profile is not None:
            merged = cast(DictConfig, OmegaConf.merge(merged, profile))
    if "modes" in merged:
        del merged["modes"]
    resolved = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Composed config must resolve to a mapping")
    return cast(dict[str, object], resolved)


class ConfigBase(BaseModel):
    """Apply strict validation and readable YAML aliases to every domain."""

    model_config = ConfigDict(
        populate_by_name=True, extra="forbid", protected_namespaces=(), allow_inf_nan=False
    )


# Declare the compact experiment, artifact, and physical domain value objects.
class ArtifactConfig(ConfigBase):
    root_dir: str = Field(alias="root-dir")
    alias: str | None
    enabled: bool


class DataGenerationArtifactConfig(ConfigBase):
    root_dir: str = Field(alias="root-dir")
    alias: str | None


class InitialStateConfig(ConfigBase):
    theta_rad: float = Field(alias="theta-rad")
    omega_rad_s: float = Field(alias="omega-rad-s")


class ExperimentConfig(ConfigBase):
    run_id: str = Field(alias="run-id")
    episode_id: int = Field(alias="episode-id")
    max_steps: int = Field(alias="max-steps", gt=0)
    stop_on_goal: bool = Field(alias="stop-on-goal")
    initial_state: InitialStateConfig = Field(alias="initial-state")


class DataGenerationExperimentConfig(ConfigBase):
    run_id: str = Field(alias="run-id")
    episode_id: int = Field(alias="episode-id")
    max_steps: int = Field(alias="max-steps", gt=0)
    stop_on_goal: bool = Field(alias="stop-on-goal")


class CoordinatorConfig(ConfigBase):
    node_id: str = Field(alias="node-id")
    mode: Literal["synchronous"]
    event_trigger: bool = Field(alias="event-trigger")
    decision_interval_steps: int = Field(alias="decision-interval-steps", gt=0)
    debug_log_every_n_steps: int = Field(alias="debug-log-every-n-steps", gt=0)


class SimulationConfig(ConfigBase):
    timestep_s: float = Field(alias="timestep-s", gt=0.0)
    pace_s: float = Field(alias="pace-s", ge=0.0)


class PendulumConfig(ConfigBase):
    mass_kg: float = Field(alias="mass-kg", gt=0.0)
    gravity_m_s2: float = Field(alias="gravity-m-s2", gt=0.0)
    length_m: float = Field(alias="length-m", gt=0.0)
    damping_nms: float = Field(alias="damping-nms", ge=0.0)
    torque_limit_nm: float = Field(alias="torque-limit-nm", gt=0.0)
    theta_limit_abs_rad: float = Field(alias="theta-limit-abs-rad", gt=0.0)
    omega_limit_abs_rad_s: float = Field(alias="omega-limit-abs-rad-s", gt=0.0)


class GoalConfig(ConfigBase):
    angle_tolerance_rad: float = Field(alias="angle-tolerance-rad", gt=0.0)
    omega_tolerance_rad_s: float = Field(alias="omega-tolerance-rad-s", gt=0.0)
    hold_steps: int = Field(alias="hold-steps", ge=0)


class EnvironmentConfig(ConfigBase):
    node_id: str = Field(alias="node-id")
    simulation: SimulationConfig
    pendulum: PendulumConfig
    goal: GoalConfig


class VisualizationConfig(ConfigBase):
    update_every: int = Field(alias="update-every", gt=0)


class VisualizationRootConfig(ConfigBase):
    visualization: VisualizationConfig


class MPCConfig(ConfigBase):
    """Configuration for the CasADi split-ratio MPC controller."""

    controller: Literal["IP-dynamics-naturalPeriod"]
    prediction_horizon_natural_periods: float = Field(
        alias="prediction-horizon-natural-periods", gt=0.0
    )
    split_ratios: list[float] = Field(alias="split-ratios")

    @field_validator("split_ratios")
    @classmethod
    def _split_ratios_must_be_valid(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("split-ratios must contain at least one value")
        if any(ratio <= 0.0 or ratio > 1.0 or not math.isfinite(ratio) for ratio in value):
            raise ValueError("split-ratios entries must be finite values in (0, 1]")
        return value


class RLConfig(ConfigBase):
    """Frozen critic paths and online RL controls for intelligent and train modes."""

    critic_artifact_dir: str = Field(alias="critic-artifact-dir")
    time_model_artifact_dir: str = Field(alias="time-model-artifact-dir")
    gamma: float = Field(ge=0.0, le=1.0)
    time_weight: float = Field(alias="time-weight", ge=0.0)
    action_weight: float = Field(alias="action-weight", ge=0.0)
    compute_weight: float = Field(alias="compute-weight", ge=0.0)
    fail_penalty: float = Field(alias="fail-penalty", ge=0.0)
    exploration_epsilon: float = Field(alias="exploration-epsilon", ge=0.0, le=1.0)
    exploration_temperature: float = Field(alias="exploration-temperature", gt=0.0)
    training_learning_rate: float = Field(alias="training-learning-rate", gt=0.0)
    training_batch_size: int = Field(alias="training-batch-size", gt=0)
    training_updates_per_transition: int = Field(alias="training-updates-per-transition", gt=0)
    training_epochs: int = Field(alias="training-epochs", gt=0)
    training_seed: int = Field(alias="training-seed")
    training_output_root: str = Field(alias="training-output-root")


class RootConfig(ConfigBase):
    """Composed physical MPC episode without any RL domain."""

    experiment: ExperimentConfig
    coordinator: CoordinatorConfig
    environment: EnvironmentConfig
    mpc: MPCConfig
    artifacts: ArtifactConfig


class RLRootConfig(RootConfig):
    """Composed intelligent or training episode with explicit RL ownership."""

    rl: RLConfig


class DataGenerationConfig(ConfigBase):
    """Monte Carlo data-generation controls for offline RL."""

    episodes: int = Field(gt=0)
    seed: int | None
    visual_artifacts: bool = Field(alias="visual-artifacts")
    theta_rad_sample_range: list[float] = Field(alias="theta-rad-sample-range")
    omega_eq_scale_sample_range: list[float] = Field(alias="omega-eq-scale-sample-range")
    gamma: float = Field(ge=0.0, le=1.0)
    bbar_min: float = Field(alias="bbar-min", ge=0.0, le=1.0)
    bbar_max: float = Field(alias="bbar-max", ge=0.0, le=1.0)
    hbar_min: float = Field(alias="hbar-min", ge=0.0)
    hbar_max: float = Field(alias="hbar-max", ge=0.0)
    time_weight: float = Field(alias="time-weight", ge=0.0)
    action_weight: float = Field(alias="action-weight", ge=0.0)
    compute_weight: float = Field(alias="compute-weight", ge=0.0)
    fail_penalty: float = Field(alias="fail-penalty", ge=0.0)

    @field_validator("theta_rad_sample_range", "omega_eq_scale_sample_range")
    @classmethod
    def _sample_ranges_must_be_valid(cls, value: list[float]) -> list[float]:
        if len(value) != 2 or any(not math.isfinite(bound) for bound in value):
            raise ValueError("sample ranges must contain exactly two finite values")
        if value[1] < value[0]:
            raise ValueError("sample range upper bound must not be below its lower bound")
        return value

    @field_validator("bbar_max")
    @classmethod
    def _bbar_range_must_be_ordered(cls, value: float, info: ValidationInfo) -> float:
        if "bbar_min" in info.data and value < info.data["bbar_min"]:
            raise ValueError("bbar-max must be greater than or equal to bbar-min")
        return value

    @field_validator("hbar_max")
    @classmethod
    def _hbar_range_must_be_ordered(cls, value: float, info: ValidationInfo) -> float:
        if "hbar_min" in info.data and value < info.data["hbar_min"]:
            raise ValueError("hbar-max must be greater than or equal to hbar-min")
        return value


class DataGenerationRootConfig(ConfigBase):
    """Composed Monte Carlo config without coordinator, initial-state, or RL domains."""

    experiment: DataGenerationExperimentConfig
    environment: EnvironmentConfig
    mpc: MPCConfig
    artifacts: DataGenerationArtifactConfig
    data_generation: DataGenerationConfig = Field(alias="data-generation")
