"""Typed configuration schema."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator

from inverted_pendulum.utils.time import realtime

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "default-config.yaml"
INTELLIGENT_CONFIG_PATH = CONFIG_ROOT / "intelligent-config.yaml"
ONLINE_TRAINING_CONFIG_PATH = CONFIG_ROOT / "online-training-config.yaml"
DATA_GENERATION_CONFIG_PATH = CONFIG_ROOT / "data-generation-config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> RootConfig:
    """Load the complete default episode config and validate every required field."""

    resolved = _load_resolved_config(config_path)
    return RootConfig.model_validate(resolved)


def load_intelligent_config() -> RootConfig:
    """Load the deterministic critic-deployment episode config."""

    return load_config(INTELLIGENT_CONFIG_PATH)


def load_online_training_config() -> RootConfig:
    """Load the exploratory online fitted-Q training episode config."""

    return load_config(ONLINE_TRAINING_CONFIG_PATH)


def load_data_generation_config(
    config_path: Path = DATA_GENERATION_CONFIG_PATH,
) -> DataGenerationRootConfig:
    """Load the complete Monte Carlo data-generation config."""

    resolved = _load_resolved_config(config_path)
    return DataGenerationRootConfig.model_validate(resolved)


def _load_resolved_config(config_path: Path) -> dict:
    """Resolve one YAML config file before Pydantic validation."""

    # Bind the realtime resolver only at the YAML boundary.
    OmegaConf.register_new_resolver("realtime", realtime, replace=True)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)
    return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)


class ConfigBase(BaseModel):
    """Common strict schema settings for human-readable YAML aliases."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        protected_namespaces=(),
        allow_inf_nan=False,
    )


class ArtifactConfig(ConfigBase):
    """Artifact recording controls for normal experiment runs."""

    root_dir: str = Field(alias="root-dir")
    alias: str | None
    enabled: bool


class DataGenerationArtifactConfig(ConfigBase):
    """Artifact directory controls required by Monte Carlo data generation."""

    root_dir: str = Field(alias="root-dir")
    alias: str | None


class InitialStateConfig(ConfigBase):
    """Initial physical state for the pendulum environment."""

    theta_rad: float = Field(alias="theta-rad")
    omega_rad_s: float = Field(alias="omega-rad-s")


class ExperimentConfig(ConfigBase):
    """Top-level experiment controls owned by the Coordinator."""

    run_id: str = Field(alias="run-id")
    episode_id: int = Field(alias="episode-id")
    max_steps: int = Field(alias="max-steps", gt=0)
    stop_on_goal: bool = Field(alias="stop-on-goal")
    initial_state: InitialStateConfig = Field(alias="initial-state")


class DataGenerationExperimentConfig(ConfigBase):
    """Episode identity and stopping policy used by Monte Carlo generation."""

    run_id: str = Field(alias="run-id")
    episode_id: int = Field(alias="episode-id")
    max_steps: int = Field(alias="max-steps", gt=0)
    stop_on_goal: bool = Field(alias="stop-on-goal")


class CoordinatorConfig(ConfigBase):
    """Coordinator timing and synchronous execution policy."""

    node_id: str = Field(alias="node-id")
    mode: Literal["synchronous"]
    event_trigger: bool = Field(alias="event-trigger")
    decision_interval_steps: int = Field(alias="decision-interval-steps", gt=0)
    debug_log_every_n_steps: int = Field(alias="debug-log-every-n-steps", gt=0)


class SimulationConfig(ConfigBase):
    """Simulation clock controls kept separate from wall-clock pacing."""

    timestep_s: float = Field(alias="timestep-s", gt=0.0)
    pace_s: float = Field(alias="pace-s", ge=0.0)


class PendulumConfig(ConfigBase):
    """Physical constants and conservative state bounds for the pendulum."""

    mass_kg: float = Field(alias="mass-kg", gt=0.0)
    gravity_m_s2: float = Field(alias="gravity-m-s2", gt=0.0)
    length_m: float = Field(alias="length-m", gt=0.0)
    damping_nms: float = Field(alias="damping-nms", ge=0.0)
    torque_limit_nm: float = Field(alias="torque-limit-nm", gt=0.0)
    theta_limit_abs_rad: float = Field(alias="theta-limit-abs-rad", gt=0.0)
    omega_limit_abs_rad_s: float = Field(alias="omega-limit-abs-rad-s", gt=0.0)


class GoalConfig(ConfigBase):
    """Goal-set thresholds used by the Environment and Coordinator."""

    angle_tolerance_rad: float = Field(alias="angle-tolerance-rad", gt=0.0)
    omega_tolerance_rad_s: float = Field(alias="omega-tolerance-rad-s", gt=0.0)
    hold_steps: int = Field(alias="hold-steps", ge=0)


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
    """Fully composed normal episode configuration."""

    experiment: ExperimentConfig
    coordinator: CoordinatorConfig
    environment: EnvironmentConfig
    mpc: MPCConfig
    rl: RLConfig
    artifacts: ArtifactConfig


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
    hbar_min: float = Field(alias="hbar-min", ge=0.0, le=1.0)
    hbar_max: float = Field(alias="hbar-max", ge=0.0, le=1.0)
    time_weight: float = Field(alias="time-weight", ge=0.0)
    action_weight: float = Field(alias="action-weight", ge=0.0)
    compute_weight: float = Field(alias="compute-weight", ge=0.0)
    fail_penalty: float = Field(alias="fail-penalty", ge=0.0)

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


class DataGenerationRootConfig(ConfigBase):
    """Complete Monte Carlo generation configuration."""

    experiment: DataGenerationExperimentConfig
    environment: EnvironmentConfig
    mpc: MPCConfig
    artifacts: DataGenerationArtifactConfig
    data_generation: DataGenerationConfig = Field(alias="data-generation")
