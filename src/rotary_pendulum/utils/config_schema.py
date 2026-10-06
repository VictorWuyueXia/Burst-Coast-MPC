"""Strict domain-composed configuration for the rotary-pendulum scenario."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
PHYSICS_CONFIG_PATH = CONFIG_ROOT / "physics.yaml"
MISSION_CONFIG_PATH = CONFIG_ROOT / "mission.yaml"
RUNTIME_CONFIG_PATH = CONFIG_ROOT / "runtime.yaml"
MPC_CONFIG_PATH = CONFIG_ROOT / "mpc.yaml"
ARTIFACT_CONFIG_PATH = CONFIG_ROOT / "artifacts.yaml"
VISUAL_CONFIG_PATH = CONFIG_ROOT / "visual.yaml"
EPISODE_CONFIG_PATHS = (
    PHYSICS_CONFIG_PATH,
    MISSION_CONFIG_PATH,
    RUNTIME_CONFIG_PATH,
    MPC_CONFIG_PATH,
    ARTIFACT_CONFIG_PATH,
)
RuntimeMode = Literal["realtime", "headless"]


class ConfigBase(BaseModel):
    """Apply strict validation and readable YAML aliases to every config group."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        protected_namespaces=(),
        allow_inf_nan=False,
    )


class SimulationConfig(ConfigBase):
    """Define the fixed prediction, integration, and wall-clock intervals."""

    timestep_s: float = Field(alias="timestep-s", gt=0.0)
    pace_s: float = Field(alias="pace-s", ge=0.0)


class RotaryPendulumConfig(ConfigBase):
    """Own the primitive physical quantities of the rotary pendulum."""

    gravity_m_s2: float = Field(alias="gravity-m-s2", gt=0.0)
    arm_mass_kg: float = Field(alias="arm-mass-kg", gt=0.0)
    arm_length_m: float = Field(alias="arm-length-m", gt=0.0)
    pendulum_mass_kg: float = Field(alias="pendulum-mass-kg", gt=0.0)
    pendulum_length_m: float = Field(alias="pendulum-length-m", gt=0.0)
    rotary_damping_nms: float = Field(alias="rotary-damping-nms", ge=0.0)
    pendulum_damping_nms: float = Field(alias="pendulum-damping-nms", ge=0.0)
    torque_limit_nm: float = Field(alias="torque-limit-nm", gt=0.0)


class InitialStateConfig(ConfigBase):
    """Define the four unwrapped initial state coordinates."""

    theta_rad: float = Field(alias="theta-rad")
    alpha_rad: float = Field(alias="alpha-rad")
    omega_rad_s: float = Field(alias="omega-rad-s")
    nu_rad_s: float = Field(alias="nu-rad-s")


class ExperimentConfig(ConfigBase):
    """Define episode identity, duration, and initial physical state."""

    run_id: str = Field(alias="run-id", min_length=1)
    episode_id: int = Field(alias="episode-id", ge=0)
    max_steps: int = Field(alias="max-steps", gt=0)
    initial_state: InitialStateConfig = Field(alias="initial-state")


class GoalConfig(ConfigBase):
    """Define the complete upright-state goal set and hold duration."""

    theta_tolerance_rad: float = Field(alias="theta-tolerance-rad", gt=0.0)
    beta_tolerance_rad: float = Field(alias="beta-tolerance-rad", gt=0.0)
    omega_tolerance_rad_s: float = Field(alias="omega-tolerance-rad-s", gt=0.0)
    nu_tolerance_rad_s: float = Field(alias="nu-tolerance-rad-s", gt=0.0)
    hold_steps: int = Field(alias="hold-steps", gt=0)


class MPCConfig(ConfigBase):
    """Define the exact burst-coast search and its objective weights."""

    prediction_horizon_natural_periods: float = Field(
        alias="prediction-horizon-natural-periods",
        gt=0.0,
    )
    split_ratios: list[float] = Field(alias="split-ratios")
    terminal_swing_energy_weight: float = Field(alias="terminal-swing-energy-weight", gt=0.0)
    phase_chasing_weight: float = Field(alias="phase-chasing-weight", gt=0.0)
    energy_transition_width: float = Field(alias="energy-transition-width", gt=0.0)
    local_energy_shell_width: float = Field(alias="local-energy-shell-width", gt=0.0)
    pendulum_local_weight: float = Field(alias="pendulum-local-weight", gt=0.0)
    rotary_local_weight: float = Field(alias="rotary-local-weight", gt=0.0)
    arm_angle_soft_penalty_weight: float = Field(alias="arm-angle-soft-penalty-weight", gt=0.0)
    torque_slew_weight: float = Field(alias="torque-slew-weight", gt=0.0)

    @field_validator("split_ratios")
    @classmethod
    def _split_ratios_must_be_valid(cls, values: list[float]) -> list[float]:
        if not values or any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in values
        ):
            raise ValueError("split-ratios must contain finite values in (0, 1]")
        return values


class ArtifactConfig(ConfigBase):
    """Define the mandatory experiment artifact directory and optional alias."""

    root_dir: str = Field(alias="root-dir", min_length=1)
    alias: str | None


class VisualizationConfig(ConfigBase):
    """Define how often the realtime figure redraws recorded dynamics."""

    update_every: int = Field(alias="update-every", gt=0)


class EpisodeConfig(ConfigBase):
    """Compose only the domains required by rotary MPC execution."""

    simulation: SimulationConfig
    rotary_pendulum: RotaryPendulumConfig = Field(alias="rotary-pendulum")
    experiment: ExperimentConfig
    goal: GoalConfig
    mpc: MPCConfig
    artifacts: ArtifactConfig


def load_episode_config(
    config_paths: tuple[Path, ...] = EPISODE_CONFIG_PATHS,
    *,
    runtime_mode: RuntimeMode = "realtime",
) -> EpisodeConfig:
    """Merge and validate the domains required by one rotary MPC episode."""

    return EpisodeConfig.model_validate(_load_domains(config_paths, runtime_mode))


def load_visualization_config(config_path: Path = VISUAL_CONFIG_PATH) -> VisualizationConfig:
    """Load the visualization domain only for interactive execution."""

    return VisualizationConfig.model_validate(_load_domains((config_path,))["visualization"])


def _load_domains(
    config_paths: tuple[Path, ...], runtime_mode: RuntimeMode | None = None
) -> dict[str, Any]:
    """Merge required domains, apply one runtime profile, and resolve interpolation."""

    if not config_paths:
        raise ValueError("At least one config path is required")
    missing_paths = [path for path in config_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Config file not found: {missing_paths[0]}")
    loaded_domains = tuple(OmegaConf.load(path) for path in config_paths)
    merged = cast(DictConfig, OmegaConf.merge(*loaded_domains))
    if runtime_mode is not None:
        profile = OmegaConf.select(merged, f"modes.{runtime_mode}")
        if profile is None:
            raise ValueError(f"Config runtime profile not found: {runtime_mode}")
        merged = cast(DictConfig, OmegaConf.merge(merged, profile))
    if "modes" in merged:
        del merged["modes"]
    resolved = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Composed config must resolve to a mapping")
    return cast(dict[str, Any], resolved)
