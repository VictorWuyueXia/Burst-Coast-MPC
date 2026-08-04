"""Strict configuration schema for the rotary-pendulum simulation."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "default-config.yaml"


class ConfigBase(BaseModel):
    """Apply strict validation and readable YAML aliases to every config group."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        protected_namespaces=(),
        allow_inf_nan=False,
    )


class SimulationConfig(ConfigBase):
    """Define the fixed simulation step and wall-clock pacing interval."""

    timestep_s: float = Field(alias="timestep-s", gt=0.0)
    pace_s: float = Field(alias="pace-s", ge=0.0)


class VisualizationConfig(ConfigBase):
    """Define how often the realtime figure redraws from recorded dynamics."""

    update_every: int = Field(alias="update-every", gt=0)


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
    """Define reproducibility, duration, and the initial physical state."""

    run_id: str = Field(alias="run-id", min_length=1)
    seed: int
    max_steps: int = Field(alias="max-steps", gt=0)
    initial_state: InitialStateConfig = Field(alias="initial-state")


class GoalConfig(ConfigBase):
    """Define the complete upright-state goal set and hold duration."""

    theta_tolerance_rad: float = Field(alias="theta-tolerance-rad", gt=0.0)
    beta_tolerance_rad: float = Field(alias="beta-tolerance-rad", gt=0.0)
    omega_tolerance_rad_s: float = Field(alias="omega-tolerance-rad-s", gt=0.0)
    nu_tolerance_rad_s: float = Field(alias="nu-tolerance-rad-s", gt=0.0)
    hold_steps: int = Field(alias="hold-steps", gt=0)


class RootConfig(ConfigBase):
    """Compose the complete rotary-pendulum simulation configuration."""

    simulation: SimulationConfig
    visualization: VisualizationConfig
    rotary_pendulum: RotaryPendulumConfig = Field(alias="rotary-pendulum")
    experiment: ExperimentConfig
    goal: GoalConfig


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> RootConfig:
    """Load one atomic YAML configuration and validate every required field."""

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return RootConfig.model_validate(OmegaConf.to_container(OmegaConf.load(config_path)))
