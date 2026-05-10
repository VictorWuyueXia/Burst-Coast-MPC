"""Typed configuration schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigBase(BaseModel):
    """Base model that accepts human-readable YAML aliases."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", protected_namespaces=())


class LoggingConfig(ConfigBase):
    """Runtime logging controls."""

    level: str = "INFO"


class ArtifactConfig(ConfigBase):
    """Artifact recording controls for local experiment runs."""

    root_dir: str = Field(default="artifacts/experiments", alias="root-dir")
    alias: str | None = None
    enabled: bool = True


class InitialStateConfig(ConfigBase):
    """Initial physical state for the pendulum environment."""

    theta_rad: float = Field(default=0.0, alias="theta-rad")
    omega_rad_s: float = Field(default=0.0, alias="omega-rad-s")


class DefaultActionConfig(ConfigBase):
    """Default action used by the Coordinator."""

    u_nm: float = Field(default=0.0, alias="u-nm")
    source: str = "zero_torque"


class ExperimentConfig(ConfigBase):
    """Top-level experiment controls owned by the Coordinator."""

    run_id: str = Field(default="pendulum_baseline", alias="run-id")
    global_seed: int = Field(default=1, alias="global-seed")
    episode_id: int = Field(default=0, alias="episode-id")
    max_steps: int = Field(default=25, alias="max-steps")
    stop_on_goal: bool = Field(default=True, alias="stop-on-goal")
    initial_state: InitialStateConfig = Field(
        default_factory=InitialStateConfig, alias="initial-state"
    )
    default_action: DefaultActionConfig = Field(
        default_factory=DefaultActionConfig, alias="default-action"
    )

    @field_validator("max_steps")
    @classmethod
    def _max_steps_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            msg = "max-steps must be positive"
            raise ValueError(msg)
        return value


class CoordinatorConfig(ConfigBase):
    """Coordinator timing and synchronous execution policy."""

    node_id: str = Field(default="Coordinator", alias="node-id")
    mode: Literal["synchronous"] = "synchronous"
    decision_interval_steps: int = Field(default=1, alias="decision-interval-steps")
    debug_log_every_n_steps: int = Field(default=1, alias="debug-log-every-n-steps")

    @field_validator("decision_interval_steps", "debug_log_every_n_steps")
    @classmethod
    def _step_counts_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            msg = "step counts must be positive"
            raise ValueError(msg)
        return value


class SimulationConfig(ConfigBase):
    """Simulation clock controls kept separate from wall-clock pacing."""

    timestep_s: float = Field(default=0.02, alias="timestep-s")
    pace_s: float = Field(default=0.02, alias="pace-s")
    max_rollout_steps: int = Field(default=10000, alias="max-rollout-steps")

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

    mass_kg: float = Field(default=1.0, alias="mass-kg")
    gravity_m_s2: float = Field(default=9.80665, alias="gravity-m-s2")
    length_m: float = Field(default=1.0, alias="length-m")
    damping_nms: float = Field(default=0.05, alias="damping-nms")
    torque_limit_nm: float = Field(default=4.0, alias="torque-limit-nm")
    theta_limit_abs_rad: float = Field(default=12.566370614359172, alias="theta-limit-abs-rad")
    omega_limit_abs_rad_s: float = Field(default=40.0, alias="omega-limit-abs-rad-s")

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

    angle_tolerance_rad: float = Field(default=0.05, alias="angle-tolerance-rad")
    omega_tolerance_rad_s: float = Field(default=0.05, alias="omega-tolerance-rad-s")
    hold_steps: int = Field(default=25, alias="hold-steps")

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

    node_id: str = Field(default="Environment", alias="node-id")
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    pendulum: PendulumConfig = Field(default_factory=PendulumConfig)
    goal: GoalConfig = Field(default_factory=GoalConfig)


class RuntimeConfig(ConfigBase):
    """Resource limits for local runs."""

    node_id: str = Field(default="Runtime", alias="node-id")
    max_worker_threads: int = Field(default=2, alias="max-worker-threads")
    blas_threads: int = Field(default=1, alias="blas-threads")
    torch_threads: int = Field(default=1, alias="torch-threads")
    cpu_affinity: list[int] = Field(default_factory=list, alias="cpu-affinity")
    set_env: bool = Field(default=True, alias="set-env")

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
    runtime: RuntimeConfig
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
