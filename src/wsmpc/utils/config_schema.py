"""Typed configuration schema."""

from __future__ import annotations

import math
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


class MPCCostConfig(ConfigBase):
    """Energy-phase objective weights used by the nonlinear MPC problem."""

    epsilon_phi: float = Field(default=1.0e-6, alias="epsilon-phi")
    sigma_energy: float = Field(default=0.5, alias="sigma-energy")
    q_energy: float = Field(default=1.0, alias="q-energy")
    q_phase: float = Field(default=0.0, alias="q-phase")
    q_local: float = Field(default=0.0, alias="q-local")
    q_terminal: float = Field(default=5.0, alias="q-terminal")
    rho_saturation: float = Field(default=0.0, alias="rho-saturation")
    rho_delta_u: float = Field(default=1.0e-3, alias="rho-delta-u")
    q_phase_diag: list[float] = Field(default_factory=lambda: [1.0, 1.0], alias="q-phase-diag")
    q_local_diag: list[float] = Field(default_factory=lambda: [1.0, 1.0], alias="q-local-diag")

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

    enabled: bool = False
    split_ratios: list[float] = Field(
        default_factory=lambda: [0.1, 0.2, 0.3],
        alias="split-ratios",
    )
    solve_candidates_in_parallel: bool = Field(default=False, alias="solve-candidates-in-parallel")
    max_parallel_workers: int | None = Field(default=None, alias="max-parallel-workers")
    horizon_steps_override: int | None = Field(default=None, alias="horizon-steps-override")
    prediction_horizon_rule: Literal["half-natural-period"] = Field(
        default="half-natural-period",
        alias="prediction-horizon-rule",
    )
    coast_mode: Literal["zero"] = Field(default="zero", alias="coast-mode")
    solver: Literal["ipopt"] = "ipopt"
    ipopt_print_level: int = Field(default=0, alias="ipopt-print-level")
    solver_max_iterations: int = Field(default=100, alias="solver-max-iterations")
    solver_tolerance: float = Field(default=1.0e-6, alias="solver-tolerance")
    cost: MPCCostConfig = Field(default_factory=MPCCostConfig)

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

    @field_validator("max_parallel_workers", "horizon_steps_override")
    @classmethod
    def _optional_step_counts_must_be_positive(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            msg = "optional MPC counts must be positive when provided"
            raise ValueError(msg)
        return value

    @field_validator("ipopt_print_level")
    @classmethod
    def _ipopt_print_level_must_be_nonnegative(cls, value: int) -> int:
        if value < 0:
            msg = "ipopt-print-level must be nonnegative"
            raise ValueError(msg)
        return value

    @field_validator("solver_max_iterations")
    @classmethod
    def _solver_iterations_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            msg = "solver-max-iterations must be positive"
            raise ValueError(msg)
        return value

    @field_validator("solver_tolerance")
    @classmethod
    def _solver_tolerance_must_be_positive(cls, value: float) -> float:
        if value <= 0.0 or not math.isfinite(value):
            msg = "solver-tolerance must be finite and positive"
            raise ValueError(msg)
        return value


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
    mpc: MPCConfig = Field(default_factory=MPCConfig)
    runtime: RuntimeConfig
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
