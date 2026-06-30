import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import inverted_pendulum.utils.config_schema as config_schema
from inverted_pendulum.mpc.controller import prediction_horizon_steps
from inverted_pendulum.utils.config_schema import (
    DataGenerationRootConfig,
    RootConfig,
    load_config,
    load_data_generation_config,
    load_intelligent_config,
    load_online_training_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads_as_atomic_file() -> None:
    config = load_config()

    assert isinstance(config, RootConfig)
    assert config.environment.simulation.pace_s == config.environment.simulation.timestep_s
    assert config.experiment.run_id == "pendulum_baseline"
    assert config.coordinator.event_trigger is True
    assert config.mpc.controller == "IP-dynamics-naturalPeriod"
    assert config.rl.training_updates_per_transition == 8
    assert not hasattr(config.mpc, "cost")
    assert not hasattr(config, "runtime")
    assert not hasattr(config.environment.simulation, "max_rollout_steps")


def test_intelligent_config_uses_frozen_critic_for_deployment() -> None:
    config = load_intelligent_config()

    assert isinstance(config, RootConfig)
    assert config.artifacts.alias == "intelligent"
    assert config.experiment.run_id == "pendulum_intelligent"
    assert config.environment.simulation.pace_s == config.environment.simulation.timestep_s
    assert config.rl.critic_artifact_dir.endswith("structured-critic_20260614T215834")
    assert config.rl.time_model_artifact_dir.endswith("linear_20260615T191818")
    assert config.rl.exploration_epsilon == 0.02
    assert config.rl.exploration_temperature == 0.5


def test_frozen_time_model_is_positive_on_deployment_grid() -> None:
    config = load_intelligent_config()
    critic_dir = REPO_ROOT / config.rl.critic_artifact_dir
    time_model_dir = REPO_ROOT / config.rl.time_model_artifact_dir
    critic_config = json.loads((critic_dir / "config.json").read_text(encoding="utf-8"))
    time_model = json.loads((time_model_dir / "final_model.json").read_text(encoding="utf-8"))
    axis = np.linspace(0.0, 1.0, int(critic_config["action-grid-count"]))
    bbar_grid, hbar_grid = np.meshgrid(axis, axis, indexing="xy")
    full_horizon_steps = prediction_horizon_steps(config.environment)
    horizon_steps = np.maximum(1, np.rint(hbar_grid * full_horizon_steps)).astype(np.int64)
    burst_steps = np.maximum(1, np.rint(bbar_grid * 0.5 * horizon_steps)).astype(np.int64)
    compute_time_s = np.full(bbar_grid.shape, float(time_model["intercept_s"]))

    for term in time_model["all_terms"]:
        if term["name"] == "horizon_steps":
            compute_time_s += float(term["coefficient_s"]) * horizon_steps
        elif term["name"] == "burst_steps":
            compute_time_s += float(term["coefficient_s"]) * burst_steps
        elif term["name"] == "burst_horizon_steps":
            compute_time_s += float(term["coefficient_s"]) * burst_steps * horizon_steps
        else:
            raise ValueError(f"Unsupported frozen time-model term: {term['name']}")

    assert float(compute_time_s.min()) > 0.0


def test_online_training_config_uses_frozen_critic_for_exploration() -> None:
    config = load_online_training_config()

    assert isinstance(config, RootConfig)
    assert config.artifacts.alias == "online-training"
    assert config.experiment.run_id == "pendulum_online_training"
    assert config.environment.simulation.pace_s == 0.0
    assert config.rl.critic_artifact_dir.endswith("structured-critic_20260614T215834")
    assert config.rl.time_model_artifact_dir.endswith("linear_20260615T191818")
    assert config.rl.exploration_epsilon == 0.10
    assert config.rl.exploration_temperature == 2.0
    assert config.rl.training_updates_per_transition == 8


def test_data_generation_config_loads_with_event_trigger_disabled() -> None:
    config = load_data_generation_config()

    assert isinstance(config, DataGenerationRootConfig)
    assert not hasattr(config, "coordinator")
    assert not hasattr(config, "runtime")
    assert not hasattr(config.experiment, "initial_state")
    assert not hasattr(config.artifacts, "enabled")
    assert config.data_generation.seed is None
    assert config.data_generation.visual_artifacts is True
    assert 0.0 <= config.data_generation.bbar_min <= config.data_generation.bbar_max <= 1.0
    assert 0.0 <= config.data_generation.hbar_min <= config.data_generation.hbar_max <= 1.0


def test_missing_package_fails_loudly() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(config_schema.CONFIG_ROOT / "missing-config.yaml")


def test_missing_parameter_fails_schema_validation(tmp_path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir(parents=True)
    partial_path = config_root / "partial-config.yaml"
    partial_path.write_text(
        "artifacts:\n  root-dir: artifacts/experiments\n  alias: null\n  enabled: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(partial_path)


def test_unknown_mpc_controller_fails_schema_validation(tmp_path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir(parents=True)
    invalid_path = config_root / "invalid-config.yaml"
    text = config_schema.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    invalid_path.write_text(
        text.replace("IP-dynamics-naturalPeriod", "missing-controller"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(invalid_path)


def test_invalid_data_generation_bounds_fail_schema_validation(tmp_path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir(parents=True)
    invalid_path = config_root / "invalid-data-generation-config.yaml"
    text = config_schema.DATA_GENERATION_CONFIG_PATH.read_text(encoding="utf-8")
    invalid_path.write_text(
        text.replace("bbar-min: 0.01", "bbar-min: 1.1"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_data_generation_config(invalid_path)
