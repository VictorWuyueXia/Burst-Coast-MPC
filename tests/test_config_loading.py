import pytest
from pydantic import ValidationError

import wsmpc.utils.config_schema as config_schema
from wsmpc.utils.config_schema import (
    DataGenerationRootConfig,
    RootConfig,
    load_config,
    load_data_generation_config,
)


def test_default_config_loads_as_atomic_file() -> None:
    config = load_config()

    assert isinstance(config, RootConfig)
    assert config.environment.simulation.pace_s == config.environment.simulation.timestep_s
    assert config.experiment.run_id == "pendulum_baseline"
    assert config.coordinator.event_trigger is True
    assert config.mpc.controller == "IP-dynamics-naturalPeriod"
    assert not hasattr(config.mpc, "cost")
    assert not hasattr(config, "runtime")
    assert not hasattr(config.environment.simulation, "max_rollout_steps")

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
