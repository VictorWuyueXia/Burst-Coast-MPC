import pytest
from pydantic import ValidationError

import wsmpc.utils.config_schema as config_schema
from wsmpc.utils.config_schema import (
    DataGenerationRootConfig,
    RootConfig,
    load_config,
    load_data_generation_config,
)


def test_standard_config_loads_as_atomic_package() -> None:
    config = load_config("standard")

    assert isinstance(config, RootConfig)
    assert config.environment.simulation.pace_s == config.environment.simulation.timestep_s
    assert config.experiment.run_id == "pendulum_baseline"
    assert config.coordinator.event_trigger is True
    assert config.mpc.controller == "IP-dynamics-naturalPeriod"
    assert not hasattr(config.mpc, "cost")
    assert not hasattr(config.environment.simulation, "max_rollout_steps")
    assert not hasattr(config.runtime, "torch_threads")


def test_default_config_enables_event_trigger() -> None:
    config = load_config("default")

    assert config.coordinator.event_trigger is True


def test_data_generation_config_loads_with_event_trigger_disabled() -> None:
    config = load_data_generation_config()

    assert isinstance(config, DataGenerationRootConfig)
    assert config.coordinator.event_trigger is False
    assert config.runtime.max_worker_threads == 1
    assert config.data_generation.visual_artifacts is True
    assert 0.0 <= config.data_generation.bbar_min <= config.data_generation.bbar_max <= 1.0
    assert 0.0 <= config.data_generation.hbar_min <= config.data_generation.hbar_max <= 1.0


def test_missing_package_fails_loudly() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("missing_package")


def test_missing_parameter_fails_schema_validation(tmp_path, monkeypatch) -> None:
    config_root = tmp_path / "configs"
    partial_dir = config_root / "partial"
    partial_dir.mkdir(parents=True)
    (partial_dir / "config.yaml").write_text(
        "artifacts:\n  root-dir: artifacts/experiments\n  alias: null\n  enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_schema, "CONFIG_ROOT", config_root)

    with pytest.raises(ValidationError):
        load_config("partial")


def test_unknown_mpc_controller_fails_schema_validation(tmp_path, monkeypatch) -> None:
    config_root = tmp_path / "configs"
    invalid_dir = config_root / "invalid"
    invalid_dir.mkdir(parents=True)
    text = (config_schema.CONFIG_ROOT / "default" / "config.yaml").read_text(encoding="utf-8")
    (invalid_dir / "config.yaml").write_text(
        text.replace("IP-dynamics-naturalPeriod", "missing-controller"),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_schema, "CONFIG_ROOT", config_root)

    with pytest.raises(ValidationError):
        load_config("invalid")


def test_invalid_data_generation_bounds_fail_schema_validation(tmp_path, monkeypatch) -> None:
    config_root = tmp_path / "configs"
    invalid_dir = config_root / "invalid-data-generation"
    invalid_dir.mkdir(parents=True)
    text = (config_schema.CONFIG_ROOT / "data-generation" / "config.yaml").read_text(
        encoding="utf-8"
    )
    (invalid_dir / "config.yaml").write_text(
        text.replace("bbar-min: 0.1", "bbar-min: 1.1"),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_schema, "CONFIG_ROOT", config_root)

    with pytest.raises(ValidationError):
        load_data_generation_config("invalid-data-generation")
