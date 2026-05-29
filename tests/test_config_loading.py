import pytest
from pydantic import ValidationError

import wsmpc.utils.config_schema as config_schema
from wsmpc.utils.config_schema import RootConfig, load_config


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
