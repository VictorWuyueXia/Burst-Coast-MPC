import pytest
from pydantic import ValidationError

import wsmpc.utils.loaders as loaders
from wsmpc.utils.config_schema import RootConfig
from wsmpc.utils.loaders import load_config


def test_standard_config_loads_as_atomic_package() -> None:
    config = load_config("standard")

    assert isinstance(config, RootConfig)
    assert config.environment.simulation.pace_s == config.environment.simulation.timestep_s
    assert config.experiment.run_id == "pendulum_baseline"


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
    monkeypatch.setattr(loaders, "CONFIG_ROOT", config_root)

    with pytest.raises(ValidationError):
        load_config("partial")
