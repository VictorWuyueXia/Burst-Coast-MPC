import logging
import shutil
from pathlib import Path

import wsmpc.utils.loaders as loaders
from wsmpc.utils.loaders import load_config, load_config_with_fallbacks, warn_default_fallbacks


def test_standard_config_loads_as_atomic_package() -> None:
    config = load_config()

    assert config.environment.simulation.pace_s == config.environment.simulation.timestep_s
    assert config.experiment.run_id == "pendulum_baseline"


def test_missing_package_falls_back_to_default_and_warns(caplog) -> None:
    config, fallbacks = load_config_with_fallbacks("missing_package")

    assert config.experiment.run_id == "pendulum_baseline"
    assert fallbacks[0].parameter == "config-package"

    with caplog.at_level(logging.WARNING):
        warn_default_fallbacks(logging.getLogger("test"), fallbacks)

    assert "parameter=config-package" in caplog.text


def test_missing_parameter_uses_default_value_and_warns(tmp_path, monkeypatch, caplog) -> None:
    config_root = tmp_path / "configs"
    default_dir = config_root / "default"
    partial_dir = config_root / "partial"
    default_dir.mkdir(parents=True)
    partial_dir.mkdir(parents=True)
    shutil.copyfile(
        Path("configs/default/config.yaml"),
        default_dir / "config.yaml",
    )
    (partial_dir / "config.yaml").write_text("logging:\n  level: DEBUG\n", encoding="utf-8")
    monkeypatch.setattr(loaders, "CONFIG_ROOT", config_root)

    config, fallbacks = load_config_with_fallbacks("partial")

    assert config.environment.simulation.timestep_s == 0.02
    assert any(fallback.parameter == "environment.simulation.timestep-s" for fallback in fallbacks)

    with caplog.at_level(logging.WARNING):
        warn_default_fallbacks(logging.getLogger("test"), fallbacks)

    assert "parameter=environment.simulation.timestep-s value=0.02" in caplog.text
