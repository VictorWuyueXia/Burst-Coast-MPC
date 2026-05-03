from pathlib import Path

from wsmpc.config.loaders import load_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_realtime_resolver_matches_timestep() -> None:
    config = load_config(config_dir=CONFIG_DIR)

    assert config.environment.simulation.pace_s == config.environment.simulation.timestep_s


def test_numeric_pace_override_wins() -> None:
    config = load_config(
        config_dir=CONFIG_DIR,
        overrides=["environment.simulation.pace-s=0.0"],
    )

    assert config.environment.simulation.pace_s == 0.0
