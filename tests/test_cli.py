from typer.testing import CliRunner

from wsmpc.cli import app
from wsmpc.utils.config_schema import load_config


def _require_matplotlib() -> None:
    import pytest

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


def test_cli_run_episode(monkeypatch) -> None:
    runner = CliRunner()
    config = load_config("default")
    config.experiment.max_steps = 2
    config.environment.simulation.timestep_s = 0.25
    config.environment.simulation.pace_s = 0.0
    config.runtime.max_worker_threads = 1
    config.artifacts.enabled = False

    def load_config_stub(package_name: str):
        return config

    monkeypatch.setattr("wsmpc.cli.load_config", load_config_stub)

    result = runner.invoke(
        app,
        ["run-episode", "--config-package", "default", "--no-visual"],
    )

    assert result.exit_code == 0, result.output
    assert "pendulum_baseline" in result.output


def test_cli_run_episode_writes_artifacts_with_alias(tmp_path, monkeypatch) -> None:
    _require_matplotlib()
    runner = CliRunner()
    config = load_config("default")
    config.artifacts.root_dir = str(tmp_path)
    config.experiment.max_steps = 2
    config.environment.simulation.timestep_s = 0.25
    config.environment.simulation.pace_s = 0.0
    config.runtime.max_worker_threads = 1

    def load_config_stub(package_name: str):
        return config

    monkeypatch.setattr("wsmpc.cli.load_config", load_config_stub)

    result = runner.invoke(
        app,
        [
            "run-episode",
            "--config-package",
            "default",
            "--alias",
            "smoke",
            "--no-visual",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "pendulum_baseline" in result.output
    assert "artifact_dir" in result.output
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    for name in ["states", "energy", "phase", "commands"]:
        assert (run_dirs[0] / "figures" / f"{name}.png").exists()
