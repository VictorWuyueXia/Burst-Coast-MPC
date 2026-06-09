from typer.testing import CliRunner

from wsmpc.cli import app
from wsmpc.utils.config_schema import load_config, load_data_generation_config


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


def test_cli_generate_mc_data_writes_step_and_rl_artifacts(tmp_path, monkeypatch) -> None:
    _require_matplotlib()
    runner = CliRunner()
    config = load_data_generation_config()
    config.artifacts.root_dir = str(tmp_path)
    config.data_generation.episodes = 1
    config.data_generation.bbar_min = 1.0
    config.data_generation.bbar_max = 1.0
    config.data_generation.hbar_min = 1.0
    config.data_generation.hbar_max = 1.0
    config.data_generation.visual_artifacts = True
    config.experiment.max_steps = 2
    config.environment.simulation.timestep_s = 0.25
    config.environment.simulation.pace_s = 0.0
    config.environment.goal.hold_steps = 999
    config.runtime.max_worker_threads = 1

    def load_data_generation_config_stub():
        return config

    monkeypatch.setattr("wsmpc.cli.load_data_generation_config", load_data_generation_config_stub)

    result = runner.invoke(app, ["generate-mc-data"])

    assert result.exit_code == 0, result.output
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "steps.csv").exists()
    assert (run_dirs[0] / "rl_steps.csv").exists()
    for name in ["states", "energy", "phase", "commands"]:
        assert (run_dirs[0] / "figures" / f"{name}.png").exists()
    assert "epochs" in result.output
    assert "rl_steps" in result.output


def test_cli_generate_mc_data_epochs_create_separate_artifacts(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    config = load_data_generation_config()
    config.artifacts.root_dir = str(tmp_path)
    config.artifacts.alias = "mc"
    config.data_generation.episodes = 1
    config.data_generation.bbar_min = 1.0
    config.data_generation.bbar_max = 1.0
    config.data_generation.hbar_min = 1.0
    config.data_generation.hbar_max = 1.0
    config.data_generation.visual_artifacts = False
    config.experiment.max_steps = 1
    config.environment.simulation.timestep_s = 0.25
    config.environment.simulation.pace_s = 0.0
    config.environment.goal.hold_steps = 999
    config.runtime.max_worker_threads = 1

    def load_data_generation_config_stub():
        return config

    monkeypatch.setattr("wsmpc.cli.load_data_generation_config", load_data_generation_config_stub)

    result = runner.invoke(app, ["generate-mc-data", "--epochs", "2"])

    assert result.exit_code == 0, result.output
    run_dirs = sorted(tmp_path.iterdir())
    assert len(run_dirs) == 2
    assert run_dirs[0].name.startswith("mc-epoch-1_")
    assert run_dirs[1].name.startswith("mc-epoch-2_")
    for run_dir in run_dirs:
        assert (run_dir / "steps.csv").exists()
        assert (run_dir / "rl_steps.csv").exists()
