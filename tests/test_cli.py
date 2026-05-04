from typer.testing import CliRunner

from wsmpc.cli import app


def test_cli_run_episode() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["run-episode"])

    assert result.exit_code == 0, result.output
    assert "pendulum_baseline" in result.output


def test_cli_run_episode_with_explicit_standard_config() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["run-episode", "--config-package", "standard"])

    assert result.exit_code == 0, result.output
    assert "pendulum_baseline" in result.output
