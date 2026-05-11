from typer.testing import CliRunner

from wsmpc.cli import TerminalPauseController, app


def test_cli_run_episode() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["run-episode", "--config-package", "default", "--no-artifacts"])

    assert result.exit_code == 0, result.output
    assert "pendulum_baseline" in result.output


def test_cli_run_episode_writes_artifacts(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-episode",
            "--config-package",
            "default",
            "--artifact-root",
            str(tmp_path),
            "--run-alias",
            "smoke",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "pendulum_baseline" in result.output
    assert "artifact_dir" in result.output
    assert len(list(tmp_path.iterdir())) == 1


def test_cli_run_episode_no_artifacts_does_not_create_root(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-episode",
            "--config-package",
            "default",
            "--artifact-root",
            str(tmp_path / "unused"),
            "--no-artifacts",
        ],
    )

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "unused").exists()


def test_terminal_pause_controller_disabled_does_not_block() -> None:
    controller = TerminalPauseController(enabled=False)

    with controller:
        controller.wait_if_paused()

    assert controller.enabled is False
