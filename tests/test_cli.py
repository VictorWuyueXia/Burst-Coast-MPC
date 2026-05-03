from pathlib import Path

from typer.testing import CliRunner

from wsmpc.cli import app


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_cli_run_episode() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-episode",
            "--config-dir",
            str(CONFIG_DIR),
            "--max-steps",
            "2",
            "--pace-s",
            "0.0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "pendulum_baseline" in result.output
