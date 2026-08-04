from typer.testing import CliRunner

from bcmpc import app


def test_rotary_task_defaults_to_physics_simulation(monkeypatch) -> None:
    runner = CliRunner()
    calls = []

    def run_rotary_stub(*, console) -> None:
        calls.append(console)

    monkeypatch.setattr("bcmpc.run_rotary_pendulum_mode", run_rotary_stub)

    result = runner.invoke(app, ["--rotary-pendulum"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1


def test_task_selection_rejects_multiple_scenarios() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["--inverted-pendulum", "--rotary-pendulum"],
    )

    assert result.exit_code != 0
    assert "Select exactly one task" in result.output


def test_rotary_mpc_path_remains_unimplemented() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["--rotary-pendulum", "mpc-only", "--no-visual"],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)
