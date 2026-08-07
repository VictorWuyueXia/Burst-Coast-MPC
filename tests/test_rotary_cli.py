from typer.testing import CliRunner

from bcmpc import app


def test_rotary_task_defaults_to_realtime_mpc(monkeypatch) -> None:
    runner = CliRunner()
    calls = []

    def run_rotary_stub(*, alias, no_visual, console) -> None:
        calls.append((alias, no_visual, console))

    monkeypatch.setattr("bcmpc.run_rotary_pendulum_mode", run_rotary_stub)

    result = runner.invoke(app, ["--rotary-pendulum"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][:2] == (None, False)


def test_task_selection_rejects_multiple_scenarios() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["--inverted-pendulum", "--rotary-pendulum"],
    )

    assert result.exit_code != 0
    assert "Select exactly one task" in result.output


def test_rotary_mpc_only_routes_alias_and_headless_policy(monkeypatch) -> None:
    runner = CliRunner()
    calls = []

    def run_rotary_stub(*, alias, no_visual, console) -> None:
        calls.append((alias, no_visual, console))

    monkeypatch.setattr("bcmpc.run_rotary_pendulum_mode", run_rotary_stub)

    result = runner.invoke(
        app,
        ["--rotary-pendulum", "mpc-only", "--alias", "trial", "--no-visual"],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][:2] == ("trial", True)
