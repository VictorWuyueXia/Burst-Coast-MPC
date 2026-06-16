"""Top-level command line entry point for Burst-Coast MPC experiments."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from burst_coast_mpc.intelligent_mode import run_intelligent_mode
from burst_coast_mpc.monte_carlo_mode import run_monte_carlo_mode
from burst_coast_mpc.mpc_only_mode import run_mpc_only_mode
from burst_coast_mpc.train_mode import run_train_mode

app = typer.Typer(help="Burst-Coast MPC research CLI.", invoke_without_command=True)
console = Console()


def _selected_task(inverted_pendulum: bool, cr3bp: bool) -> str:
    """Resolve the single task selected by the top-level CLI flags."""

    if inverted_pendulum == cr3bp:
        msg = "Select exactly one task with --inverted-pendulum or --cr3bp"
        raise typer.BadParameter(msg)
    if cr3bp:
        msg = "CR3BP task scaffold exists, but runtime implementation is not available yet"
        raise NotImplementedError(msg)
    return "inverted_pendulum"


@app.callback()
def select_task(
    ctx: typer.Context,
    inverted_pendulum: Annotated[
        bool,
        typer.Option(
            "--inverted-pendulum",
            help="Run the inverted pendulum task.",
        ),
    ] = False,
    cr3bp: Annotated[
        bool,
        typer.Option(
            "--cr3bp",
            help="Run the CR3BP task.",
        ),
    ] = False,
) -> None:
    """Select one task package before running a Burst-Coast MPC command."""

    # Bind the task once at the command boundary so subcommands remain task-explicit.
    ctx.obj = _selected_task(inverted_pendulum, cr3bp)
    if ctx.invoked_subcommand is None:
        run_mpc_only_mode(ctx.obj, alias=None, no_visual=False, console=console)
        raise typer.Exit()


@app.command("mpc-only")
def mpc_only_command(
    ctx: typer.Context,
    alias: Annotated[
        str | None,
        typer.Option(
            "--alias",
            help="Human-readable alias prefix for the artifact run directory.",
        ),
    ] = None,
    no_visual: Annotated[
        bool,
        typer.Option(
            "--no-visual",
            help="Run without realtime plots or pendulum animation.",
        ),
    ] = False,
) -> None:
    """Run one MPC-only experiment episode."""

    run_mpc_only_mode(ctx.obj, alias=alias, no_visual=no_visual, console=console)


@app.command("intelligent")
def intelligent_command(
    ctx: typer.Context,
    alias: Annotated[
        str | None,
        typer.Option(
            "--alias",
            help="Human-readable alias prefix for the artifact run directory.",
        ),
    ] = None,
    no_visual: Annotated[
        bool,
        typer.Option(
            "--no-visual",
            help="Run without realtime plots or pendulum animation.",
        ),
    ] = False,
) -> None:
    """Run RL critic deployment mode without online learning."""

    run_intelligent_mode(ctx.obj, alias=alias, no_visual=no_visual, console=console)


@app.command("train")
def train_command(
    ctx: typer.Context,
    alias: Annotated[
        str | None,
        typer.Option(
            "--alias",
            help="Human-readable alias prefix for the artifact run directory.",
        ),
    ] = None,
    no_visual: Annotated[
        bool,
        typer.Option(
            "--no-visual",
            help="Run without realtime plots or pendulum animation.",
        ),
    ] = False,
) -> None:
    """Run exploratory online RL policy training mode."""

    run_train_mode(ctx.obj, alias=alias, no_visual=no_visual, console=console)


@app.command("montecarlo")
def monte_carlo_command(
    ctx: typer.Context,
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            help="Independent Monte Carlo artifact runs to generate.",
        ),
    ] = 1,
) -> None:
    """Generate sequential Monte Carlo datasets for offline RL."""

    if epochs <= 0:
        msg = "--epochs must be positive"
        raise typer.BadParameter(msg)
    run_monte_carlo_mode(ctx.obj, epochs=epochs, console=console)


def main() -> None:
    """Run the Typer command application."""

    app()


if __name__ == "__main__":
    main()
