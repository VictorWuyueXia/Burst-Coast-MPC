"""Root command interface for Burst-Coast MPC experiment modes."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from bringup.intelligent_mode import run_intelligent_mode
from bringup.monte_carlo_mode import run_monte_carlo_mode
from bringup.mpc_only_mode import run_mpc_only_mode
from bringup.train_mode import run_train_mode

app = typer.Typer(help="Run one selected Burst-Coast MPC task mode.", invoke_without_command=True)
console = Console()


def _selected_task(inverted_pendulum: bool, cr3bp: bool) -> str:
    """Resolve the one task selected by the root command flags."""

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
        typer.Option("--inverted-pendulum", help="Run the inverted-pendulum task package."),
    ] = False,
    cr3bp: Annotated[
        bool,
        typer.Option("--cr3bp", help="Run the CR3BP task package."),
    ] = False,
) -> None:
    """Select a task; without a subcommand, run its MPC baseline."""

    # Bind the task once so each mode receives the same explicit package selection.
    ctx.obj = _selected_task(inverted_pendulum, cr3bp)
    if ctx.invoked_subcommand is None:
        run_mpc_only_mode(ctx.obj, alias=None, no_visual=False, console=console)
        raise typer.Exit()


@app.command("mpc-only")
def mpc_only(
    ctx: typer.Context,
    alias: Annotated[
        str | None,
        typer.Option("--alias", help="Prefix the generated artifact directory."),
    ] = None,
    no_visual: Annotated[
        bool,
        typer.Option("--no-visual", help="Disable realtime diagnostics and animation."),
    ] = False,
) -> None:
    """Run one deterministic MPC episode for the selected task."""

    run_mpc_only_mode(ctx.obj, alias=alias, no_visual=no_visual, console=console)


@app.command("intelligent")
def intelligent(
    ctx: typer.Context,
    alias: Annotated[
        str | None,
        typer.Option("--alias", help="Prefix the generated artifact directory."),
    ] = None,
    no_visual: Annotated[
        bool,
        typer.Option("--no-visual", help="Disable realtime diagnostics and animation."),
    ] = False,
    with_exploration: Annotated[
        bool,
        typer.Option("--with-exploration", help="Sample critic-scored actions probabilistically."),
    ] = False,
) -> None:
    """Deploy the learned critic to select an MPC burst-coast plan."""

    run_intelligent_mode(
        ctx.obj,
        alias=alias,
        no_visual=no_visual,
        with_exploration=with_exploration,
        console=console,
    )


@app.command("train")
def train(
    ctx: typer.Context,
    alias: Annotated[
        str | None,
        typer.Option("--alias", help="Prefix each generated epoch artifact directory."),
    ] = None,
    no_visual: Annotated[
        bool,
        typer.Option("--no-visual", help="Disable realtime diagnostics and animation."),
    ] = False,
) -> None:
    """Run online critic training through repeated RL-guided MPC episodes."""

    run_train_mode(ctx.obj, alias=alias, no_visual=no_visual, console=console)


@app.command("montecarlo")
def montecarlo(
    ctx: typer.Context,
    epochs: Annotated[
        int,
        typer.Option("--epochs", help="Generate this many independent Monte Carlo runs."),
    ] = 1,
) -> None:
    """Generate Monte Carlo transition data for offline critic training."""

    if epochs <= 0:
        msg = "--epochs must be positive"
        raise typer.BadParameter(msg)
    run_monte_carlo_mode(ctx.obj, epochs=epochs, console=console)


def main() -> None:
    """Launch the root command interface."""

    app()


if __name__ == "__main__":
    main()