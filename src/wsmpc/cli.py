"""Top-level command line entry point for wake-sleep MPC experiments."""

from __future__ import annotations

import logging
import select
import sys
import termios
import tty
from types import TracebackType
from typing import Annotated, Any

import typer
from rich.console import Console

from wsmpc.coordinator import Coordinator
from wsmpc.utils.artifacts import (
    ArtifactWriter,
    attach_run_log_handler,
    detach_run_log_handler,
)
from wsmpc.utils.loaders import STANDARD_PACKAGE, load_config
from wsmpc.utils.logging import EpisodeHooks, configure_logging, episode_output, run_episode
from wsmpc.utils.resources import configure_runtime_resources

app = typer.Typer(help="Wake-sleep MPC research CLI.")
console = Console()


class TerminalPauseController:
    """Nonblocking single-key pause and resume controls for interactive runs."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled and sys.stdin.isatty()
        self._paused = False
        self._original_terminal_attrs: list[Any] | None = None

    def __enter__(self) -> TerminalPauseController:
        if self.enabled:
            self._original_terminal_attrs = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            console.print("Controls: press 's' to pause simulation, 'r' to resume.")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.enabled and self._original_terminal_attrs is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._original_terminal_attrs)

    def wait_if_paused(self) -> None:
        if not self.enabled:
            return
        self._consume_pause_command()
        while self._paused:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            key = sys.stdin.read(1).lower()
            if key == "r":
                self._paused = False
                console.print("Simulation resumed. Press 's' to pause again.")

    def _consume_pause_command(self) -> None:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return
        key = sys.stdin.read(1).lower()
        if key == "s":
            self._paused = True
            console.print("Simulation paused. Press 'r' to resume.")


@app.callback()
def cli_root() -> None:
    """Wake-sleep MPC research CLI."""


def load_runtime_context(config_package: str):
    """Load config and configure logging."""

    configure_logging()
    config = load_config(config_package)
    logger = logging.getLogger("wsmpc")
    logger.debug(
        "identity=CLI status=running action=load_runtime_context "
        "action_result=config_validated config_package=%s",
        config_package,
    )
    return config, logger


@app.command("run-episode")
def run_episode_command(
    config_package: Annotated[
        str,
        typer.Option(
            "--config-package",
            "-c",
            help="Config package under configs/ to load.",
        ),
    ] = STANDARD_PACKAGE,
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
    """Run one MPC experiment episode."""

    config, logger = load_runtime_context(config_package)
    configure_runtime_resources(config.runtime, logger=logger)

    if alias is not None:
        config.artifacts.alias = alias

    artifact_writer: ArtifactWriter | None = None
    run_log_handler: logging.Handler | None = None
    if config.artifacts.enabled:
        artifact_writer = ArtifactWriter.create(
            config.artifacts.root_dir,
            alias=config.artifacts.alias,
            config_package=config_package,
            cli_args={
                "config_package": config_package,
                "alias": alias,
                "no_visual": no_visual,
            },
        )
        artifact_writer.write_config(config)
        artifact_writer.open_step_writer()
        run_log_handler = attach_run_log_handler(logger, artifact_writer.run_dir)
        logger.info(
            "identity=Artifacts status=initialized action=create_run_directory "
            "action_result=ready run_dir=%s",
            artifact_writer.run_dir,
        )

    realtime_plot = None
    if not no_visual:
        from wsmpc.visualization.realtime import RealtimeEpisodePlot

        realtime_plot = RealtimeEpisodePlot(
            config.environment.pendulum,
            phase_epsilon_phi=config.mpc.cost.epsilon_phi,
            update_every=1,
            include_animation=True,
        )

    coordinator = Coordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        config.mpc,
        config.runtime,
        logger=logger,
    )

    def on_episode_start(observation) -> None:
        if realtime_plot is not None:
            realtime_plot.start_animation(observation)

    pause_controller = TerminalPauseController(enabled=not no_visual)

    def on_before_step(observation) -> None:
        pause_controller.wait_if_paused()

    def on_step(observation, record) -> None:
        if artifact_writer is not None:
            artifact_writer.write_step(record)
        if realtime_plot is not None:
            realtime_plot.add_step(observation, record)

    def on_episode_finish(summary) -> None:
        if artifact_writer is not None:
            artifact_writer.write_summary(summary)
        if realtime_plot is not None:
            realtime_plot.finish()

    hooks = EpisodeHooks(
        on_episode_start=on_episode_start,
        on_before_step=on_before_step,
        on_step=on_step,
        on_episode_finish=on_episode_finish,
    )

    try:
        with pause_controller:
            result = run_episode(coordinator, hooks)
    except Exception:
        if artifact_writer is not None:
            artifact_writer.finalize_manifest(completed=False, status="failed")
        detach_run_log_handler(logger, run_log_handler)
        raise

    if artifact_writer is not None:
        artifact_writer.finalize_manifest(
            completed=result.summary.status != "interrupted",
            status=result.summary.status,
        )
    detach_run_log_handler(logger, run_log_handler)

    artifact_dir = str(artifact_writer.run_dir) if artifact_writer is not None else None
    console.print(episode_output(result.summary, artifact_dir=artifact_dir))


def main() -> None:
    """Console-script wrapper."""

    app()


if __name__ == "__main__":
    main()
