"""Top-level command line entry point for wake-sleep MPC experiments."""

from __future__ import annotations

import logging
import select
import sys
import termios
import tty
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any

import typer
from rich.console import Console

from wsmpc.utils.artifacts import (
    ArtifactWriter,
    attach_run_log_handler,
    detach_run_log_handler,
)
from wsmpc.utils.config_schema import RootConfig
from wsmpc.utils.loaders import STANDARD_PACKAGE, load_config_with_fallbacks, warn_default_fallbacks
from wsmpc.utils.logging import configure_logging
from wsmpc.utils.resources import configure_runtime_resources

app = typer.Typer(help="Wake-sleep MPC research CLI.")
console = Console()


class TerminalPauseController:
    """Nonblocking single-key pause and resume controls for interactive runs."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled and sys.stdin.isatty()
        self._paused = False
        self._original_terminal_attrs: list[Any] | None = None

    def __enter__(self) -> TerminalPauseController:
        """Enable cbreak input so single-key commands do not need Enter."""

        if self.enabled:
            # Save original terminal state and set terminal to cbreak (for single key detection)
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
        """Restore the user's terminal mode after the run finishes."""

        if self.enabled and self._original_terminal_attrs is not None:
            # Restore original terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._original_terminal_attrs)

    def wait_if_paused(self) -> None:
        """Poll for pause commands and block until resume when paused."""

        if not self.enabled:
            return
        # Check if user wants to pause the simulation
        self._consume_pause_command()
        while self._paused:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            # Resume if 'r' is pressed
            key = sys.stdin.read(1).lower()
            if key == "r":
                self._paused = False
                console.print("Simulation resumed. Press 's' to pause again.")

    def _consume_pause_command(self) -> None:
        """Process any pending terminal keypress before a simulator step."""

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return
        # Pause if 's' is pressed
        key = sys.stdin.read(1).lower()
        if key == "s":
            self._paused = True
            console.print("Simulation paused. Press 'r' to resume.")


@app.callback()
def cli_root() -> None:
    """Wake-sleep MPC research CLI."""


def load_checked_runtime_context(
    config_package: str = STANDARD_PACKAGE,
) -> tuple[RootConfig, logging.Logger]:
    """Load config, validate schema, and configure logging."""

    # Load experiment configuration from package (with fallback to defaults if needed)
    config, fallbacks = load_config_with_fallbacks(config_package)

    # Configure logging for the experiment run
    configure_logging(config.logging.level)
    logger = logging.getLogger("wsmpc")
    warn_default_fallbacks(logger, fallbacks)
    logger.debug(
        "identity=CLI status=running action=load_checked_runtime_context "
        "action_result=config_validated config_package=%s",
        config_package,
    )

    return config, logger


@app.command("run-episode")
def run_episode(
    config_package: Annotated[
        str,
        typer.Option(
            "--config-package",
            "-c",
            help="Config package under configs/ to load.",
        ),
    ] = STANDARD_PACKAGE,
    artifact_root: Annotated[
        Path | None,
        typer.Option(
            "--artifact-root",
            help="Directory where experiment artifact run folders are created.",
        ),
    ] = None,
    run_alias: Annotated[
        str | None,
        typer.Option(
            "--run-alias",
            help="Human-readable alias prefix for the artifact run directory.",
        ),
    ] = None,
    no_artifacts: Annotated[
        bool,
        typer.Option(
            "--no-artifacts",
            help="Run without writing experiment artifacts.",
        ),
    ] = False,
    visualize: Annotated[
        bool,
        typer.Option(
            "--visualize",
            help="Show realtime Matplotlib diagnostics during the episode.",
        ),
    ] = False,
    animate: Annotated[
        bool,
        typer.Option(
            "--animate",
            help="Show a realtime Matplotlib pendulum animation during the episode.",
        ),
    ] = False,
    viz_update_every: Annotated[
        int,
        typer.Option(
            "--viz-update-every",
            min=1,
            help="Update visualization windows every N emitted records.",
        ),
    ] = 1,
) -> None:
    """Run one experiment episode."""

    # Load config and logger first
    config, logger = load_checked_runtime_context(config_package)

    # Set core resource usage limits (threads, etc) before heavy imports
    configure_runtime_resources(config.runtime, logger=logger)

    # Import Coordinator for experiment logic after resource limits are set
    from wsmpc.coordinator import Coordinator

    # Set custom artifact directory and alias if specified
    if artifact_root is not None:
        config.artifacts.root_dir = str(artifact_root)
    if run_alias is not None:
        config.artifacts.alias = run_alias

    # Prepare artifact writer if enabled
    artifact_writer: ArtifactWriter | None = None
    run_log_handler: logging.Handler | None = None
    artifacts_enabled = config.artifacts.enabled and not no_artifacts
    cli_args = {
        "config_package": config_package,
        "artifact_root": str(artifact_root) if artifact_root is not None else None,
        "run_alias": run_alias,
        "no_artifacts": no_artifacts,
        "visualize": visualize,
        "animate": animate,
        "viz_update_every": viz_update_every,
    }
    if artifacts_enabled:
        # Create artifact directory and record config/command metadata for this run
        artifact_writer = ArtifactWriter.create(
            config.artifacts.root_dir,
            alias=config.artifacts.alias,
            config_package=config_package,
            cli_args=cli_args,
        )
        artifact_writer.write_config(config)
        artifact_writer.open_step_writer()
        # Attach log handler to save run log as artifact
        run_log_handler = attach_run_log_handler(logger, artifact_writer.run_dir)
        logger.info(
            "identity=Artifacts status=initialized action=create_run_directory "
            "action_result=ready run_dir=%s",
            artifact_writer.run_dir,
        )

    # Initialize visualization objects if requested
    realtime_plot = None
    pendulum_animation = None
    if visualize or animate:
        try:
            from wsmpc.visualization.realtime import PendulumAnimation, RealtimeEpisodePlot
        except ImportError as exc:
            # Show error if visualization dependencies are missing
            raise typer.BadParameter(
                "visualization requires matplotlib; install project dependencies first"
            ) from exc

        if visualize:
            # Live plot for diagnostics
            realtime_plot = RealtimeEpisodePlot(
                config.environment.pendulum,
                update_every=viz_update_every,
                include_animation=animate,
            )
        if animate and realtime_plot is None:
            # Animated visualization (pendulum movement)
            pendulum_animation = PendulumAnimation(
                config.environment.pendulum,
                update_every=viz_update_every,
            )

    # Instantiate Coordinator (controls the experiment/episode)
    coordinator = Coordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        logger=logger,
    )
    action_provider = None
    if config.mpc.enabled:
        # Enable MPC controller if specified in config
        from wsmpc.mpc.controller import CasadiMPCController
        from wsmpc.mpc.types import MPCControllerContext

        mpc_controller = CasadiMPCController(
            MPCControllerContext(
                environment=config.environment,
                experiment=config.experiment,
                mpc=config.mpc,
                runtime=config.runtime,
            ),
            logger=logger,
        )
        action_provider = mpc_controller.select_action

    def on_episode_start(observation) -> None:
        """Initialize live animation state from the reset observation."""

        # Start visualization from initial observation (if enabled)
        if realtime_plot is not None:
            realtime_plot.start_animation(observation)
        elif pendulum_animation is not None:
            pendulum_animation.start(observation)

    def on_before_step(observation) -> None:
        """Apply interactive terminal controls before advancing simulated time."""

        # Pause simulation if user requests via terminal key
        pause_controller.wait_if_paused()

    def on_step(observation, record) -> None:
        """Fan completed environment transitions out to active subscribers."""

        # Record data step as artifact and update visualization
        if artifact_writer is not None:
            artifact_writer.write_step(record)
        if realtime_plot is not None:
            realtime_plot.add_step(observation, record)
        elif pendulum_animation is not None:
            pendulum_animation.add_step(observation, record)

    def on_episode_finish(summary) -> None:
        """Write final artifacts and flush visualization state."""

        # Finalize and write summary of episode (for artifacts/visuals)
        if artifact_writer is not None:
            artifact_writer.write_summary(summary)
        if realtime_plot is not None:
            realtime_plot.finish()
        elif pendulum_animation is not None:
            pendulum_animation.finish()

    try:
        # Run the entire episode with callbacks for logging, pause, and visualization
        with TerminalPauseController() as pause_controller:
            result = coordinator.run_episode(
                action_provider=action_provider,
                on_episode_start=on_episode_start,
                on_before_step=on_before_step,
                on_step=on_step,
                on_episode_finish=on_episode_finish,
            )
    except Exception:
        # On error: mark run as failed and detach artifact log handler
        if artifact_writer is not None:
            artifact_writer.finalize_manifest(completed=False, status="failed")
        detach_run_log_handler(logger, run_log_handler)
        raise
    # Finalize run manifests and detach log handlers after successful run
    if artifact_writer is not None:
        artifact_writer.finalize_manifest(
            completed=result.summary.status != "interrupted",
            status=result.summary.status,
        )
    detach_run_log_handler(logger, run_log_handler)

    # Print a short summary of the run to the terminal
    output = {
        "run_id": result.summary.run_id,
        "status": result.summary.status,
        "total_steps": result.summary.total_steps,
        "final_t_sec": round(result.summary.final_t_sec, 6),
        "goal_reached": result.summary.goal_reached,
    }
    if artifact_writer is not None:
        output["artifact_dir"] = str(artifact_writer.run_dir)
    console.print(output)


def main() -> None:
    """Console-script wrapper."""

    # Launch the CLI application (processes command line)
    app()


if __name__ == "__main__":
    main()
