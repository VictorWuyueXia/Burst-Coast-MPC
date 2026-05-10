"""Top-level command line entry point for wake-sleep MPC experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

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


@app.callback()
def cli_root() -> None:
    """Wake-sleep MPC research CLI."""


def load_checked_runtime_context(
    config_package: str = STANDARD_PACKAGE,
) -> tuple[RootConfig, logging.Logger]:
    """Load config, validate schema, and configure logging."""

    # Load the selected atomic config package; package choice is the only config CLI knob.
    config, fallbacks = load_config_with_fallbacks(config_package)

    # Configure logs immediately after schema validation so later setup steps are visible.
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

    # Load config and logger first.
    config, logger = load_checked_runtime_context(config_package)

    # Apply conservative resource limits once per command before simulation-heavy imports.
    configure_runtime_resources(config.runtime, logger=logger)

    # Import after resource limiting so native math libraries see conservative thread settings.
    from wsmpc.coordinator import Coordinator

    # set the artifact root and alias
    if artifact_root is not None:
        config.artifacts.root_dir = str(artifact_root)
    if run_alias is not None:
        config.artifacts.alias = run_alias

    # create the artifact writer 
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
        artifact_writer = ArtifactWriter.create(
            config.artifacts.root_dir,
            alias=config.artifacts.alias,
            config_package=config_package,
            cli_args=cli_args,
        )
        artifact_writer.write_config(config)
        artifact_writer.open_step_writer()
        run_log_handler = attach_run_log_handler(logger, artifact_writer.run_dir)
        logger.info(
            "identity=Artifacts status=initialized action=create_run_directory "
            "action_result=ready run_dir=%s",
            artifact_writer.run_dir,
        )

    # create the visualization state
    realtime_plot = None
    pendulum_animation = None
    if visualize or animate:
        try:
            from wsmpc.visualization.realtime import PendulumAnimation, RealtimeEpisodePlot
        except ImportError as exc:
            raise typer.BadParameter(
                "visualization requires matplotlib; install project dependencies first"
            ) from exc

        if visualize:
            realtime_plot = RealtimeEpisodePlot(update_every=viz_update_every)
        if animate:
            pendulum_animation = PendulumAnimation(
                config.environment.pendulum,
                update_every=viz_update_every,
            )

    # Instantiate the Coordinator with the loaded atomic config package.
    coordinator = Coordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        logger=logger,
    )

    def on_episode_start(observation) -> None:
        """Initialize live animation state from the reset observation."""

        if pendulum_animation is not None:
            pendulum_animation.start(observation)

    def on_step(observation, record) -> None:
        """Fan completed environment transitions out to active subscribers."""

        if artifact_writer is not None:
            artifact_writer.write_step(record)
        if realtime_plot is not None:
            realtime_plot.add_step(observation, record)
        if pendulum_animation is not None:
            pendulum_animation.add_step(observation, record)

    def on_episode_finish(summary) -> None:
        """Write final artifacts and flush visualization state."""

        if artifact_writer is not None:
            artifact_writer.write_summary(summary)
        if realtime_plot is not None:
            realtime_plot.finish()
        if pendulum_animation is not None:
            pendulum_animation.finish()

    try:
        # run the episode with the callbacks
        result = coordinator.run_episode(
            on_episode_start=on_episode_start,
            on_step=on_step,
            on_episode_finish=on_episode_finish,
        )
    except Exception:
        if artifact_writer is not None:
            artifact_writer.finalize_manifest(completed=False, status="failed")
        detach_run_log_handler(logger, run_log_handler)
        raise
    # finalize the artifacts and detach the log handler
    if artifact_writer is not None:
        artifact_writer.finalize_manifest(completed=True, status=result.summary.status)
    detach_run_log_handler(logger, run_log_handler)

    # Print a short human-facing summary while detailed traces remain in logs.
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

    app()


if __name__ == "__main__":
    main()
