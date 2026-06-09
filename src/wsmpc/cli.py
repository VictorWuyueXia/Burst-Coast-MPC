"""Top-level command line entry point for wake-sleep MPC experiments."""

from __future__ import annotations

import logging
from typing import Annotated

import typer
from rich.console import Console

from wsmpc.coordinator import Coordinator
from wsmpc.data_generation import MonteCarloDataGenerator
from wsmpc.utils.artifacts import (
    ArtifactWriter,
    attach_run_log_handler,
    detach_run_log_handler,
)
from wsmpc.utils.config_schema import (
    DATA_GENERATION_PACKAGE,
    STANDARD_PACKAGE,
    load_config,
    load_data_generation_config,
)
from wsmpc.utils.logging import ThirdPersonObservers, configure_logging, episode_output
from wsmpc.utils.resources import configure_runtime_resources
from wsmpc.visualization.artifact_plots import create_artifact_figures

app = typer.Typer(help="Wake-sleep MPC research CLI.")
console = Console()


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

    def observe_episode_start(observation) -> None:
        if realtime_plot is not None:
            realtime_plot.start_animation(observation)

    def observe_after_step(observation, record) -> None:
        if artifact_writer is not None:
            artifact_writer.write_step(record)
        if realtime_plot is not None:
            realtime_plot.add_step(observation, record)

    def observe_episode_finish(summary) -> None:
        if artifact_writer is not None:
            artifact_writer.write_summary(summary)
        if realtime_plot is not None:
            realtime_plot.finish()

    third_person_observers = ThirdPersonObservers(
        at_episode_start=observe_episode_start,
        after_step=observe_after_step,
        at_episode_finish=observe_episode_finish,
    )

    result = coordinator.run_episode(third_person_observers)

    if artifact_writer is not None:
        from matplotlib import pyplot as plt

        figures = create_artifact_figures(result.records, config.environment)
        for name, figure in figures.items():
            artifact_writer.write_figure(name, figure)
        for figure in figures.values():
            plt.close(figure)
        artifact_writer.finalize_manifest(
            completed=result.summary.status != "interrupted",
            status=result.summary.status,
        )
    detach_run_log_handler(logger, run_log_handler)

    artifact_dir = str(artifact_writer.run_dir) if artifact_writer is not None else None
    console.print(episode_output(result.summary, artifact_dir=artifact_dir))


@app.command("generate-mc-data")
def generate_mc_data_command(
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            help="Independent Monte Carlo artifact runs to generate.",
        ),
    ] = 1,
) -> None:
    """Generate one sequential Monte Carlo dataset for offline RL."""

    configure_logging()
    config = load_data_generation_config()
    if epochs <= 0:
        msg = "--epochs must be positive"
        raise typer.BadParameter(msg)
    logger = logging.getLogger("wsmpc")
    configure_runtime_resources(config.runtime, logger=logger)

    artifact_dirs: list[str] = []
    total_rl_steps = 0
    base_alias = config.artifacts.alias
    for epoch_index in range(epochs):
        epoch_config = config.model_copy(deep=True)
        epoch_config.data_generation.seed = config.data_generation.seed + epoch_index
        epoch_config.experiment.episode_id = (
            config.experiment.episode_id + epoch_index * config.data_generation.episodes
        )
        if epochs > 1:
            alias_root = base_alias or DATA_GENERATION_PACKAGE
            epoch_config.artifacts.alias = f"{alias_root}-epoch-{epoch_index + 1}"

        artifact_writer = ArtifactWriter.create(
            epoch_config.artifacts.root_dir,
            alias=epoch_config.artifacts.alias,
            config_package=DATA_GENERATION_PACKAGE,
            cli_args={
                "config_package": DATA_GENERATION_PACKAGE,
                "epochs": epochs,
                "epoch_index": epoch_index,
            },
        )
        artifact_writer.write_config(epoch_config)
        artifact_writer.open_step_writer()
        run_log_handler = attach_run_log_handler(logger, artifact_writer.run_dir)
        logger.info(
            "identity=Artifacts status=initialized action=create_run_directory "
            "action_result=ready run_dir=%s",
            artifact_writer.run_dir,
        )

        generator = MonteCarloDataGenerator(epoch_config, logger)
        step_records, rl_records = generator.run(artifact_writer)
        artifact_writer.write_rl_steps(rl_records)
        if epoch_config.data_generation.visual_artifacts:
            from matplotlib import pyplot as plt

            figures = create_artifact_figures(step_records, epoch_config.environment)
            for name, figure in figures.items():
                artifact_writer.write_figure(name, figure)
            for figure in figures.values():
                plt.close(figure)
        artifact_writer.finalize_manifest(completed=True, status="monte_carlo_generated")
        detach_run_log_handler(logger, run_log_handler)
        artifact_dirs.append(str(artifact_writer.run_dir))
        total_rl_steps += len(rl_records)

    console.print(
        {
            "run_id": config.experiment.run_id,
            "epochs": epochs,
            "rl_steps": total_rl_steps,
            "artifact_dirs": artifact_dirs,
        }
    )


def main() -> None:
    """Console-script wrapper."""

    app()


if __name__ == "__main__":
    main()
