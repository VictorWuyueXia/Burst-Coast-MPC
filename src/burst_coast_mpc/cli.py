"""Top-level command line entry point for Burst-Coast MPC experiments."""

from __future__ import annotations

import logging
from typing import Annotated

import typer
from rich.console import Console

from burst_coast_mpc.coordinator import Coordinator
from burst_coast_mpc.data_generation import MonteCarloDataGenerator
from inverted_pendulum.utils.artifacts import (
    ArtifactWriter,
    attach_run_log_handler,
    detach_run_log_handler,
)
from inverted_pendulum.utils.config_schema import (
    load_config,
    load_data_generation_config,
)
from inverted_pendulum.utils.logging import (
    ThirdPersonObservers,
    configure_logging,
    episode_output,
)
from inverted_pendulum.visualization.artifact_plots import (
    create_artifact_figures,
    create_rl_timeseries_figure,
)

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
        run_episode_command(ctx)
        raise typer.Exit()


@app.command("simulate")
def run_episode_command(
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
    """Run one MPC experiment episode."""

    # 1. Load the fixed runtime config and bind one logger for the command.
    if ctx.obj != "inverted_pendulum":
        msg = "Only the inverted pendulum task has a runtime implementation"
        raise NotImplementedError(msg)
    configure_logging()
    config = load_config()
    logger = logging.getLogger("burst_coast_mpc")
    if alias is not None:
        config.artifacts.alias = alias

    # 2. Open artifact streams before the coordinator emits per-step records.
    artifact_writer: ArtifactWriter | None = None
    run_log_handler: logging.Handler | None = None
    if config.artifacts.enabled:
        artifact_writer = ArtifactWriter.create(
            config.artifacts.root_dir,
            alias=config.artifacts.alias,
            config_package="default-config",
            cli_args={
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

    # 3. Create realtime diagnostics only when the operator has not disabled visuals.
    realtime_plot = None
    if not no_visual:
        from inverted_pendulum.visualization.realtime import RealtimeEpisodePlot

        realtime_plot = RealtimeEpisodePlot(
            config.environment.pendulum,
            update_every=1,
            include_animation=True,
        )

    # 4. Bridge coordinator events into the active visualization and artifact sinks.
    def observe_episode_start(observation) -> None:
        if realtime_plot is not None:
            realtime_plot.start_animation(observation)

    def observe_after_step(observation, record) -> None:
        if artifact_writer is not None:
            artifact_writer.write_step(record)
        if realtime_plot is not None:
            realtime_plot.add_step(observation, record)

    # 5. Run the coordinator-owned episode loop with only the required observers.
    coordinator = Coordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        config.mpc,
        logger=logger,
    )
    result = coordinator.run_episode(
        ThirdPersonObservers(
            at_episode_start=observe_episode_start,
            after_step=observe_after_step,
        )
    )

    # 6. Write post-run summary and figures after the complete record list is known.
    if artifact_writer is not None:
        from matplotlib import pyplot as plt

        artifact_writer.write_summary(result.summary)
        figures = create_artifact_figures(result.records, config.environment)
        for name, figure in figures.items():
            artifact_writer.write_figure(name, figure)
        for figure in figures.values():
            plt.close(figure)
        artifact_writer.finalize_manifest(completed=True, status=result.summary.status)
        assert run_log_handler is not None
        detach_run_log_handler(logger, run_log_handler)
    if realtime_plot is not None:
        realtime_plot.finish()

    # 7. Emit one concise terminal summary for scripts and human inspection.
    artifact_dir = None
    if artifact_writer is not None:
        artifact_dir = str(artifact_writer.run_dir)
    console.print(episode_output(result.summary, artifact_dir=artifact_dir))


@app.command("montecarlo")
def generate_mc_data_command(
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

    # 1. Load the dedicated Monte Carlo config and validate the one CLI sweep parameter.
    if ctx.obj != "inverted_pendulum":
        msg = "Only the inverted pendulum task has a runtime implementation"
        raise NotImplementedError(msg)
    configure_logging()
    config = load_data_generation_config()
    if epochs <= 0:
        msg = "--epochs must be positive"
        raise typer.BadParameter(msg)
    logger = logging.getLogger("burst_coast_mpc")

    artifact_dirs: list[str] = []
    total_rl_steps = 0
    base_alias = config.artifacts.alias
    for epoch_index in range(epochs):
        # 2. Derive the epoch-local config so seeds, IDs, and aliases remain disjoint.
        epoch_config = config.model_copy(deep=True)
        if config.data_generation.seed is not None:
            epoch_config.data_generation.seed = config.data_generation.seed + epoch_index
        epoch_config.experiment.episode_id = (
            config.experiment.episode_id + epoch_index * config.data_generation.episodes
        )
        if epochs > 1:
            if base_alias is None:
                epoch_config.artifacts.alias = f"data-generation-epoch-{epoch_index + 1}"
            else:
                epoch_config.artifacts.alias = f"{base_alias}-epoch-{epoch_index + 1}"

        # 3. Open one artifact run before dense simulation records are generated.
        artifact_writer = ArtifactWriter.create(
            epoch_config.artifacts.root_dir,
            alias=epoch_config.artifacts.alias,
            config_package="data-generation-config",
            cli_args={
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

        # 4. Generate dense step records plus return-labeled RL transition rows.
        generator = MonteCarloDataGenerator(epoch_config, logger)
        step_records, rl_records = generator.run(artifact_writer)
        artifact_writer.write_rl_steps(rl_records)
        from matplotlib import pyplot as plt

        # 5. Save RL diagnostics first, then optional dense episode diagnostics.
        rl_figure = create_rl_timeseries_figure(rl_records)
        artifact_writer.write_figure("rl_timeseries", rl_figure)
        plt.close(rl_figure)
        if epoch_config.data_generation.visual_artifacts:
            figures = create_artifact_figures(step_records, epoch_config.environment)
            for name, figure in figures.items():
                artifact_writer.write_figure(name, figure)
            for figure in figures.values():
                plt.close(figure)
        artifact_writer.finalize_manifest(completed=True, status="monte_carlo_generated")
        detach_run_log_handler(logger, run_log_handler)
        artifact_dirs.append(str(artifact_writer.run_dir))
        total_rl_steps += len(rl_records)

    # 6. Report the generated artifact roots and row count for downstream scripts.
    console.print(
        {
            "run_id": config.experiment.run_id,
            "epochs": epochs,
            "rl_steps": total_rl_steps,
            "artifact_dirs": artifact_dirs,
        }
    )


def main() -> None:
    """Run the Typer command application."""

    app()


if __name__ == "__main__":
    main()
