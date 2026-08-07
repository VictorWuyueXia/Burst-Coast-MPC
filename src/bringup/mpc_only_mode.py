"""MPC-only simulation mode bring-up."""

from __future__ import annotations

import logging

from rich.console import Console

from bringup.epoch_coordinator import EpochCoordinator
from inverted_pendulum.utils.artifacts import (
    ArtifactWriter,
    attach_run_log_handler,
    detach_run_log_handler,
)
from inverted_pendulum.utils.config_schema import (
    load_mpc_only_config,
    load_visualization_config,
)
from inverted_pendulum.utils.logging import (
    ThirdPersonObservers,
    configure_logging,
    episode_output,
)
from inverted_pendulum.utils.messages import StateObs, StepRecord
from inverted_pendulum.visualization.artifact_plots import create_artifact_figures


def run_mpc_only_mode(
    task: str,
    *,
    alias: str | None,
    no_visual: bool,
    console: Console,
) -> None:
    """Run one independent MPC-only episode."""

    # 1. Load the fixed runtime config and bind one logger for the command.
    if task != "inverted_pendulum":
        msg = "Only the inverted pendulum task has a runtime implementation"
        raise NotImplementedError(msg)
    configure_logging()
    config = load_mpc_only_config()
    logger = logging.getLogger("bcmpc")
    if alias is not None:
        config.artifacts.alias = alias

    # 2. Open artifact streams before the coordinator emits per-step records.
    artifact_writer: ArtifactWriter | None = None
    run_log_handler: logging.Handler | None = None
    if config.artifacts.enabled:
        artifact_writer = ArtifactWriter.create(
            config.artifacts.root_dir,
            alias=config.artifacts.alias,
            config_package="mpc-only",
            cli_args={
                "mode": "mpc-only",
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

        visualization = load_visualization_config()
        realtime_plot = RealtimeEpisodePlot(
            config.environment.pendulum,
            update_every=visualization.update_every,
            include_animation=True,
        )

    # 4. Bridge coordinator events into the active visualization and artifact sinks.
    def observe_episode_start(observation: StateObs) -> None:
        if realtime_plot is not None:
            realtime_plot.start_animation(observation)

    def observe_after_step(observation: StateObs, record: StepRecord) -> None:
        if artifact_writer is not None:
            artifact_writer.write_step(record)
        if realtime_plot is not None:
            realtime_plot.add_step(observation, record)

    # 5. Run the epoch coordinator with the normal MPC candidate selection path.
    coordinator = EpochCoordinator(
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
    output = episode_output(result.summary, artifact_dir=artifact_dir)
    output["mode"] = "mpc-only"
    console.print(output)
