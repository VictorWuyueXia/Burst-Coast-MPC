"""RL critic plus MPC deployment mode bring-up."""

from __future__ import annotations

import logging

from rich.console import Console

from burst_coast_mpc.epoch_coordinator import EpochCoordinator
from inverted_pendulum.RL.policy import StructuredCriticPolicy
from inverted_pendulum.utils.artifacts import (
    ArtifactWriter,
    attach_run_log_handler,
    detach_run_log_handler,
)
from inverted_pendulum.utils.config_schema import load_config
from inverted_pendulum.utils.logging import (
    ThirdPersonObservers,
    configure_logging,
    episode_output,
)
from inverted_pendulum.visualization.artifact_plots import create_artifact_figures


def run_intelligent_mode(
    task: str,
    *,
    alias: str | None,
    no_visual: bool,
    with_exploration: bool,
    console: Console,
) -> None:
    """Run one RL-grid-selected MPC deployment episode."""

    # Load runtime config, frozen critic policy, and command logger.
    if task != "inverted_pendulum":
        msg = "Only the inverted pendulum task has a runtime implementation"
        raise NotImplementedError(msg)
    configure_logging()
    config = load_config()
    logger = logging.getLogger("burst_coast_mpc")
    if alias is not None:
        config.artifacts.alias = alias
    policy = StructuredCriticPolicy(config.rl, config.environment)

    # Open artifact streams before dense and RL records are emitted.
    artifact_writer: ArtifactWriter | None = None
    run_log_handler: logging.Handler | None = None
    if config.artifacts.enabled:
        artifact_writer = ArtifactWriter.create(
            config.artifacts.root_dir,
            alias=config.artifacts.alias,
            config_package="default-config",
            cli_args={
                "mode": "intelligent",
                "alias": alias,
                "no_visual": no_visual,
                "with_exploration": with_exploration,
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

    # Create realtime diagnostics only when the operator has not disabled visuals.
    realtime_plot = None
    if not no_visual:
        from inverted_pendulum.visualization.realtime import RealtimeEpisodePlot

        realtime_plot = RealtimeEpisodePlot(
            config.environment.pendulum,
            update_every=1,
            include_animation=True,
        )

    # Bridge coordinator events into the active visualization and artifact sinks.
    def observe_episode_start(observation) -> None:
        if realtime_plot is not None:
            realtime_plot.start_animation(observation)

    def observe_after_step(observation, record) -> None:
        if artifact_writer is not None:
            artifact_writer.write_step(record)
        if realtime_plot is not None:
            realtime_plot.add_step(observation, record)

    # Run critic-selected burst-horizon replanning with MPC torque solves.
    coordinator = EpochCoordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        config.mpc,
        logger=logger,
        rl_policy=policy,
        rl_config=config.rl,
        rl_explore=with_exploration,
    )
    result = coordinator.run_episode(
        ThirdPersonObservers(
            at_episode_start=observe_episode_start,
            after_step=observe_after_step,
        )
    )

    # Write dense episode artifacts, RL transitions, and figures.
    if artifact_writer is not None:
        from matplotlib import pyplot as plt

        artifact_writer.write_summary(result.summary)
        artifact_writer.write_rl_steps(result.rl_records)
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

    # Emit one concise terminal summary for scripts and human inspection.
    artifact_dir = None
    if artifact_writer is not None:
        artifact_dir = str(artifact_writer.run_dir)
    output = episode_output(result.summary, artifact_dir=artifact_dir)
    output["mode"] = "intelligent"
    output["with_exploration"] = with_exploration
    output["rl_steps"] = len(result.rl_records)
    console.print(output)
