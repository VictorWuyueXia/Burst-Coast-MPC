"""Online RL training mode bring-up."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from bringup.epoch_coordinator import EpochCoordinator
from inverted_pendulum.RL.online_training import OnlinePolicyTrainer
from inverted_pendulum.RL.policy import StructuredCriticPolicy
from inverted_pendulum.utils.artifacts import (
    ArtifactWriter,
    attach_run_log_handler,
    detach_run_log_handler,
)
from inverted_pendulum.utils.config_schema import (
    load_online_training_config,
    load_visualization_config,
)
from inverted_pendulum.utils.logging import (
    ThirdPersonObservers,
    configure_logging,
    episode_output,
)
from inverted_pendulum.utils.messages import StateObs, StepRecord
from inverted_pendulum.visualization.artifact_plots import create_artifact_figures

if TYPE_CHECKING:
    from inverted_pendulum.visualization.realtime import RealtimeEpisodePlot


def run_train_mode(
    task: str,
    *,
    alias: str | None,
    no_visual: bool,
    console: Console,
) -> None:
    """Run exploratory RL+MPC epochs and fine-tune the critic after each episode."""

    # Load runtime config, frozen critic policy, and command logger.
    if task != "inverted_pendulum":
        msg = "Only the inverted pendulum task has a runtime implementation"
        raise NotImplementedError(msg)
    configure_logging()
    config = load_online_training_config()
    logger = logging.getLogger("bcmpc")
    if alias is not None:
        config.artifacts.alias = alias
    policy = StructuredCriticPolicy(config.rl, config.environment, config.mpc)
    visualization = None if no_visual else load_visualization_config()
    base_alias = config.artifacts.alias
    if config.artifacts.enabled and config.rl.training_epochs > 1 and base_alias is None:
        msg = "Multi-epoch online training requires a configured artifact alias"
        raise ValueError(msg)
    epoch_outputs: list[dict[str, object]] = []
    total_rl_steps = 0
    snapshot_dir: Path | None = None

    # Repeat full online episodes while carrying the same updated critic forward.
    for epoch_index in range(config.rl.training_epochs):
        epoch_config = config.model_copy(deep=True)
        epoch_config.experiment.episode_id = config.experiment.episode_id + epoch_index
        if config.rl.training_epochs > 1:
            epoch_config.artifacts.alias = f"{base_alias}-epoch-{epoch_index + 1}"

        # Open artifact streams before dense and RL records are emitted.
        artifact_writer: ArtifactWriter | None = None
        run_log_handler: logging.Handler | None = None
        if epoch_config.artifacts.enabled:
            artifact_writer = ArtifactWriter.create(
                epoch_config.artifacts.root_dir,
                alias=epoch_config.artifacts.alias,
                config_package="online-training",
                cli_args={
                    "mode": "train",
                    "alias": alias,
                    "no_visual": no_visual,
                    "training_epochs": config.rl.training_epochs,
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

        # Create realtime diagnostics only when the operator has not disabled visuals.
        realtime_plot: RealtimeEpisodePlot | None = None
        if visualization is not None:
            from inverted_pendulum.visualization.realtime import RealtimeEpisodePlot

            realtime_plot = RealtimeEpisodePlot(
                epoch_config.environment.pendulum,
                update_every=visualization.update_every,
                include_animation=True,
            )

        # Bridge coordinator events into the active visualization and artifact sinks.
        def observe_episode_start(
            observation: StateObs,
            realtime_plot: RealtimeEpisodePlot | None = realtime_plot,
        ) -> None:
            if realtime_plot is not None:
                realtime_plot.start_animation(observation)

        def observe_after_step(
            observation: StateObs,
            record: StepRecord,
            artifact_writer: ArtifactWriter | None = artifact_writer,
            realtime_plot: RealtimeEpisodePlot | None = realtime_plot,
        ) -> None:
            if artifact_writer is not None:
                artifact_writer.write_step(record)
            if realtime_plot is not None:
                realtime_plot.add_step(observation, record)

        # Run exploratory cost-softmax replanning and collect return-labeled transitions.
        coordinator = EpochCoordinator(
            epoch_config.coordinator,
            epoch_config.environment,
            epoch_config.experiment,
            epoch_config.mpc,
            logger=logger,
            rl_policy=policy,
            rl_config=epoch_config.rl,
            rl_explore=True,
        )
        result = coordinator.run_episode(
            ThirdPersonObservers(
                at_episode_start=observe_episode_start,
                after_step=observe_after_step,
            )
        )

        # Write episode artifacts before fitting the online update snapshot.
        if artifact_writer is not None:
            from matplotlib import pyplot as plt

            artifact_writer.write_summary(result.summary)
            artifact_writer.write_rl_steps(result.rl_records)
            figures = create_artifact_figures(result.records, epoch_config.environment)
            for name, figure in figures.items():
                artifact_writer.write_figure(name, figure)
            for figure in figures.values():
                plt.close(figure)
            artifact_writer.finalize_manifest(completed=True, status=result.summary.status)
            assert run_log_handler is not None
            detach_run_log_handler(logger, run_log_handler)
        if realtime_plot is not None:
            realtime_plot.finish()

        # Fine-tune the carried critic in place so the next epoch starts from this update.
        snapshot_dir = OnlinePolicyTrainer(policy, epoch_config.rl).fit(result.rl_records)
        artifact_dir = None
        if artifact_writer is not None:
            artifact_dir = str(artifact_writer.run_dir)
        epoch_output = episode_output(result.summary, artifact_dir=artifact_dir)
        epoch_output["epoch"] = epoch_index + 1
        epoch_output["rl_steps"] = len(result.rl_records)
        epoch_output["snapshot_dir"] = str(snapshot_dir)
        epoch_outputs.append(epoch_output)
        total_rl_steps += len(result.rl_records)

    assert snapshot_dir is not None
    output = {
        "mode": "train",
        "training_epochs": config.rl.training_epochs,
        "rl_steps": total_rl_steps,
        "snapshot_dir": str(snapshot_dir),
        "epoch_artifact_dirs": [
            item["artifact_dir"] for item in epoch_outputs if "artifact_dir" in item
        ],
    }
    console.print(output)
