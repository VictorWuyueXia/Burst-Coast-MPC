"""Monte Carlo data generation mode bring-up."""

from __future__ import annotations

import logging

from rich.console import Console

from burst_coast_mpc.data_generation import MonteCarloDataGenerator
from inverted_pendulum.utils.artifacts import (
    ArtifactWriter,
    attach_run_log_handler,
    detach_run_log_handler,
)
from inverted_pendulum.utils.config_schema import load_data_generation_config
from inverted_pendulum.utils.logging import configure_logging
from inverted_pendulum.visualization.artifact_plots import (
    create_artifact_figures,
    create_rl_timeseries_figure,
)


def run_monte_carlo_mode(task: str, *, epochs: int, console: Console) -> None:
    """Generate sequential Monte Carlo datasets for offline RL."""

    # 1. Load the dedicated Monte Carlo config and validate the one CLI sweep parameter.
    if task != "inverted_pendulum":
        msg = "Only the inverted pendulum task has a runtime implementation"
        raise NotImplementedError(msg)
    configure_logging()
    config = load_data_generation_config()
    if epochs <= 0:
        msg = "--epochs must be positive"
        raise ValueError(msg)
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
                "mode": "montecarlo",
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
            "mode": "montecarlo",
            "epochs": epochs,
            "rl_steps": total_rl_steps,
            "artifact_dirs": artifact_dirs,
        }
    )
