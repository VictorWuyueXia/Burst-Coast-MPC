"""Top-level command line entry point for wake-sleep MPC experiments."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from wsmpc.config.loaders import load_config
from wsmpc.core.logging import configure_logging
from wsmpc.core.resources import configure_runtime_resources


app = typer.Typer(help="Wake-sleep MPC research CLI.")
console = Console()


@app.command("run-episode")
def run_episode(
    config_name: str = typer.Option("config", "--config-name", help="Root config selector name."),
    config_dir: Path | None = typer.Option(None, "--config-dir", help="Directory containing configs."),
    experiment: str | None = typer.Option(None, "--experiment", help="Experiment config name."),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Override experiment max steps."),
    pace_s: float | None = typer.Option(None, "--pace-s", help="Override environment pace-s."),
    log_level: str | None = typer.Option(None, "--log-level", help="Override logging level."),
) -> None:
    """Run one deterministic episode."""

    # Convert CLI conveniences into OmegaConf dot-list overrides.
    overrides: list[str] = []
    if max_steps is not None:
        overrides.append(f"experiment.max-steps={max_steps}")
    if pace_s is not None:
        overrides.append(f"environment.simulation.pace-s={pace_s}")
    if log_level is not None:
        overrides.append(f"logging.level={log_level}")

    # Load and validate config before importing simulation-heavy runtime modules.
    config = load_config(
        config_name=config_name,
        config_dir=config_dir,
        experiment_name=experiment,
        overrides=overrides,
    )
    configure_logging(config.logging.level)
    logger = logging.getLogger("wsmpc")

    # Apply resource limits before constructing the Environment, which imports NumPy math paths.
    configure_runtime_resources(config.runtime, logger=logger)

    # Import after resource limiting so native math libraries see conservative thread settings.
    from wsmpc.coordinator import Coordinator

    coordinator = Coordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        logger=logger,
    )
    result = coordinator.run_episode()

    # Print a short human-facing summary while detailed traces remain in logs.
    console.print(
        {
            "run_id": result.summary.run_id,
            "status": result.summary.status,
            "total_steps": result.summary.total_steps,
            "final_t_sec": round(result.summary.final_t_sec, 6),
            "goal_reached": result.summary.goal_reached,
        }
    )


def main() -> None:
    """Console-script wrapper."""

    app()


if __name__ == "__main__":
    main()
