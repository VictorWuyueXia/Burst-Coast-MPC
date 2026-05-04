"""Top-level command line entry point for wake-sleep MPC experiments."""

from __future__ import annotations

import logging

import typer
from rich.console import Console

from wsmpc.utils.loaders import STANDARD_PACKAGE, load_config_with_fallbacks, warn_default_fallbacks
from wsmpc.utils.logging import configure_logging
from wsmpc.utils.resources import configure_runtime_resources
from wsmpc.utils.config_schema import RootConfig


app = typer.Typer(help="Wake-sleep MPC research CLI.")
console = Console()


def load_checked_runtime_context(config_package: str = STANDARD_PACKAGE) -> tuple[RootConfig, logging.Logger]:
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
    config_package: str = typer.Option(
        STANDARD_PACKAGE,
        "--config-package",
        "-c",
        help="Config package under configs/ to load.",
    ),
) -> None:
    """Run one experiment episode."""

    # Load config and logger first.
    config, logger = load_checked_runtime_context(config_package)

    # Apply conservative resource limits once per command before simulation-heavy imports.
    configure_runtime_resources(config.runtime, logger=logger)

    # Import after resource limiting so native math libraries see conservative thread settings.
    from wsmpc.coordinator import Coordinator

    # Instantiate the Coordinator with the loaded atomic config package.
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
