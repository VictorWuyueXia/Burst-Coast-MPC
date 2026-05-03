from pathlib import Path

from wsmpc.config.loaders import load_config
from wsmpc.coordinator import Coordinator


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_coordinator_runs_short_baseline_episode() -> None:
    config = load_config(
        config_dir=CONFIG_DIR,
        overrides=[
            "experiment.max-steps=3",
            "environment.goal.hold-steps=999",
            "environment.simulation.pace-s=0.0",
        ],
    )
    coordinator = Coordinator(config.coordinator, config.environment, config.experiment)

    result = coordinator.run_episode()

    assert result.summary.status == "max_steps_reached"
    assert result.summary.total_steps == 3
    assert result.summary.records_emitted == 3
