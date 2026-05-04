from wsmpc.utils.loaders import load_config
from wsmpc.coordinator import Coordinator


def test_coordinator_runs_short_baseline_episode() -> None:
    config = load_config()
    config.experiment.max_steps = 3
    config.environment.goal.hold_steps = 999
    config.environment.simulation.pace_s = 0.0
    coordinator = Coordinator(config.coordinator, config.environment, config.experiment)

    result = coordinator.run_episode()

    assert result.summary.status == "max_steps_reached"
    assert result.summary.total_steps == 3
    assert result.summary.records_emitted == 3
