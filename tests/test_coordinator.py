from wsmpc.coordinator import Coordinator
from wsmpc.utils.loaders import load_config


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


def test_coordinator_invokes_episode_callbacks() -> None:
    config = load_config()
    config.experiment.max_steps = 4
    config.environment.goal.hold_steps = 999
    config.environment.simulation.pace_s = 0.0
    coordinator = Coordinator(config.coordinator, config.environment, config.experiment)
    events: list[tuple[str, int]] = []

    result = coordinator.run_episode(
        on_episode_start=lambda observation: events.append(("start", observation.t_index)),
        on_step=lambda observation, record: events.append(("step", record.t_index)),
        on_episode_finish=lambda summary: events.append(("finish", summary.final_t_index)),
    )

    assert result.summary.records_emitted == 4
    assert events[0] == ("start", 0)
    assert [event for event in events if event[0] == "step"] == [
        ("step", 1),
        ("step", 2),
        ("step", 3),
        ("step", 4),
    ]
    assert events[-1] == ("finish", 4)


def test_coordinator_invokes_before_step_callback_before_each_step() -> None:
    config = load_config()
    config.experiment.max_steps = 3
    config.environment.goal.hold_steps = 999
    config.environment.simulation.pace_s = 0.0
    coordinator = Coordinator(config.coordinator, config.environment, config.experiment)
    before_steps: list[int] = []

    result = coordinator.run_episode(
        on_before_step=lambda observation: before_steps.append(observation.t_index),
    )

    assert result.summary.records_emitted == 3
    assert before_steps == [0, 1, 2]


def test_coordinator_returns_interrupted_summary_from_keyboard_interrupt() -> None:
    config = load_config()
    config.experiment.max_steps = 4
    config.environment.goal.hold_steps = 999
    config.environment.simulation.pace_s = 0.0
    coordinator = Coordinator(config.coordinator, config.environment, config.experiment)
    finished_statuses: list[str] = []

    def interrupt_after_first_step(observation, record) -> None:
        raise KeyboardInterrupt

    result = coordinator.run_episode(
        on_step=interrupt_after_first_step,
        on_episode_finish=lambda summary: finished_statuses.append(summary.status),
    )

    assert result.summary.status == "interrupted"
    assert result.summary.records_emitted == 1
    assert result.summary.final_observation is not None
    assert finished_statuses == ["interrupted"]
