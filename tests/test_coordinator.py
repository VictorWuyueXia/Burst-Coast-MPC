import logging

from wsmpc.coordinator import Coordinator
from wsmpc.utils.loaders import load_config
from wsmpc.utils.logging import EpisodeHooks, run_episode


def _fast_coordinator(config) -> Coordinator:
    config.experiment.max_steps = 3
    config.environment.goal.hold_steps = 999
    config.environment.simulation.pace_s = 0.0
    config.environment.simulation.timestep_s = 0.25
    config.runtime.max_worker_threads = 1
    return Coordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        config.mpc,
        config.runtime,
        logger=logging.getLogger("test"),
    )


def test_coordinator_runs_short_mpc_episode() -> None:
    coordinator = _fast_coordinator(load_config("standard"))

    result = run_episode(coordinator)

    assert result.summary.status == "max_steps_reached"
    assert result.summary.total_steps == 3
    assert result.summary.records_emitted == 3


def test_coordinator_invokes_episode_hooks() -> None:
    config = load_config("standard")
    config.experiment.max_steps = 4
    config.environment.goal.hold_steps = 999
    config.environment.simulation.pace_s = 0.0
    config.environment.simulation.timestep_s = 0.25
    config.runtime.max_worker_threads = 1
    coordinator = Coordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        config.mpc,
        config.runtime,
        logger=logging.getLogger("test"),
    )
    events: list[tuple[str, int]] = []
    hooks = EpisodeHooks(
        on_episode_start=lambda observation: events.append(("start", observation.t_index)),
        on_step=lambda observation, record: events.append(("step", record.t_index)),
        on_episode_finish=lambda summary: events.append(("finish", summary.final_t_index)),
    )

    result = run_episode(coordinator, hooks)

    assert result.summary.records_emitted == 4
    assert events[0] == ("start", 0)
    assert [event for event in events if event[0] == "step"] == [
        ("step", 1),
        ("step", 2),
        ("step", 3),
        ("step", 4),
    ]
    assert events[-1] == ("finish", 4)


def test_coordinator_invokes_before_step_hook() -> None:
    coordinator = _fast_coordinator(load_config("standard"))
    before_steps: list[int] = []
    hooks = EpisodeHooks(
        on_before_step=lambda observation: before_steps.append(observation.t_index),
    )

    result = run_episode(coordinator, hooks)

    assert result.summary.records_emitted == 3
    assert before_steps == [0, 1, 2]


def test_coordinator_returns_interrupted_summary_from_keyboard_interrupt() -> None:
    config = load_config("standard")
    config.experiment.max_steps = 4
    config.environment.goal.hold_steps = 999
    config.environment.simulation.pace_s = 0.0
    config.environment.simulation.timestep_s = 0.25
    config.runtime.max_worker_threads = 1
    coordinator = Coordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        config.mpc,
        config.runtime,
        logger=logging.getLogger("test"),
    )
    finished_statuses: list[str] = []

    def interrupt_after_first_step(observation, record) -> None:
        raise KeyboardInterrupt

    hooks = EpisodeHooks(
        on_step=interrupt_after_first_step,
        on_episode_finish=lambda summary: finished_statuses.append(summary.status),
    )
    result = run_episode(coordinator, hooks)

    assert result.summary.status == "interrupted"
    assert result.summary.records_emitted == 1
    assert result.summary.final_observation is not None
    assert finished_statuses == ["interrupted"]
