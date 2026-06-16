import logging
import math

from burst_coast_mpc.epoch_coordinator import EpochCoordinator
from inverted_pendulum.RL.policy import RLActionSelection
from inverted_pendulum.utils.config_schema import load_config
from inverted_pendulum.utils.logging import ThirdPersonObservers
from inverted_pendulum.utils.messages import ActionCommand
from inverted_pendulum.utils.monte_carlo import MonteCarloAction


def _fast_coordinator(config) -> EpochCoordinator:
    config.experiment.max_steps = 3
    config.environment.goal.hold_steps = 999
    config.environment.simulation.pace_s = 0.0
    config.environment.simulation.timestep_s = 0.25
    return EpochCoordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        config.mpc,
        logger=logging.getLogger("test"),
    )


def test_coordinator_runs_short_mpc_episode() -> None:
    coordinator = _fast_coordinator(load_config())

    result = coordinator.run_episode()

    assert result.summary.status == "max_steps_reached"
    assert result.summary.total_steps == 3
    assert result.summary.records_emitted == 3


def test_coordinator_invokes_third_person_observers() -> None:
    config = load_config()
    config.experiment.max_steps = 4
    config.environment.goal.hold_steps = 999
    config.environment.simulation.pace_s = 0.0
    config.environment.simulation.timestep_s = 0.25
    coordinator = EpochCoordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        config.mpc,
        logger=logging.getLogger("test"),
    )
    events: list[tuple[str, int]] = []
    third_person_observers = ThirdPersonObservers(
        at_episode_start=lambda observation: events.append(("start", observation.t_index)),
        after_step=lambda observation, record: events.append(("step", record.t_index)),
        at_episode_finish=lambda summary: events.append(("finish", summary.final_t_index)),
    )

    result = coordinator.run_episode(third_person_observers)

    assert result.summary.records_emitted == 4
    assert events[0] == ("start", 0)
    assert [event for event in events if event[0] == "step"] == [
        ("step", 1),
        ("step", 2),
        ("step", 3),
        ("step", 4),
    ]
    assert events[-1] == ("finish", 4)


def test_coordinator_invokes_before_step_observer() -> None:
    coordinator = _fast_coordinator(load_config())
    before_steps: list[int] = []
    third_person_observers = ThirdPersonObservers(
        before_step=lambda observation: before_steps.append(observation.t_index),
    )

    result = coordinator.run_episode(third_person_observers)

    assert result.summary.records_emitted == 3
    assert before_steps == [0, 1, 2]


def test_event_trigger_forces_replanning_at_zero_and_pi_sections() -> None:
    config = load_config()
    config.experiment.max_steps = 2
    config.experiment.stop_on_goal = False
    config.environment.goal.hold_steps = 999
    config.environment.simulation.pace_s = 0.0
    config.environment.simulation.timestep_s = 0.25
    coordinator = EpochCoordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        config.mpc,
        logger=logging.getLogger("test"),
    )
    force_replans: list[bool] = []

    class RecordingController:
        def select_action(self, observation, *, force_replan: bool) -> ActionCommand:
            force_replans.append(force_replan)
            return ActionCommand(
                run_id=observation.run_id,
                episode_id=observation.episode_id,
                t_index=observation.t_index,
                t_sec=observation.t_sec,
                u_nm=0.0,
                source="test",
                early_wake_flag=force_replan,
                plan_id=f"mock-{observation.t_index}",
            )

    coordinator.mpc_controller = RecordingController()
    config.experiment.initial_state.theta_rad = 0.0
    result = coordinator.run_episode()

    assert force_replans == [True, True]
    assert [record.early_wake_flag for record in result.records] == [True, True]

    force_replans.clear()
    config.experiment.initial_state.theta_rad = math.pi
    result = coordinator.run_episode()

    assert force_replans == [True, True]
    assert [record.early_wake_flag for record in result.records] == [True, True]

    force_replans.clear()
    config.experiment.initial_state.theta_rad = 1.0
    coordinator.run_episode()

    assert force_replans == [False, False]


def test_epoch_coordinator_records_rl_transitions_from_policy() -> None:
    config = load_config()
    config.experiment.max_steps = 2
    config.experiment.stop_on_goal = False
    config.coordinator.event_trigger = False
    config.environment.goal.hold_steps = 999
    config.environment.simulation.pace_s = 0.0
    config.environment.simulation.timestep_s = 0.25

    class FixedPolicy:
        def select_action(self, observation, *, explore: bool) -> RLActionSelection:
            return RLActionSelection(
                action=MonteCarloAction(
                    bbar=0.5,
                    hbar=0.5,
                    horizon_steps=2,
                    burst_steps=1,
                    coast_steps=1,
                ),
                q_value=1.0,
                probability=1.0,
                mode="rl_test",
            )

    coordinator = EpochCoordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        config.mpc,
        logger=logging.getLogger("test"),
        rl_policy=FixedPolicy(),
        rl_config=config.rl,
    )

    result = coordinator.run_episode()

    assert result.summary.records_emitted == 2
    assert len(result.rl_records) == 1
    assert result.rl_records[0].burst_steps == 1
    assert result.rl_records[0].horizon_steps == 2
    assert result.rl_records[0].done is True
    assert result.rl_records[0].return_cost == result.rl_records[0].step_cost
