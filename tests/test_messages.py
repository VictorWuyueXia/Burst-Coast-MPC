from wsmpc.utils.messages import ActionCommand


def test_action_command_round_trip_json() -> None:
    command = ActionCommand(
        run_id="run",
        episode_id=0,
        t_index=1,
        t_sec=0.02,
        u_nm=0.5,
        source="zero_torque",
        early_wake_flag=False,
    )

    loaded = ActionCommand.model_validate_json(command.model_dump_json())

    assert loaded == command
