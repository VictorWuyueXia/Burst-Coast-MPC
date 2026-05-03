from wsmpc.core.messages import ActionCommand, NodeStatus


def test_action_command_round_trip_json() -> None:
    command = ActionCommand(
        run_id="run",
        episode_id=0,
        t_index=1,
        t_sec=0.02,
        u=0.5,
        source="zero_torque",
    )

    loaded = ActionCommand.model_validate_json(command.model_dump_json())

    assert loaded == command


def test_node_status_alias_round_trip() -> None:
    status = NodeStatus(
        **{
            "node-id": "Coordinator",
            "identity": "Coordinator",
            "status": "running",
            "action": "episode_start",
            "action_result": "initialized",
            "updated_wall_time_s": 1.0,
        }
    )

    dumped = status.model_dump(by_alias=True)

    assert dumped["node-id"] == "Coordinator"
    assert "node_id" not in dumped
