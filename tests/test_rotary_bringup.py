"""End-to-end rotary bringup and CLI routing checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from rotary_pendulum.utils.config_schema import EPISODE_CONFIG_PATHS, load_episode_config
from rotary_pendulum.utils.messages import ActionPlan, CandidateRecord


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_rotary_bringup_rolls_out_each_complete_plan_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replan only after H actions and retain the complete session contract."""

    from bringup import rotary_pendulum_mode

    config = load_episode_config(EPISODE_CONFIG_PATHS, runtime_mode="headless")
    config.experiment.max_steps = 6
    config.goal.hold_steps = 999
    config.artifacts.root_dir = str(tmp_path)
    config.artifacts.alias = "rollout"
    solve_indices: list[int] = []
    previous_torques: list[float] = []

    class ControllerStub:
        """Return one burst action followed by two exact coast actions."""

        def __init__(self, *_: object) -> None:
            pass

        def solve_plan(
            self,
            observation: object,
            previous_applied_torque_nm: float,
            replan_index: int,
        ) -> ActionPlan:
            solve_indices.append(observation.t_index)
            previous_torques.append(previous_applied_torque_nm)
            candidate = CandidateRecord(
                split_ratio=1.0 / 3.0,
                horizon_steps=3,
                burst_steps=1,
                coast_steps=2,
                objective_value=float(2 - replan_index),
                solve_time_s=0.001,
                solver_status="Solve_Succeeded",
                selected=True,
            )
            return ActionPlan(
                plan_id=f"stub-{replan_index}",
                replan_index=replan_index,
                hbar=3.0 * config.simulation.timestep_s,
                bbar=1.0 / 3.0,
                horizon_steps=3,
                burst_steps=1,
                coast_steps=2,
                objective_value=candidate.objective_value,
                solve_time_s=candidate.solve_time_s,
                solver_status=candidate.solver_status,
                torques_nm=np.array([0.001, 0.0, 0.0], dtype=np.float64),
                predicted_states=np.zeros((4, 4), dtype=np.float64),
                candidates=(candidate,),
            )

    monkeypatch.setattr(
        rotary_pendulum_mode,
        "load_episode_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(rotary_pendulum_mode, "RotaryMPCController", ControllerStub)
    summary = rotary_pendulum_mode.run_rotary_pendulum_mode(
        alias=None,
        no_visual=True,
        console=Console(width=120),
    )

    assert summary.steps == 6
    assert summary.replans == 2
    assert solve_indices == [0, 3]
    assert previous_torques == pytest.approx([0.0, 0.0])
    run_dir = next(tmp_path.iterdir())
    required_files = {
        "config.json",
        "metadata.json",
        "manifest.json",
        "run.log",
        "steps.csv",
        "plans.csv",
        "summary.json",
        "interpretation_summary.md",
    }
    assert required_files <= {path.name for path in run_dir.iterdir()}
    assert {path.name for path in (run_dir / "figures").iterdir()} == {
        "states.png",
        "energy.png",
        "phase.png",
        "commands.png",
        "mpc_diagnostics.png",
    }
    step_file = (run_dir / "steps.csv").open(encoding="utf-8", newline="")
    step_rows = list(csv.DictReader(step_file))
    step_file.close()
    assert [row["replan_flag"] for row in step_rows] == [
        "True",
        "False",
        "False",
        "True",
        "False",
        "False",
    ]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_counts"] == {"steps": 6, "plans": 2}
    report = (run_dir / "interpretation_summary.md").read_text(encoding="utf-8")
    assert "±90 degree arm soft limit" in report
