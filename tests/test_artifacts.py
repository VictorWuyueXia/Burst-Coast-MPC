import csv
import json
import logging
import re

import pytest

from burst_coast_mpc.coordinator import Coordinator
from inverted_pendulum.utils.artifacts import RL_STEP_CSV_HEADERS, STEP_CSV_HEADERS, ArtifactWriter
from inverted_pendulum.utils.config_schema import load_config
from inverted_pendulum.utils.logging import ThirdPersonObservers
from inverted_pendulum.utils.messages import RLStepRecord


def _require_matplotlib() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


def test_artifact_writer_records_short_episode(tmp_path) -> None:
    config = load_config()
    config.experiment.max_steps = 3
    config.environment.goal.hold_steps = 999
    config.environment.simulation.pace_s = 0.0
    config.environment.simulation.timestep_s = 0.25
    config.artifacts.root_dir = str(tmp_path)
    config.artifacts.alias = "Smoke Run"
    writer = ArtifactWriter.create(
        config.artifacts.root_dir,
        alias=config.artifacts.alias,
        config_package="default-config",
        cli_args={"source": "test"},
    )
    writer.write_config(config)
    writer.open_step_writer()
    coordinator = Coordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        config.mpc,
        logger=logging.getLogger("test"),
    )
    third_person_observers = ThirdPersonObservers(
        after_step=lambda observation, record: writer.write_step(record),
        at_episode_finish=lambda summary: writer.write_summary(summary),
    )

    result = coordinator.run_episode(third_person_observers)
    _require_matplotlib()
    from matplotlib import pyplot as plt

    from inverted_pendulum.visualization.artifact_plots import create_artifact_figures

    figures = create_artifact_figures(result.records, config.environment)
    for name, figure in figures.items():
        writer.write_figure(name, figure)
        plt.close(figure)
    writer.finalize_manifest(completed=True, status=result.summary.status)

    assert re.fullmatch(r"smoke-run_\d{8}T\d{6}", writer.run_dir.name)
    for name in [
        "config.json",
        "metadata.json",
        "steps.csv",
        "summary.json",
        "run.log",
        "manifest.json",
    ]:
        assert (writer.run_dir / name).exists()
    for name in ["states", "energy", "phase", "commands"]:
        figure_path = writer.run_dir / "figures" / f"{name}.png"
        assert figure_path.exists()
        assert figure_path.stat().st_size > 0

    with (writer.run_dir / "steps.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows
    assert tuple(rows[0].keys()) == STEP_CSV_HEADERS
    assert "theta-rad" in rows[0]
    assert "u-applied-nm" in rows[0]
    assert len(rows) == result.summary.records_emitted

    manifest = json.loads((writer.run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((writer.run_dir / "summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_format_version"] == 1
    assert manifest["row_counts"]["steps"] == summary["records_emitted"]
    assert manifest["completed"] is True
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    expected_figures = {
        "figures/states.png",
        "figures/energy.png",
        "figures/phase.png",
        "figures/commands.png",
    }
    assert expected_figures <= manifest_paths
    assert summary["final_observation"]["theta-rad"] == result.summary.final_observation.theta_rad


def test_artifact_writer_records_rl_steps(tmp_path) -> None:
    config = load_config()
    writer = ArtifactWriter.create(
        tmp_path,
        alias="rl",
        config_package="test",
        cli_args={"source": "test"},
    )
    record = RLStepRecord(
        run_id="run",
        episode_id=0,
        replan_index=0,
        start_t_index=0,
        start_t_sec=0.0,
        end_t_index=2,
        end_t_sec=0.02,
        s_sin_theta=0.0,
        s_cos_theta=1.0,
        s_omega_rad_s=0.0,
        bbar=0.5,
        hbar=0.8,
        burst_steps=2,
        horizon_steps=4,
        next_s_sin_theta=0.1,
        next_s_cos_theta=0.99,
        next_s_omega_rad_s=0.2,
        done=False,
        step_cost=1.0,
        return_cost=1.5,
        u_nm_json="[0.1, 0.2]",
        solve_time_s=0.03,
        plan_id="plan",
    )

    writer.write_config(config)
    writer.write_rl_steps([record])
    writer.finalize_manifest(completed=True, status="done")

    with (writer.run_dir / "rl_steps.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    manifest = json.loads((writer.run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert tuple(rows[0].keys()) == RL_STEP_CSV_HEADERS
    assert rows[0]["plan-id"] == "plan"
    assert manifest["row_counts"]["rl_steps"] == 1
