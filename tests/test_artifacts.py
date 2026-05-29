import csv
import json
import logging
import re

import pytest

from wsmpc.coordinator import Coordinator
from wsmpc.utils.artifacts import STEP_CSV_HEADERS, ArtifactWriter
from wsmpc.utils.config_schema import load_config
from wsmpc.utils.logging import ThirdPersonObservers


def _require_matplotlib() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


def test_artifact_writer_records_short_episode(tmp_path) -> None:
    config = load_config("default")
    config.experiment.max_steps = 3
    config.environment.goal.hold_steps = 999
    config.environment.simulation.pace_s = 0.0
    config.environment.simulation.timestep_s = 0.25
    config.runtime.max_worker_threads = 1
    config.artifacts.root_dir = str(tmp_path)
    config.artifacts.alias = "Smoke Run"
    writer = ArtifactWriter.create(
        config.artifacts.root_dir,
        alias=config.artifacts.alias,
        config_package="default",
        cli_args={"source": "test"},
    )
    writer.write_config(config)
    writer.open_step_writer()
    coordinator = Coordinator(
        config.coordinator,
        config.environment,
        config.experiment,
        config.mpc,
        config.runtime,
        logger=logging.getLogger("test"),
    )
    third_person_observers = ThirdPersonObservers(
        after_step=lambda observation, record: writer.write_step(record),
        at_episode_finish=lambda summary: writer.write_summary(summary),
    )

    result = coordinator.run_episode(third_person_observers)
    _require_matplotlib()
    from matplotlib import pyplot as plt

    from wsmpc.visualization.artifact_plots import create_artifact_figures

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
