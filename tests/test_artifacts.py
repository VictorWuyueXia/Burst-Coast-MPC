import csv
import json
import logging
import re

from wsmpc.coordinator import Coordinator
from wsmpc.utils.artifacts import STEP_CSV_HEADERS, ArtifactWriter
from wsmpc.utils.config_schema import load_config
from wsmpc.utils.logging import EpisodeHooks, run_episode


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
    hooks = EpisodeHooks(
        on_step=lambda observation, record: writer.write_step(record),
        on_episode_finish=lambda summary: writer.write_summary(summary),
    )

    result = run_episode(coordinator, hooks)
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
    assert summary["final_observation"]["theta-rad"] == result.summary.final_observation.theta_rad
