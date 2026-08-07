"""Reproducible machine and human artifacts for rotary-pendulum MPC episodes."""

from __future__ import annotations

import csv
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from bringup import __version__
from rotary_pendulum.utils.config_schema import EpisodeConfig, VisualizationConfig
from rotary_pendulum.utils.messages import ActionPlan, EpisodeSummary, StepRecord

ARTIFACT_FORMAT_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]
STEP_CSV_HEADERS = tuple(field.name for field in fields(StepRecord))
PLAN_CSV_HEADERS = (
    "plan_id",
    "replan_index",
    "replan_t_sec",
    "split_ratio",
    "hbar",
    "bbar",
    "horizon_steps",
    "burst_steps",
    "coast_steps",
    "objective_value",
    "candidate_solve_time_s",
    "total_solve_time_s",
    "solver_status",
    "selected",
    "torques_nm_json",
    "predicted_states_json",
)


class RotaryArtifactWriter:
    """Own one session directory and write its resolved execution evidence."""

    def __init__(
        self,
        config: EpisodeConfig,
        *,
        alias: str | None,
        cli_args: dict[str, Any],
        config_sources: tuple[Path, ...],
        visualization: VisualizationConfig | None,
    ) -> None:
        # Create one explicit session location and immediately mark it as running.
        self.started_at = datetime.now().astimezone()
        safe_alias = (
            ""
            if alias is None
            else re.sub(
                r"-{2,}",
                "-",
                re.sub(r"[^a-z0-9_-]+", "-", alias.lower()).strip("-_"),
            )
        )
        timestamp = self.started_at.strftime("%Y%m%dT%H%M%S%f")
        directory_name = f"{safe_alias}_{timestamp}" if safe_alias else timestamp
        self.run_dir = Path(config.artifacts.root_dir) / directory_name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.cli_args = cli_args
        self.config_sources = tuple(path.resolve() for path in config_sources)
        self.timestep_s = config.simulation.timestep_s
        self.finished_at: datetime | None = None
        self.step_count = 0
        self.plan_count = 0
        (self.run_dir / "run.log").touch()
        resolved_config = config.model_dump(mode="json", by_alias=True)
        if visualization is not None:
            resolved_config["visualization"] = visualization.model_dump(mode="json", by_alias=True)
        self._write_json("config.json", resolved_config)
        self._write_metadata()
        self._write_manifest(completed=False, status="running")

    def log(self, message: str) -> None:
        """Append one timestamped controller event to the session log."""

        file = (self.run_dir / "run.log").open("a", encoding="utf-8")
        file.write(f"{datetime.now().astimezone().isoformat()} {message}\n")
        file.close()

    def finalize(
        self,
        records: list[StepRecord],
        plans: list[ActionPlan],
        summary: EpisodeSummary,
        figures: dict[str, Any],
    ) -> None:
        """Write dense, decision-level, visual, and interpretive episode outputs."""

        if not records or not plans:
            raise ValueError("A completed rotary MPC episode requires step and plan records")

        # Write every dynamics transition as a stable flat table.
        step_file = (self.run_dir / "steps.csv").open("w", newline="", encoding="utf-8")
        step_writer: csv.DictWriter[str] = csv.DictWriter(
            step_file,
            fieldnames=STEP_CSV_HEADERS,
        )
        step_writer.writeheader()
        for record in records:
            step_writer.writerow(asdict(record))
        step_file.close()
        self.step_count = len(records)

        # Preserve all split candidates while attaching full trajectories only to the winner.
        first_step_by_replan = {
            record.replan_index: record for record in records if record.replan_flag
        }
        plan_file = (self.run_dir / "plans.csv").open("w", newline="", encoding="utf-8")
        plan_writer: csv.DictWriter[str] = csv.DictWriter(
            plan_file,
            fieldnames=PLAN_CSV_HEADERS,
        )
        plan_writer.writeheader()
        for plan in plans:
            replan_t_sec = first_step_by_replan[plan.replan_index].t_sec - self.timestep_s
            for candidate in plan.candidates:
                plan_writer.writerow(
                    {
                        "plan_id": plan.plan_id,
                        "replan_index": plan.replan_index,
                        "replan_t_sec": replan_t_sec,
                        "split_ratio": candidate.split_ratio,
                        "hbar": plan.hbar,
                        "bbar": candidate.burst_steps / candidate.horizon_steps,
                        "horizon_steps": candidate.horizon_steps,
                        "burst_steps": candidate.burst_steps,
                        "coast_steps": candidate.coast_steps,
                        "objective_value": candidate.objective_value,
                        "candidate_solve_time_s": candidate.solve_time_s,
                        "total_solve_time_s": plan.solve_time_s,
                        "solver_status": candidate.solver_status,
                        "selected": candidate.selected,
                        "torques_nm_json": json.dumps(plan.torques_nm.tolist())
                        if candidate.selected
                        else "",
                        "predicted_states_json": json.dumps(plan.predicted_states.tolist())
                        if candidate.selected
                        else "",
                    }
                )
        plan_file.close()
        self.plan_count = sum(len(plan.candidates) for plan in plans)

        # Save typed outcome data and the five human-readable diagnostic figures.
        self._write_json("summary.json", asdict(summary))
        figures_dir = self.run_dir / "figures"
        figures_dir.mkdir()
        for name, figure in figures.items():
            figure.savefig(figures_dir / f"{name}.png", dpi=150, bbox_inches="tight")

        # State the run result and its formulation-level interpretation in a shallow report.
        minimum_energy_error = min(abs(record.normalized_energy_error) for record in records)
        minimum_beta_error = min(abs(record.beta_rad) for record in records)
        selected_splits = np.asarray([plan.bbar for plan in plans], dtype=np.float64)
        report = (
            "# Rotary-Pendulum MPC Interpretation\n\n"
            f"The episode ended with `{summary.status}` after {summary.steps} dynamics steps "
            f"and {summary.replans} complete MPC replans. Goal reached: "
            f"`{str(summary.goal_reached).lower()}`.\n\n"
            f"- Final normalized energy error: {summary.final_normalized_energy_error:.6f}\n"
            f"- Closest normalized energy error: {minimum_energy_error:.6f}\n"
            f"- Closest wrapped upright pendulum error: {minimum_beta_error:.6f} rad\n"
            f"- Mean selected burst fraction B/H: {selected_splits.mean():.6f}\n\n"
            "The objective combines terminal pendulum-relative swing energy, burst-only phase "
            "chasing and actuation continuation, energy-gated terminal capture, and the ±90 "
            "degree arm soft limit. If target swing energy is approached without satisfying the "
            "full-state goal, inspect the local capture errors; if it is not approached, the "
            "burst-coast structure or pumping objective remains limiting.\n"
        )
        report_file = (self.run_dir / "interpretation_summary.md").open("w", encoding="utf-8")
        report_file.write(report)
        report_file.close()

        self.finished_at = datetime.now().astimezone()
        self._write_metadata()
        self._write_manifest(completed=True, status=summary.status)

    def _write_metadata(self) -> None:
        """Capture source, runtime, platform, and Git state for reproduction."""

        git_command = ["git", "-C", str(REPO_ROOT)]
        commit = subprocess.run(
            [*git_command, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [*git_command, "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        self._write_json(
            "metadata.json",
            {
                "run_dir": str(self.run_dir),
                "config_sources": [str(path) for path in self.config_sources],
                "cli_args": self.cli_args,
                "package_version": __version__,
                "python_version": sys.version,
                "platform": platform.platform(),
                "git_commit": commit,
                "git_dirty": dirty,
                "wall_started_at": self.started_at.isoformat(),
                "wall_finished_at": self.finished_at.isoformat() if self.finished_at else None,
            },
        )

    def _write_manifest(self, *, completed: bool, status: str) -> None:
        """Index every artifact file, size, row count, and completion state."""

        files = [
            {"path": path.relative_to(self.run_dir).as_posix(), "bytes": path.stat().st_size}
            for path in sorted(item for item in self.run_dir.rglob("*") if item.is_file())
        ]
        self._write_json(
            "manifest.json",
            {
                "artifact_format_version": ARTIFACT_FORMAT_VERSION,
                "run_dir": str(self.run_dir),
                "files": files,
                "row_counts": {"steps": self.step_count, "plans": self.plan_count},
                "completed": completed,
                "completion_status": status,
            },
        )

    def _write_json(self, name: str, data: Any) -> None:
        """Write deterministic indented JSON for direct human and machine inspection."""

        file = (self.run_dir / name).open("w", encoding="utf-8")
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
        file.close()
