"""Experiment artifact directory and file writers."""

from __future__ import annotations

import csv
import json
import logging
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from wsmpc import __version__
from wsmpc.utils.config_schema import RootConfig
from wsmpc.utils.messages import ExperimentSummary, RLStepRecord, StepRecord

ARTIFACT_FORMAT_VERSION = 1
STEP_CSV_HEADERS = tuple(
    field.alias or name for name, field in StepRecord.model_fields.items()
)
RL_STEP_CSV_HEADERS = tuple(
    field.alias or name for name, field in RLStepRecord.model_fields.items()
)


def create_run_directory(
    root_dir: str | Path,
    alias: str | None = None,
    *,
    started_at: datetime | None = None,
) -> Path:
    """Create one collision-free run directory with a local timestamp suffix."""

    timestamp = (started_at or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S")
    safe_alias = sanitize_alias(alias)
    directory_name = f"{safe_alias}_{timestamp}" if safe_alias else timestamp

    # Create the parent root separately so an existing run directory still fails loudly.
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    run_dir = root_path / directory_name
    run_dir.mkdir(exist_ok=False)
    return run_dir


def sanitize_alias(alias: str | None) -> str:
    """Return a filesystem-safe alias using lowercase letters, numbers, hyphens, and underscores."""

    if alias is None:
        return ""
    normalized = re.sub(r"[^a-z0-9_-]+", "-", alias.lower()).strip("-_")
    return re.sub(r"-{2,}", "-", normalized)


def attach_run_log_handler(logger: logging.Logger, run_dir: str | Path) -> logging.Handler:
    """Attach a plain file handler that captures structured experiment logs."""

    log_path = Path(run_dir) / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    return handler


def detach_run_log_handler(logger: logging.Logger, handler: logging.Handler | None) -> None:
    """Remove and close a run-specific file handler."""

    if handler is None:
        return
    logger.removeHandler(handler)
    handler.close()


class ArtifactWriter:
    """Write recreate-focused experiment artifacts into one run directory."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        config_package: str,
        cli_args: dict[str, Any],
        started_at: datetime | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.config_package = config_package
        self.cli_args = cli_args
        self.started_at = started_at or datetime.now().astimezone()
        self.finished_at: datetime | None = None
        self.step_count = 0
        self.rl_step_count = 0
        self._step_file = None
        self._step_writer: csv.DictWriter[str] | None = None

        # Create placeholder files early so the manifest represents the full run contract.
        (self.run_dir / "run.log").touch(exist_ok=True)
        self._write_metadata()
        self._write_manifest(completed=False, status="running")

    @classmethod
    def create(
        cls,
        root_dir: str | Path,
        *,
        alias: str | None,
        config_package: str,
        cli_args: dict[str, Any],
    ) -> ArtifactWriter:
        """Create a new run directory and return a writer bound to it."""

        started_at = datetime.now().astimezone()
        run_dir = create_run_directory(root_dir, alias, started_at=started_at)
        return cls(
            run_dir,
            config_package=config_package,
            cli_args=cli_args,
            started_at=started_at,
        )

    def write_config(self, config: RootConfig) -> None:
        """Write the resolved runtime config using human-readable YAML aliases."""

        self._write_json(
            "config.json",
            config.model_dump(mode="json", by_alias=True),
        )

    def open_step_writer(self) -> None:
        """Open the stable CSV record stream and write its header row."""

        if self._step_writer is not None:
            return
        self._step_file = (self.run_dir / "steps.csv").open("w", newline="", encoding="utf-8")
        self._step_writer = csv.DictWriter(self._step_file, fieldnames=STEP_CSV_HEADERS)
        self._step_writer.writeheader()
        self._step_file.flush()

    def write_step(self, record: StepRecord) -> None:
        """Append one dense step record to the CSV artifact."""

        if self._step_writer is None:
            msg = "Step writer is not open; call open_step_writer before write_step"
            raise RuntimeError(msg)

        # Serialize through Pydantic so CSV columns match the public message aliases.
        row = record.model_dump(mode="json", by_alias=True)
        self._step_writer.writerow({header: row[header] for header in STEP_CSV_HEADERS})
        self.step_count += 1
        if self._step_file is not None:
            self._step_file.flush()

    def write_summary(self, summary: ExperimentSummary) -> None:
        """Write the final typed episode summary with nested observation aliases."""

        self._write_json(
            "summary.json",
            summary.model_dump(mode="json", by_alias=True),
        )

    def write_rl_steps(self, records: list[RLStepRecord]) -> None:
        """Write replanning-level RL transitions after Monte Carlo returns are known."""

        path = self.run_dir / "rl_steps.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer: csv.DictWriter[str] = csv.DictWriter(file, fieldnames=RL_STEP_CSV_HEADERS)
            writer.writeheader()
            for record in records:
                row = record.model_dump(mode="json", by_alias=True)
                writer.writerow({header: row[header] for header in RL_STEP_CSV_HEADERS})
        self.rl_step_count = len(records)

    def write_figure(self, name: str, figure: Any) -> Path:
        """Save one diagnostic Matplotlib figure into the figures artifact directory."""

        figures_dir = self.run_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        path = figures_dir / f"{name}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def finalize_manifest(
        self,
        *,
        completed: bool,
        status: str,
    ) -> None:
        """Close open streams and write final metadata plus manifest status."""

        self.close()
        self.finished_at = datetime.now().astimezone()
        self._write_metadata()
        self._write_manifest(completed=completed, status=status)

    def close(self) -> None:
        """Close any open artifact streams."""

        if self._step_file is not None:
            self._step_file.close()
            self._step_file = None
            self._step_writer = None

    def __enter__(self) -> ArtifactWriter:
        """Return the active writer for context-manager use."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Ensure open file handles are released if a caller exits early."""

        self.close()

    def _write_metadata(self) -> None:
        """Write run metadata that helps reproduce the local execution context."""

        git_commit, git_dirty = _read_git_state()
        metadata = {
            "run_dir": str(self.run_dir),
            "config_package": self.config_package,
            "cli_args": self.cli_args,
            "package_version": __version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "wall_started_at": self.started_at.isoformat(),
            "wall_finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }
        self._write_json("metadata.json", metadata)

    def _write_manifest(self, *, completed: bool, status: str) -> None:
        """Write a compact index of files and row counts in the run directory."""

        manifest = {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "run_dir": str(self.run_dir),
            "files": self._file_entries(),
            "row_counts": {"steps": self.step_count, "rl_steps": self.rl_step_count},
            "completed": completed,
            "completion_status": status,
        }
        self._write_json("manifest.json", manifest)

    def _file_entries(self) -> list[dict[str, Any]]:
        """Return existing artifact files with byte sizes for quick integrity checks."""

        entries: list[dict[str, Any]] = []
        for path in sorted(item for item in self.run_dir.rglob("*") if item.is_file()):
            relative_path = path.relative_to(self.run_dir).as_posix()
            entries.append({"path": relative_path, "bytes": path.stat().st_size})
        return entries

    def _write_json(self, name: str, data: Any) -> None:
        """Write deterministic, inspectable JSON."""

        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")


def _read_git_state() -> tuple[str, bool]:
    """Read Git commit and dirty state for reproducible artifacts."""

    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    dirty_result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )

    commit = commit_result.stdout.strip()
    if not commit:
        msg = "Git returned an empty commit hash"
        raise RuntimeError(msg)
    dirty = bool(dirty_result.stdout.strip())
    return commit, dirty
