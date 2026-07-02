from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from burst_coast_mpc.epoch_coordinator import EpochCoordinator
from inverted_pendulum.RL.policy import StructuredCriticPolicy
from inverted_pendulum.utils.config_schema import (
    InitialStateConfig,
    RootConfig,
    load_data_generation_config,
    load_online_training_config,
)
from inverted_pendulum.utils.monte_carlo import sample_uniform_initial_state

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_ROOT = REPO_ROOT / "artifacts" / "model-snapshots"
ANALYSIS_ROOT = REPO_ROOT / "artifacts" / "analysis"
SNAPSHOT_STRIDE = 16
MAX_STEP_MULTIPLIER = 2
SWEEP_SEED_MODULUS = 2**32 - 1


@dataclass(frozen=True)
class EvaluationCase:
    case_id: int
    theta_rad: float
    omega_rad_s: float


@dataclass(frozen=True)
class ControllerSpec:
    controller_family: str
    controller_name: str
    epoch_index: int | None
    critic_artifact_dir: str | None


@dataclass(frozen=True)
class ControllerResult:
    controller_family: str
    controller_name: str
    epoch_index: int | None
    critic_artifact_dir: str | None
    cases: int
    success_rate: float
    mean_completion_time_s: float
    mean_solver_compute_time_s: float
    mean_torque_abs_integral_nm_s: float
    mean_total_steps: float
    mean_rl_segments: float
    mean_wall_time_s: float


class SolveTimeLogHandler(logging.Handler):
    def __init__(self, controller_name: str, case_id: int) -> None:
        super().__init__(level=logging.INFO)
        self.controller_name = controller_name
        self.case_id = case_id
        self.solve_time_s = 0.0
        self.solve_events = 0

    def emit(self, record: logging.LogRecord) -> None:
        # Parse only the structured solve-time token emitted by selected-plan logs.
        tokens = {
            part.split("=", maxsplit=1)[0]: part.split("=", maxsplit=1)[1]
            for part in record.getMessage().split()
            if "=" in part
        }
        if "solve_time_s" in tokens:
            self.solve_events += 1
            solve_time_s = float(tokens["solve_time_s"])
            self.solve_time_s += solve_time_s
            print(
                f"[solve] {self.controller_name} case={self.case_id} "
                f"n={self.solve_events} t={tokens['t_sec']}s solve={solve_time_s:.4f}s "
                f"cum={self.solve_time_s:.2f}s",
                flush=True,
            )


def current_sweep_seed() -> int:
    # Encode wall-clock time into the NumPy-compatible seed range.
    timestamp_text = datetime.now().astimezone().strftime("%Y%m%d%H%M%S%f")
    return int(timestamp_text) % SWEEP_SEED_MODULUS


def discover_controller_specs(base_config: RootConfig) -> list[ControllerSpec]:
    # Keep the structural baselines ahead of the learned online progression.
    specs = [
        ControllerSpec("mpc", "event-triggered-mpc", None, None),
        ControllerSpec("offline", "offline-critic", 0, base_config.rl.critic_artifact_dir),
    ]
    snapshots = sorted(path for path in SNAPSHOT_ROOT.glob("online-critic_*") if path.is_dir())
    selected = [
        (snapshot_index, snapshot_dir)
        for snapshot_index, snapshot_dir in enumerate(snapshots, start=1)
        if (snapshot_index - 1) % SNAPSHOT_STRIDE == 0
    ]
    if not selected:
        msg = f"No online critic snapshots found under {SNAPSHOT_ROOT}"
        raise FileNotFoundError(msg)
    specs.extend(
        ControllerSpec(
            "online",
            f"online-critic-epoch-{snapshot_index:03d}",
            snapshot_index,
            str(snapshot_dir.relative_to(REPO_ROOT)),
        )
        for snapshot_index, snapshot_dir in selected
    )
    return specs


def sample_evaluation_cases(sweep_seed: int) -> list[EvaluationCase]:
    # Use the existing Monte Carlo domain while changing the seed for each sweep.
    data_config = load_data_generation_config()
    rng = np.random.default_rng(sweep_seed)
    cases: list[EvaluationCase] = []
    for case_id in range(data_config.data_generation.episodes):
        initial_state = sample_uniform_initial_state(
            rng,
            data_config.data_generation,
            data_config.environment,
        )
        cases.append(EvaluationCase(case_id, initial_state.theta_rad, initial_state.omega_rad_s))
    return cases


def evaluate_controller(
    spec: ControllerSpec,
    cases: list[EvaluationCase],
    base_config: RootConfig,
) -> ControllerResult:
    # Mutate only controller identity, critic path, horizon, and initial condition.
    completion_time_s: list[float] = []
    solver_compute_time_s: list[float] = []
    torque_abs_integral_nm_s: list[float] = []
    total_steps: list[int] = []
    rl_segments: list[int] = []
    wall_time_s: list[float] = []
    successes = 0
    for case in cases:
        config = base_config.model_copy(deep=True)
        config.experiment.max_steps = base_config.experiment.max_steps * MAX_STEP_MULTIPLIER
        config.experiment.run_id = f"online_snapshot_progress_{spec.controller_name}"
        config.experiment.episode_id = case.case_id
        config.experiment.initial_state = InitialStateConfig(
            theta_rad=case.theta_rad, omega_rad_s=case.omega_rad_s
        )
        handler = SolveTimeLogHandler(spec.controller_name, case.case_id)
        logger_name = f"online_snapshot_progress.{spec.controller_name}.{case.case_id}"
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        policy = None
        rl_config = None
        if spec.critic_artifact_dir is not None:
            config.rl.critic_artifact_dir = spec.critic_artifact_dir
            policy = StructuredCriticPolicy(config.rl, config.environment)
            rl_config = config.rl
        coordinator = EpochCoordinator(
            config.coordinator,
            config.environment,
            config.experiment,
            config.mpc,
            logger=logger,
            rl_policy=policy,
            rl_config=rl_config,
            rl_explore=False,
        )
        result = coordinator.run_episode()
        torque = np.fromiter((record.u_applied_nm for record in result.records), dtype=np.float64)
        completion_time_s.append(result.summary.final_t_sec)
        solver_compute_time_s.append(handler.solve_time_s)
        torque_abs_integral_nm_s.append(
            float(np.sum(np.abs(torque)) * config.environment.simulation.timestep_s)
        )
        total_steps.append(result.summary.total_steps)
        rl_segments.append(len(result.rl_records))
        wall_time_s.append(result.summary.total_wall_time_s)
        successes += int(result.summary.goal_reached)
    return ControllerResult(
        controller_family=spec.controller_family,
        controller_name=spec.controller_name,
        epoch_index=spec.epoch_index,
        critic_artifact_dir=spec.critic_artifact_dir,
        cases=len(cases),
        success_rate=successes / len(cases),
        mean_completion_time_s=float(np.mean(completion_time_s)),
        mean_solver_compute_time_s=float(np.mean(solver_compute_time_s)),
        mean_torque_abs_integral_nm_s=float(np.mean(torque_abs_integral_nm_s)),
        mean_total_steps=float(np.mean(total_steps)),
        mean_rl_segments=float(np.mean(rl_segments)),
        mean_wall_time_s=float(np.mean(wall_time_s)),
    )


def write_artifacts(
    rows: list[ControllerResult],
    cases: list[EvaluationCase],
    sweep_seed: int,
) -> Path:
    # Keep one complete artifact directory for the sampled setup and metrics.
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    output_dir = ANALYSIS_ROOT / f"controller-progress-mc_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    result_file = (output_dir / "controller_progress.csv").open("w", newline="", encoding="utf-8")
    fieldnames = [field.name for field in fields(ControllerResult)]
    writer = csv.DictWriter(result_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    result_file.close()
    payload = {
        "snapshot_stride": SNAPSHOT_STRIDE,
        "max_step_multiplier": MAX_STEP_MULTIPLIER,
        "sweep_seed": sweep_seed,
        "cases": [asdict(case) for case in cases],
        "results": [asdict(row) for row in rows],
        "energy_metric": "mean(sum(abs(u_applied_nm)) * timestep_s)",
    }
    metadata_file = (output_dir / "summary.json").open("w", encoding="utf-8")
    json.dump(payload, metadata_file, indent=2, sort_keys=True)
    metadata_file.write("\n")
    metadata_file.close()

    # Plot online progression against the two structural baselines.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    online_rows = [row for row in rows if row.controller_family == "online"]
    baseline_rows = [row for row in rows if row.controller_family != "online"]
    indices = np.array([row.epoch_index for row in online_rows], dtype=np.int64)
    metrics = (
        ("mean_completion_time_s", "mean completion time s"),
        ("mean_solver_compute_time_s", "mean solver compute time s"),
        ("mean_torque_abs_integral_nm_s", "mean torque integral N m s"),
    )
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis, (metric_name, ylabel) in zip(axes, metrics, strict=True):
        values = np.array([getattr(row, metric_name) for row in online_rows], dtype=np.float64)
        axis.plot(indices, values, marker="o", linewidth=1.4, label="online critics")
        for baseline in baseline_rows:
            axis.axhline(
                getattr(baseline, metric_name),
                linestyle="--",
                linewidth=1.1,
                label=baseline.controller_name,
            )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[2].set_xlabel("online snapshot epoch")
    axes[0].legend(loc="best")
    figure.suptitle("Controller Structure Performance on One Monte Carlo Sweep")
    figure.tight_layout()
    figure.savefig(output_dir / "structure_metric_trends.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_dir


def main() -> None:
    base_config = load_online_training_config()
    specs = discover_controller_specs(base_config)
    sweep_seed = current_sweep_seed()
    cases = sample_evaluation_cases(sweep_seed)
    rows: list[ControllerResult] = []
    started_at = time.monotonic()
    print(
        f"evaluating {len(specs)} controllers on {len(cases)} MC case(s), seed={sweep_seed}",
        flush=True,
    )
    for position, spec in enumerate(specs, start=1):
        print(f"[{position:02d}/{len(specs):02d}] start {spec.controller_name}", flush=True)
        row = evaluate_controller(spec, cases, base_config)
        rows.append(row)
        print(
            f"[{position:02d}/{len(specs):02d}] done {row.controller_name} "
            f"success={row.success_rate:.2f} time={row.mean_completion_time_s:.2f}s "
            f"compute={row.mean_solver_compute_time_s:.2f}s "
            f"energy={row.mean_torque_abs_integral_nm_s:.3f}Nm*s",
            flush=True,
        )
    print(f"writing artifacts after {(time.monotonic() - started_at) / 60.0:.1f}min", flush=True)
    output_dir = write_artifacts(rows, cases, sweep_seed)
    print(output_dir.relative_to(REPO_ROOT), flush=True)


if __name__ == "__main__":
    main()
