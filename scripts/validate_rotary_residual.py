"""Run analytical-controller comparisons only; this script cannot call training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import jax
import numpy as np

from rotary_pendulum.RL.jax_residual_evaluation import evaluate, validation_resets, write_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    campaign = json.loads(arguments.config.read_text())
    settings = campaign["shared"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1", "2", "3", "0,1,2,3", ""):
        raise ValueError("Explicitly select only physical GPUs 0–3, or an empty CPU device list")
    root = arguments.output
    machine = root / "machine-scannables"
    human = root / "human-readables"
    machine.mkdir(parents=True, exist_ok=False)
    human.mkdir(parents=True, exist_ok=False)
    (machine / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    sources = [
        Path(__file__).resolve().relative_to(Path.cwd()),
        *Path("src/rotary_pendulum/RL").glob("jax_residual*.py"),
        Path("src/rotary_pendulum/RL/jax_td3.py"),
        *Path("src/rotary_pendulum/environment").glob("jax_*.py"),
        Path("src/rotary_pendulum/environment/dynamics.py"),
        Path("src/rotary_pendulum/utils/config_schema.py"),
        Path("src/rotary_pendulum/configs/physics.yaml"),
        Path("src/rotary_pendulum/configs/mission.yaml"),
        Path("pyproject.toml"),
    ]
    snapshot = machine / "source_snapshot"
    for source in sources:
        target = snapshot / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    (machine / "provenance.json").write_text(
        json.dumps(
            {
                "jax_version": jax.__version__,
                "devices": [str(device) for device in jax.devices()],
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "training_executed": False,
                "sha256": {
                    str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                    for source in sources
                },
            },
            indent=2,
        )
        + "\n"
    )
    initial, deadlines, labels = validation_resets(
        settings["validation_seed"],
        settings["episodes_per_stratum"],
        [20, 40, 60],
    )
    print(
        f"devices={jax.devices()}, paired validation lanes={len(labels)}; NO TRAINING", flush=True
    )
    summary = []
    for trial in campaign["trials"]:
        started = time.monotonic()
        experiment = {**settings, **trial, "evaluation_max_s": 60}
        print(f"START {trial['name']}: compiling/running heuristic validation", flush=True)
        evaluate_jit = jax.jit(
            lambda state, horizon, experiment=experiment, mode=trial["mode"]: evaluate(
                state, horizon, experiment, mode
            )
        )
        traces = jax.device_get(evaluate_jit(initial, deadlines))
        if not all(np.isfinite(value).all() for value in traces.values()):
            raise FloatingPointError(f"Nonfinite validation trace: {trial['name']}")
        print(
            f"SIMULATED {trial['name']} in {time.monotonic() - started:.1f}s; saving plots",
            flush=True,
        )
        rows = write_validation(root / trial["name"], traces, labels, experiment)
        for deadline in (20, 40, 60):
            for stratum in ("downward", "moving", "near", "tight", "probe"):
                selected = [
                    row
                    for row in rows
                    if row["deadline_s"] == deadline and row["stratum"] == stratum
                ]
                summary.append(
                    {
                        "trial": trial["name"],
                        "deadline_s": deadline,
                        "stratum": stratum,
                        "episodes": len(selected),
                        **{
                            key: float(np.mean([row[key] for row in selected]))
                            for key in (
                                "success",
                                "arm_violation",
                                "return",
                                "energy_error_mean",
                                "target_energy_reached",
                                "upright_visited",
                                "peak_arm_rad",
                                "peak_pendulum_speed_rad_s",
                                "filter_fraction",
                                "best_effort_fraction",
                                "duration_s",
                            )
                        },
                    }
                )
        handle = (machine / "summary.csv").open("w", newline="")
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
        handle.close()
        outcome = [
            row for row in summary if row["trial"] == trial["name"] and row["deadline_s"] == 60
        ]
        print(
            f"DONE {trial['name']}: "
            + "; ".join(
                f"{row['stratum']} hold={row['success']:.1%}, "
                f"energy≥90%={row['target_energy_reached']:.1%}, "
                f"arm excursion={row['arm_violation']:.1%}"
                for row in outcome
            ),
            flush=True,
        )
        jax.clear_caches()
    (human / "interpretation_summary.md").write_text(
        "# Heuristic-only validation\n\nNo training was executed. Each trial directory contains "
        "labelled trajectory plots and complete numerical records. Campaign interpretation "
        "must be added after inspecting those plots. Summary rates use paired resets and "
        "three deadlines; the four deterministic probes are reported separately.\n"
    )


if __name__ == "__main__":
    main()
