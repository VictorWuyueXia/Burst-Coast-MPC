"""Fit a sparse compute-time model from Monte Carlo RL artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "experiments"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "cmp-time-fitting"
SEEDS = (11, 23, 37, 53, 71)
THREAD_COUNT = 8
CV_FOLDS = 5
TEST_FRACTION = 0.25
ALPHAS = np.logspace(-4, 1, 90)
BASE_NAMES = [
    "bbar",
    "hbar",
    "burst_steps",
    "horizon_steps",
    "coast_steps",
    "burst_fraction",
    "horizon_duration_s",
    "abs_sin_theta",
    "one_minus_cos_theta",
    "abs_omega_normalized",
    "abs_energy_error",
]
CANDIDATES = (
    {"name": "linear-action", "width": 7, "powers": (1,), "logs": False},
    {"name": "cubic-action", "width": 7, "powers": (1, 2, 3), "logs": False},
    {"name": "power-log-action", "width": 7, "powers": (1, 2, 3, 4), "logs": True},
    {"name": "power-log-state-action", "width": 11, "powers": (1, 2, 3, 4), "logs": True},
)


def load_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for path in sorted(ARTIFACT_ROOT.glob("monte-carlo-data*/rl_steps.csv")):
        config_file = (path.parent / "config.json").open(encoding="utf-8")
        config = json.load(config_file)
        config_file.close()
        pendulum = config["environment"]["pendulum"]
        constants = {
            "run_dir": path.parent.name,
            "timestep_s": float(config["environment"]["simulation"]["timestep-s"]),
            "gravity_m_s2": float(pendulum["gravity-m-s2"]),
            "length_m": float(pendulum["length-m"]),
            "mass_kg": float(pendulum["mass-kg"]),
        }

        # Attach the minimal run constants needed for state-energy features.
        csv_file = path.open(encoding="utf-8", newline="")
        for row in csv.DictReader(csv_file):
            rows.append(
                constants
                | {
                    "s_sin_theta": float(row["s-sin-theta"]),
                    "s_cos_theta": float(row["s-cos-theta"]),
                    "s_omega_rad_s": float(row["s-omega-rad-s"]),
                    "bbar": float(row["bbar"]),
                    "hbar": float(row["hbar"]),
                    "burst_steps": float(row["burst-steps"]),
                    "horizon_steps": float(row["horizon-steps"]),
                    "solve_time_s": float(row["solve-time-s"]),
                }
            )
        csv_file.close()
    return rows


def array(rows: list[dict[str, float | str]], name: str) -> np.ndarray:
    return np.array([row[name] for row in rows], dtype=float)


def build_base(rows: list[dict[str, float | str]]) -> tuple[np.ndarray, np.ndarray]:
    timestep = array(rows, "timestep_s")
    gravity = array(rows, "gravity_m_s2")
    length = array(rows, "length_m")
    mass = array(rows, "mass_kg")
    sin_theta = array(rows, "s_sin_theta")
    cos_theta = array(rows, "s_cos_theta")
    omega = array(rows, "s_omega_rad_s")
    bbar = array(rows, "bbar")
    hbar = array(rows, "hbar")
    burst = array(rows, "burst_steps")
    horizon = array(rows, "horizon_steps")

    # Express action scale and pendulum phase-energy state in vectorized coordinates.
    inertia = mass * length**2
    target_energy = 2.0 * mass * gravity * length
    energy = 0.5 * inertia * omega**2 + mass * gravity * length * (1.0 + cos_theta)
    omega_ref = 2.0 * np.sqrt(gravity / length)
    base = np.column_stack(
        [
            bbar,
            hbar,
            burst,
            horizon,
            horizon - burst,
            burst / horizon,
            horizon * timestep,
            np.abs(sin_theta),
            1.0 - cos_theta,
            np.abs(omega) / omega_ref,
            np.abs((energy - target_energy) / target_energy),
        ]
    )
    return base, array(rows, "solve_time_s")


def expand_terms(
    candidate: dict[str, object],
    base: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    width = int(candidate["width"])
    scoped_base = base[:, :width]
    scoped_names = BASE_NAMES[:width]
    matrices: list[np.ndarray] = []
    names: list[str] = []
    penalties: list[float] = []

    # Penalize higher powers more heavily by shrinking their standardized columns.
    for power in candidate["powers"]:
        degree = int(power)
        matrices.append(scoped_base**degree)
        names.extend(f"{name}^{degree}" if degree > 1 else name for name in scoped_names)
        penalties.extend([float(2 ** (degree - 1))] * len(scoped_names))
    if bool(candidate["logs"]):
        matrices.append(np.log1p(scoped_base))
        names.extend(f"log1p_{name}" for name in scoped_names)
        penalties.extend([6.0] * len(scoped_names))
    return np.column_stack(matrices), names, np.array(penalties, dtype=float)


def fit_model(x_train: np.ndarray, y_train: np.ndarray, penalty: np.ndarray, seed: int):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x_train) / penalty
    model = LassoCV(
        alphas=ALPHAS,
        cv=CV_FOLDS,
        fit_intercept=True,
        max_iter=1_000_000,
        n_jobs=THREAD_COUNT,
        random_state=seed,
        selection="cyclic",
    )
    model.fit(scaled, y_train)
    return model, scaler


def score(
    model: LassoCV,
    scaler: StandardScaler,
    penalty: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
):
    pred = model.predict(scaler.transform(x) / penalty)
    return {
        "rmse_s": float(math.sqrt(mean_squared_error(y, pred))),
        "mae_s": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def run_ablation(base: np.ndarray, target: np.ndarray) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        x, names, penalty = expand_terms(candidate, base)
        for seed in SEEDS:
            split = train_test_split(x, target, test_size=TEST_FRACTION, random_state=seed)
            x_train, x_test, y_train, y_test = split
            model, scaler = fit_model(x_train, y_train, penalty, seed)
            records.append(
                {
                    "candidate": candidate["name"],
                    "seed": seed,
                    "alpha": float(model.alpha_),
                    "nonzero_terms": int(np.count_nonzero(model.coef_)),
                    "term_count": len(names),
                    **score(model, scaler, penalty, x_test, y_test),
                }
            )
    return records


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        subset = [row for row in records if row["candidate"] == candidate["name"]]
        summary.append(
            {
                "candidate": candidate["name"],
                "mean_rmse_s": float(np.mean([row["rmse_s"] for row in subset])),
                "mean_mae_s": float(np.mean([row["mae_s"] for row in subset])),
                "mean_r2": float(np.mean([row["r2"] for row in subset])),
                "mean_nonzero_terms": float(np.mean([row["nonzero_terms"] for row in subset])),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    file = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    file.close()


def save_model(output_dir: Path, rows, base, target, selected: dict[str, object]) -> None:
    candidate = next(item for item in CANDIDATES if item["name"] == selected["candidate"])
    x, names, penalty = expand_terms(candidate, base)
    model, scaler = fit_model(x, target, penalty, int(selected["seed"]))
    coefficients = model.coef_ / penalty
    nonzero = np.flatnonzero(np.abs(coefficients) > 0.0)

    # Store every standardized term so zeroed coefficients remain reproducible.
    all_terms = [
        {
            "name": names[index],
            "mean": float(scaler.mean_[index]),
            "scale": float(scaler.scale_[index]),
            "penalty_weight": float(penalty[index]),
            "standardized_coefficient_s": float(coefficients[index]),
        }
        for index in range(len(names))
    ]
    payload = {
        "artifact_root": str(ARTIFACT_ROOT),
        "row_count": len(rows),
        "run_dirs": sorted({str(row["run_dir"]) for row in rows}),
        "selected_ablation": selected,
        "candidate": candidate["name"],
        "seed": int(selected["seed"]),
        "alpha": float(model.alpha_),
        "intercept_s": float(model.intercept_),
        "all_terms": all_terms,
        "terms": [all_terms[index] for index in nonzero],
    }

    json_file = (output_dir / "final_model.json").open("w", encoding="utf-8")
    json.dump(payload, json_file, indent=2, sort_keys=True)
    json_file.write("\n")
    json_file.close()
    text_file = (output_dir / "model_equation.txt").open("w", encoding="utf-8")
    text_file.write(f"solve_time_s = {model.intercept_:.10g}\n")
    for index in nonzero:
        text_file.write(f"  + ({coefficients[index]:.10g}) * z({names[index]})\n")
    text_file.close()


def main() -> None:
    rows = load_rows()
    base, target = build_base(rows)
    output_dir = OUTPUT_ROOT / f"lasso_{datetime.now().astimezone():%Y%m%dT%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)

    # Bound vectorized CPU kernels for Apple Silicon shared-memory execution.
    with threadpool_limits(limits=THREAD_COUNT):
        ablation = run_ablation(base, target)
        summary = summarize(ablation)
        best_summary = min(
            summary,
            key=lambda row: (float(row["mean_rmse_s"]), float(row["mean_nonzero_terms"])),
        )
        selected = min(
            [row for row in ablation if row["candidate"] == best_summary["candidate"]],
            key=lambda row: (float(row["rmse_s"]), int(row["nonzero_terms"])),
        )
        write_csv(output_dir / "ablation.csv", ablation)
        write_csv(output_dir / "candidate_summary.csv", summary)
        save_model(output_dir, rows, base, target, selected)

    print(
        {
            "rows": len(rows),
            "best_candidate": selected["candidate"],
            "best_seed": selected["seed"],
            "best_rmse_s": selected["rmse_s"],
            "output_dir": str(output_dir),
        }
    )


if __name__ == "__main__":
    main()
