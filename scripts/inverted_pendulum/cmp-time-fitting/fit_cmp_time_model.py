"""Fit a nonnegative burst-horizon compute-time model from Monte Carlo artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/private/tmp/burst-coast-mpc-matplotlib"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "MonteCarloData"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "cmp-time-fitting"
SEEDS = (11, 23, 37, 53, 71)
ALPHAS = np.logspace(-5, 0, 80)
THREAD_COUNT = 8
CV_FOLDS = 5
TEST_FRACTION = 0.25
MIN_ABSOLUTE_RMSE_IMPROVEMENT_S = 0.01
MIN_RELATIVE_RMSE_IMPROVEMENT = 0.02
CANDIDATES = (
    {"name": "linear-steps", "interaction": False},
    {"name": "bilinear-steps", "interaction": True},
)


def load_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for path in sorted(ARTIFACT_ROOT.glob("monte-carlo-data*/rl_steps.csv")):
        csv_file = path.open(encoding="utf-8", newline="")
        for row in csv.DictReader(csv_file):
            rows.append(
                {
                    "run_dir": path.parent.name,
                    "hbar": float(row["hbar"]),
                    "bbar": float(row["bbar"]),
                    "horizon_steps": float(row["horizon-steps"]),
                    "burst_steps": float(row["burst-steps"]),
                    "solve_time_s": float(row["solve-time-s"]),
                }
            )
        csv_file.close()
    return rows


def column(rows: list[dict[str, float | str]], name: str) -> np.ndarray:
    return np.array([row[name] for row in rows], dtype=float)


def build_base(rows: list[dict[str, float | str]]) -> tuple[np.ndarray, np.ndarray]:
    return np.column_stack([column(rows, "horizon_steps"), column(rows, "burst_steps")]), column(
        rows,
        "solve_time_s",
    )


def expand_terms(
    candidate: dict[str, object],
    base: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    horizon_steps = base[:, 0]
    burst_steps = base[:, 1]
    matrices = [horizon_steps, burst_steps]
    names = ["horizon_steps", "burst_steps"]

    # Add the physical interaction only when it improves held-out solve-time fit enough.
    if candidate["interaction"]:
        matrices.append(burst_steps * horizon_steps)
        names.append("burst_horizon_steps")

    return np.column_stack(matrices), names


def fit_model(x_train: np.ndarray, y_train: np.ndarray) -> tuple[LassoCV, StandardScaler]:
    scaler = StandardScaler(with_mean=False)
    model = LassoCV(
        alphas=ALPHAS,
        cv=CV_FOLDS,
        fit_intercept=False,
        max_iter=1_000_000,
        positive=True,
        random_state=0,
        selection="cyclic",
    )
    model.fit(scaler.fit_transform(x_train), y_train)
    return model, scaler


def original_coefficients(model: LassoCV, scaler: StandardScaler) -> tuple[float, np.ndarray]:
    coefficients = model.coef_ / scaler.scale_
    intercept = float(model.intercept_)
    if scaler.with_mean:
        intercept -= float(np.dot(coefficients, scaler.mean_))
    return intercept, coefficients


def predict(model: LassoCV, scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    return model.predict(scaler.transform(x))


def score(model: LassoCV, scaler: StandardScaler, x: np.ndarray, y: np.ndarray):
    prediction = predict(model, scaler, x)
    return {
        "rmse_s": float(math.sqrt(mean_squared_error(y, prediction))),
        "mae_s": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
    }


def run_ablation(base: np.ndarray, target: np.ndarray) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        x, names = expand_terms(candidate, base)
        for seed in SEEDS:
            split = train_test_split(x, target, test_size=TEST_FRACTION, random_state=seed)
            x_train, x_test, y_train, y_test = split
            model, scaler = fit_model(x_train, y_train)
            _, coefficients = original_coefficients(model, scaler)
            records.append(
                {
                    "candidate": candidate["name"],
                    "seed": seed,
                    "alpha": float(model.alpha_),
                    "nonzero_terms": int(np.count_nonzero(np.abs(coefficients) > 1.0e-12)),
                    "term_count": len(names),
                    **score(model, scaler, x_test, y_test),
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


def improvement_rows(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    by_name = {str(row["candidate"]): row for row in summary}
    base_rmse = float(by_name["linear-steps"]["mean_rmse_s"])
    rows: list[dict[str, object]] = []
    for name in ["bilinear-steps"]:
        rmse = float(by_name[name]["mean_rmse_s"])
        improvement = base_rmse - rmse
        relative = improvement / base_rmse
        rows.append(
            {
                "candidate": name,
                "mean_rmse_s": rmse,
                "rmse_improvement_vs_linear_steps_s": improvement,
                "relative_improvement_vs_linear_steps": relative,
                "passes_absolute_threshold": improvement >= MIN_ABSOLUTE_RMSE_IMPROVEMENT_S,
                "passes_relative_threshold": relative >= MIN_RELATIVE_RMSE_IMPROVEMENT,
                "necessary": (improvement >= MIN_ABSOLUTE_RMSE_IMPROVEMENT_S)
                and (relative >= MIN_RELATIVE_RMSE_IMPROVEMENT),
            }
        )
    return rows


def select_candidate(
    ablation: list[dict[str, object]],
    summary: list[dict[str, object]],
) -> dict[str, object]:
    improvements = improvement_rows(summary)
    useful = {row["candidate"] for row in improvements if row["necessary"]}
    if not useful:
        selected_name = "linear-steps"
    else:
        selected_name = min(
            [row for row in summary if row["candidate"] in useful],
            key=lambda row: (float(row["mean_rmse_s"]), float(row["mean_nonzero_terms"])),
        )["candidate"]
    return min(
        [row for row in ablation if row["candidate"] == selected_name],
        key=lambda row: (float(row["rmse_s"]), int(row["nonzero_terms"])),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    file = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    file.close()


def model_package(base: np.ndarray, target: np.ndarray, selected: dict[str, object]):
    candidate = next(item for item in CANDIDATES if item["name"] == selected["candidate"])
    x, names = expand_terms(candidate, base)
    model, scaler = fit_model(x, target)
    intercept, coefficients = original_coefficients(model, scaler)
    terms = [
        {
            "name": names[index],
            "coefficient_s": float(coefficients[index]),
        }
        for index in range(len(names))
    ]
    return candidate, model, scaler, intercept, terms


def latex_name(name: str) -> str:
    names = {
        "horizon_steps": "H",
        "burst_steps": "B",
        "burst_horizon_steps": "BH",
    }
    return names[name]


def write_equation(path: Path, intercept: float, terms: list[dict[str, float | str]]) -> None:
    file = path.open("w", encoding="utf-8")
    file.write("$$\n")
    file.write("\\hat t = ")
    file.write(f"{intercept:.10g}\n")
    for term in terms:
        coefficient = float(term["coefficient_s"])
        sign = "+" if coefficient >= 0.0 else "-"
        file.write(f"  {sign} {abs(coefficient):.10g}\\,{latex_name(str(term['name']))}\n")
    file.write("$$\n")
    file.close()


def grid_prediction(candidate, model: LassoCV, scaler: StandardScaler, base: np.ndarray):
    horizon_steps = np.linspace(base[:, 0].min(), base[:, 0].max(), 80)
    burst_steps = np.linspace(base[:, 1].min(), base[:, 1].max(), 80)
    horizon_grid, burst_grid = np.meshgrid(horizon_steps, burst_steps)
    grid_base = np.column_stack([horizon_grid.ravel(), burst_grid.ravel()])
    x_grid, _ = expand_terms(candidate, grid_base)
    pred_grid = predict(model, scaler, x_grid).reshape(horizon_grid.shape)
    return horizon_grid, burst_grid, pred_grid


def write_plots(output_dir, candidate, model, scaler, base, target) -> None:
    horizon_grid, burst_grid, pred_grid = grid_prediction(candidate, model, scaler, base)
    color_min = min(float(target.min()), float(pred_grid.min()))
    color_max = max(float(target.max()), float(pred_grid.max()))

    # Render the fitted solve-time surface over the realized MPC dimension plane.
    figure, axis = plt.subplots(figsize=(7.0, 5.2))
    contour = axis.contourf(
        horizon_grid,
        burst_grid,
        pred_grid,
        levels=24,
        cmap="viridis",
        vmin=color_min,
        vmax=color_max,
    )
    axis.scatter(
        base[:, 0],
        base[:, 1],
        c=target,
        s=16,
        alpha=0.75,
        cmap="viridis",
        vmin=color_min,
        vmax=color_max,
    )
    axis.set_xlabel("horizon steps")
    axis.set_ylabel("burst steps")
    axis.set_title("Compute-time fit, 2D")
    figure.colorbar(contour, ax=axis, label="predicted solve time [s]")
    figure.tight_layout()
    figure.savefig(output_dir / "fit_surface_2d.png", dpi=160)

    # Render measured rows against the fitted surface in physical seconds.
    figure = plt.figure(figsize=(7.0, 5.2))
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        horizon_grid,
        burst_grid,
        pred_grid,
        cmap="viridis",
        alpha=0.50,
        vmin=color_min,
        vmax=color_max,
    )
    axis.scatter(
        base[:, 0],
        base[:, 1],
        target,
        c=target,
        s=18,
        alpha=0.75,
        cmap="viridis",
        vmin=color_min,
        vmax=color_max,
    )
    axis.set_xlabel("horizon steps")
    axis.set_ylabel("burst steps")
    axis.set_zlabel("solve time [s]")
    axis.set_title("Compute-time fit, 3D")
    figure.colorbar(surface, ax=axis, label="solve time [s]", shrink=0.72)
    figure.tight_layout()
    figure.savefig(output_dir / "fit_surface_3d.png", dpi=160)


def save_model(output_dir: Path, rows, base, target, selected: dict[str, object]) -> None:
    candidate, model, scaler, intercept, terms = model_package(base, target, selected)
    payload = {
        "artifact_root": str(ARTIFACT_ROOT),
        "row_count": len(rows),
        "run_dirs": sorted({str(row["run_dir"]) for row in rows}),
        "selected_ablation": selected,
        "candidate": candidate["name"],
        "alpha": float(model.alpha_),
        "minimum_absolute_rmse_improvement_s": MIN_ABSOLUTE_RMSE_IMPROVEMENT_S,
        "minimum_relative_rmse_improvement": MIN_RELATIVE_RMSE_IMPROVEMENT,
        "intercept_s": intercept,
        "all_terms": terms,
    }

    json_file = (output_dir / "final_model.json").open("w", encoding="utf-8")
    json.dump(payload, json_file, indent=2, sort_keys=True)
    json_file.write("\n")
    json_file.close()
    write_equation(output_dir / "model_equation.md", intercept, terms)
    write_plots(output_dir, candidate, model, scaler, base, target)


def main() -> None:
    rows = load_rows()
    base, target = build_base(rows)
    output_dir = OUTPUT_ROOT / f"lasso_{datetime.now().astimezone():%Y%m%dT%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)

    # Bound vectorized CPU kernels for Apple Silicon shared-memory execution.
    with threadpool_limits(limits=THREAD_COUNT):
        ablation = run_ablation(base, target)
        summary = summarize(ablation)
        term_ablation = improvement_rows(summary)
        selected = select_candidate(ablation, summary)
        write_csv(output_dir / "ablation.csv", ablation)
        write_csv(output_dir / "candidate_summary.csv", summary)
        write_csv(output_dir / "term_ablation.csv", term_ablation)
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
    plt.show(block=True)


if __name__ == "__main__":
    main()
