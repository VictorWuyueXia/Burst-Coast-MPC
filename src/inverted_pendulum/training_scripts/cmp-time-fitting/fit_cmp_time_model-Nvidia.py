"""Fit a compact compute-time model on the Ubuntu NVIDIA workstation data."""
# ruff: noqa: E402

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

# Resolve Ubuntu workstation data and output locations.
REPO_ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "inverted_pendulum" / "MonteCarloData"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "inverted_pendulum" / "cmp-time-fitting"
MPLCONFIGDIR = Path("/tmp/burst-coast-mpc-matplotlib")
MPLBACKEND = "Agg"
THREAD_COUNT = "12"

# Pin headless plotting and BLAS thread counts before numerical imports.
os.environ["MPLBACKEND"] = MPLBACKEND
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
os.environ["OMP_NUM_THREADS"] = THREAD_COUNT
os.environ["OPENBLAS_NUM_THREADS"] = THREAD_COUNT
os.environ["MKL_NUM_THREADS"] = THREAD_COUNT
os.environ["NUMEXPR_NUM_THREADS"] = THREAD_COUNT

# Import numerical and modeling libraries after environment setup.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Expose fitting, validation, and plotting hyperparameters as file-top variables.
FitResult = tuple[LassoCV, StandardScaler, float, np.ndarray]
RUN_LABEL = "nvidia-4070"
OUTPUT_PREFIX = "lasso"
CSV_PATTERN = "monte-carlo-data*/rl_steps.csv"
SEEDS = (11, 23, 37, 53, 71)
ALPHAS = np.logspace(-5, 0, 80)
CV_FOLDS = 5
TEST_FRACTION = 0.25
LASSO_MAX_ITER = 1_000_000
LASSO_RANDOM_STATE = 0
COEFFICIENT_ZERO_TOLERANCE = 1.0e-12
MIN_ABSOLUTE_RMSE_IMPROVEMENT_S = 0.01
MIN_RELATIVE_RMSE_IMPROVEMENT = 0.02
GRID_POINT_COUNT = 80
PLOT_DPI = 160
PLOT_FIGSIZE = (7.0, 5.2)
CANDIDATES = (  # Enumerate the candidate feature families tested by cross-validation.
    {"name": "bilinear", "log_terms": ()},
    {"name": "bilinear-hlog", "log_terms": ("hbar_log1p_hbar",)},
    {"name": "bilinear-blog", "log_terms": ("bbar_log1p_bbar",)},
    {"name": "bilinear-log", "log_terms": ("hbar_log1p_hbar", "bbar_log1p_bbar")},
)
LATEX_TERM_NAMES = {  # Map fitted term names to report equation notation.
    "hbar": "\\tilde H",
    "bbar": "\\tilde B",
    "bbar_hbar": "\\tilde B\\tilde H",
    "hbar_log1p_hbar": "\\tilde H\\log(1+\\tilde H)",
    "bbar_log1p_bbar": "\\tilde B\\log(1+\\tilde B)",
}

# Read Monte Carlo rows into aligned metadata, feature, and target arrays.
def load_rows() -> tuple[list[dict[str, float | str]], np.ndarray, np.ndarray]:
    rows: list[dict[str, float | str]] = []
    for path in sorted(ARTIFACT_ROOT.glob(CSV_PATTERN)):
        csv_file = path.open(encoding="utf-8", newline="")
        for row in csv.DictReader(csv_file):
            rows.append(
                {
                    "run_dir": path.parent.name,
                    "hbar": float(row["hbar"]),
                    "bbar": float(row["bbar"]),
                    "solve_time_s": float(row["solve-time-s"]),
                }
            )
        csv_file.close()
    base = np.array([[row["hbar"], row["bbar"]] for row in rows], dtype=float)
    target = np.array([row["solve_time_s"] for row in rows], dtype=float)
    return rows, base, target

# Build the design matrix with candidate-specific log interactions.
def feature_matrix(candidate: dict[str, object], base: np.ndarray) -> tuple[np.ndarray, list[str]]:
    hbar = base[:, 0]
    bbar = base[:, 1]
    matrices = [hbar, bbar, bbar * hbar]
    names = ["hbar", "bbar", "bbar_hbar"]
    log_terms = candidate["log_terms"]
    if "hbar_log1p_hbar" in log_terms:
        matrices.append(hbar * np.log1p(hbar))
        names.append("hbar_log1p_hbar")
    if "bbar_log1p_bbar" in log_terms:
        matrices.append(bbar * np.log1p(bbar))
        names.append("bbar_log1p_bbar")
    return np.column_stack(matrices), names

# Fit scaled LassoCV and convert coefficients back to raw feature units.
def fit_lasso(x_train: np.ndarray, y_train: np.ndarray) -> FitResult:
    scaler = StandardScaler()
    model = LassoCV(
        alphas=ALPHAS, cv=CV_FOLDS, fit_intercept=True, max_iter=LASSO_MAX_ITER,
        random_state=LASSO_RANDOM_STATE, selection="cyclic",
    )
    model.fit(scaler.fit_transform(x_train), y_train)
    coefficients = model.coef_ / scaler.scale_
    intercept = float(model.intercept_ - np.dot(coefficients, scaler.mean_))
    return model, scaler, intercept, coefficients

# Score candidates across fixed splits and select the simplest useful form.
def evaluate_candidates(base: np.ndarray, target: np.ndarray):
    ablation: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    term_ablation: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        x, names = feature_matrix(candidate, base)
        for seed in SEEDS:
            x_train, x_test, y_train, y_test = train_test_split(
                x, target, test_size=TEST_FRACTION, random_state=seed
            )
            model, scaler, _, coefficients = fit_lasso(x_train, y_train)
            prediction = model.predict(scaler.transform(x_test))
            ablation.append(
                {
                    "candidate": candidate["name"],
                    "seed": seed,
                    "alpha": float(model.alpha_),
                    "nonzero_terms": int(
                        np.count_nonzero(np.abs(coefficients) > COEFFICIENT_ZERO_TOLERANCE)
                    ),
                    "term_count": len(names),
                    "rmse_s": float(math.sqrt(mean_squared_error(y_test, prediction))),
                    "mae_s": float(mean_absolute_error(y_test, prediction)),
                    "r2": float(r2_score(y_test, prediction)),
                }
            )
        rows = [row for row in ablation if row["candidate"] == candidate["name"]]
        summary.append(
            {
                "candidate": candidate["name"],
                "mean_rmse_s": float(np.mean([row["rmse_s"] for row in rows])),
                "mean_mae_s": float(np.mean([row["mae_s"] for row in rows])),
                "mean_r2": float(np.mean([row["r2"] for row in rows])),
                "mean_nonzero_terms": float(np.mean([row["nonzero_terms"] for row in rows])),
            }
        )

    by_name = {str(row["candidate"]): row for row in summary}
    base_rmse = float(by_name["bilinear"]["mean_rmse_s"])
    for name in ("bilinear-hlog", "bilinear-blog", "bilinear-log"):
        rmse = float(by_name[name]["mean_rmse_s"])
        improvement = base_rmse - rmse
        relative = improvement / base_rmse
        term_ablation.append(
            {
                "candidate": name,
                "mean_rmse_s": rmse,
                "rmse_improvement_vs_bilinear_s": improvement,
                "relative_improvement_vs_bilinear": relative,
                "passes_absolute_threshold": improvement >= MIN_ABSOLUTE_RMSE_IMPROVEMENT_S,
                "passes_relative_threshold": relative >= MIN_RELATIVE_RMSE_IMPROVEMENT,
                "necessary": (improvement >= MIN_ABSOLUTE_RMSE_IMPROVEMENT_S)
                and (relative >= MIN_RELATIVE_RMSE_IMPROVEMENT),
            }
        )
    # Prefer the bilinear baseline unless a log term clears both thresholds.
    useful = {row["candidate"] for row in term_ablation if row["necessary"]}
    selected_name = "bilinear"
    if useful:
        selected_name = min(
            [row for row in summary if row["candidate"] in useful],
            key=lambda row: (float(row["mean_rmse_s"]), float(row["mean_nonzero_terms"])),
        )["candidate"]
    selected = min(
        [row for row in ablation if row["candidate"] == selected_name],
        key=lambda row: (float(row["rmse_s"]), int(row["nonzero_terms"])),
    )
    return ablation, summary, term_ablation, selected

# Persist a rectangular metric table with stable field ordering.
def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    csv_file = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    csv_file.close()


def write_outputs(
    rows: list[dict[str, float | str]],
    base: np.ndarray,
    target: np.ndarray,
    ablation: list[dict[str, object]],
    summary: list[dict[str, object]],
    term_ablation: list[dict[str, object]],
    selected: dict[str, object],
) -> Path:
    # Create one timestamped report directory with metrics, model, equation, and plots.
    output_dir = OUTPUT_ROOT / f"{OUTPUT_PREFIX}_{datetime.now().astimezone():%Y%m%dT%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, artifact_rows in (
        ("ablation.csv", ablation),
        ("candidate_summary.csv", summary),
        ("term_ablation.csv", term_ablation),
    ):
        write_csv(output_dir / name, artifact_rows)
    candidate = next(item for item in CANDIDATES if item["name"] == selected["candidate"])
    x, names = feature_matrix(candidate, base)
    model, scaler, intercept, coefficients = fit_lasso(x, target)
    terms = [
        {"name": names[index], "coefficient_s": float(coefficients[index])}
        for index in range(len(names))
    ]
    # Serialize the final model package for downstream controller tuning.
    payload = {
        "run_label": RUN_LABEL,
        "artifact_root": str(ARTIFACT_ROOT),
        "thread_count": int(THREAD_COUNT),
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
    # Write the fitted equation in manuscript-ready LaTeX form.
    equation_file = (output_dir / "model_equation.md").open("w", encoding="utf-8")
    equation_file.write("$$\n")
    equation_file.write(f"\\hat t = {intercept:.10g}\n")
    for term in terms:
        coefficient = float(term["coefficient_s"])
        sign = "+" if coefficient >= 0.0 else "-"
        equation_file.write(
            f"  {sign} {abs(coefficient):.10g}\\,{LATEX_TERM_NAMES[str(term['name'])]}\n"
        )
    equation_file.write("$$\n")
    equation_file.close()
    # Predict the fitted surface over a regular normalized action grid.
    hbar = np.linspace(base[:, 0].min(), base[:, 0].max(), GRID_POINT_COUNT)
    bbar = np.linspace(base[:, 1].min(), base[:, 1].max(), GRID_POINT_COUNT)
    hbar_grid, bbar_grid = np.meshgrid(hbar, bbar)
    grid_base = np.column_stack([hbar_grid.ravel(), bbar_grid.ravel()])
    x_grid, _ = feature_matrix(candidate, grid_base)
    pred_grid = model.predict(scaler.transform(x_grid)).reshape(hbar_grid.shape)
    # Render the fitted solve-time surface over measured action samples.
    figure, axis = plt.subplots(figsize=PLOT_FIGSIZE)
    contour = axis.contourf(hbar_grid, bbar_grid, pred_grid, levels=24, cmap="viridis")
    axis.scatter(base[:, 0], base[:, 1], c="black", s=16, alpha=0.68)
    axis.set_xlabel("hbar")
    axis.set_ylabel("bbar")
    axis.set_title("Compute-time fit, 2D")
    figure.colorbar(contour, ax=axis, label="predicted solve time [s]")
    figure.tight_layout()
    figure.savefig(output_dir / "fit_surface_2d.png", dpi=PLOT_DPI)
    plt.close(figure)
    # Render measured solve times against the same fitted surface in 3D.
    figure = plt.figure(figsize=PLOT_FIGSIZE)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(hbar_grid, bbar_grid, pred_grid, cmap="viridis", alpha=0.72)
    axis.scatter(base[:, 0], base[:, 1], target, c="black", s=18)
    axis.set_xlabel("hbar")
    axis.set_ylabel("bbar")
    axis.set_zlabel("solve time [s]")
    axis.set_title("Compute-time fit, 3D")
    figure.tight_layout()
    figure.savefig(output_dir / "fit_surface_3d.png", dpi=PLOT_DPI)
    plt.close(figure)
    return output_dir

# Run the complete workflow and print the selected artifact location.
def main() -> None:
    rows, base, target = load_rows()
    ablation, summary, term_ablation, selected = evaluate_candidates(base, target)
    output_dir = write_outputs(rows, base, target, ablation, summary, term_ablation, selected)
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
