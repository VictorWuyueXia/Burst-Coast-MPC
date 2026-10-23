"""Recreate solver comparison tables and figures from recorded solves; takes a run directory."""

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
machine, human = root / "machine-scannables", root / "human-readables"
design = json.loads((machine / "design.json").read_text())
stream = (machine / "solves.csv").open()
records = list(csv.DictReader(stream))
stream.close()
summary = []
for config_id, settings in enumerate(design["configurations"]):
    selected = [row for row in records if int(row["configuration"]) == config_id]
    accepted = [row for row in selected if row["accepted"] == "True"]
    summary.append(
        dict(
            configuration=config_id,
            solver=settings["solver"],
            horizon=settings["horizon"],
            attempted=len(selected),
            accepted=len(accepted),
            completed=sum(int(row["decision"]) == design["decisions"] - 1 for row in accepted),
            median_objective=float(np.median([float(row["objective"]) for row in accepted]))
            if accepted
            else None,
            p50_s=float(np.median([float(row["elapsed_s"]) for row in selected])),
            p95_s=float(np.quantile([float(row["elapsed_s"]) for row in selected], 0.95)),
        )
    )
(machine / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

# A numerical table stays machine-readable; this figure shows rates and elapsed time.
figure, axes = plt.subplots(
    1, 2, figsize=(12, max(4, len(summary) * 0.35)), constrained_layout=True
)
labels = [f"{row['configuration']}: {row['solver']} / {row['horizon'] * 0.1:g}s" for row in summary]
axes[0].barh(labels, [row["accepted"] / row["attempted"] for row in summary])
axes[0].set(xlabel="Accepted plans / attempted solves", xlim=(0, 1))
axes[1].barh(labels, [row["p95_s"] for row in summary])
axes[1].axvline(0.1, color="red", linestyle="--", label="100 ms decision period")
axes[1].set(xlabel="95th percentile solver + validation time (s)")
axes[1].legend()
figure.savefig(human / "solver_comparison.png", dpi=160)
plt.close(figure)
print(f"Rebuilt summary and figure for {len(records)} attempted solves", flush=True)
