"""Compare frozen GPU studies: OUTPUT_DIRECTORY RUN_DIRECTORY ...; no numerical reruns."""

import csv
import itertools
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
machine, human = root / "machine-scannables", root / "human-readables"
machine.mkdir(parents=True, exist_ok=True)
human.mkdir(exist_ok=True)
configurations, paired, integrity, episodes = [], [], [], []
for source in map(Path, sys.argv[2:]):
    location = source / "machine-scannables"
    design = json.loads((location / "design.json").read_text())
    stream = (location / "solves.csv").open()
    records = list(csv.DictReader(stream))
    stream.close()
    starts = np.load(location / "starts.npz")["states"]
    for index, settings in enumerate(design["configurations"]):
        rows = [row for row in records if int(row["configuration"]) == index]
        if not rows:
            continue
        accepted = [row for row in rows if row["accepted"] == "True"]
        first = [row for row in rows if int(row["decision"]) == 0]
        costs = {
            int(row["episode"]): float(row["objective"])
            for row in first
            if row["accepted"] == "True"
        }
        summary = dict(
            run=source.name,
            configuration=index,
            settings=settings,
            states=starts.tolist(),
            costs=costs,
            attempts=len(rows),
            accepted=len(accepted),
            completed=sum(int(r["decision"]) == design["decisions"] - 1 for r in accepted),
            decisions=design["decisions"],
            initial_count=len(starts),
            median_batch_s=float(np.median([float(r["elapsed_s"]) for r in rows])),
            median_throughput_s=float(np.median([float(r["throughput_s"]) for r in rows])),
            median_first_objective=float(np.median(list(costs.values()))) if costs else None,
            mean_first_objective=float(np.mean(list(costs.values()))) if costs else None,
        )
        for metric in ("executed_arm_rms_rad", "executed_upright_rms_rad", "arm_margin_rad"):
            summary[metric] = (
                float(np.median([float(r[metric]) for r in accepted])) if accepted else None
            )
        configurations.append(summary)
        # Audit every accepted retained plan and every transition, independently of the CSV.
        prior = None
        verified = shifted = 0
        paths = [[] for _ in starts]
        for step in sorted((location / f"config-{index:02d}").glob("step-*")):
            archive = np.load(step / "input.npz")
            inputs = dict(archive)
            archive.close()
            archive = np.load(step / "plans.npz")
            plans = dict(archive)
            archive.close()
            mask = inputs["alive"] & plans["accepted"].any(axis=-1)
            ids = np.flatnonzero(mask)
            selected = plans["selected"][ids]
            for episode, seed in zip(ids, selected, strict=True):
                paths[episode].append(plans["refined"][episode, seed, 0])
            angles = plans["refined"][ids, selected, ..., 0]
            assert np.isfinite(plans["refined"][ids, selected]).all()
            assert (np.abs(angles) < settings["plant"]["observation_scales"][0]).all()
            assert (np.abs(plans["plans"][ids, selected, :-1]) <= 1).all()
            assert plans["physical_pass"][ids, selected].all()
            if prior is not None:
                live = inputs["alive"]
                np.testing.assert_allclose(inputs["states"][live], prior[0][live], atol=1e-12)
                np.testing.assert_allclose(
                    inputs["seeds"][live, 0, :-1], prior[1][live, 1:], atol=1e-12
                )
                shifted += int(live.sum())
            batch = np.arange(len(starts))
            prior = (
                plans["refined"][batch, plans["selected"], 0, -1],
                plans["plans"][batch, plans["selected"], :-1],
            )
            verified += len(ids)
        integrity.append(
            dict(run=source.name, configuration=index, verified=verified, shifted=shifted)
        )
        print(f"Audited {source.name}:{index}: {verified} accepted, {shifted} shifts", flush=True)
        completed_episodes = []
        for episode, path in enumerate(paths):
            if not path:
                continue
            trajectory = np.concatenate(path)
            upright = np.arctan2(-np.sin(trajectory[:, 1]), -np.cos(trajectory[:, 1]))
            complete = len(path) == design["decisions"]
            item = dict(
                run=source.name,
                configuration=index,
                episode=episode,
                completed=complete,
                executed_decisions=len(path),
                arm_rms_rad=float(np.sqrt(np.mean(trajectory[:, 0] ** 2))),
                upright_rms_rad=float(np.sqrt(np.mean(upright**2))),
                final_second_upright_rms_rad=float(np.sqrt(np.mean(upright[-1000:] ** 2))),
                final_second_arm_rms_rad=float(np.sqrt(np.mean(trajectory[-1000:, 0] ** 2))),
                final_second_in_5deg=bool(
                    complete and len(path) >= 10 and np.all(np.abs(upright[-1000:]) < np.deg2rad(5))
                ),
            )
            episodes.append(item)
            if complete:
                completed_episodes.append(item)
        summary["final_second_in_5deg"] = sum(r["final_second_in_5deg"] for r in completed_episodes)
        for metric in (
            "arm_rms_rad",
            "upright_rms_rad",
            "final_second_upright_rms_rad",
            "final_second_arm_rms_rad",
        ):
            summary["mean_" + metric] = (
                float(np.mean([r[metric] for r in completed_episodes]))
                if completed_episodes
                else None
            )

for first, second in itertools.combinations(configurations, 2):
    a, b = first["settings"], second["settings"]
    if (
        a["horizon"] != b["horizon"]
        or a["plant"] != b["plant"]
        or first["states"] != second["states"]
    ):
        continue
    common = sorted(first["costs"].keys() & second["costs"].keys())
    if not common:
        continue
    delta = np.array([second["costs"][k] - first["costs"][k] for k in common])
    paired.append(
        dict(
            first=f"{first['run']}:{first['configuration']}",
            second=f"{second['run']}:{second['configuration']}",
            common=len(common),
            second_wins=int((delta < -1e-6).sum()),
            first_wins=int((delta > 1e-6).sum()),
            median_second_minus_first=float(np.median(delta)),
            mean_second_minus_first=float(np.mean(delta)),
            worst_second_minus_first=float(delta.max()),
        )
    )

(machine / "comparison.json").write_text(
    json.dumps(configurations, indent=2, allow_nan=False) + "\n"
)
(machine / "paired.json").write_text(json.dumps(paired, indent=2, allow_nan=False) + "\n")
(machine / "audit.json").write_text(json.dumps(integrity, indent=2) + "\n")
(machine / "episodes.json").write_text(json.dumps(episodes, indent=2) + "\n")
# Paired horizon/weight panel: no mixed-objective ranking.
figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for solver, marker in (("gpu_sqp", "o"), ("gpu_cem", "x")):
    selected = [
        c
        for c in configurations
        if c["settings"]["solver"] == solver
        and c["settings"]["horizon"] == 10
        and c["settings"]["plant"]["centering_cost_per_s"] == 1.0
        and c["decisions"] == 1
        and c["initial_count"] == 64
    ]
    axes[0].scatter(
        [c["median_batch_s"] for c in selected],
        [c["mean_first_objective"] for c in selected],
        marker=marker,
        label=solver.replace("gpu_", "GPU ").upper(),
    )
    for c in selected:
        if c["run"].endswith("cem-refine"):
            axes[0].annotate(
                f"{c['settings']['population']} x {c['settings']['generations']}",
                (c["median_batch_s"], c["mean_first_objective"]),
                xytext=(-4, 5),
                textcoords="offset points",
                ha="right",
                fontsize=7,
            )
selected = [c for c in configurations if c["decisions"] > 1]
for solver, marker in (("gpu_sqp", "o"), ("gpu_cem", "x")):
    points = [c for c in selected if c["settings"]["solver"] == solver and c["completed"]]
    axes[1].scatter(
        [np.rad2deg(c["mean_final_second_arm_rms_rad"]) for c in points],
        [np.rad2deg(c["mean_final_second_upright_rms_rad"]) for c in points],
        marker=marker,
        label=solver.replace("gpu_", "GPU ").upper(),
    )
    for c in points:
        axes[1].annotate(
            f"w={c['settings']['plant']['centering_cost_per_s']:g}"
            + (", spread 0.8" if c["run"].endswith("confirm-cem") else ""),
            (
                np.rad2deg(c["mean_final_second_arm_rms_rad"]),
                np.rad2deg(c["mean_final_second_upright_rms_rad"]),
            ),
            fontsize=8,
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
        )
axes[1].set(
    xlabel="Final-second arm RMS (degrees)",
    xlim=(0, 55),
    ylabel="Final-second upright-error RMS (degrees)",
    title="5 s replanning: 32-episode means\nw = centering rate; default CEM spread 0.5",
)
axes[1].legend(loc="lower right")
axes[0].set(
    xlabel="Whole-batch solve + audit latency (s)",
    ylabel="Mean first-plan cost",
    title=("64 paired states: 1 s horizon, centering rate 1\n"
           "Labels: samples x generations, spread 0.8"),
)
axes[0].legend()
figure.savefig(human / "gpu_tradeoff.png", dpi=180)
plt.close(figure)
print(
    f"Compared {len(configurations)} configurations; "
    f"audited {sum(r['verified'] for r in integrity)} accepted plans"
)
