"""Summarize teacher/learner campaigns and independently audit recorded upright holds."""

import csv
import io
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Keep complete tables separate from concise plots; reject any false success label."""
    root = Path(sys.argv[1])
    machine, human = root / "machine-scannables", root / "human-readables"
    machine.mkdir(exist_ok=True)
    human.mkdir(exist_ok=True)
    results, learning, supervised, teachers, audits = [], [], [], [], []
    for wave in sorted(root.glob("wave-*")):
        for run in sorted(wave.iterdir()):
            settings_path = run / "machine-scannables/settings.json"
            if not settings_path.exists():
                continue
            settings = json.loads(settings_path.read_text())
            best = json.loads((run / "machine-scannables/best.json").read_text())
            result = {
                "wave": wave.name,
                "run": run.name,
                "seed": settings["seed"],
                "deadline_s": settings["deadline_s"],
                "best_success": best["success_rate"],
                "best_return": best["mean_return"],
                "best_transitions": best["transitions"],
                "completed_transitions": 0,
                "training_successes": 0,
            }
            history = run / "machine-scannables/history.jsonl"
            if history.exists():
                lines = [json.loads(line) for line in history.read_text().splitlines()]
                if lines:
                    result["completed_transitions"] = lines[-1]["transitions"]
                    result["training_successes"] = sum(line["successes"] for line in lines)
            for stage in [run / "pretrained", *sorted(run.glob("validation-*"))]:
                path = stage / "machine-scannables/episodes.csv"
                if not path.exists():
                    continue
                rows = list(csv.DictReader(path.open()))
                ordinary = [row for row in rows if row["stratum"] != "probe"]
                learning.append(
                    {
                        "wave": wave.name,
                        "run": run.name,
                        "transitions": 0
                        if stage.name == "pretrained"
                        else int(stage.name.split("-")[1]),
                        "success": np.mean([row["success"] == "True" for row in ordinary]),
                        "arm_violation": np.mean(
                            [row["arm_violation"] == "True" for row in ordinary]
                        ),
                        "duration_s": np.mean([float(row["duration_s"]) for row in ordinary]),
                    }
                )
            pretrain = run / "machine-scannables/pretrain_history.jsonl"
            if pretrain.exists():
                records = [json.loads(line) for line in pretrain.read_text().splitlines()]
                for record in records:
                    if "validation_mse" in record:
                        supervised.append({"wave": wave.name, "run": run.name, **record})
            results.append(result)
    for source in sorted(root.glob("teacher-*/*/machine-scannables/episodes.csv")):
        rows = list(csv.DictReader(source.open()))
        for label in ("downward", "moving", "near", "tight"):
            subset = [row for row in rows if row["stratum"] == label]
            for deadline in (30, 40):
                teachers.append(
                    {
                        "wave": source.parents[2].name,
                        "trial": source.parents[1].name,
                        "stratum": label,
                        "deadline_s": deadline,
                        "episodes": len(subset),
                        "success": np.mean(
                            [
                                r["success"] == "True" and float(r["duration_s"]) <= deadline
                                for r in subset
                            ]
                        ),
                        "arm_violation": np.mean([r["arm_violation"] == "True" for r in subset]),
                    }
                )
    for source in sorted(root.glob("heldout-*/*/*-policy/machine-scannables/trajectories.npz")):
        data = np.load(source, allow_pickle=False)
        x, elapsed = data["physics_x"], data["elapsed_s"]
        lanes = x.shape[1]
        beta = np.arctan2(np.sin(x[..., 1] - np.pi), np.cos(x[..., 1] - np.pi))
        goal = source.parents[1].name.split("-")[0]
        membership = abs(beta) < np.deg2rad(5)
        required = 50
        if goal == "strict":
            membership = (abs(beta) <= 0.08) & (abs(x[..., 0]) <= 0.08)
            membership &= (abs(x[..., 2]) <= 0.15) & (abs(x[..., 3]) <= 0.20)
            required = 5
        enabled = np.arange(5)[None, None, :] < np.rint(elapsed[..., None] * 50)
        hold, maximum = np.zeros(lanes, int), np.zeros(lanes, int)
        for inside, active in zip(
            membership.transpose(0, 2, 1).reshape(-1, lanes),
            enabled.transpose(0, 2, 1).reshape(-1, lanes),
            strict=True,
        ):
            hold = np.where(active, np.where(inside, hold + 1, 0), hold)
            maximum = np.maximum(maximum, hold)
        np.testing.assert_array_equal(data["success"], maximum >= required)
        for label in ("downward", "moving", "near", "tight"):
            selected = data["labels"] == label
            audits.append(
                {
                    "wave": source.parents[3].name,
                    "run": source.parents[2].name,
                    "stratum": label,
                    "goal": goal,
                    "episodes": int(selected.sum()),
                    "deadline_s": float(data["deadline_s"][selected][0]),
                    "success": float(data["success"][selected].mean()),
                    "arm_violation": float(data["arm_violation"][selected].mean()),
                    "duration_s": float(data["duration_s"][selected].mean()),
                    "mean_longest_hold_s": float(maximum[selected].mean() * 0.02),
                    "minimum_success_hold_s": float(
                        maximum[selected & data["success"]].min() * 0.02
                    )
                    if np.any(selected & data["success"])
                    else None,
                }
            )
        data.close()
        print(f"Verified all hold labels: {source.parents[2].name}", flush=True)
    for name, rows in (
        ("runs", results),
        ("learning", learning),
        ("supervised", supervised),
        ("teachers", teachers),
        ("heldout", audits),
    ):
        if rows:
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            (machine / f"{name}.csv").write_text(buffer.getvalue())
    figure, axes = plt.subplots(1, 3, figsize=(21, 5), layout="constrained")
    for wave, axis in zip(("wave-1", "wave-2", "wave-3"), axes, strict=True):
        for run in sorted({row["run"] for row in learning if row["wave"] == wave}):
            rows = sorted(
                [r for r in learning if r["wave"] == wave and r["run"] == run],
                key=lambda r: r["transitions"],
            )
            axis.plot(
                [r["transitions"] / 1e6 for r in rows],
                [r["success"] * 100 for r in rows],
                marker=".",
                label=run,
            )
        axis.set(
            title=wave,
            xlabel="Online transitions [millions]",
            ylabel="Validation success [%]",
            ylim=(-2, 102),
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.savefig(human / "training_comparison.png", dpi=160)
    plt.close(figure)
    if audits:
        audits = [row for row in audits if row["goal"] == "angle"]
        groups = list(dict.fromkeys((r["wave"], r["run"]) for r in audits))
        figure, axes = plt.subplots(
            1, 2, figsize=(15, max(5, len(groups) * 0.4)), layout="constrained"
        )
        for axis, metric, title in zip(
            axes,
            ("success", "arm_violation"),
            ("Held-out sustained capture [%]", "Any arm-bound crossing [%]"),
            strict=True,
        ):
            values = np.array(
                [
                    [r[metric] * 100 for r in audits if (r["wave"], r["run"]) == group]
                    for group in groups
                ]
            )
            axis.imshow(values, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
            axis.set(
                xticks=range(4),
                xticklabels=["Downward", "Moving", "Near", "Tight"],
                yticks=range(len(groups)),
                yticklabels=[g[1] for g in groups],
                title=title,
            )
            for index in np.ndindex(values.shape):
                axis.text(
                    index[1],
                    index[0],
                    f"{values[index]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if values[index] > 60 else "black",
                )
        figure.savefig(human / "heldout_comparison.png", dpi=160)
        plt.close(figure)
    failure = root / "numerical-diagnostic/scratch/machine-scannables/failure_transitions.npz"
    if failure.exists():
        from rotary_pendulum.environment.dynamics import rk4_step
        from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL
        from rotary_pendulum.RL.jax_residual_control import (
            ARM_SPEED_SCALE,
            PENDULUM_SPEED_SCALE,
            POLICY_TORQUE_NM,
        )

        data = np.load(failure, allow_pickle=False)
        invalid = ~np.isfinite(data["next_observation"]).all(axis=1)
        observation = data["observation"][invalid].astype(float)
        state = np.stack(
            (
                observation[:, 0] * np.pi,
                np.arctan2(observation[:, 1], observation[:, 2]),
                observation[:, 3] * float(ARM_SPEED_SCALE),
                observation[:, 4] * float(PENDULUM_SPEED_SCALE),
            ),
            axis=-1,
        )
        torque = data["action"][invalid].astype(float) * POLICY_TORQUE_NM
        numerical, traces = (
            [],
            {"initial": state, "torque": torque, "lanes": np.flatnonzero(invalid)},
        )
        for timestep in (0.02, 0.002, 0.001, 0.0005):
            following = state.copy()
            for _ in range(round(0.1 / timestep)):
                following = rk4_step(following, torque, timestep, PHYSICAL, MODEL)
            traces[f"final_{timestep}"] = following
            numerical.append(
                {"timestep_s": timestep, "all_finite": bool(np.isfinite(following).all())}
            )
        data.close()
        np.savez_compressed(machine / "integration_failure_replay.npz", **traces)
        (machine / "integration_failure_replay.json").write_text(
            json.dumps(
                {
                    "initial_state_from_observation": state.tolist(),
                    "torque_nm": torque.tolist(),
                    "replays": numerical,
                    "max_difference_1ms_05ms": np.max(
                        abs(traces["final_0.001"] - traces["final_0.0005"]), axis=0
                    ).tolist(),
                },
                indent=2,
            )
        )
    print(
        json.dumps(
            {
                "training_runs": len(results),
                "finite_transitions": sum(r["completed_transitions"] for r in results),
                "heldout_rows": len(audits),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
