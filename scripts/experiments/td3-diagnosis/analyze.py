"""Aggregate TD3 validation outcomes and learning diagnostics without loading JAX."""

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
    """Write numerical campaign tables and learning figures, grouped by training wave."""
    root = Path(sys.argv[1])
    human, machine = root / "human-readables", root / "machine-scannables"
    human.mkdir(parents=True, exist_ok=True)
    machine.mkdir(parents=True, exist_ok=True)
    rows, diagnostics = [], []
    for wave in sorted(root.glob("wave-*")):
        trials = sorted(wave.glob("*/machine-scannables/history.jsonl"))
        if not trials:
            continue
        figure, axes = plt.subplots(3, 3, figsize=(16, 12))
        for path in trials:
            trial = path.parent.parent
            history = [json.loads(line) for line in path.read_text().splitlines()]
            settings = json.loads((path.parent / "settings.json").read_text())
            points = []
            for evaluation in sorted(
                trial.glob("validation-*/machine-scannables/episodes.csv"),
                key=lambda p: int(p.parent.parent.name.split("-")[1]),
            ):
                count = int(evaluation.parent.parent.name.split("-")[1])
                episodes = list(csv.DictReader(io.StringIO(evaluation.read_text())))
                summary = {}
                for label in ("downward", "moving", "near", "tight"):
                    subset = [row for row in episodes if row["stratum"] == label]
                    row = {
                        "wave": wave.name,
                        "trial": trial.name,
                        "transitions": count,
                        "stratum": label,
                        "seed": settings["seed"],
                        "success": np.mean([item["success"] == "True" for item in subset]),
                        "arm_violation": np.mean(
                            [item["arm_violation"] == "True" for item in subset]
                        ),
                        **{
                            name: np.mean([float(item[name]) for item in subset])
                            for name in (
                                "return",
                                "duration_s",
                                "filter_fraction",
                                "peak_pendulum_speed_rad_s",
                            )
                        },
                    }
                    rows.append(row)
                    summary[label] = row["success"]
                points.append((count, summary))
            if points:
                for axis, label in zip(axes[0], ("downward", "moving", "near"), strict=True):
                    axis.plot(
                        [point[0] for point in points],
                        [point[1][label] for point in points],
                        label=trial.name,
                    )
                    axis.set(title=f"{label}: capture", ylabel="Episode fraction", ylim=(0, 1.05))
            for axis, name in zip(
                axes[1:].flat,
                (
                    "critic_loss",
                    "target_mean",
                    "saturation_fraction",
                    "filter_fraction",
                    "controllable_fraction",
                    "replay_terminal_fraction",
                ),
                strict=True,
            ):
                selected = [row for row in history if name in row]
                chunks = np.array_split(np.arange(len(selected)), min(80, len(selected)))
                axis.plot(
                    [np.mean([selected[i]["transitions"] for i in chunk]) for chunk in chunks],
                    [np.mean([selected[i][name] for i in chunk]) for chunk in chunks],
                    label=trial.name,
                )
                axis.set(title=name.replace("_", " "), xlabel="Collected transitions")
            tail = history[-min(256, len(history)) :]
            diagnostics.append(
                {
                    "wave": wave.name,
                    "trial": trial.name,
                    "transitions": history[-1]["transitions"],
                    "training_successes": sum(row["successes"] for row in history),
                    "training_completed": sum(row["completed"] for row in history),
                    **{
                        name: float(np.mean([row[name] for row in tail]))
                        for name in (
                            "critic_loss",
                            "target_mean",
                            "filter_fraction",
                            "saturation_fraction",
                            "controllable_fraction",
                        )
                    },
                }
            )
        for axis in axes.flat:
            axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(human / f"{wave.name}.png", dpi=140)
        plt.close(figure)
        print(f"analyzed {wave.name}: {len(trials)} runs", flush=True)
    for name, records in (("evaluations", rows), ("training_diagnostics", diagnostics)):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        (machine / f"{name}.csv").write_text(buffer.getvalue())
    outcomes, calibration = [], []
    for path in sorted(root.glob("audit-*/*/machine-scannables/outcomes.csv")):
        trial = path.parent.parent.name
        for row in csv.DictReader(io.StringIO(path.read_text())):
            outcomes.append({"trial": trial, **row})
        calibration.append(
            {"trial": trial, **json.loads((path.parent / "diagnostics.json").read_text())}
        )
    if outcomes:
        for name, records in (("heldout_outcomes", outcomes), ("heldout_diagnostics", calibration)):
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
            (machine / f"{name}.csv").write_text(buffer.getvalue())
        figure, axes = plt.subplots(1, 2, figsize=(15, 7))
        for axis, goal in zip(axes, ("angle", "strict"), strict=True):
            trials = sorted(
                {
                    row["trial"]
                    for row in outcomes
                    if row["trial"].startswith("strict") == (goal == "strict")
                }
            )
            strata = ("downward", "moving") if goal == "angle" else ("tight", "near")
            for index, label in enumerate(strata):
                selected = [
                    row
                    for row in outcomes
                    if row["goal"] == goal
                    and row["controller"] == "policy"
                    and row["stratum"] == label
                    and row["trial"] in trials
                ]
                axis.scatter(
                    [100 * float(row["success"]) for row in selected],
                    [trials.index(row["trial"]) + index * 0.18 for row in selected],
                    label=label,
                    s=45,
                )
            if goal == "strict":
                reference = next(row for row in outcomes if row["controller"] == "linear")
                axis.scatter(
                    100 * float(reference["success"]),
                    len(trials),
                    marker="*",
                    s=130,
                    color="black",
                    label="Local linear controller, tight",
                )
                trials.append("Local linear controller")
            axis.set(
                yticks=range(len(trials)),
                yticklabels=trials,
                xlim=(-2, 103),
                xlabel="Held-out capture (%)",
                title=f"Trained for {goal} capture",
            )
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(human / "heldout_results.png", dpi=150)
        plt.close(figure)
    for path in root.glob("**/human-readables/interpretation_summary.md"):
        if not path.read_text().startswith("# Validation plots"):
            continue
        settings = json.loads((path.parent.parent / "machine-scannables/settings.json").read_text())
        (path.parent / "interpretation_summary-CN.md").write_text(
            "# 验证图说明\n\nθ 为臂角；α 为从下垂测量的摆角，范围为 −180° 至 180°；"
            "β 为相对直立的环绕误差。ω 与 ν 分别为臂和摆角速度。相对摆动能量比"
            "不包含随臂运动的动能，不等同于摆体总机械能。\n\n"
            "过滤模式：0 接受，1 零扭矩，2 制动，3 所有候选预测均越界时取最小偏离。"
            "红线表示臂限位。保持比例为一表示连续五个 20 ms 目标样本后提前成功。"
            f"本次目标模式为 {settings['goal_mode']}，精确容差见实验约定。\n\n"
            "每类随机初态展示最长截止时间下的中位和最佳回报轨迹；探针全部展示。"
            "完整轨迹、选取索引及结局位于相邻 machine-scannables；总体结论另见人工报告。\n"
        )
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
