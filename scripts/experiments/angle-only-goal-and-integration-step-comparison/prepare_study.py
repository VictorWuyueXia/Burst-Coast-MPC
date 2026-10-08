"""Prepare isolated angle-only studies while preserving all physical time intervals."""

from __future__ import annotations

import difflib
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    repository = Path(__file__).resolve().parents[3]
    directory = Path(__file__).resolve().parent
    original = directory.parent / "torque-and-arm-angle-range-comparison"
    output = repository / (
        "artifacts/rotary_pendulum/experiment-results/"
        "angle-only-goal-and-integration-step-comparison/raw-runs/prepared"
    )
    output.mkdir(parents=True, exist_ok=False)
    for interval in (0.02, 0.01):
        name = f"dt{round(1000 * interval)}"
        snapshot = output / name
        snapshot.mkdir()
        shutil.copytree(
            repository / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__")
        )
        (snapshot / "scripts").mkdir()
        shutil.copy2(repository / "scripts/validate_rotary_heuristic.py", snapshot / "scripts")
        shutil.copy2(repository / "pyproject.toml", snapshot)
        runner = snapshot / original.relative_to(repository)
        runner.mkdir(parents=True)
        shutil.copy2(original / "run_study.py", runner)
        hold = round(0.1 / interval)
        replacements = {
            "src/rotary_pendulum/environment/jax_dynamics.py": [
                ("PHYSICS_DT_S = 0.02", f"PHYSICS_DT_S = {interval}"),
                ("one 20 ms physics interval", f"one {1000 * interval:g} ms physics interval"),
            ],
            "src/rotary_pendulum/configs/mission.yaml": [
                ("beta-tolerance-rad: 0.08", f"beta-tolerance-rad: {float(np.deg2rad(15))!r}"),
                ("hold-steps: 5", f"hold-steps: {hold}"),
            ],
            "src/rotary_pendulum/environment/jax_environment.py": [
                ("HOLD_PHYSICS_STEPS = 5", f"HOLD_PHYSICS_STEPS = {hold}"),
                ("MAX_PHYSICS_STEPS = 1000", f"MAX_PHYSICS_STEPS = {round(20 / interval)}"),
                (
                    """        inside_goal = (
            (jnp.abs(upright_error) <= GOAL.beta_tolerance_rad)
            & (jnp.abs(next_x[..., 2]) <= GOAL.omega_tolerance_rad_s)
            & (jnp.abs(next_x[..., 3]) <= GOAL.nu_tolerance_rad_s)
        )""",
                    "        inside_goal = jnp.abs(upright_error) <= GOAL.beta_tolerance_rad",
                ),
            ],
            "src/rotary_pendulum/heuristic/decoder.py": [
                (
                    "recovery_steps <= 5 or recovery_steps % 5",
                    f"recovery_steps <= {hold} or recovery_steps % {hold}",
                ),
                ("multiple of five, greater than five", f"multiple of {hold}, greater than {hold}"),
                ("(index >= 5) & (index % 5 == 0)", f"(index >= {hold}) & (index % {hold} == 0)"),
                ("index == 4", f"index == {hold - 1}"),
                ("predict(grid, 5)", f"predict(grid, {hold})"),
                ("predict(middle, 5)", f"predict(middle, {hold})"),
            ],
            "src/rotary_pendulum/heuristic/evaluation.py": [
                ("length=5)", f"length={hold})"),
            ],
            "src/rotary_pendulum/heuristic/artifacts.py": [
                ("0.02 *", f"{interval} *"),
                ("np.arange(1, 6)", f"np.arange(1, {hold + 1})"),
                ("axhline(0.08,", "axhline(np.deg2rad(15),"),
            ],
            "scripts/validate_rotary_heuristic.py": [
                ('campaign["recovery_steps"] <= 5', f'campaign["recovery_steps"] <= {hold}'),
                ('campaign["recovery_steps"] % 5', f'campaign["recovery_steps"] % {hold}'),
                ("multiple of five, greater than five", f"multiple of {hold}, greater than {hold}"),
                (
                    '"capture_requires_arm_centering": False,',
                    '"capture_requires_arm_centering": False,'
                    f'\n                "physics_step_s": {interval},'
                    '\n                "goal_angle_deg": 15.0,\n                "goal_hold_s": 0.1,'
                    '\n                "goal_requires_velocity": False,',
                ),
            ],
        }
        patch = []
        for relative, edits in replacements.items():
            target = snapshot / relative
            before = target.read_text()
            after = before
            for old, new in edits:
                assert old in after, (relative, old)
                after = after.replace(old, new)
            target.write_text(after)
            patch.extend(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile="a/" + relative,
                    tofile="b/" + relative,
                )
            )
        (snapshot / "goal_timing.patch").write_text("".join(patch))
        (
            repository
            / (
                "artifacts/rotary_pendulum/experiment-results/"
                "angle-only-goal-and-integration-step-comparison/records"
            )
            / f"{name}_goal_timing.patch"
        ).write_text("".join(patch))
        settings = {
            "episodes_per_stratum": 64,
            "seeds": [20261008, 20261009, 20261010],
            "decisions": 200,
            "chunk_decisions": 10,
            "cases": [
                {
                    "name": f"{name}_t{torque:g}_a{arm}",
                    "torque_multiplier": torque,
                    "arm_limit_deg": float(arm),
                    "recovery_steps": 4 * hold,
                    "work_weight": 1.0,
                }
                for torque in (0.5, 1.0, 2.0)
                for arm in (90, 180)
            ],
        }
        (
            repository
            / (
                "artifacts/rotary_pendulum/experiment-results/"
                "angle-only-goal-and-integration-step-comparison/records"
            )
            / f"integration_{round(1000 * interval)}_milliseconds_confirmation.json"
        ).write_text(json.dumps(settings, indent=2) + "\n")
        settings["episodes_per_stratum"], settings["seeds"] = 16, [20261011]
        (
            repository
            / (
                "artifacts/rotary_pendulum/experiment-results/"
                "angle-only-goal-and-integration-step-comparison/records"
            )
            / f"integration_{round(1000 * interval)}_milliseconds_initial_parameter_screening.json"
        ).write_text(json.dumps(settings, indent=2) + "\n")
        settings["episodes_per_stratum"], settings["decisions"], settings["chunk_decisions"] = (
            1,
            2,
            2,
        )
        (
            repository
            / (
                "artifacts/rotary_pendulum/experiment-results/"
                "angle-only-goal-and-integration-step-comparison/records"
            )
            / f"integration_{round(1000 * interval)}_milliseconds_smoke.json"
        ).write_text(json.dumps(settings, indent=2) + "\n")
        settings = json.loads((original / "tuning.json").read_text())
        settings["cases"].append(
            {
                "name": "both_h20_w0.01",
                "torque_multiplier": 2,
                "arm_limit_deg": 180.0,
                "recovery_steps": 20,
                "work_weight": 0.01,
            }
        )
        for case in settings["cases"]:
            case["name"] = name + "_" + case["name"]
            case["recovery_steps"] *= hold // 5
        (
            repository
            / (
                "artifacts/rotary_pendulum/experiment-results/"
                "angle-only-goal-and-integration-step-comparison/records"
            )
            / f"integration_{round(1000 * interval)}_milliseconds_tuning.json"
        ).write_text(json.dumps(settings, indent=2) + "\n")
        print(f"Prepared {name}: physics={interval}s, control=0.1s, goal_hold=0.1s", flush=True)


if __name__ == "__main__":
    main()
