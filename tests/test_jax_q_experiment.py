"""Resolved reward contract and restricted GPU-launch preflight checks."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rotary_pendulum.RL.jax_experiment import validate_experiment
from scripts.launch_rotary_q_wave import main


def test_dense_example_validates_and_legacy_or_invalid_rewards_are_rejected() -> None:
    path = Path("docs/10-rl-five-action-local-capture/machine-scannables/dense_experiment.json")
    experiment = json.loads(path.read_text())
    assert validate_experiment(experiment) == 16
    for change in (
        {"contract_id": "rotary-q-prior-v2"},
        {"capture_weight": 1.0},
        {"arm_failure_cost": 5.0},
        {"time_cost_per_s": 1.0},
        {"upright_widths": [0.16, 0.0, 0.3]},
        {"energy_cost_per_s": float("nan")},
        {"reward_revision": 0},
    ):
        with pytest.raises(ValueError):
            validate_experiment({**experiment, **change})


@pytest.mark.parametrize("indices", [[4], [7], [0, 0], [], [0, 1, 2, 3, 4], ["0"]])
def test_gpu_preflight_rejects_before_writing_or_launching(tmp_path: Path, indices: list) -> None:
    plan_path = tmp_path / "wave.json"
    wave_dir = tmp_path / "not-created"
    plan_path.write_text(
        json.dumps({"wave_dir": str(wave_dir), "trials": [{"gpu": index} for index in indices]})
    )
    with patch.object(sys, "argv", ["launch_rotary_q_wave.py", str(plan_path)]):
        with pytest.raises(ValueError, match="physical GPUs"):
            main()
    assert not wave_dir.exists()
