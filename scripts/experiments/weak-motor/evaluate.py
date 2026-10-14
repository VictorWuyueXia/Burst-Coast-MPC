"""Evaluate a saved actor with an explicitly declared motor torque limit."""

import json
import sys
from pathlib import Path

import flax.serialization
import jax
import numpy as np

from rotary_pendulum.RL.jax_residual_evaluation import evaluate, validation_resets, write_validation
from rotary_pendulum.RL.jax_td3 import initialize


def main() -> None:
    """Restore actor parameters and save fresh physical trajectories at the requested cap."""
    settings = json.loads(Path(sys.argv[1]).read_text())
    root = Path(settings["run_dir"])
    learner = flax.serialization.from_bytes(
        initialize(jax.random.PRNGKey(settings["seed"]), settings),
        Path(settings["checkpoint_path"]).read_bytes(),
    )
    initial, deadlines, labels = validation_resets(
        settings["validation_seed"], settings["episodes_per_stratum"], [settings["deadline_s"]]
    )
    print(
        f"Evaluating {settings['torque_limit_nm'] * 1000:g} mN m on {len(labels)} starts",
        flush=True,
    )
    traces = jax.device_get(
        jax.jit(lambda: evaluate(initial, deadlines, settings, "policy", learner["actor"]))()
    )
    if not np.isfinite(traces["physics_x"]).all():
        raise FloatingPointError("Nonfinite physical trajectory")
    assert np.max(abs(traces["applied_nm"])) <= settings["torque_limit_nm"] + 1e-8
    print("Rollout complete; saving physical records and plots", flush=True)
    rows = write_validation(root, traces, labels, settings)
    summary = {
        label: {
            "success": np.mean([row["success"] for row in rows if row["stratum"] == label]),
            "arm_violation": np.mean(
                [row["arm_violation"] for row in rows if row["stratum"] == label]
            ),
        }
        for label in ("downward", "moving", "near", "tight")
    }
    (root / "machine-scannables/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
