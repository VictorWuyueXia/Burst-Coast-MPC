"""CPU-only integration evidence for dense rewards; no learning or checkpoint selection."""

import csv
import io
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_environment import EnvState
from rotary_pendulum.RL.jax_evaluation import evaluate
from rotary_pendulum.RL.jax_experiment import validate_experiment
from rotary_pendulum.RL.jax_q import QNetwork
from rotary_pendulum.RL.jax_validation_plots import plot_validation


def main() -> None:
    """Evaluate fixed starts and save reproducible component and trajectory evidence."""

    experiment_path = Path(
        "docs/10-rl-five-action-local-capture/machine-scannables/dense_experiment.json"
    )
    experiment = json.loads(experiment_path.read_text())
    validate_experiment(experiment)
    output = Path("artifacts/rotary_pendulum/q-prior-v3/implementation-check")
    machine = output / "machine-scannables"
    machine.mkdir(parents=True, exist_ok=False)
    network = QNetwork(tuple(experiment["hidden_widths"]), experiment["activation_name"])
    params = jax.tree.map(
        jnp.zeros_like, network.init(jax.random.key(31), jnp.zeros((1, 7), dtype=jnp.float32))
    )
    rows = []
    all_trajectories = {}
    for mode in ("greedy", "lookahead_zero"):
        compiled = jax.jit(
            lambda states, mode_=mode: evaluate({"params": params}, states, mode_, experiment)
        )
        retained = {}
        for index, (name, extent) in enumerate(
            (("tight", [0.08, 0.12, 0.15, 0.30]), ("near", [0.25, 0.25, 1.0, 1.0]))
        ):
            x = jax.random.uniform(
                jax.random.key(320 + index), (16, 4), minval=-1.0, maxval=1.0
            ) * jnp.array(extent)
            x = x.at[:, 1].add(jnp.pi).at[0].set(jnp.array([0.0, jnp.pi, 0.0, 0.0]))
            zero_i = jnp.zeros(16, dtype=jnp.int32)
            zero_b = jnp.zeros(16, dtype=jnp.bool_)
            states = EnvState(x, zero_i, zero_i, zero_b, zero_b, zero_b)
            print(f"dense-validation controller={mode} stratum={name} episodes=16", flush=True)
            metrics, trajectories = jax.device_get(compiled(states))
            for key in ("state", "reward_components", "time_s"):
                if not np.isfinite(trajectories[key]).all():
                    raise FloatingPointError(f"Nonfinite diagnostic {mode}/{name}/{key}")
            rows.append(
                {
                    "controller": mode,
                    "stratum": name,
                    **{key: np.asarray(value).item() for key, value in metrics.items()},
                }
            )
            retained.update(
                {f"{name}_{key}": np.asarray(value) for key, value in trajectories.items()}
            )
            print(
                f"dense-validation complete success={float(metrics['success_rate']):.4f} "
                f"excursion={float(metrics['arm_violation_rate']):.4f}",
                flush=True,
            )
        human = output / "human-readables" / mode
        human.mkdir(parents=True)
        plot_validation(human, retained)
        all_trajectories.update({f"{mode}_{key}": value for key, value in retained.items()})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (machine / "evaluation.csv").write_text(buffer.getvalue())
    (machine / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    np.savez_compressed(machine / "trajectories.npz", **all_trajectories)  # type: ignore[arg-type]
    print(f"dense-validation artifacts={output}", flush=True)


if __name__ == "__main__":
    main()
