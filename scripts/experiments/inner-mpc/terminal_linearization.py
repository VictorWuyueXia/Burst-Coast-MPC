"""CPU derivatives of the frozen actor and both composed terminal-cost branches."""

import hashlib
import json
from pathlib import Path

import casadi as ca
import numpy as np


def terminal_model(weights: dict[str, np.ndarray], divisors: np.ndarray) -> ca.Function:
    """Decode saved Dense/SiLU arrays while retaining all physical-state derivatives."""

    # Express the original five observations in local upright-error coordinates.
    assert divisors.shape == (5,) and np.all(np.isfinite(divisors)) and np.all(divisors > 0)
    error = ca.MX.sym("error", 4)
    observed = ca.times(
        ca.vertcat(error[0], -ca.sin(error[1]), -ca.cos(error[1]), error[2], error[3]),
        ca.DM(1 / divisors),
    )
    outputs = []
    for name in ("actor", "q1", "q2"):
        layers = sorted({int(key.split(".")[1]) for key in weights if key.startswith(name + ".")})
        assert layers and layers == list(range(len(layers)))
        hidden = observed if name == "actor" else ca.vertcat(observed, outputs[0])
        for index in layers:
            kernel = np.asarray(weights[f"{name}.{index}.kernel"], dtype=np.float64)
            bias = np.asarray(weights[f"{name}.{index}.bias"], dtype=np.float64)
            assert kernel.shape == (hidden.numel(), bias.size) and bias.ndim == 1
            assert np.all(np.isfinite(kernel)) and np.all(np.isfinite(bias))
            hidden = ca.DM(kernel).T @ hidden + ca.DM(bias)
            if index != layers[-1]:
                hidden = ca.times(hidden, 1 / (1 + ca.exp(-hidden)))
        assert hidden.numel() == 1
        outputs.append(ca.tanh(hidden) if name == "actor" else -hidden)

    # Differentiate each critic after substituting the actor and physical observation map.
    values = ca.vertcat(*outputs)
    return ca.Function(
        "terminal_derivatives",
        [error],
        [
            values,
            ca.jacobian(values, error),
            ca.hessian(outputs[1], error)[0],
            ca.hessian(outputs[2], error)[0],
        ],
        ["error"],
        ["values", "jacobian", "cost1_hessian", "cost2_hessian"],
    )


if __name__ == "__main__":
    # Require the original frozen bundle and verify its checksum before reading model arrays.
    repository = Path(__file__).resolve().parents[3]
    bundle = repository / "artifacts/rotary_pendulum/1-TD3_20261016/machine-scannables"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    checksum = hashlib.sha256((bundle / "weights.npz").read_bytes()).hexdigest()
    assert checksum == manifest["weights_sha256"]
    archive = np.load(bundle / "weights.npz", allow_pickle=False)
    arrays = {name: archive[name] for name in archive.files}
    archive.close()
    divisors = np.asarray(manifest["observation"]["divisors"], dtype=np.float64)

    # Preserve both branches because a critic tie does not have one ordinary terminal Hessian.
    print("Building frozen terminal derivatives on CPU", flush=True)
    model = terminal_model(arrays, divisors)
    origin = np.zeros(4)
    result = {name: np.asarray(value) for name, value in model(error=origin).items()}
    costs = result["values"].ravel()[1:]
    destination = (
        repository
        / "artifacts/rotary_pendulum/inner-mpc/mpc-rl-local-analysis/machine-scannables"
    )
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "terminal_at_upright.npz", error=origin, **result)
    record = dict(
        scope="Frozen actor and terminal-cost derivatives; not the executed MPC policy Jacobian",
        checkpoint_sha256=checksum,
        casadi_version=ca.__version__,
        state_order=["arm_rad", "upright_error_rad", "arm_speed_rad_s", "pendulum_speed_rad_s"],
        output_order=["normalized_actor_action", "negative_q1", "negative_q2"],
        observation_divisors=divisors.tolist(),
        actor_at_upright=float(result["values"][0, 0]),
        terminal_costs=costs.tolist(),
        critic_branch_gap=float(abs(costs[0] - costs[1])),
        active_terminal_branches=np.flatnonzero(costs == costs.max()).tolist(),
        array_shapes={name: list(value.shape) for name, value in arrays.items()},
    )
    (destination / "terminal_at_upright.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2), flush=True)
