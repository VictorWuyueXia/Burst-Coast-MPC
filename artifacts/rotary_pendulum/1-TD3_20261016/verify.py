"""CPU parity against archived Flax outputs, batching, gradients and warm timing."""

import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from inference import TD3Model

root = Path(__file__).resolve().parent
model = TD3Model(root)
reference = np.load(root / "machine-scannables/reference_vectors.npz", allow_pickle=False)
state = jnp.asarray(reference["state"])
torque = jnp.asarray(reference["torque_nm"])
predict = jax.jit(model)
result = predict(state, torque)
errors = {}
for name, actual in result.items():
    expected = reference[name]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    errors[name] = float(np.max(np.abs(np.asarray(actual) - expected)))

single = predict(state[0], torque[0])
for name in result:
    np.testing.assert_allclose(single[name], result[name][0], rtol=2e-5, atol=2e-5)
broadcast = predict(state[:4].reshape(2, 2, 4), 0.0)
assert broadcast["q_min"].shape == (2, 2)
np.testing.assert_allclose(
    broadcast["q_min"].reshape(4), predict(state[:4], jnp.zeros(4))["q_min"], rtol=2e-5, atol=2e-5
)
assert bool(jnp.max(jnp.abs(result["actor_torque_nm"])) <= jnp.float32(model.max_torque))

# Gradients through both the SI-unit normalization and the neural networks.
gradients = jax.grad(lambda x, u: model(x, u)["q_min"], argnums=(0, 1))(state[5], torque[5])
assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradients)
start = perf_counter()
for _ in range(100):
    predict(state[0], torque[0])["q_min"].block_until_ready()
single_ms = (perf_counter() - start) * 10
start = perf_counter()
for _ in range(30):
    predict(state, torque)["q_min"].block_until_ready()
batch_ms = (perf_counter() - start) * 1000 / 30
report = {
    "passed": True,
    "device": str(jax.devices()[0]),
    "reference_cases": len(state),
    "reference": "Original training-snapshot Flax online actor and critics",
    "rtol": 2e-5,
    "atol": 2e-5,
    "max_absolute_error": errors,
    "single_and_multiaxis_batch": "passed",
    "finite_state_torque_gradients": "passed",
    "warm_single_call_ms": single_ms,
    "warm_1024_batch_ms": batch_ms,
    "timing_scope": "Full API including actor and both critics at supplied and actor actions",
    "q_calibration_validated": False,
}
(root / "machine-scannables/verification.json").write_text(json.dumps(report, indent=2) + "\n")
reference.close()
print(json.dumps(report, indent=2))
