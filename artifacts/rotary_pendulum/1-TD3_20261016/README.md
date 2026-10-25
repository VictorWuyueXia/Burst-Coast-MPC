# TD3_20261016

Portable inference package for the selected stationary rotary-pendulum TD3 model.
**Maximum torque: ±0.00459 N·m; policy interval: 0.1 s.**
Exported 2026-10-16 from the 2026-10-13 training campaign, not retrained.

Install `requirements.txt` in a Python 3.12 environment. The package needs only
JAX, jaxlib and NumPy; no repository imports, simulator, Flax or GPU are required.
Copy this entire directory anywhere. From that directory:

```bash
python -m pip install -r requirements.txt
JAX_PLATFORMS=cpu python verify.py
```

```python
from pathlib import Path
import jax
import jax.numpy as jnp
from inference import TD3Model

model = TD3Model(Path.cwd())  # Directory containing this README.
predict = jax.jit(model)       # Construct and compile once; reuse.
state = jnp.array([0.0, jnp.pi, 0.0, 0.0])  # Upright, stationary.
result = predict(state, 0.0)  # Physical applied torque in N·m.
print(result["q1"], result["q2"], result["q_min"])
print(result["actor_torque_nm"], result["v_actor"])
```

For imports from elsewhere, add the package directory to your application's
module search path and pass its absolute path to `TD3Model`. Set
`JAX_PLATFORMS=cpu` before importing JAX to explicitly select CPU execution.
The first call compiles; subsequent calls with the same shapes reuse compilation.
The API is differentiable with `jax.grad`; each new batch shape may compile again.

| Input/output | Meaning |
|---|---|
| `state` | Shape `(..., 4)`: unwrapped arm angle, pendulum angle, arm speed, pendulum speed; radians and rad/s |
| Pendulum convention | Zero points **down**; ±π points **up**. Do not supply upright error in place of pendulum angle. |
| `torque_nm` | Scalar or batch broadcastable to `state.shape[:-1]`; actual torque, not normalized action |
| `q1`, `q2`, `q_min` | Two online critics at the supplied torque, and their elementwise minimum |
| `actor_torque_nm` | Direct learned torque command, already scaled to ±0.00459 N·m |
| `v_actor` | Minimum of the two critics at the actor's action; candidate continuation value |

All outputs follow the state batch shape. Supply finite inputs and bounded torque.
No clipping, angle wrapping, braking filter or hidden controller is added.
The arm angle must remain unwrapped because its distance from zero affects cost.
Normalization scales are **not speed limits**. Invalid torque/state values can
produce numerical outputs but have no established physical or predictive validity.

Read [interpretation_summary.md](human-readables/interpretation_summary.md) before
using Q in MPC: it defines the reward, dynamics assumptions, selection evidence and
limits. Q accuracy is **not** established by successful actor rollouts.

`machine-scannables/manifest.json` stores exact architecture, normalization,
physics, torque limits, discount, dependencies and source/checkpoint hashes.
`weights.npz` contains 400,643 float32 parameters: actor and two online critics.
There are no optimizer or target-network states; this is an inference export.
`training_settings.json` preserves the original settings, including unused legacy
fields; the manifest and interpretation specify the active stationary contract.
`selection.json` and `heldout_comparison.csv` preserve selection/test evidence.
`reference_vectors.npz` contains 1,024 original Flax inference examples.
`verification.json` records parity, batching, gradient and CPU timing checks.
