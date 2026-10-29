"""Standalone float32 TD3 inference; state in radians/rad/s, torque in N m."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


class TD3Model:
    """Load once, then use jax.jit(model) for batched, differentiable inference."""

    def __init__(self, package_dir):
        root = Path(package_dir) / "machine-scannables"
        self.manifest = json.loads((root / "manifest.json").read_text())
        weights_path = root / "weights.npz"
        if hashlib.sha256(weights_path.read_bytes()).hexdigest() != self.manifest["weights_sha256"]:
            raise ValueError("Weights checksum does not match manifest")
        archive = np.load(weights_path, allow_pickle=False)
        self.weights = {}
        for name, inputs in (("actor", 5), ("q1", 6), ("q2", 6)):
            layers = []
            for index, outputs in enumerate((256, 256, 256, 1)):
                kernel = archive[f"{name}.{index}.kernel"]
                bias = archive[f"{name}.{index}.bias"]
                if kernel.shape != (inputs, outputs) or bias.shape != (outputs,):
                    raise ValueError(f"Invalid {name} layer {index} shape")
                if kernel.dtype != np.float32 or bias.dtype != np.float32:
                    raise ValueError("Weights must be float32")
                layers.append((jnp.asarray(kernel), jnp.asarray(bias)))
                inputs = outputs
            self.weights[name] = tuple(layers)
        archive.close()
        self.scales = self.manifest["observation"]["divisors"]
        self.max_torque = self.manifest["limits"]["max_torque_nm"]

    def __call__(self, state, torque_nm):
        """Return actor torque, both Qs, their minimum and actor-continuation value.

        state: (..., 4) = unwrapped arm angle, pendulum angle (zero DOWN),
        arm speed, pendulum speed. torque_nm broadcasts to state.shape[:-1].
        Caller must supply finite states and torque within the documented cap.
        Inputs are never clipped; out-of-domain outputs are extrapolation.
        """
        state = jnp.asarray(state, dtype=jnp.float32)
        if state.shape[-1] != 4:
            raise ValueError("state must have trailing dimension 4")
        theta, alpha, omega, nu = jnp.moveaxis(state, -1, 0)
        observation = jnp.stack(
            (
                theta / self.scales[0],
                jnp.sin(alpha),
                jnp.cos(alpha),
                omega / self.scales[3],
                nu / self.scales[4],
            ),
            axis=-1,
        )
        hidden = observation
        for kernel, bias in self.weights["actor"][:-1]:
            hidden = jax.nn.silu(hidden @ kernel + bias)
        kernel, bias = self.weights["actor"][-1]
        action = jnp.tanh((hidden @ kernel + bias)[..., 0])
        supplied = jnp.broadcast_to(jnp.asarray(torque_nm, jnp.float32), theta.shape)
        actions = jnp.stack((supplied / self.max_torque, action), axis=0)
        features = jnp.broadcast_to(observation, (2,) + observation.shape)
        critic_input = jnp.concatenate((features, actions[..., None]), axis=-1)
        values = []
        for name in ("q1", "q2"):
            hidden = critic_input
            for kernel, bias in self.weights[name][:-1]:
                hidden = jax.nn.silu(hidden @ kernel + bias)
            kernel, bias = self.weights[name][-1]
            values.append((hidden @ kernel + bias)[..., 0])
        return {
            "actor_torque_nm": action * self.max_torque,
            "q1": values[0][0],
            "q2": values[1][0],
            "q_min": jnp.minimum(values[0][0], values[1][0]),
            "v_actor": jnp.minimum(values[0][1], values[1][1]),
        }
