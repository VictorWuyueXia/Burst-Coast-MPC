"""Physical-body energy encoding and a strictly energy-only work policy."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from rotary_pendulum.environment.jax_dynamics import MODEL

TARGET_ENERGY_J = 2.0 * MODEL.gravity_torque_nm
WORK_LIMIT_J = 0.04 * TARGET_ENERGY_J


def encode(x: Array) -> Array:
    """Return arm kinetic, full pendulum kinetic, and potential energy in joules."""

    _, alpha, omega, nu = jnp.moveaxis(x, -1, 0)
    sine, cosine = jnp.sin(alpha), jnp.cos(alpha)
    arm = 0.5 * MODEL.arm_inertia_kg_m2 * omega**2
    # The pendulum body moves with the arm as well as about its hinge.
    # Keeping the full cross term makes the three entries sum to physical energy.
    carried_inertia = (
        MODEL.base_inertia_kg_m2 - MODEL.arm_inertia_kg_m2 + MODEL.pendulum_inertia_kg_m2 * sine**2
    )
    pendulum = (
        0.5 * carried_inertia * omega**2
        + MODEL.coupling_inertia_kg_m2 * cosine * omega * nu
        + 0.5 * MODEL.pendulum_inertia_kg_m2 * nu**2
    )
    potential = MODEL.gravity_torque_nm * (1.0 - cosine)
    return jnp.stack((arm, pendulum, potential), axis=-1)


def policy(energy: Array, work_gain: Array, kinetic_weight: Array) -> Array:
    """Request bounded signed work from three energies, without physical phase."""

    # Weight one regulates total energy; larger weights discourage arm storage.
    kinetic = kinetic_weight * energy[..., 0] + energy[..., 1]
    potential_deficit = TARGET_ENERGY_J - energy[..., 2]
    requested_work = work_gain * (potential_deficit - kinetic)
    return jnp.clip(requested_work, -WORK_LIMIT_J, WORK_LIMIT_J)
