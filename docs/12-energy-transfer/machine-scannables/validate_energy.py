"""Reproduce the energy-coordinate assessment with nominal CPU dynamics."""

from pathlib import Path

import casadi as ca
import numpy as np
from omegaconf import OmegaConf

from rotary_pendulum.environment.dynamics import (
    ModelConstants,
    derive_model,
    energy_components,
    rk4_step,
    state_derivative,
)
from rotary_pendulum.utils.config_schema import PHYSICS_CONFIG_PATH, RotaryPendulumConfig


def energy_ledger(x, u, physical: RotaryPendulumConfig, model: ModelConstants):
    """Compute three body energies, their rates, and four power-flow terms."""

    _, alpha, omega, nu = np.moveaxis(np.asarray(x), -1, 0)
    sine, cosine = np.sin(alpha), np.cos(alpha)
    inertia = model.pendulum_inertia_kg_m2
    coupling = model.coupling_inertia_kg_m2
    gravity = model.gravity_torque_nm
    arm_inertia = model.arm_inertia_kg_m2
    carried_inertia = model.base_inertia_kg_m2 - arm_inertia + inertia * sine**2
    acceleration = state_derivative(x, u, physical, model)[..., 2:]
    arm_acceleration, pendulum_acceleration = np.moveaxis(acceleration, -1, 0)

    energy = np.stack(
        (
            0.5 * arm_inertia * omega**2,
            0.5 * carried_inertia * omega**2
            + coupling * cosine * omega * nu
            + 0.5 * inertia * nu**2,
            gravity * (1.0 - cosine),
        ),
        axis=-1,
    )
    arm_loss = physical.rotary_damping_nms * omega**2
    pendulum_loss = physical.pendulum_damping_nms * nu**2
    input_power = u * omega
    arm_rate = arm_inertia * omega * arm_acceleration
    transfer = input_power - arm_loss - arm_rate
    potential_rate = gravity * sine * nu
    pendulum_kinetic_rate = (
        carried_inertia * omega * arm_acceleration
        + inertia * sine * cosine * nu * omega**2
        + coupling * cosine * (arm_acceleration * nu + omega * pendulum_acceleration)
        - coupling * sine * omega * nu**2
        + inertia * nu * pendulum_acceleration
    )
    rates = np.stack((arm_rate, pendulum_kinetic_rate, potential_rate), axis=-1)
    power = np.stack((input_power, transfer, arm_loss, pendulum_loss), axis=-1)
    return energy, rates, power


def main():
    """Audit identities, phase aliasing, work inversion, and integration accuracy."""

    directory = Path(__file__).resolve().parent
    physical = RotaryPendulumConfig.model_validate(
        OmegaConf.to_container(OmegaConf.load(PHYSICS_CONFIG_PATH), resolve=True)[
            "rotary-pendulum"
        ]
    )
    model = derive_model(physical)
    rng = np.random.default_rng(20261007)
    x = rng.uniform([-1.5, -np.pi, -3.0, -8.0], [1.5, np.pi, 3.0, 8.0], (512, 4))
    u = rng.uniform(-0.00918, 0.00918, 512)
    energy, rates, power = energy_ledger(x, u, physical, model)
    metrics = []
    print("Checking Lagrangian and power identities on 512 parallel states", flush=True)

    # Differentiate L independently of the implemented generalized-force equations.
    q, velocity, torque = ca.SX.sym("q", 2), ca.SX.sym("v", 2), ca.SX.sym("u")
    kinetic = (
        0.5 * (model.base_inertia_kg_m2 + model.pendulum_inertia_kg_m2 * ca.sin(q[1])**2)
        * velocity[0]**2
        + model.coupling_inertia_kg_m2 * ca.cos(q[1]) * velocity[0] * velocity[1]
        + 0.5 * model.pendulum_inertia_kg_m2 * velocity[1]**2
    )
    lagrangian = kinetic - model.gravity_torque_nm * (1 - ca.cos(q[1]))
    momentum = ca.gradient(lagrangian, velocity)
    force = ca.vertcat(
        torque - physical.rotary_damping_nms * velocity[0],
        -physical.pendulum_damping_nms * velocity[1],
    )
    acceleration = ca.solve(
        ca.jacobian(momentum, velocity),
        force + ca.gradient(lagrangian, q) - ca.jacobian(momentum, q) @ velocity,
    )
    lagrange_ode = ca.Function("lagrange_ode", [q, velocity, torque], [acceleration])
    derived = np.asarray(lagrange_ode.map(512)(x[:, :2].T, x[:, 2:].T, u[None, :])).T
    error = np.max(np.abs(derived - state_derivative(x, u, physical, model)[:, 2:]))
    metrics.append(("lagrangian_acceleration_max_error", error, "rad_s2", 1e-10))
    error = np.max(np.abs(energy.sum(axis=-1) - energy_components(x, physical, model)[2]))
    metrics.append(("total_energy_max_error", error, "J", 1e-14))
    error = np.max(np.abs(rates.sum(axis=-1) - (power[:, 0] - power[:, 2:].sum(axis=-1))))
    metrics.append(("total_power_max_error", error, "W", 1e-12))
    error = np.max(np.abs(rates[:, 1:].sum(axis=-1) - power[:, 1] + power[:, 3]))
    metrics.append(("pendulum_pool_power_max_error", error, "W", 1e-12))
    direction = state_derivative(x, u, physical, model)
    difference = (
        energy_ledger(x + 1e-6 * direction, u, physical, model)[0]
        - energy_ledger(x - 1e-6 * direction, u, physical, model)[0]
    ) / 2e-6
    metrics.append(("directional_derivative_max_error", np.max(abs(difference - rates)), "W", 1e-7))
    assert np.all(energy >= 0.0)

    print("Checking identical energies with different derivatives", flush=True)
    pairs = np.array([[0.0, 0.7, 1.0, 2.0], [0.0, -0.7, 1.0, 2.0]])
    pair_energy, pair_rates, pair_power = energy_ledger(pairs, 0.002, physical, model)
    np.testing.assert_allclose(pair_energy[0], pair_energy[1], atol=1e-15, rtol=0)
    assert abs(pair_rates[0, 2] - pair_rates[1, 2]) > 0.01
    np.savetxt(
        directory / "phase_alias.csv",
        np.column_stack((pairs, np.full(2, 0.002), pair_energy, pair_rates, pair_power)),
        delimiter=",", comments="",
        header="theta_rad,alpha_rad,omega_rad_s,nu_rad_s,torque_nm,arm_kinetic_J,"
        "pendulum_kinetic_J,potential_J,arm_rate_W,pendulum_kinetic_rate_W,potential_rate_W,"
        "input_power_W,transfer_power_W,arm_loss_W,pendulum_loss_W",
    )

    # Identical three-energy observations and exactly zero commanded/delivered work.
    coast = pairs.copy()
    for _ in range(100):
        coast = rk4_step(coast, 0.0, 0.001, physical, model)
    coast_energy = energy_ledger(coast, 0.0, physical, model)[0]
    assert np.max(abs(coast_energy[0] - coast_energy[1])) > 0.001
    np.savetxt(
        directory / "coast_alias.csv",
        np.column_stack((pairs, pair_energy, np.zeros(2), coast, coast_energy)),
        delimiter=",", comments="",
        header="initial_theta_rad,initial_alpha_rad,initial_omega_rad_s,initial_nu_rad_s,"
        "initial_arm_kinetic_J,initial_pendulum_kinetic_J,initial_potential_J,work_J,"
        "final_theta_rad,final_alpha_rad,final_omega_rad_s,final_nu_rad_s,"
        "final_arm_kinetic_J,final_pendulum_kinetic_J,final_potential_J",
    )

    print("Integrating mirrored startup torques and refining the work balance", flush=True)
    rows = []
    for dt in (0.02, 0.002, 0.001):
        # Two lanes share time integration; accumulate work and damping at RK4 stages.
        state, accumulated = np.zeros((2, 4)), np.zeros((2, 2))
        torques = np.array([0.003, -0.003])
        for _ in range(round(0.1 / dt)):
            stages, powers = [], []
            for stage in range(4):
                current = state if stage == 0 else state + (
                    dt * (0.5 if stage < 3 else 1.0) * stages[-1]
                )
                stages.append(state_derivative(current, torques, physical, model))
                stage_power = energy_ledger(current, torques, physical, model)[2]
                powers.append(np.column_stack((stage_power[:, 0], stage_power[:, 2:].sum(axis=-1))))
            state += dt * np.einsum("i,ijk->jk", [1, 2, 2, 1], stages) / 6
            accumulated += dt * np.einsum("i,ijk->jk", [1, 2, 2, 1], powers) / 6
        final_energy = energy_ledger(state, torques, physical, model)[0]
        residual = final_energy.sum(axis=-1) - accumulated[:, 0] + accumulated[:, 1]
        np.testing.assert_allclose(state[0], -state[1], atol=1e-14, rtol=0)
        np.testing.assert_allclose(accumulated[0], accumulated[1], atol=1e-14, rtol=0)
        assert np.all(accumulated[:, 0] > 0)
        rows.extend(
            np.column_stack((np.full(2, dt), torques, state, final_energy, accumulated, residual))
        )
    np.savetxt(
        directory / "startup_work.csv", rows, delimiter=",", comments="",
        header="dt_s,torque_nm,theta_rad,alpha_rad,omega_rad_s,nu_rad_s,arm_kinetic_J,"
        "pendulum_kinetic_J,potential_J,input_work_J,damping_loss_J,balance_residual_J",
    )
    assert abs(rows[4][-1]) < abs(rows[2][-1]) < abs(rows[0][-1])
    metrics.append(("refined_startup_balance_error", abs(rows[4][-1]), "J", 1e-10))
    np.savetxt(directory / "identity_checks.csv", np.array(metrics, dtype=object),
               fmt="%s", delimiter=",", comments="", header="metric,value,unit,pass_upper_bound")
    for name, value, unit, bound in metrics:
        assert value < bound, (name, value, bound)
        print(f"PASS {name}: {value:.6g} {unit} < {bound:g}", flush=True)
    print("PASS phase alias and equal-work opposite-torque counterexamples", flush=True)


if __name__ == "__main__":
    main()
