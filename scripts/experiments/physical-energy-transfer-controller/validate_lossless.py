"""Audit the lossless energy surface, squared cost, and rejected-action coast."""

import json
from pathlib import Path

import matplotlib
import numpy as np
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from validate_energy import energy_ledger

from rotary_pendulum.environment.dynamics import derive_model, rk4_step
from rotary_pendulum.utils.config_schema import PHYSICS_CONFIG_PATH, RotaryPendulumConfig


def main():
    """Run vectorized physical probes and write separate data and explanatory plots."""

    directory = Path(__file__).resolve().parents[3] / (
        "artifacts/rotary_pendulum/experiment-results/physical-energy-transfer-controller/records"
    )
    human = directory.parent / "figures"
    human.mkdir(exist_ok=True)
    physical = RotaryPendulumConfig.model_validate(
        OmegaConf.to_container(OmegaConf.load(PHYSICS_CONFIG_PATH), resolve=True)["rotary-pendulum"]
    ).model_copy(update={"rotary_damping_nms": 0.0, "pendulum_damping_nms": 0.0})
    model = derive_model(physical)
    target = 2.0 * model.gravity_torque_nm
    initial = np.array([[0.0, 0.7, 1.0, 2.0], [1.55, 0.0, 1.0, 0.0]])
    checks, records = [], []
    print("Checking lossless coast at 20, 2, and 1 ms; two parallel probes", flush=True)
    for dt in (0.02, 0.002, 0.001):
        state = initial.copy()
        history = [state.copy()]
        for _ in range(round(1.0 / dt)):
            state = rk4_step(state, np.zeros(2), dt, physical, model)
            history.append(state.copy())
        history = np.asarray(history)
        energy = energy_ledger(history, np.zeros(2), physical, model)[0]
        momentum = (
            model.base_inertia_kg_m2 + model.pendulum_inertia_kg_m2 * np.sin(history[..., 1]) ** 2
        ) * history[..., 2] + model.coupling_inertia_kg_m2 * np.cos(history[..., 1]) * history[
            ..., 3
        ]
        drift = np.max(abs(energy.sum(axis=-1) - energy[0].sum(axis=-1)))
        momentum_drift = np.max(abs(momentum - momentum[0]))
        checks.append((dt, drift, momentum_drift))
        time = dt * np.arange(len(history))
        records.extend(
            np.column_stack(
                (
                    np.full(len(history), dt),
                    time,
                    history[:, 0],
                    energy[:, 0],
                    energy[:, 0].sum(axis=-1),
                    momentum[:, 0],
                )
            )
        )
        print(
            f"dt={dt:g}: energy drift={drift:.3g} J; momentum drift={momentum_drift:.3g}",
            flush=True,
        )
    assert checks[2][1] < checks[1][1] < checks[0][1]
    assert checks[2][1] < 1e-9
    assert checks[2][2] < 1e-9
    assert np.max(abs(energy[:, 0] - energy[0, 0])) > 0.001
    np.testing.assert_allclose(history[100, 1, 0], 1.65, atol=1e-12, rtol=0)
    assert history[100, 1, 0] > np.pi / 2
    np.savetxt(
        directory / "lossless_coast.csv",
        records,
        delimiter=",",
        comments="",
        header="dt_s,time_s,theta_rad,alpha_rad,omega_rad_s,nu_rad_s,"
        "arm_kinetic_J,pendulum_kinetic_J,potential_J,total_J,arm_momentum_kg_m2_s",
    )

    print("Checking the finite-interval torque curve and work-selected surface", flush=True)
    torques = np.linspace(-0.00918, 0.00918, 257)
    states = np.broadcast_to(initial[0], (len(torques), 4)).copy()
    for _ in range(100):
        states = rk4_step(states, torques, 0.001, physical, model)
    endpoints = energy_ledger(states, torques, physical, model)[0]
    work = torques * (states[:, 0] - initial[0, 0])
    surface_residual = endpoints.sum(axis=-1) - energy[0, 0].sum() - work
    assert np.max(abs(surface_residual)) < 1e-9
    inertia = model.base_inertia_kg_m2 + model.pendulum_inertia_kg_m2 * np.sin(states[:, 1]) ** 2
    coupling = model.coupling_inertia_kg_m2 * np.cos(states[:, 1])
    fraction = (
        model.arm_inertia_kg_m2
        * model.pendulum_inertia_kg_m2
        / (inertia * model.pendulum_inertia_kg_m2 - coupling**2)
    )
    assert np.all((fraction > 0) & (fraction < 1))
    allocation = (
        states[:, 2, None]
        * torques[:, None]
        * np.column_stack((fraction, 1.0 - fraction, np.zeros_like(fraction)))
    )
    allocation_error = np.max(
        abs(
            energy_ledger(states, torques, physical, model)[1]
            - energy_ledger(states, 0.0, physical, model)[1]
            - allocation
        )
    )
    assert allocation_error < 1e-12
    cost = 0.5 * np.sum((endpoints / target - np.array([0.0, 0.0, 1.0])) ** 2, axis=-1)
    np.savetxt(
        directory / "lossless_torque_curve.csv",
        np.column_stack((torques, states, endpoints, work, surface_residual, cost)),
        delimiter=",",
        comments="",
        header="torque_nm,theta_rad,alpha_rad,omega_rad_s,nu_rad_s,arm_kinetic_J,"
        "pendulum_kinetic_J,potential_J,work_J,surface_residual_J,squared_energy_cost",
    )

    print("Checking squared-energy cost near upright", flush=True)
    beta = np.geomspace(0.001, 0.1, 31)
    local = np.zeros((len(beta), 4))
    local[:, 1] = np.pi + beta
    local_energy = energy_ledger(local, 0.0, physical, model)[0]
    linear = (target - local_energy[:, 2]) / target
    squared = 0.5 * linear**2
    slope = np.polyfit(np.log(beta), np.log(squared), 1)[0]
    assert abs(slope - 4.0) < 0.01
    np.savetxt(
        directory / "lossless_cost.csv",
        np.column_stack((beta, linear, squared)),
        delimiter=",",
        comments="",
        header="upright_error_rad,linear_height_cost,squared_energy_cost",
    )
    evidence = {
        "rotary_damping_nms": 0.0,
        "pendulum_damping_nms": 0.0,
        "shared_physics_config_changed": False,
        "target_energy_J": target,
        "coast_initial_state": initial[0].tolist(),
        "coast_initial_energy_J": energy[0, 0].tolist(),
        "coast_energy_after_100ms_J": energy[100, 0].tolist(),
        "coast_refinement_columns": ["dt_s", "max_energy_drift_J", "max_momentum_drift"],
        "coast_refinement": checks,
        "max_surface_residual_J": float(np.max(abs(surface_residual))),
        "max_instantaneous_allocation_error_W": float(allocation_error),
        "squared_cost_local_log_slope": float(slope),
        "boundary_probe_initial": initial[1].tolist(),
        "boundary_probe_arm_angle_after_100ms_rad": float(history[100, 1, 0]),
        "arm_bound_rad": float(np.pi / 2),
    }
    (directory / "lossless_checks.json").write_text(json.dumps(evidence, indent=2) + "\n")

    figure = plt.figure(figsize=(16, 5), layout="constrained")
    figure.get_layout_engine().set(wspace=0.12)
    axis = figure.add_subplot(131)
    for index, label in enumerate(("Arm kinetic", "Pendulum kinetic", "Potential")):
        axis.plot(time, 1000 * energy[:, 0, index], label=label)
    axis.plot(time, 1000 * energy[:, 0].sum(axis=-1), "k--", label="Total (conserved)")
    axis.set(xlabel="Coast time [s]", ylabel="Energy [mJ]", title="Zero damping + zero torque")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)

    axis = figure.add_subplot(132, projection="3d")
    normalized = energy[:, 0] / energy[0, 0].sum()
    plane = Poly3DCollection([np.eye(3)], alpha=0.12, facecolor="gray")
    axis.add_collection3d(plane)
    axis.plot(*normalized.T, color="tab:blue", label="Actual coast trajectory")
    axis.scatter(*normalized[0], color="black", s=25, label="Initial energy point")
    axis.set(
        xlabel="Arm kinetic / H₀",
        ylabel="Pendulum kinetic / H₀",
        zlabel="Potential / H₀",
        title="Same plane, changing point\nH₀ = initial total energy",
        xlim=(0, 1),
        ylim=(0, 1),
        zlim=(0, 1),
    )
    axis.tick_params(labelsize=8)
    axis.legend(fontsize=7, loc="upper left")

    axis = figure.add_subplot(133)
    axis.loglog(beta, linear, label="Linear height deficit")
    axis.loglog(beta, squared, label="Squared energy distance")
    axis.set(
        xlabel="Upright angle error [rad]",
        ylabel="Dimensionless cost",
        title="Squared energy cost is fourth-order locally",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.savefig(human / "lossless_energy_geometry.png", dpi=170)
    plt.close(figure)
    print(
        "PASS conservation, changing components, torque curve, boundary counterexample, cost slope",
        flush=True,
    )


if __name__ == "__main__":
    main()
