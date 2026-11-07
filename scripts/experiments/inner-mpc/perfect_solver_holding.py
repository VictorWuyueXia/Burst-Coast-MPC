"""Local nonlinear holding analysis for the unchanged, exactly optimized MPC objective."""

import hashlib
import json
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.linalg import expm, null_space, solve_discrete_lyapunov

from mpc_sensitivity import mpc_sensitivity
from terminal_linearization import terminal_model

# Authenticate the frozen network and reuse the original nonlinear optimization problem.
repository = Path(__file__).resolve().parents[3]
root = repository / "artifacts/rotary_pendulum"
bundle = root / "1-TD3_20261016/machine-scannables"
manifest = json.loads((bundle / "manifest.json").read_text())
assert hashlib.sha256((bundle / "weights.npz").read_bytes()).hexdigest() == (
    manifest["weights_sha256"]
)
archive = np.load(bundle / "weights.npz", allow_pickle=False)
weights = {name: archive[name] for name in archive.files}
archive.close()
terminal = terminal_model(weights, np.array(manifest["observation"]["divisors"]))
source = root / "inner-mpc/20261022-lqr-cost-fine-confirm-16/machine-scannables"
design = json.loads((source / "design.json").read_text())
previous = root / "inner-mpc/actual-holding-study-20261025/machine-scannables"
destination = root / "inner-mpc/perfect-solver-holding-20261025/machine-scannables"
destination.mkdir(parents=True, exist_ok=True)
model = mpc_sensitivity(
    terminal, dict(plant=design["plant"], mechanics=manifest["derived_physics"])
)
reference = np.load(previous / "point_equilibrium_2.npz", allow_pickle=False)
center = reference["error"].copy()
variables = reference["variables"].copy()
dual = reference["constraint_multipliers"].copy()
reference.close()
weight = np.diag(design["plant"]["lqr_state_weights"])

# Refine the zero-action rest point, then test independent starts and nearby states.
jobs = [dict(name="root_0", radius=0., error=center.copy(), initial=variables.copy())]
records, solutions = [], []
for job in jobs:
    print(f"\nOffline one-horizon solve: {job['name']}", flush=True)
    error = job["error"]
    answer = model["solver"](
        x0=job["initial"], p=error, lbx=model["lower_variables"],
        ubx=model["upper_variables"], lbg=model["lower_constraints"],
        ubg=model["upper_constraints"],
    )
    assert model["solver"].stats()["success"], model["solver"].stats()["return_status"]
    variables, dual = np.array(answer["x"]).ravel(), np.array(answer["lam_g"]).ravel()
    bound_dual = np.array(answer["lam_x"]).ravel()
    values = model["values"](error, variables, dual)
    gradient = np.array(values[2]).ravel() + bound_dual
    constraint = np.array(values[1]).ravel()
    active = np.r_[np.arange(40), np.flatnonzero(abs(dual[40:]) > 1e-6) + 40]
    bounds = np.flatnonzero(abs(bound_dual) > 1e-6)
    constraint_y, constraint_e = np.array(values[5]), np.array(values[6])
    active_y = np.vstack([constraint_y[active], np.eye(51)[bounds]])
    active_e = np.vstack([constraint_e[active], np.zeros((len(bounds), 4))])
    hessian, mixed = np.array(values[3]), np.array(values[4])
    kkt = np.block([[hessian, active_y.T],
                    [active_y, np.zeros((len(active_y), len(active_y)))]])
    sensitivity = np.linalg.solve(kkt, -np.vstack([mixed, active_e]))[:51]
    gain = sensitivity[0]
    tangent = null_space(active_y)
    curvature = float(np.linalg.eigvalsh(tangent.T @ hessian @ tangent).min())
    following, flow_e, flow_a, dense = model["execution"](error, float(variables[0]))
    following, dense = np.array(following).ravel(), np.array(dense)
    closed = np.array(flow_e) + np.array(flow_a) @ gain[None, :]
    row = dict(
        name=job["name"], radius=job["radius"], error=error.tolist(), objective=float(answer["f"]),
        first_action=float(variables[0]), gain=gain.tolist(), reduced_curvature=curvature,
        stationarity_inf=float(abs(gradient).max()),
        dynamics_residual_inf=float(abs(constraint[:40]).max()),
        terminal_gap=float(np.diff(np.array(values[8]).ravel())[0]),
        active_inequalities=active[40:].tolist(), active_bounds=bounds.tolist(),
        spectral_radius=float(abs(np.linalg.eigvals(closed)).max()),
    )
    assert row["stationarity_inf"] < 1e-7 and row["dynamics_residual_inf"] < 1e-8
    assert curvature > 0, row
    if job["name"] == "root_0":
        center[0] -= variables[0] / gain[0]
        jobs.append(dict(name="center", radius=0., error=center.copy(), initial=variables.copy()))
    elif job["name"] == "center":
        assert abs(variables[0]) < 1e-11
        center_variables, center_sensitivity = variables.copy(), sensitivity.copy()
        metric = solve_discrete_lyapunov(closed.T, weight)
        metric = (metric + metric.T) / 2
        factor = np.linalg.cholesky(metric).T
        inverse = np.linalg.inv(metric)
        inverse_factor = np.linalg.inv(factor)
        q = np.linalg.norm(factor @ closed @ inverse_factor, 2)
        coordinate_factors = np.sqrt(np.diag(inverse))
        numerical_defect = float(np.linalg.norm(factor @ (following-center)))
        np.savez_compressed(
            destination / "center.npz", center=center, variables=variables, multipliers=dual,
            sensitivity=sensitivity, gain=gain, closed_loop=closed, metric=metric,
        )

        # Multiple starts assess local-basin consistency; they do not certify global optimality.
        starts = [dict(name="start_zero", controls=np.zeros(10))]
        actor_state, actor_actions = center.copy(), []
        for index in range(10):
            action = float(terminal(error=actor_state)["values"][0])
            actor_actions.append(action)
            actor_state = np.array(model["command"](actor_state, action)[0]).ravel()
        starts.append(dict(name="start_actor", controls=np.array(actor_actions)))
        alternative = np.load(previous / "point_negative_arm.npz", allow_pickle=False)
        starts.append(dict(name="start_other_basin", controls=alternative["variables"][:10]))
        alternative.close()
        for start in starts:
            endpoints = np.array(model["prediction"](center, start["controls"][None, :])[0])
            epigraph = np.array(terminal(error=endpoints[:, -1])["values"])[1:].max()
            initial = np.r_[start["controls"], epigraph, endpoints.flatten(order="F")]
            jobs.append(dict(name=start["name"], radius=0., error=center.copy(), initial=initial))

        # Probe coordinate-extreme boundary directions and fixed mixed directions in the metric.
        directions = (inverse / coordinate_factors).T
        directions = np.vstack([directions, -directions])
        worst = np.linalg.svd(factor @ closed @ inverse_factor)[2][0]
        directions = np.vstack([directions, inverse_factor @ worst, -inverse_factor @ worst])
        for radius in (0.05, 0.15, 0.30, 0.60):
            for index, direction in enumerate(directions):
                offset = radius * direction
                jobs.append(dict(
                    name=f"boundary_{radius:.2f}_{index}", radius=radius,
                    error=center+offset, initial=center_variables+center_sensitivity @ offset,
                ))
        mixed_directions = np.array([[1, 1, 1, 1], [1, -1, 1, -1]], dtype=float) / 2
        mixed_directions = np.vstack([mixed_directions, -mixed_directions])
        for radius in (0.15, 0.60):
            for index, direction in enumerate(mixed_directions):
                offset = radius * inverse_factor @ direction
                jobs.append(dict(name=f"mixed_{radius:.2f}_{index}", radius=radius,
                                 error=center+offset,
                                 initial=center_variables+center_sensitivity @ offset))
    if job["name"] != "root_0":
        offset, next_offset = error-center, following-center
        norm = float(np.linalg.norm(factor @ offset))
        row.update(
            jacobian_norm=float(np.linalg.norm(factor @ closed @ inverse_factor, 2)),
            dense_abs_max=np.max(abs(dense), axis=1).tolist(),
        )
        if norm > 1e-10:
            row.update(
                norm_ratio=float(np.linalg.norm(factor @ next_offset) / norm),
                remainder_coefficient=float(np.linalg.norm(
                    factor @ (next_offset - np.array(solutions[1]["closed"]) @ offset)
                ) / norm**2),
                jacobian_change_over_radius=float(np.linalg.norm(
                    factor @ (closed-solutions[1]["closed"]) @ inverse_factor, 2
                ) / norm),
            )
    records.append(row)
    solutions.append(dict(variables=variables, closed=closed, error=error, dense=dense))
    (destination / "point_diagnostics.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(row, indent=2), flush=True)

# Derive the inter-command linear tube from the differentiated refined execution nodes.
state, action = ca.MX.sym("tube_state", 4), ca.MX.sym("tube_action")
dense = model["execution"](state, action)[3]
dense_derivatives = ca.Function("dense_derivatives", [state, action], [
    ca.jacobian(ca.vec(dense), state), ca.jacobian(ca.vec(dense), action),
])
dense_e, dense_a = dense_derivatives(center, 0.)
gain = np.array(records[1]["gain"])
tube_jacobians = (np.array(dense_e) + np.array(dense_a) @ gain[None, :]).reshape(200, 4, 4)
tube_jacobians = np.concatenate([np.eye(4)[None, :, :], tube_jacobians])
tube_rows = np.einsum("tij,jk->tik", tube_jacobians, inverse_factor)
tube_factors = np.linalg.norm(tube_rows, axis=2).max(axis=0)

# Compare refined integration with the analytic held-input linear flow of the same plant.
mechanics = manifest["derived_physics"]
base, hinge, coupling, gravity = [mechanics[name] for name in (
    "base_inertia_kg_m2", "pendulum_inertia_kg_m2",
    "coupling_inertia_kg_m2", "gravity_torque_nm",
)]
determinant = base*hinge-coupling**2
continuous = np.zeros((4, 4))
continuous[0, 2] = continuous[1, 3] = 1
continuous[2, 1], continuous[3, 1] = coupling*gravity/determinant, base*gravity/determinant
torque = design["plant"]["torque_limit_nm"]
input_row = np.array([0., 0., hinge*torque/determinant, coupling*torque/determinant])
augmented = np.zeros((5, 5))
augmented[:4, :4], augmented[:4, 4] = continuous, input_row
flows = expm(np.linspace(0., 0.1, 201)[:, None, None]*augmented)
held_flow = flows[:, :4, :4] + flows[:, :4, 4, None]*gain[None, None, :]
scaled = factor @ continuous @ inverse_factor
lognorm = np.linalg.eigvalsh((scaled+scaled.T)/2).max()
derivative_bound = np.exp(0.1*lognorm)*np.linalg.norm(
    factor @ (continuous+np.outer(input_row, gain)) @ inverse_factor, 2
)
np.savez_compressed(destination / "linear_flow.npz", Ac=continuous, Bc=input_row,
                    held_flow=held_flow, metric=metric)
boundaries = [row for row in records if row["radius"] > 0]
for row, point in zip(records, solutions):
    if row["radius"] > 0:
        offset = point["error"]-center
        linear_tube = np.einsum("tij,j->ti", tube_jacobians[1:], offset)
        remainder = point["dense"].T-center-linear_tube
        row["tube_remainder_coefficient"] = float(
            np.linalg.norm(remainder @ factor.T, axis=1).max() / row["radius"]**2
        )
summary = dict(
    scope="Nonlinear local optimizer diagnostics; sampled bounds are not regional certificates",
    center=center.tolist(), metric=metric.tolist(), q=float(q),
    fixed_point_numerical_defect=numerical_defect,
    coordinate_factors=coordinate_factors.tolist(),
    linear_intercommand_coordinate_factors=tube_factors.tolist(),
    exact_flow_linear_contraction=float(np.linalg.norm(factor @ held_flow[-1] @ inverse_factor, 2)),
    fine_grid_linearization_error_inf=float(abs(held_flow[-1]-solutions[1]["closed"]).max()),
    held_flow_derivative_bound=float(derivative_bound),
    radii=[dict(
        radius=radius, count=sum(row["radius"] == radius for row in boundaries),
        max_norm_ratio=max(row["norm_ratio"] for row in boundaries if row["radius"] == radius),
        max_jacobian_norm=max(
            row["jacobian_norm"] for row in boundaries if row["radius"] == radius
        ),
        max_remainder_coefficient=max(
            row["remainder_coefficient"] for row in boundaries if row["radius"] == radius
        ),
        max_tube_remainder_coefficient=max(
            row["tube_remainder_coefficient"] for row in boundaries if row["radius"] == radius
        ),
    ) for radius in (0.05, 0.15, 0.30, 0.60)],
    global_optimality_certified=False, regional_derivative_bound_certified=False,
)
np.savez_compressed(destination / "regional_probes.npz",
                    errors=np.array([row["error"] for row in records]),
                    variables=np.array([point["variables"] for point in solutions]),
                    closed_loops=np.array([point["closed"] for point in solutions]),
                    dense_tube_jacobians=tube_jacobians)
(destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(destination / "point_diagnostics.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
