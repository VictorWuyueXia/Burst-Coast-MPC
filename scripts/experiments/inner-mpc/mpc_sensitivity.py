"""Nonlinear MPC objective and local optimizer sensitivity from the actual frozen networks."""

import casadi as ca
import numpy as np


def mpc_sensitivity(terminal_model: ca.Function, point_records: dict) -> dict:
    """Build the original nominal problem and differentiated KKT quantities on CPU."""

    # Use the archived mechanics and runtime torque cap in local upright coordinates.
    plant, mechanics = point_records["plant"], point_records["mechanics"]
    base, hinge = mechanics["base_inertia_kg_m2"], mechanics["pendulum_inertia_kg_m2"]
    coupling, gravity = mechanics["coupling_inertia_kg_m2"], mechanics["gravity_torque_nm"]
    state, action = ca.MX.sym("state", 4), ca.MX.sym("action")
    sine, cosine = ca.sin(state[1]), ca.cos(state[1])
    mass11, mass12 = base + hinge * sine**2, -coupling * cosine
    torque1 = plant["torque_limit_nm"] * action - 2 * hinge * sine * cosine * state[2] * state[3]
    torque1 -= coupling * sine * state[3] ** 2
    torque2 = hinge * sine * cosine * state[2] ** 2 + gravity * sine
    determinant = mass11 * hinge - mass12**2
    ode = ca.Function("ode", [state, action], [ca.vertcat(
        state[2], state[3], (hinge * torque1 - mass12 * torque2) / determinant,
        (mass11 * torque2 - mass12 * torque1) / determinant,
    )])
    dt = ca.MX.sym("dt")
    k1 = ode(state, action)
    k2 = ode(state + dt * k1 / 2, action)
    k3 = ode(state + dt * k2 / 2, action)
    k4 = ode(state + dt * k3, action)
    micro = ca.Function("micro", [state, action, dt], [state + dt * (k1 + 2*k2 + 2*k3 + k4)/6])

    # Accumulate all 1 ms prediction nodes and the independently refined first command.
    print("Building 1 ms nonlinear prediction and 0.5 ms execution maps", flush=True)
    nominal = micro.mapaccum(100)(state, ca.repmat(action, 1, 100), ca.DM.ones(1, 100)*0.001)
    command = ca.Function("command", [state, action], [nominal[:, -1], nominal])
    error, variables = ca.MX.sym("error", 4), ca.MX.sym("variables", 51)
    endpoints = ca.reshape(variables[11:], 4, 10)
    nodes = ca.horzcat(error, endpoints[:, :9])
    propagated, dense = command.map(10)(nodes, variables[:10].T)
    prediction = command.mapaccum(10)
    coarse = micro.mapaccum(50)(state, ca.repmat(action, 1, 50), ca.DM.ones(1, 50)*0.002)
    coarse_command = ca.Function("coarse_command", [state, action], [coarse[:, -1]])
    fine = micro.mapaccum(200)(state, ca.repmat(action, 1, 200), ca.DM.ones(1, 200)*0.0005)
    execution = ca.Function("execution", [state, action], [
        fine[:, -1], ca.jacobian(fine[:, -1], state),
        ca.jacobian(fine[:, -1], action), fine,
    ])
    nodes[1, :] = ca.atan2(ca.sin(nodes[1, :]), ca.cos(nodes[1, :]))
    primary = ca.sum2(ca.DM(plant["lqr_state_weights"]).T @ (nodes**2))
    primary += ca.dot(variables[:10], variables[:10])
    costs = terminal_model(endpoints[:, -1])[0][1:]
    objective = primary + variables[10]
    constraints = ca.vertcat(
        ca.vec(propagated - endpoints), costs - variables[10],
        dense[0, :].T / plant["observation_scales"][0],
    )
    multipliers = ca.MX.sym("multipliers", 1042)
    lagrangian = objective + ca.dot(multipliers, constraints)
    gradient = ca.gradient(lagrangian, variables)

    # Preserve the full Lagrangian curvature and active-constraint state derivatives.
    print("Building nonlinear objective and KKT derivatives", flush=True)
    values = ca.Function("mpc_values", [error, variables, multipliers], [
        objective, constraints, gradient, ca.jacobian(gradient, variables),
        ca.jacobian(gradient, error), ca.jacobian(constraints, variables),
        ca.jacobian(constraints, error), endpoints, costs,
    ])
    print("Creating the local interior-point solver", flush=True)
    solver = ca.nlpsol("local_mpc", "ipopt", {
        "x": variables, "p": error, "f": objective, "g": constraints,
    }, {
        "print_time": False, "ipopt.print_level": 5, "ipopt.sb": "yes",
        "ipopt.max_iter": 60, "ipopt.tol": 1e-9, "ipopt.max_wall_time": 60.0,
        "ipopt.mu_init": 1e-6,
    })
    return dict(
        solver=solver, values=values, execution=execution, prediction=prediction, command=command,
        coarse_prediction=coarse_command.mapaccum(9),
        lower_variables=np.r_[np.full(10, -1.), np.full(41, -np.inf)],
        upper_variables=np.r_[np.full(10, 1.), np.full(41, np.inf)],
        lower_constraints=np.r_[np.zeros(40), [-np.inf, -np.inf], np.full(1000, -1.)],
        upper_constraints=np.r_[np.zeros(40), [0., 0.], np.full(1000, 1.)],
    )
