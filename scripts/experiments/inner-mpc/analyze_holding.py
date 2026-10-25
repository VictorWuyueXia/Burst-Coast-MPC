"""Local actual-network MPC sensitivity and saved-transition Lyapunov diagnostics."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space, solve_discrete_lyapunov
from scipy.optimize import linprog

from mpc_sensitivity import mpc_sensitivity
from terminal_linearization import terminal_model

# Load the authenticated network and the compact transitions reconstructed from the archive.
repository = Path(__file__).resolve().parents[3]
root = repository / "artifacts/rotary_pendulum"
bundle = root / "1-TD3_20261016/machine-scannables"
manifest = json.loads((bundle / "manifest.json").read_text())
checksum = hashlib.sha256((bundle / "weights.npz").read_bytes()).hexdigest()
assert checksum == manifest["weights_sha256"]
archive = np.load(bundle / "weights.npz", allow_pickle=False)
weights = {name: archive[name] for name in archive.files}
archive.close()
terminal = terminal_model(weights, np.array(manifest["observation"]["divisors"]))
source = root / "inner-mpc/20261022-lqr-cost-fine-confirm-16/machine-scannables"
design = json.loads((source / "design.json").read_text())
assert manifest["physical_config"]["rotary_damping_nms"] == 0
assert manifest["physical_config"]["pendulum_damping_nms"] == 0
machine = root / "inner-mpc/actual-holding-study-20261025/machine-scannables"
data = np.load(machine / "recorded_transitions.npz", allow_pickle=False)
command, successors, plans = data["command"], data["successor"], data["plans"]
summary = json.loads((machine / "holding_statistics.json").read_text())
model = mpc_sensitivity(
    terminal, dict(plant=design["plant"], mechanics=manifest["derived_physics"])
)

# Select fixed diagnostic points; these local solves are not closed-loop controller experiments.
mean = np.array(summary["mean_error"])
extreme = np.unravel_index(command[300:, :, 0].argmin(), command[300:, :, 0].shape)
cases = [
    dict(name="upright", error=np.zeros(4),
         plan=np.r_[np.zeros(10), np.array(terminal(error=np.zeros(4))["values"])[1:].max()],
         root_iteration=0),
    dict(name="tail_mean", error=mean, seed_actor=True),
    dict(name="negative_arm", error=command[300+extreme[0], extreme[1]],
         plan=plans[300+extreme[0], extreme[1]], recorded=(300+extreme[0], extreme[1])),
    dict(name="final_lane2", error=command[-1, 2], plan=plans[-1, 2], recorded=(399, 2)),
]
results, solved = [], {}
for case in cases:
    error = case["error"]
    if "seed_actor" in case:
        predicted, controls = error.copy(), []
        for index in range(10):
            action = float(terminal(error=predicted)["values"][0])
            controls.append(action)
            predicted = np.array(model["command"](predicted, action)[0]).ravel()
        case["plan"] = np.r_[controls, np.array(terminal(error=predicted)["values"])[1:].max()]
    plan = case["plan"]
    endpoints = np.array(model["prediction"](error, plan[:10].reshape(1, 10))[0])
    initial = np.r_[plan, endpoints.flatten(order="F")]
    values = model["values"](error, initial, np.zeros(1042))
    baseline = float(values[0])
    if "recorded" in case:
        decision, lane = case["recorded"]
        np.testing.assert_allclose(baseline, data["objective"][decision, lane], atol=1e-8)
        executed = np.array(model["execution"](error, float(plan[0]))[0]).ravel()
        np.testing.assert_allclose(executed, successors[decision, lane], atol=1e-9)

    # Recover the reduced objective gradient at the original plan through equality multipliers.
    multipliers = np.zeros(1042)
    multipliers[40 + np.array(values[8]).argmax()] = 1
    original = model["values"](error, initial, multipliers)
    multipliers[:40] = -np.linalg.solve(np.array(original[5])[:40, 11:].T,
                                       np.array(original[2]).ravel()[11:])
    reduced_gradient = np.array(model["values"](error, initial, multipliers)[2]).ravel()[:10]
    print(f"Local point {case['name']}: recorded/initial cost {baseline:.8f}", flush=True)
    answer = model["solver"](
        x0=initial, p=error, lbx=model["lower_variables"], ubx=model["upper_variables"],
        lbg=model["lower_constraints"], ubg=model["upper_constraints"],
    )
    status = model["solver"].stats()
    variables = np.array(answer["x"]).ravel()
    dual = np.array(answer["lam_g"]).ravel()
    bound_dual = np.array(answer["lam_x"]).ravel()
    np.savez_compressed(machine / f"point_{case['name']}.npz", error=error, variables=variables,
                        constraint_multipliers=dual, bound_multipliers=bound_dual)
    assert status["success"], (case["name"], status["return_status"])
    outputs = model["values"](error, variables, dual)
    constraint_values = np.array(outputs[1]).ravel()
    stationarity = np.array(outputs[2]).ravel() + bound_dual
    hessian, mixed = np.array(outputs[3]), np.array(outputs[4])
    constraint_z, constraint_e = np.array(outputs[5]), np.array(outputs[6])

    # Differentiate the full active-set KKT equations, retaining both critics at a tie.
    active_inequalities = np.flatnonzero(np.abs(dual[40:]) > 1e-6) + 40
    active = np.r_[np.arange(40), active_inequalities]
    bounds = np.flatnonzero(np.abs(bound_dual) > 1e-6)
    active_z = np.vstack([constraint_z[active], np.eye(51)[bounds]])
    active_e = np.vstack([constraint_e[active], np.zeros((len(bounds), 4))])
    kkt = np.block([[hessian, active_z.T],
                    [active_z, np.zeros((len(active_z), len(active_z)))]])
    sensitivity = np.linalg.solve(kkt, -np.vstack([mixed, active_e]))
    gain = sensitivity[0]
    tangent = null_space(active_z)
    reduced_eigenvalues = np.linalg.eigvalsh(tangent.T @ hessian @ tangent)
    next_state, flow_e, flow_a, dense = model["execution"](error, float(variables[0]))
    next_state = np.array(next_state).ravel()
    closed = np.array(flow_e) + np.array(flow_a) @ gain[None, :]
    record = dict(
        name=case["name"], error=error.tolist(), initial_cost=baseline,
        optimized_cost=float(answer["f"]), status=status["return_status"],
        original_reduced_gradient_inf=float(np.abs(reduced_gradient).max()),
        stationarity_inf=float(np.abs(stationarity).max()),
        dynamics_residual_inf=float(np.abs(constraint_values[:40]).max()),
        active_inequalities=active_inequalities.tolist(), active_bounds=bounds.tolist(),
        terminal_costs=np.array(outputs[8]).ravel().tolist(),
        reduced_hessian_min=float(reduced_eigenvalues.min()),
        first_action=float(variables[0]), first_action_jacobian=gain.tolist(),
        closed_loop_spectral_radius=float(np.abs(np.linalg.eigvals(closed)).max()),
        fixed_point_defect=(next_state-error).tolist(),
    )
    np.savez_compressed(machine / f"sensitivity_{case['name']}.npz",
                        gain=gain, closed_loop=closed, kkt_sensitivity=sensitivity,
                        hessian=hessian, mixed=mixed, active_z=active_z, active_e=active_e)
    results.append(record)
    solved[case["name"]] = dict(record=record, variables=variables, closed=closed, gain=gain)
    (machine / "local_sensitivities.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    if "root_iteration" in case:
        if abs(variables[0]) < 1e-8:
            solved["equilibrium"] = solved[case["name"]]
        elif case["root_iteration"] < 5:
            theta = error[0] - variables[0] / gain[0]
            assert abs(theta) < design["plant"]["observation_scales"][0]
            cases.append(dict(name=f"equilibrium_{case['root_iteration']+1}",
                              error=np.array([theta, 0., 0., 0.]), plan=variables[:11],
                              root_iteration=case["root_iteration"]+1))

# Test the MPC-derived quadratic metric against saved transitions and an empirical best residual.
if "equilibrium" in solved:
    result = solved["equilibrium"]
    center = np.array(result["record"]["error"])
    closed = result["closed"]
    assert result["record"]["closed_loop_spectral_radius"] < 1
    metric = solve_discrete_lyapunov(closed.T, np.diag(design["plant"]["lqr_state_weights"]))
    metric = (metric + metric.T) / 2
    assert np.linalg.eigvalsh(metric).min() > 0
    before, after = command[300:] - center, successors[300:] - center
    energy = np.einsum("...i,ij,...j->...", before, metric, before).ravel()
    following = np.einsum("...i,ij,...j->...", after, metric, after).ravel()
    differences = following - energy
    lp = linprog([0., 1.], A_ub=np.column_stack([differences, -np.ones(len(energy))]),
                 b_ub=-energy, bounds=[(1., None), (0., None)], method="highs")
    assert lp.success
    inverse_time, floor = lp.x
    rho, epsilon = 1 - 1 / inverse_time, floor / inverse_time
    physical_limit = np.deg2rad(5)**2 / np.linalg.inv(metric)[1, 1]
    witness = int(differences.argmax())
    report = dict(
        scope="Finite recorded-transition tests; no unsampled regional bound is inferred",
        center=center.tolist(), metric=metric.tolist(), largest_observed_level=float(energy.max()),
        largest_positive_increment=float(differences[witness]), witness_index=witness,
        positive_increment_count=int((differences > 1e-10).sum()),
        best_sampled_rho=float(rho), best_sampled_epsilon=float(epsilon),
        best_sampled_ultimate_level=float(floor), five_degree_ellipsoid_level=float(physical_limit),
        residual_certifies_five_degree_region=bool(floor < physical_limit),
    )
    # Test longer blocks and retain an actual transition that exits the proposed ellipsoid.
    path = np.concatenate([command[300:301], successors[300:]], axis=0) - center
    levels = np.einsum("...i,ij,...j->...", path, metric, path)
    blocks = []
    for length in (1, 2, 5, 10, 20, 50):
        initial, final = levels[:-length].ravel(), levels[length:].ravel()
        fit = linprog([0., 1.], A_ub=np.column_stack([final-initial, -np.ones(len(initial))]),
                      b_ub=-initial, bounds=[(1., None), (0., None)], method="highs")
        assert fit.success
        blocks.append(dict(commands=length, rho=float(1-1/fit.x[0]),
                           ultimate_level=float(fit.x[1])))
    exits = (energy.reshape(100, 32) <= physical_limit) & (
        following.reshape(100, 32) > physical_limit
    )
    choices = np.argwhere(exits)
    local, lane = choices[np.argmax((following-energy).reshape(100, 32)[exits])]
    decision, lane = 300 + int(local), int(lane)
    report.update(block_comparisons=blocks, five_degree_ellipsoid_recorded_exit_count=len(choices),
                  exit_witness=dict(decision=decision, episode=lane, level=float(physical_limit),
                                    before_level=float(energy.reshape(100, 32)[local, lane]),
                                    after_level=float(following.reshape(100, 32)[local, lane]),
                                    state=command[decision, lane].tolist(),
                                    successor=successors[decision, lane].tolist(),
                                    normalized_action=float(plans[decision, lane, 0])))
    np.savez_compressed(machine / "invariance_exit_witness.npz", metric=metric, center=center,
                        state=command[decision, lane], successor=successors[decision, lane],
                        plan=plans[decision, lane], previous_plan=plans[decision-1, lane])
    np.savez_compressed(machine / "certificate_transitions.npz", metric=metric, center=center,
                        before=before, after=after, energy=energy, following=following)
    (machine / "certificate_diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
data.close()
