"""Audit queued-trajectory certificates and CEM replacements using saved decisions only."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.linalg import eigvalsh

from mpc_sensitivity import mpc_sensitivity
from terminal_linearization import terminal_model

# Authenticate the unchanged controller inputs and reuse its decoded local dynamics.
repository = Path(__file__).resolve().parents[3]
root = repository / "artifacts/rotary_pendulum"
bundle = root / "final-results/rl-policy/machine-scannables"
manifest = json.loads((bundle / "manifest.json").read_text())
assert hashlib.sha256((bundle / "weights.npz").read_bytes()).hexdigest() == manifest["weights_sha256"]
archive = np.load(bundle / "weights.npz", allow_pickle=False)
terminal = terminal_model({key: archive[key] for key in archive.files},
                          np.array(manifest["observation"]["divisors"]))
terminal = terminal.factory("frozen_values", ["error"], ["values", "jacobian"])
archive.close()
source = root / "final-results/rl-mpc/machine-scannables/session-records"
design = json.loads((source / "design.json").read_text())
folder = root / "inner-mpc/actual-holding-study-20261025/machine-scannables"
model = mpc_sensitivity(terminal, dict(plant=design["plant"], mechanics=manifest["derived_physics"]))
reference = np.load(folder / "actor_reference_metric.npz", allow_pickle=False)
metric, center, closed = reference["metric"], reference["center"], reference["closed_loop"]
linear_factor = float(eigvalsh(closed.T @ metric @ closed, metric).max())
gamma = (1 + linear_factor) / 2
radius = np.deg2rad(5) / np.sqrt(np.linalg.inv(metric)[1, 1])

# Weight later queue entries to turn the retained-queue argument into one-step contraction.
state, actions = ca.MX.sym("state", 4), ca.MX.sym("actions", 10)
endpoints = model["prediction"](state, actions.T)[0]
nodes = ca.horzcat(state, endpoints)
errors = nodes - ca.repmat(ca.DM(center), 1, 11)
levels = ca.sum1(errors * (ca.DM(metric) @ errors))
weighted = ca.times(levels, ca.DM(gamma ** -np.arange(11)).T)
costs = terminal(endpoints[:, -1])[0][1:]
running = ca.sum2(ca.DM(design["plant"]["lqr_state_weights"]).T @ (nodes[:, :10]**2))
objective = running + ca.dot(actions, actions) + ca.mmax(costs)
queue = ca.Function("queue", [state, actions], [ca.sqrt(ca.mmax(levels)),
                    ca.sqrt(ca.mmax(weighted)), objective, nodes, levels])
coarse_end = model["coarse_prediction"](state, actions[1:].T)[:, -1]
shifted = ca.vertcat(actions[1:], terminal(coarse_end)[0][0])
warm = ca.Function("warm", [state, actions], [shifted])
tail = terminal(endpoints[:, -1])[0][0]
tail_end = model["command"](endpoints[:, -1], tail)[0]
defect = (ca.dot(endpoints[:, -1]**2, ca.DM(design["plant"]["lqr_state_weights"]))
          + tail**2 + ca.mmax(terminal(tail_end)[0][1:]) - ca.mmax(costs))
terminal_defect = ca.Function("terminal_defect", [state, actions], [defect])
objective_gradient = ca.Function("objective_gradient", [state, actions],
                                 [ca.gradient(objective, actions)])
actor_closed_jacobian = ca.Function("actor_closed_jacobian", [state],
    [ca.jacobian(model["command"](state, terminal(state)[0][0])[0], state)])

# Reconstruct every selected holding queue and its actual shifted warm competitor in batches.
data = np.load(folder / "recorded_transitions.npz", allow_pickle=False)
states, plans = data["command"][299:], data["plans"][299:, :, :10]
selected_norms, selected_costs, forecasts, warm_norms, warm_costs = [], [], [], [], []
warm_plans, ideal_norms, defects = [], [], []
for start in range(0, 101, 10):
    stop = min(start + 10, 101)
    count = (stop-start) * 32
    x = states[start:stop].reshape(-1, 4).T
    u = plans[start:stop].reshape(-1, 10).T
    result = queue.map(count)(x, u)
    selected_norms.append(np.array(result[1]).reshape(-1, 32))
    selected_costs.append(np.array(result[2]).reshape(-1, 32))
    forecasts.append(np.array(result[3]).reshape(4, count, 11).transpose(1, 2, 0))
    defects.append(np.array(terminal_defect.map(count)(x, u)).reshape(-1, 32))
    print(f"Selected queues reconstructed: {299+start} through {298+stop}", flush=True)
norms, objectives, forecast, terminal_defects = map(np.concatenate,
    (selected_norms, selected_costs, forecasts, defects))
forecast = forecast.reshape(101, 32, 11, 4)
stage = data["stage_states"][299:].copy()
stage[..., 1] = np.arctan2(-np.sin(stage[..., 1]), -np.cos(stage[..., 1]))
stage_error = float(np.abs(forecast[:, :, :10] - stage).max())
cost_error = float(np.abs(objectives - data["objective"][299:]).max())
print(f"Rollout parity: states={stage_error:.12g}, objective={cost_error:.12g}", flush=True)
assert stage_error < 1e-7 and cost_error < 1e-5
norm_tolerance = 10*np.sqrt(eigvalsh(metric).max())*stage_error*gamma**-5
cost_tolerance = 10*cost_error
for start in range(0, 100, 10):
    count = 10 * 32
    x = states[start+1:start+11].reshape(-1, 4).T
    u = plans[start:start+10].reshape(-1, 10).T
    shifted = np.array(warm.map(count)(x, u))
    result = queue.map(count)(x, shifted)
    warm_norms.append(np.array(result[1]).reshape(10, 32))
    warm_costs.append(np.array(result[2]).reshape(10, 32))
    warm_plans.append(shifted.T.reshape(10, 32, 10))
    old_forecast = forecast[start:start+10].reshape(count, 11, 4)
    old_end = old_forecast[:, -1].T
    appended = np.array(terminal.map(count)(old_end)[0])[0]
    ideal = np.vstack((u[1:], appended))
    ideal_result = queue.map(count)(old_forecast[:, 1].T, ideal)
    ideal_norms.append(np.array(ideal_result[1]).reshape(10, 32))
    print(f"Warm and ideal queues reconstructed at decisions {300+start} through {309+start}", flush=True)
warm_norm, warm_cost, warm_plan, ideal_norm = map(np.concatenate,
    (warm_norms, warm_costs, warm_plans, ideal_norms))

# Keep switching, objective preference, and integration mismatch as separate measured quantities.
before, after = norms[:-1], norms[1:]
retained_nine = np.all(plans[1:, :, :9] == plans[:-1, :, 1:], axis=-1)
retained_full = np.max(np.abs(plans[1:] - warm_plan), axis=-1) < 1e-8
replacement = ~retained_nine
increment, replacement_gap = after-before, after-warm_norm
cost_gap = objectives[1:] - warm_cost
exit_mask = (before <= radius) & (after > radius)
certificate_gap = after - np.sqrt(gamma)*before
model_gap = warm_norm - ideal_norm
index = np.unravel_index(np.where(replacement, replacement_gap, -np.inf).argmax(), before.shape)
exit_indices = np.argwhere(exit_mask)
summary = dict(
    scope="Saved selected plans only; no CEM rerun or unsampled regional verification",
    center=center.tolist(), metric=metric.tolist(), linear_squared_factor=linear_factor,
    chosen_squared_factor=gamma, five_degree_radius=float(radius), transitions=before.size,
    forecast_parity_max_error=stage_error, objective_parity_max_error=cost_error,
    retained_first_nine=int(retained_nine.sum()), retained_full_tolerance_1e8=int(retained_full.sum()),
    norm_comparison_tolerance=float(norm_tolerance), cost_comparison_tolerance=float(cost_tolerance),
    replacements=int(replacement.sum()), all_queue_increases=int((increment > norm_tolerance).sum()),
    replacement_queue_increases=int(((increment > norm_tolerance) & replacement).sum()),
    replacement_worse_than_warm=int(((replacement_gap > norm_tolerance) & replacement).sum()),
    replacement_lower_cost_but_worse_queue=int(((cost_gap < -cost_tolerance) &
                                              (replacement_gap > norm_tolerance) & replacement).sum()),
    selected_more_costly_than_warm=int((cost_gap > cost_tolerance).sum()),
    max_objective_gap=float(cost_gap.max()), min_objective_gap=float(cost_gap.min()),
    max_warm_vs_ideal_norm_error=float(np.abs(model_gap).max()),
    max_ideal_contraction_residual=float((ideal_norm - np.sqrt(gamma)*before).max()),
    max_selected_contraction_residual=float(certificate_gap.max()),
    necessary_sample_residual_radius=float(certificate_gap.max() / (1-np.sqrt(gamma))),
    max_terminal_actor_dissipation_defect=float(terminal_defects.max()),
    min_terminal_actor_dissipation_defect=float(terminal_defects.min()),
    queued_five_degree_exits=len(exit_indices),
    queued_five_degree_exits_on_replacements=int((exit_mask & replacement).sum()),
    queue_radius_min=float(norms[1:].min()), queue_radius_max=float(norms[1:].max()),
    maximum_predicted_upright_error_rad=float(np.abs(forecast[1:, :, :, 1]).max()),
)
nominal_closed = np.array(actor_closed_jacobian(center))
summary["nominal_actor_linear_squared_factor"] = float(eigvalsh(
    nominal_closed.T @ metric @ nominal_closed, metric).max())

# Test whether critic-guided improvement alone preserves the zero of the actor queue metric.
zero = np.zeros(10)
gradient = np.array(objective_gradient(center, zero)).ravel()
probe = zero.copy()
probe[-1] = -np.sign(gradient[-1]) * 1e-3
zero_result, probe_result = queue(center, zero), queue(center, probe)
fine = model["execution"].mapaccum(10)(center, probe.reshape(1, 10))
fine_dense = np.array(fine[3]).T
probe_forecast = np.array(probe_result[3]).T
nominal_dense = np.array(model["prediction"](center, probe.reshape(1, 10))[1]).T
refinement = np.abs(fine_dense[1::2] - nominal_dense).max(axis=0)
assert float(probe_result[2]) < float(zero_result[2])
assert np.abs(nominal_dense[:, 0]).max() < design["plant"]["observation_scales"][0]
assert np.abs(fine_dense[:, 0]).max() < design["plant"]["observation_scales"][0]
assert np.all(refinement < np.array([1e-3, 1e-3, 1e-2, 1e-2]))
summary["zero_queue_probe"] = dict(
    scope="One feasible plan comparison, not a regenerated CEM random draw",
    last_action_gradient=float(gradient[-1]), last_action=float(probe[-1]),
    zero_cost=float(zero_result[2]), probe_cost=float(probe_result[2]),
    zero_norm=float(zero_result[1]), probe_norm=float(probe_result[1]),
    refinement_all_nodes_error=refinement.tolist(),
    terminal_branches=np.array(terminal(center)[0]).ravel()[1:].tolist(),
)
probe_plus, probe_minus = zero.copy(), zero.copy()
probe_plus[-1], probe_minus[-1] = 1e-6, -1e-6
finite_difference = (float(queue(center, probe_plus)[2])-float(queue(center, probe_minus)[2]))/2e-6
assert abs(finite_difference-gradient[-1]) < 1e-6
summary["zero_queue_probe"]["gradient_finite_difference_error"] = float(abs(finite_difference-gradient[-1]))
np.savez_compressed(folder / "plan_memory_zero_probe.npz", center=center, metric=metric,
                    gamma=gamma, gradient=gradient, probe=probe, forecast=probe_forecast)

# Save witnesses in complete precision so objective and queue inequalities can be reproduced.
for label, chosen in (("replacement", index), ("set_exit", exit_indices[0])):
    local, lane = map(int, chosen)
    entry = dict(decision=300+local, episode=lane,
                 before_norm=float(before[local, lane]), after_norm=float(after[local, lane]),
                 warm_norm=float(warm_norm[local, lane]),
                 selected_cost=float(objectives[local+1, lane]), warm_cost=float(warm_cost[local, lane]),
                 replacement=bool(replacement[local, lane]),
                 state=states[local+1, lane].tolist(), before_state=states[local, lane].tolist())
    summary[label+"_witness"] = entry
    # Rational comparisons validate stored forecasts; they do not certify integration rounding.
    exact_levels = []
    for points in (forecast[local, lane], forecast[local+1, lane]):
        levels = []
        for i, point in enumerate(points):
            difference = [Fraction(float(point[j]))-Fraction(float(center[j])) for j in range(4)]
            level = sum(difference[j]*Fraction(float(metric[j, k]))*difference[k]
                        for j in range(4) for k in range(4)) / Fraction(gamma)**i
            levels.append(level)
        exact_levels.append(max(levels))
    exact_threshold = Fraction(float(radius))**2
    assert exact_levels[0] < exact_threshold < exact_levels[1]
    entry["strict_stored_forecast_exit_rational"] = True
    entry["rational_squared_levels"] = [str(value) for value in exact_levels]
    np.savez_compressed(folder / ("plan_memory_"+label+"_witness.npz"),
                        center=center, metric=metric, gamma=gamma, radius=radius,
                        before_state=states[local, lane], state=states[local+1, lane],
                        before_plan=plans[local, lane], selected_plan=plans[local+1, lane],
                        warm_plan=warm_plan[local, lane],
                        before_forecast=forecast[local, lane],
                        selected_forecast=forecast[local+1, lane],
                        warm_forecast=np.array(queue(states[local+1, lane], warm_plan[local, lane])[3]).T)
np.savez_compressed(folder / "plan_memory_audit.npz", selected_norm=norms, warm_norm=warm_norm,
                    ideal_norm=ideal_norm, objectives=objectives, warm_cost=warm_cost,
                    replacement=replacement, retained_full=retained_full,
                    forecast=forecast, terminal_actor_defect=terminal_defects)
(folder / "plan_memory_audit.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2), flush=True)
