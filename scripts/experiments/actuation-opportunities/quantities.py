"""Energy authority and return-section diagnostics for archived rotary motion."""

import numpy as np

from rotary_pendulum.environment.dynamics import energy_components, state_derivative


def observables(x, torque, physical, model):
    """Compute the proposed task coordinates and their full, affine energy rates."""
    alpha, omega, nu = x[:, 1:].T
    a = model.base_inertia_kg_m2 + model.pendulum_inertia_kg_m2 * np.sin(alpha) ** 2
    b = model.coupling_inertia_kg_m2 * np.cos(alpha)
    determinant = a * model.pendulum_inertia_kg_m2 - b**2
    potential = model.gravity_torque_nm * (1 - np.cos(alpha))
    total = energy_components(x, physical, model)[2]
    swing = 0.5 * model.pendulum_inertia_kg_m2 * nu**2 + potential
    target = 2 * model.gravity_torque_nm
    sensitivity = -model.pendulum_inertia_kg_m2 * nu * b / determinant
    drift = state_derivative(x, 0.0, physical, model)
    swing_drift = nu * (
        model.pendulum_inertia_kg_m2 * drift[:, 3] + model.gravity_torque_nm * np.sin(alpha)
    )
    total_drift = -physical.rotary_damping_nms * omega**2 - physical.pendulum_damping_nms * nu**2
    return dict(
        theta_rad=x[:, 0],
        alpha_rad=alpha,
        omega_rad_s=omega,
        nu_rad_s=nu,
        torque_nm=torque,
        total_j=total,
        swing_j=swing,
        momentum_nms=a * omega + b * nu,
        s_total_rad_s=omega,
        s_swing_rad_s=sensitivity,
        s_body_rad_s=omega
        * (1 - model.arm_inertia_kg_m2 * model.pendulum_inertia_kg_m2 / determinant),
        c_total=(total - target) * omega,
        c_swing=(swing - target) * sensitivity,
        total_drift_w=total_drift,
        swing_drift_w=swing_drift,
        total_input_w=omega * torque,
        swing_input_w=sensitivity * torque,
        total_rate_w=total_drift + omega * torque,
        swing_rate_w=swing_drift + sensitivity * torque,
        phi_rad=np.arctan2(sensitivity, omega),
        psi_rad=np.unwrap(np.arctan2(-nu / model.natural_frequency_rad_s, alpha)),
    )


def cycle_metrics(record):
    """Measure positive downward returns, phase reversals, and actual coast runs."""
    t, alpha = record["time_s"], record["alpha_rad"]
    s = np.column_stack((record["s_total_rad_s"], record["s_swing_rad_s"]))
    radius = np.linalg.norm(s, axis=1)
    # Do not unwrap across near-origin gaps: direction uses adjacent valid samples only.
    dphi = np.angle(np.exp(1j * np.diff(record["phi_rad"])))
    reversals = {}
    for threshold in (0.001, 0.01, 0.05):
        valid = radius > threshold * np.sqrt(np.mean(radius**2))
        increments = dphi[valid[:-1] & valid[1:] & (abs(dphi) > 1e-8)]
        reversals[str(threshold)] = dict(
            origin_fraction=float(np.mean(~valid)),
            reverse_fraction=float(min(np.mean(increments > 0), np.mean(increments < 0))),
            absolute_turns=float(np.sum(abs(increments)) / (2 * np.pi)),
            signed_turns=float(np.sum(increments) / (2 * np.pi)),
        )
    crossing = np.flatnonzero(
        (np.sin(alpha[:-1]) < 0)
        & (np.sin(alpha[1:]) >= 0)
        & (np.cos(alpha[1:]) > 0)
        & (record["nu_rad_s"][1:] > 0)
    )
    fraction = -np.sin(alpha[crossing]) / (np.sin(alpha[crossing + 1]) - np.sin(alpha[crossing]))
    section_time = t[crossing] + fraction * (t[crossing + 1] - t[crossing])
    section = np.column_stack(
        [
            record[key][crossing] + fraction * (record[key][crossing + 1] - record[key][crossing])
            for key in ("theta_rad", "omega_rad_s", "nu_rad_s", "total_j", "momentum_nms")
        ]
    )
    zero = abs(record["torque_nm"][1:]) < 1e-10
    changes = np.diff(np.r_[False, zero, False].astype(int))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    durations = t[ends] - t[starts]
    result = dict(
        duration_s=float(t[-1] - t[0]),
        positive_downward_crossings=len(crossing),
        zero_torque_fraction=float(np.sum(np.diff(t) * zero) / (t[-1] - t[0])),
        longest_zero_torque_s=float(np.max(durations, initial=0)),
        sensitivity_same_sign_fraction=float(np.mean(s[:, 0] * s[:, 1] > 0)),
        task_conflict_fraction=float(np.mean(record["c_total"] * record["c_swing"] < 0)),
        phase=reversals,
        sensitivity_singular_value_ratio=float(
            np.linalg.svd(s - s.mean(axis=0), compute_uv=False)[1]
            / np.linalg.svd(s - s.mean(axis=0), compute_uv=False)[0]
        ),
        section_times_s=section_time.tolist(),
        section_states=section.tolist(),
    )
    if len(section_time) >= 3:
        periods = np.diff(section_time)
        speeds = np.column_stack((record["omega_rad_s"], record["nu_rad_s"]))
        scale = np.sqrt(np.mean(np.sum(speeds**2, axis=1)))
        result.update(
            period_mean_s=float(periods.mean()),
            period_cv=float(periods.std() / periods.mean()),
            period_min_s=float(periods.min()),
            period_max_s=float(periods.max()),
            reduced_return_error=float(
                np.max(np.linalg.norm(np.diff(section[:, 1:3], axis=0), axis=1)) / scale
            ),
            arm_drift_per_cycle_rad=float(np.mean(np.diff(section[:, 0]))),
        )
    return result
