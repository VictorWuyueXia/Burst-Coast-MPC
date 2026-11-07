"""Center energy pulses on conservative orbits and check the nonlinear arm path."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL, PHYSICS_DT_S
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD
from rotary_pendulum.heuristic.energy import TARGET_ENERGY_J
from rotary_pendulum.heuristic.orbital_controller import (
    _PHASE_TICK,
    ARM_MARGIN_RAD,
    CAPTURE_LIMITS,
    CENTERS,
    OrbitalMemory,
)
from rotary_pendulum.heuristic.orbital_phase import orbital_invariants, orbital_phase
from rotary_pendulum.heuristic.orbital_prediction import HORIZON_TICKS, predict_candidates

MOMENTUM_SCALE = np.sqrt(2 * MODEL.base_inertia_kg_m2 * TARGET_ENERGY_J)
LOCAL_ENTRY = np.array([1.5, 0.20, 2.0, 3.0])
MAX_PULSE_TICKS = 5


def energy_pulse(
    x: NDArray[np.float64],
    memory: OrbitalMemory,
    prediction: tuple,
    orbit: tuple,
    goal: tuple,
) -> tuple[float, int, float, float, float]:
    """Test one duration and bounded torque corrections against the passive path."""
    horizons, plant, coast_x, coast_peak, coast_first = prediction
    chart, center, width, offset, phase_step = orbit
    energy, momentum, requested, momentum_weight, ticks, launch, momentum_target = goal
    cap = plant[0]
    continued = memory.pulse_ticks > 0 and chart and memory.chart == chart
    arm_bound = float(ARM_LIMIT_RAD) - ARM_MARGIN_RAD
    displacement = coast_x[ticks - 1, 0, 0] - x[0]
    # Include the energy-based torque estimate only when the passive displacement resolves it.
    if launch or continued:
        torques = np.array([cap if launch else memory.pulse_torque])
    else:
        estimate = requested / displacement if abs(displacement) >= 1e-5 else 0.0
        steer = (momentum_target - momentum) / (ticks * PHYSICS_DT_S)
        torques = np.clip(np.array([-cap, -cap / 2, estimate, steer, cap / 2, cap]), -cap, cap)
    schedules = np.column_stack((torques, np.full(len(torques), ticks), np.zeros(len(torques))))
    states, energies, peaks, _, first = map(
        np.asarray, predict_candidates(x, schedules, horizons, plant)
    )
    if not np.isfinite(states).all():
        raise FloatingPointError("Nonfinite energy pulse prediction")
    delivered = energies[ticks - 1] - energy
    eligible = (peaks[:, 1] <= arm_bound) & (torques != 0)
    # Require the energy and momentum objective to improve before accepting a new pulse.
    if not (launch or continued):
        end_p = momentum + torques * ticks * PHYSICS_DT_S
        momentum_cost = momentum_weight * ((end_p - momentum_target) / MOMENTUM_SCALE) ** 2
        passive_cost = ((energy - TARGET_ENERGY_J) / TARGET_ENERGY_J) ** 2
        passive_cost += momentum_weight * ((momentum - momentum_target) / MOMENTUM_SCALE) ** 2
        endpoint_cost = ((energies[ticks - 1] - TARGET_ENERGY_J) / TARGET_ENERGY_J) ** 2
        eligible &= endpoint_cost + momentum_cost < passive_cost - 1e-8
        score = ((delivered - requested) / TARGET_ENERGY_J) ** 2 + momentum_cost
    else:
        score = np.zeros(len(torques))
    if chart and not launch and np.any(eligible):
        # Check the full powered path at control instants and the next command at finer steps.
        active = np.flatnonzero(eligible)
        fine = np.asarray(_PHASE_TICK(np.broadcast_to(x, (len(active), 4)), torques[active]))
        path = np.concatenate((fine, states[1:ticks, active]), axis=0)
        h_path, p_path = orbital_invariants(path, PHYSICAL, MODEL)
        branches, _, phases = orbital_phase(path, h_path, p_path, MODEL)
        distance = (phases - center + np.pi) % (2 * np.pi) - np.pi
        within = np.all((branches == chart) & (np.abs(distance) <= width), axis=0)
        if not continued:
            within &= (distance[-1] >= 0) & (np.abs(offset + distance[-1]) <= phase_step)
        eligible[active] &= within
    if not np.any(eligible):
        return 0.0, 0, 0.0, float(coast_peak[0, 1]), float(coast_first[0])
    index = int(np.argmin(np.where(eligible, score, np.inf)))
    return (
        float(torques[index]),
        ticks - 1,
        float(delivered[index]),
        *map(float, (peaks[index, 1], first[index])),
    )


def refined_step(
    x: NDArray[np.float64],
    memory: OrbitalMemory,
    plant: tuple[float, NDArray[np.float64]],
    tuning: tuple[float, float, float, float, float],
) -> tuple[float | None, OrbitalMemory, dict]:
    """Select a 20 ms command from the measured state and report finite-path rejection."""
    cap, gain = plant
    energy_gain, width, brake_gain, momentum_weight, centering_rate = tuning
    # Request inward mean arm drift during swing-up when a centering rate is specified.
    momentum_target = -MODEL.base_inertia_kg_m2 * centering_rate * x[0]
    prediction_plant = (cap, gain, brake_gain)
    H, p = orbital_invariants(x, PHYSICAL, MODEL)
    branch, period, phase = orbital_phase(x, H, p, MODEL)
    chart = int(branch)
    cycle = float(period) if chart else MODEL.natural_period_s
    horizons = np.minimum(np.ceil(np.array([0.5, 1.0]) * cycle / PHYSICS_DT_S), HORIZON_TICKS)
    horizons = horizons.astype(np.int32)
    arm_bound = float(ARM_LIMIT_RAD) - ARM_MARGIN_RAD
    coast_x, _, coast_peak, _, coast_first = map(
        np.asarray, predict_candidates(x, np.zeros((1, 3)), horizons, prediction_plant)
    )
    if not np.isfinite(coast_x).all():
        raise FloatingPointError("Nonfinite zero torque prediction")
    coast_safe = coast_peak[0] <= arm_bound
    first_peak, selected_peak = float(coast_first[0]), float(coast_peak[0, 1])
    requested = float(energy_gain * (TARGET_ENERGY_J - H))

    # Retain one attempted pulse per phase interval and account for command quantization.
    distance = (phase - CENTERS + np.pi) % (2 * np.pi) - np.pi
    nearest = int(np.argmin(np.abs(distance)))
    offset = float(distance[nearest])
    window = nearest + 1 if chart and abs(offset) <= width else 0
    used = memory.window if chart == memory.chart and window == memory.window else 0
    phase_step = 2 * np.pi * PHYSICS_DT_S / cycle
    centered_ticks = round(-2 * offset / phase_step) if window else 0
    entry = bool(window and not used and 1 <= centered_ticks <= MAX_PULSE_TICKS)
    continuing = bool(window and used and memory.pulse_ticks > 0)
    launching = memory.mode == "launch" and memory.pulse_ticks > 0
    launch = launching or (not memory.launch_used and bool(np.all(x[1:] == 0)))
    mode, remaining, local_ticks, delivered = "coast", 0, 0, 0.0
    torque: float | None = 0.0

    # The physical-state entry test is separate from the one-second experimental score.
    error = x.copy()
    error[1] = np.arctan2(-np.sin(x[1]), -np.cos(x[1]))
    was_local = memory.mode == "local"
    local_box = 2 * LOCAL_ENTRY if was_local else LOCAL_ENTRY
    if np.all(np.abs(error) < local_box):
        local_horizons = np.array([HORIZON_TICKS // 2, HORIZON_TICKS], dtype=np.int32)
        schedule = np.array([[memory.local_torque, memory.local_ticks, 2.0]])
        states, _, peaks, commands, first = map(
            np.asarray, predict_candidates(x, schedule, local_horizons, prediction_plant)
        )
        end = states[-1, 0].copy()
        end[1] = np.arctan2(-np.sin(end[1]), -np.cos(end[1]))
        if not np.isfinite(states).all():
            raise FloatingPointError("Nonfinite upright feedback prediction")
        contraction = np.linalg.norm(end / LOCAL_ENTRY) <= 0.65 * np.linalg.norm(
            error / LOCAL_ENTRY
        )
        settled = np.all(np.abs(end) < CAPTURE_LIMITS)
        if peaks[0, 1] <= arm_bound and (was_local or contraction or settled):
            torque, mode = float(commands[0]), "local"
            local_ticks = (memory.local_ticks - 1) % 5
            selected_peak, first_peak = float(peaks[0, 1]), float(first[0])

    # Prefer arm protection to a planned energy pulse when zero torque is unsafe.
    if mode == "coast" and not np.all(coast_safe):
        ticks = np.repeat(np.arange(1, MAX_PULSE_TICKS + 1), 5)
        cancel = np.clip(-p / (ticks * PHYSICS_DT_S), -cap, cap)
        torques = np.tile([-cap, -cap / 2, 0.0, cap / 2, cap], MAX_PULSE_TICKS)
        torques[2::5] = cancel[2::5]
        schedules = np.vstack(([0, 0, 1], np.column_stack((torques, ticks, np.zeros(len(ticks))))))
        states, _, peaks, commands, first = map(
            np.asarray, predict_candidates(x, schedules, horizons, prediction_plant)
        )
        if not np.isfinite(states).all():
            raise FloatingPointError("Nonfinite arm recovery prediction")
        _, final_p = orbital_invariants(states[horizons[1] - 1], PHYSICAL, MODEL)
        scores = (peaks[:, 1] / arm_bound) ** 2 + (final_p / MOMENTUM_SCALE) ** 2
        eligible = (peaks[:, 1] <= arm_bound) & np.r_[True, ticks <= horizons[1]]
        if np.any(eligible):
            index = int(np.argmin(np.where(eligible, scores, np.inf)))
            torque, mode = float(commands[index]), "brake"
            selected_peak, first_peak = float(peaks[index, 1]), float(first[index])
        else:
            torque, mode = None, "no_screened_action"
    elif mode == "coast" and (entry or continuing or launch or not chart):
        # In the nonperiodic region, choose a one-tick energy and momentum correction.
        ticks = (
            memory.pulse_ticks
            if continuing or launching
            else MAX_PULSE_TICKS
            if launch
            else centered_ticks
        )
        ticks = min(ticks if chart or launch else 1, int(horizons[1]))
        torque, remaining, delivered, selected_peak, first_peak = energy_pulse(
            x,
            memory,
            (horizons, prediction_plant, coast_x, coast_peak, coast_first),
            (chart, CENTERS[nearest], width, offset, phase_step),
            (H, p, requested, momentum_weight, ticks, launch, momentum_target),
        )
        if torque:
            mode = "launch" if launch else "pulse" if chart else "aperiodic"
        if launch and not torque:
            torque, mode = None, "launch_rejected"
    if entry or mode == "local":
        used = window
    following = OrbitalMemory(
        mode,
        chart,
        used,
        float(torque) if mode in ("pulse", "launch") else 0.0,
        remaining if mode in ("pulse", "launch") else 0,
        memory.launch_used or launch,
        0,
        float(torque) if mode == "local" else 0.0,
        local_ticks,
    )
    return (
        torque,
        following,
        {
            "mode": mode,
            "chart": chart,
            "period_s": float(period),
            "phase_rad": float(phase),
            "window": window,
            "window_entry": int(entry),
            "energy_j": float(H),
            "momentum": float(p),
            "momentum_target": float(momentum_target),
            "requested_j": requested,
            "predicted_delta_j": delivered,
            "planned_ticks": remaining + 1 if mode in ("pulse", "aperiodic", "launch") else 0,
            "coast_half_peak_rad": float(coast_peak[0, 0]),
            "coast_full_peak_rad": float(coast_peak[0, 1]),
            "selected_peak_rad": selected_peak,
            "first_tick_peak_rad": first_peak,
        },
    )
