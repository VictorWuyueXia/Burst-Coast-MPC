"""Supervise orbital energy bursts with nonlinear arm screening and local capture."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import numpy as np
from numpy.typing import NDArray

from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL, PHYSICS_DT_S, rk4_step
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, GOAL
from rotary_pendulum.heuristic.energy import TARGET_ENERGY_J
from rotary_pendulum.heuristic.orbital_phase import orbital_invariants, orbital_phase
from rotary_pendulum.heuristic.orbital_prediction import (
    AUDIT_SUBSTEPS,
    HORIZON_TICKS,
    predict_candidates,
)

CAPTURE_LIMITS = np.array(
    [
        GOAL.theta_tolerance_rad,
        GOAL.beta_tolerance_rad,
        GOAL.omega_tolerance_rad_s,
        GOAL.nu_tolerance_rad_s,
    ]
)
CAPTURE_TICKS = GOAL.hold_steps
ARM_MARGIN_RAD = np.deg2rad(0.5)
CENTERS = np.array([np.pi / 2, 3 * np.pi / 2])
_PHASE_TICK = jax.jit(partial(rk4_step, record_substeps=True, substeps=AUDIT_SUBSTEPS))


class OrbitalMemory(NamedTuple):
    """Retain only event history, command holds and the measured capture dwell."""

    mode: str = "coast"
    chart: int = -1
    window: int = 0
    pulse_torque: float = 0.0
    pulse_ticks: int = 0
    launch_used: bool = False
    capture_dwell: int = 0
    local_torque: float = 0.0
    local_ticks: int = 0


def controller_step(
    x: NDArray[np.float64],
    memory: OrbitalMemory,
    plant: tuple[float, NDArray[np.float64]],
    tuning: tuple[float, float, float],
    variant: str,
) -> tuple[float | None, OrbitalMemory, dict[str, float | int | str]]:
    """Select the next 20 ms command, or terminate with an explicit screen failure."""
    if variant not in ("gated", "ungated", "unscreened", "dense"):
        raise ValueError(f"Unknown orbital controller variant: {variant}")
    cap, gain = plant
    energy_gain, width, brake_gain = tuning
    screened, gated = variant != "unscreened", variant in ("gated", "unscreened")
    H, p = orbital_invariants(x, PHYSICAL, MODEL)
    chart_value, period, phase = orbital_phase(x, H, p, MODEL)
    chart = int(chart_value)
    distance = np.abs((phase - CENTERS + np.pi) % (2 * np.pi) - np.pi)
    window = int(np.argmin(distance)) + 1 if np.any(distance <= width) else 0
    entry = bool(
        window and (memory.chart == -1 or (chart == memory.chart and window != memory.window))
    )
    duration = float(period) if chart else MODEL.natural_period_s
    horizons = np.minimum(np.ceil(np.array([0.5, 1.0]) * duration / PHYSICS_DT_S), HORIZON_TICKS)
    horizons = horizons.astype(np.int32)
    prediction_plant = (cap, gain, brake_gain)
    arm_bound = float(ARM_LIMIT_RAD) - ARM_MARGIN_RAD
    coast_x, coast_H, coast_peak, _, coast_first = map(
        np.asarray, predict_candidates(x, np.zeros((1, 3)), horizons, prediction_plant)
    )
    coast_safe = coast_peak[0] <= arm_bound
    if not np.isfinite(coast_x).all():
        raise FloatingPointError("Nonfinite coast prediction")

    # Preserve active brake hysteresis and screen the actual local feedback schedule.
    latched_brake = memory.mode == "brake" and not np.all(
        coast_peak[0] <= arm_bound - ARM_MARGIN_RAD
    )
    error = x.copy()
    error[1] = np.arctan2(-np.sin(x[1]), -np.cos(x[1]))
    inside_release = np.all(np.abs(error) < 2 * CAPTURE_LIMITS)
    local = inside_release and (memory.mode == "local" or memory.capture_dwell >= CAPTURE_TICKS)
    demand_brake = screened and (latched_brake or (not local and not coast_safe[0]))
    launch = not memory.launch_used and bool(np.all(x[1:] == 0))
    continuing = memory.pulse_ticks > 0 and (
        not gated or (window and chart == memory.chart and window == memory.window)
    )
    offer = continuing or launch or (entry if gated else memory.pulse_ticks == 0)
    torque: float | None = 0.0
    mode, remaining, local_ticks = "coast", 0, 0
    first_peak = float(coast_first[0])
    selected_peak = float(coast_peak[0, 1])
    requested = float(energy_gain * (TARGET_ENERGY_J - H))
    delivered, candidate_count, rejected_arm, rejected_phase = 0.0, 0, 0, 0
    if local and not demand_brake:
        states, _, peak, commands, first = map(
            np.asarray,
            predict_candidates(
                x,
                np.array([[memory.local_torque, memory.local_ticks, 2.0]]),
                horizons,
                prediction_plant,
            ),
        )
        if not np.isfinite(states).all():
            raise FloatingPointError("Nonfinite local prediction")
        demand_brake = screened and peak[0, 1] > arm_bound
        if not demand_brake:
            torque, mode = float(commands[0]), "local"
            local_ticks = (memory.local_ticks or 5) - 1
            first_peak, selected_peak = float(first[0]), float(peak[0, 1])

    # Enumerate torque-duration pairs once per event and rescreen any committed remainder.
    if not local and not demand_brake and offer:
        if continuing or launch:
            ticks = np.array([memory.pulse_ticks if continuing else 1])
            values = np.array([memory.pulse_torque if continuing else cap])
        else:
            grid = np.linspace(-cap, cap, 17)
            durations = 1 if variant == "dense" else 5
            ticks, values = (
                np.repeat(np.arange(1, durations + 1), grid.size),
                np.tile(grid, durations),
            )
        schedules = np.column_stack((values, ticks, np.zeros(ticks.size)))
        states, energies, peak, _, first = map(
            np.asarray, predict_candidates(x, schedules, horizons, prediction_plant)
        )
        if not np.isfinite(states).all():
            raise FloatingPointError("Nonfinite energy-pulse prediction")
        candidate_count = len(ticks)
        candidates = np.arange(len(ticks))
        energy_change = energies[ticks - 1, candidates] - H
        residual = np.abs(energy_change - requested)
        eligible = ticks <= horizons[1]
        if screened:
            rejected_arm = int(np.count_nonzero(eligible & (peak[:, 1] > arm_bound)))
            eligible &= peak[:, 1] <= arm_bound
        if variant == "dense" and not (launch or continuing):
            _, end_p = orbital_invariants(states[0], PHYSICAL, MODEL)
            p_scale = np.sqrt(2 * MODEL.base_inertia_kg_m2 * TARGET_ENERGY_J)
            residual = ((energies[0] - TARGET_ENERGY_J) / TARGET_ENERGY_J) ** 2 + (
                end_p / p_scale
            ) ** 2
        elif not (continuing or launch):
            coast_residual = np.abs(coast_H[ticks - 1, 0] - H - requested)
            eligible &= (values != 0) & (residual <= coast_residual - 1e-4 * TARGET_ENERGY_J)
            eligible &= np.abs(H + energy_change - TARGET_ENERGY_J) < abs(H - TARGET_ENERGY_J)
        if gated and not launch and eligible.any():
            knots = (np.arange(ticks.max())[:, None] < ticks) & eligible
            pulse_x = states[: ticks.max()][knots]
            h_pred, p_pred = orbital_invariants(pulse_x, PHYSICAL, MODEL)
            charts, _, phases = orbital_phase(pulse_x, h_pred, p_pred, MODEL)
            in_window = (
                np.abs((phases - CENTERS[window - 1] + np.pi) % (2 * np.pi) - np.pi) <= width
            )
            failed = np.unique(np.nonzero(knots)[1][(charts != chart) | ~in_window])
            eligible[failed] = False
            rejected_phase = len(failed)
        if gated and not launch and eligible.any():
            # Check the next powered tick for transient chart exits missed by its endpoints.
            distinct, indices = np.unique(values, return_inverse=True)
            offered = np.unique(indices[eligible])
            fine = np.asarray(_PHASE_TICK(np.broadcast_to(x, (len(distinct), 4)), distinct))
            pulse_x = fine[:, offered].reshape(-1, 4)
            h_pred, p_pred = orbital_invariants(pulse_x, PHYSICAL, MODEL)
            charts, _, phases = orbital_phase(pulse_x, h_pred, p_pred, MODEL)
            difference = (phases - CENTERS[window - 1] + np.pi) % (2 * np.pi) - np.pi
            outside = ((charts != chart) | (np.abs(difference) > width)).reshape(-1, len(offered))
            rejected = eligible & np.isin(indices, offered[np.any(outside, axis=0)])
            eligible &= ~rejected
            rejected_phase += int(rejected.sum())
        if eligible.any():
            order = np.lexsort(
                (values, np.abs(values), ticks, np.where(eligible, residual, np.inf))
            )
            index = int(order[0])
            torque, remaining = float(values[index]), int(ticks[index] - 1)
            mode = "launch" if launch else ("pulse" if torque else "coast")
            delivered = float(energy_change[index])
            first_peak, selected_peak = float(first[index]), float(peak[index, 1])
        elif launch:
            mode, torque = "launch_rejected", None

    # Check the brake trajectory before issuing any brake command or declaring a failed screen.
    if demand_brake or (screened and mode == "coast" and not coast_safe[1]):
        states, _, peak, commands, first = map(
            np.asarray,
            predict_candidates(x, np.array([[0.0, 0.0, 1.0]]), horizons, prediction_plant),
        )
        if not np.isfinite(states).all():
            raise FloatingPointError("Nonfinite braking prediction")
        first_peak, selected_peak = float(first[0]), float(peak[0, 1])
        mode, torque = (
            ("brake", float(commands[0]))
            if selected_peak <= arm_bound
            else ("no_screened_action", None)
        )

    dwell = memory.capture_dwell if mode not in ("brake", "no_screened_action") else 0
    if memory.mode == "local" and mode != "local":
        dwell = 0
    following = OrbitalMemory(
        mode,
        chart,
        window,
        float(torque) if mode == "pulse" and torque is not None else 0.0,
        remaining,
        memory.launch_used or mode in ("launch", "launch_rejected"),
        dwell,
        float(torque) if mode == "local" and torque is not None else 0.0,
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
            "requested_j": requested,
            "predicted_delta_j": delivered,
            "candidate_count": candidate_count,
            "rejected_arm": rejected_arm,
            "rejected_phase": rejected_phase,
            "coast_half_peak_rad": float(coast_peak[0, 0]),
            "coast_full_peak_rad": float(coast_peak[0, 1]),
            "selected_peak_rad": selected_peak,
            "first_tick_peak_rad": first_peak,
        },
    )
