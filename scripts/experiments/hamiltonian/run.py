"""CPU validation of orbital energy bursts on the shared Furuta simulator."""

import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_discrete_are

from rotary_pendulum.environment import dynamics
from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL, PHYSICS_DT_S, rk4_step
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, MAX_PHYSICS_STEPS
from rotary_pendulum.heuristic.decoder import decode
from rotary_pendulum.heuristic.energy import encode, policy
from rotary_pendulum.heuristic.orbital_controller import (
    ARM_MARGIN_RAD,
    CAPTURE_LIMITS,
    CAPTURE_TICKS,
    CENTERS,
    OrbitalMemory,
    controller_step,
)
from rotary_pendulum.heuristic.orbital_phase import orbital_invariants, orbital_phase
from rotary_pendulum.heuristic.orbital_prediction import AUDIT_SUBSTEPS, predict_candidates
from rotary_pendulum.heuristic.phase_pulse_controller import refined_step

ROOT = Path(__file__).resolve().parents[3]
BASELINE = (
    ROOT / "artifacts/rotary_pendulum/final-results/rl-mpc/machine-scannables/session-records"
)
AUDIT_DT_S = PHYSICS_DT_S / AUDIT_SUBSTEPS
NOMINAL_TUNING = (0.55, np.pi / 6, 8.0, 0.5, 0.0)


def evaluate(
    starts: np.ndarray, variant: str, tuning: tuple[float, ...], output: Path
) -> list[dict]:
    """Run complete CPU sessions and store refined trajectories with decision provenance."""
    if not jax.config.x64_enabled or jax.default_backend() != "cpu":
        raise RuntimeError("Set JAX_ENABLE_X64=true and JAX_PLATFORMS=cpu for this validation")
    machine = output / "machine-scannables"
    machine.mkdir(parents=True, exist_ok=False)
    design = json.loads((BASELINE / "design.json").read_text(encoding="utf-8"))
    cap = design["plant"]["torque_limit_nm"]
    weights = np.diag(design["plant"]["lqr_state_weights"])
    upright = jnp.array([0.0, -jnp.pi, 0.0, 0.0])
    # Recover the historical feedback design from the current 100 ms sampled map.
    a, b = jax.jacfwd(
        lambda x, u: jax.lax.fori_loop(0, 5, lambda _, y: rk4_step(y, cap * u[0]), x),
        argnums=(0, 1),
    )(upright, jnp.zeros(1))
    riccati = solve_discrete_are(a, b, weights, np.ones((1, 1)))
    gain = np.linalg.solve(1 + b.T @ riccati @ b, b.T @ riccati @ a)
    poles = np.sort(np.abs(np.linalg.eigvals(a - b @ gain)))
    np.testing.assert_allclose(poles, [0.189524, 0.280134, 0.653542, 0.917225], atol=1e-5, rtol=0)
    if not np.all(poles < 1):
        raise RuntimeError("Reconstructed LQR is unstable on the discrete plant")
    plant = (cap, gain.ravel())
    sources = [Path(__file__).resolve()] + list(
        (ROOT / "src/rotary_pendulum/heuristic").glob("orbital_*.py")
    )
    sources += [ROOT / "src/rotary_pendulum/environment/jax_dynamics.py"]
    if variant == "refined":
        sources.append(ROOT / "src/rotary_pendulum/heuristic/phase_pulse_controller.py")
    physical = PHYSICAL
    controller = variant
    if variant == "gated-damping":
        frequency = 2 * np.pi / MODEL.natural_period_s
        physical = PHYSICAL.model_copy(
            update={
                "rotary_damping_nms": 0.05 * MODEL.base_inertia_kg_m2 * frequency,
                "pendulum_damping_nms": 0.05 * MODEL.pendulum_inertia_kg_m2 * frequency,
            }
        )
        controller = "gated"
    elif variant == "gated-mass":
        physical = PHYSICAL.model_copy(
            update={"pendulum_mass_kg": 1.05 * PHYSICAL.pendulum_mass_kg}
        )
        controller = "gated"
    model = dynamics.derive_model(physical)
    (machine / "design.json").write_text(
        json.dumps(
            dict(
                variant=variant,
                tuning=dict(
                    zip(
                        (
                            "k_energy",
                            "phase_half_width",
                            "brake_gain",
                            "momentum_weight",
                            "centering_rate_s_inv",
                        )[: len(tuning)],
                        tuning,
                        strict=True,
                    )
                ),
                cap_nm=cap,
                arm_limit_rad=float(ARM_LIMIT_RAD),
                lqr_gain=gain.ravel().tolist(),
                lqr_poles=poles.tolist(),
                prediction_step_s=0.001,
                execution_step_s=AUDIT_DT_S,
                session_s=MAX_PHYSICS_STEPS * PHYSICS_DT_S,
                physical=physical.model_dump(),
                source_sha256={
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sources
                },
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(machine / "starts.npy", starts)
    advance = jax.jit(lambda x, u: rk4_step(x, u, record_substeps=True, substeps=AUDIT_SUBSTEPS))
    heuristic = jax.jit(lambda x: decode(x, policy(encode(x), jnp.array(0.004)), 0.0, 1.0, cap)[0])
    for count in (1, 6, 26) if variant == "refined" else (1, 17, 85):
        jax.block_until_ready(
            predict_candidates(
                starts[0],
                np.zeros((count, 3)),
                np.array([1, 1], dtype=np.int32),
                (cap, gain.ravel(), tuning[2]),
            )
        )
    jax.block_until_ready(advance(starts[0], 0.0))
    if variant == "heuristic":
        jax.block_until_ready(heuristic(starts[0]))
    print(f"{output.name}: {len(starts)} CPU sessions, LQR poles {poles.round(6)}", flush=True)
    rows = []
    for episode, initial in enumerate(starts):
        x, memory = initial.copy(), OrbitalMemory()
        status, full_hold, first_capture = "completed", 0, np.nan
        records, histories, commands = [], [], []
        for tick in range(MAX_PHYSICS_STEPS):
            # Use the same plant for refined execution and update dwell from all executed samples.
            started = time.perf_counter()
            if controller == "heuristic":
                torque = float(heuristic(x)) if tick % 5 == 0 else commands[-1]
                record = {"mode": "heuristic"}
            elif controller == "refined":
                torque, memory, record = refined_step(x, memory, plant, tuning)
            else:
                torque, memory, record = controller_step(x, memory, plant, tuning, controller)
            record.update(time_s=tick * PHYSICS_DT_S, decision_s=time.perf_counter() - started)
            records.append(record)
            if torque is None:
                status = memory.mode
                break
            if not np.isfinite(torque) or abs(torque) > cap + 1e-12:
                raise FloatingPointError(
                    f"Invalid executed torque on episode {episode}, tick {tick}"
                )
            if physical == PHYSICAL:
                history = np.asarray(advance(x, torque))
            else:
                history = np.empty((AUDIT_SUBSTEPS, 4))
                for substep in range(AUDIT_SUBSTEPS):
                    x = dynamics.rk4_step(x, torque, AUDIT_DT_S, physical, model)
                    history[substep] = x
            if not np.isfinite(history).all():
                raise FloatingPointError(f"Nonfinite execution on episode {episode}, tick {tick}")
            histories.append(history)
            commands.append(torque)
            x = history[-1]
            error = history.copy()
            error[:, 1] = np.arctan2(-np.sin(history[:, 1]), -np.cos(history[:, 1]))
            inside = np.all(np.abs(error) < CAPTURE_LIMITS, axis=1)
            outside = np.flatnonzero(~inside)
            full_hold = len(inside) - outside[-1] - 1 if outside.size else full_hold + len(inside)
            if full_hold >= CAPTURE_TICKS * AUDIT_SUBSTEPS and np.isnan(first_capture):
                first_capture = (tick + 1) * PHYSICS_DT_S - AUDIT_DT_S * (
                    full_hold - CAPTURE_TICKS * AUDIT_SUBSTEPS
                )
            dwell = int(full_hold // AUDIT_SUBSTEPS) if memory.mode != "brake" else 0
            memory = memory._replace(capture_dwell=min(dwell, memory.capture_dwell + 1))
            if controller in ("refined", "gated", "ungated", "dense") and physical == PHYSICAL:
                bound = float(ARM_LIMIT_RAD) - ARM_MARGIN_RAD
                if record["first_tick_peak_rad"] <= bound < np.max(np.abs(history[:, 0])):
                    status = "refinement_disagreement"
                    break

        # Preserve the terminal decision and separate delivered work from net energy change.
        track = np.concatenate((initial[None], np.asarray(histories).reshape(-1, 4)))
        torque_trace = np.asarray(commands)
        modes = np.asarray([r["mode"] for r in records[: len(commands)]])
        _, _, H = dynamics.energy_components(track, physical, model)
        work = np.repeat(torque_trace, AUDIT_SUBSTEPS) * np.diff(track[:, 0])
        powered = np.abs(torque_trace) > 1e-6 * cap
        # A commanded actuation interval can include a sample with exactly zero torque.
        actuation_on = np.isin(
            modes, ("launch", "pulse", "brake", "aperiodic", "local", "heuristic")
        )
        arm_out = np.abs(track[:, 0]) >= float(ARM_LIMIT_RAD)
        beta = np.arctan2(-np.sin(track[:, 1]), -np.cos(track[:, 1]))
        error = track.copy()
        error[:, 1] = beta
        inside = np.all(np.abs(error[1:]) < CAPTURE_LIMITS, axis=1)
        edges = np.flatnonzero(np.diff(np.r_[False, inside, False]))
        phase_misses, straddled = 0, 0
        if controller in ("refined", "gated", "unscreened"):
            pulse_ticks = np.flatnonzero(modes == "pulse")
            # Evaluate the actual phase of every powered substep in a single batched audit.
            sample_indices = pulse_ticks[:, None] * AUDIT_SUBSTEPS + np.arange(
                1, AUDIT_SUBSTEPS + 1
            )
            pulse_x = track[sample_indices.ravel()]
            if pulse_x.size:
                phase_blocks = []
                for block in np.array_split(pulse_x, (len(pulse_x) + 511) // 512):
                    nominal_H, p = orbital_invariants(block, PHYSICAL, MODEL)
                    charts, _, phases = orbital_phase(block, nominal_H, p, MODEL)
                    phase_blocks.append(np.column_stack((charts, phases)))
                charts, phases = np.concatenate(phase_blocks).T
                windows = np.asarray([records[i]["window"] for i in pulse_ticks])
                chart_at_tick = np.asarray([records[i]["chart"] for i in pulse_ticks])
                distance = (
                    phases.reshape(-1, AUDIT_SUBSTEPS) - CENTERS[windows - 1, None] + np.pi
                ) % (2 * np.pi) - np.pi
                phase_misses = int(
                    np.count_nonzero(
                        (charts.reshape(distance.shape) != chart_at_tick[:, None])
                        | (np.abs(distance) > tuning[1])
                    )
                )
                boundaries = np.flatnonzero(np.diff(np.r_[False, modes == "pulse", False]))
                for begin, end in boundaries.reshape(-1, 2):
                    segment = distance[(pulse_ticks >= begin) & (pulse_ticks < end)]
                    straddled += int(segment.min() <= 0 <= segment.max())
        rows.append(
            dict(
                episode=episode,
                status=status,
                observed_s=len(commands) * PHYSICS_DT_S,
                capture_s=first_capture,
                full_hold_s=float(inside.sum()) * AUDIT_DT_S,
                longest_full_hold_s=float(np.max(edges[1::2] - edges[::2], initial=0)) * AUDIT_DT_S,
                upright_s=float(np.count_nonzero(np.abs(beta[1:]) < CAPTURE_LIMITS[1]))
                * AUDIT_DT_S,
                peak_arm_deg=np.rad2deg(np.abs(track[:, 0]).max()),
                arm_crossings=int(np.count_nonzero(arm_out & ~np.r_[False, arm_out[:-1]])),
                arm_outcome=int(arm_out.any() or status != "completed"),
                powered_s=float(powered.sum()) * PHYSICS_DT_S,
                absolute_torque_impulse_nms=float(np.abs(torque_trace).sum()) * PHYSICS_DT_S,
                peak_violation_deg=float(
                    np.rad2deg(max(0, np.abs(track[:, 0]).max() - float(ARM_LIMIT_RAD)))
                ),
                final_upright_error_deg=float(np.rad2deg(abs(beta[-1]))),
                final_arm_angle_deg=float(np.rad2deg(track[-1, 0])),
                final_arm_speed_rad_s=float(track[-1, 2]),
                final_pendulum_speed_rad_s=float(track[-1, 3]),
                switch_ons=int(np.count_nonzero(powered & ~np.r_[False, powered][:-1])),
                actuation_window_s=float(actuation_on.sum()) * PHYSICS_DT_S,
                actuation_switch_ons=int(
                    np.count_nonzero(actuation_on & ~np.r_[False, actuation_on][:-1])
                ),
                brake_entries=int(
                    np.count_nonzero(
                        (modes == "brake") & (np.concatenate(([""], modes))[:-1] != "brake")
                    )
                ),
                saturated_ticks=int(np.count_nonzero(np.abs(torque_trace) >= cap - 1e-12)),
                work_j=float(work.sum()),
                absolute_work_j=float(np.abs(work).sum()),
                local_absolute_work_j=float(
                    np.abs(work[np.repeat(modes == "local", AUDIT_SUBSTEPS)]).sum()
                ),
                phase_samples_outside=phase_misses,
                pulses_straddling_center=straddled,
                median_decision_ms=1000 * np.median([r["decision_s"] for r in records]),
                mean_decision_ms=1000 * np.mean([r["decision_s"] for r in records]),
                max_decision_ms=1000 * np.max([r["decision_s"] for r in records]),
            )
        )
        np.savez_compressed(
            machine / f"episode-{episode:03d}.npz",
            states=track,
            torque_nm=torque_trace,
            actuation_on=actuation_on,
            mechanical_energy_j=H,
            work_j=work,
            **{k: np.asarray([r[k] for r in records]) for k in records[0]},
        )
        print(
            f"{output.name} {episode + 1}/{len(starts)}: {status}, capture={first_capture:.3g} s, "
            f"peak={rows[-1]['peak_arm_deg']:.1f}°, active={rows[-1]['powered_s']:.2f} s",
            flush=True,
        )

    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (machine / "episodes.csv").write_text(table.getvalue(), encoding="utf-8", newline="")
    return rows


if __name__ == "__main__":
    # Run paired variants against the frozen baseline initial conditions.
    output = Path(sys.argv[1])
    starts_archive = np.load(BASELINE / "starts.npz", allow_pickle=False)
    official = starts_archive["states"]
    starts_archive.close()
    evaluate(official, "refined", NOMINAL_TUNING, output)
