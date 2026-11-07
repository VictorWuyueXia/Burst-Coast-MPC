"""Conservative Routh coordinates of the nonlinear Furuta plant."""

from __future__ import annotations

import jax
import numpy as np
from numpy.typing import NDArray

from rotary_pendulum.environment.dynamics import ModelConstants, energy_components
from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICAL
from rotary_pendulum.heuristic.energy import encode
from rotary_pendulum.utils.config_schema import RotaryPendulumConfig

_ENCODE = jax.jit(encode)
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(24)
_PANELS = 8
_QUADRATURE_X = ((np.arange(_PANELS)[:, None] + (_NODES + 1) / 2) / _PANELS).ravel()
_QUADRATURE_W = np.tile(_WEIGHTS / (2 * _PANELS), _PANELS)


def orbital_invariants(
    x: NDArray[np.float64], physical: RotaryPendulumConfig, model: ModelConstants
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return total mechanical energy and cyclic arm momentum over leading axes."""
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Orbital invariants require a finite state")
    # Retain the full cross inertia in both invariants.
    if physical == PHYSICAL and model == MODEL:
        energy = np.asarray(_ENCODE(x)).sum(axis=-1)
    else:
        _, _, energy = energy_components(x, physical, model)
    _, alpha, omega, nu = np.moveaxis(x, -1, 0)
    inertia = model.base_inertia_kg_m2 + model.pendulum_inertia_kg_m2 * np.sin(alpha) ** 2
    coupling = model.coupling_inertia_kg_m2 * np.cos(alpha)
    return energy, inertia * omega + coupling * nu


def orbital_phase(
    x: NDArray[np.float64],
    H: NDArray[np.float64],
    p: NDArray[np.float64],
    model: ModelConstants,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Return chart, period and phase. Chart zero denotes a physically excluded level."""
    shape = x.shape[:-1]
    alpha = ((x[..., 1].ravel() + np.pi) % (2 * np.pi)) - np.pi
    velocity = x[..., 3].ravel()
    energy, momentum = np.asarray(H).reshape(-1), np.asarray(p).reshape(-1)
    if not np.isfinite(x).all() or not np.isfinite(energy + momentum).all():
        raise ValueError("Orbital phase requires finite state and invariants")
    base, pivot, cross, gravity = (
        model.base_inertia_kg_m2,
        model.pendulum_inertia_kg_m2,
        model.coupling_inertia_kg_m2,
        model.gravity_torque_nm,
    )
    margin = 1e-5 * (2 * gravity)
    momentum2 = momentum**2
    vertical = momentum2 / (2 * base)
    # The effective potential is convex in cos(alpha), with a split well at high momentum.
    split = momentum2 * pivot > gravity * base**2
    minimum = vertical.copy()
    if split.any():
        lower, upper = np.zeros(np.count_nonzero(split)), np.ones(np.count_nonzero(split))
        for _ in range(44):
            middle = (lower + upper) / 2
            inertia = base + pivot * (1 - middle**2)
            rising = momentum2[split] * pivot * middle > gravity * inertia**2
            lower, upper = np.where(rising, lower, middle), np.where(rising, middle, upper)
        z_min = (lower + upper) / 2
        minimum[split] = gravity * (1 - z_min) + momentum2[split] / (
            2 * (base + pivot * (1 - z_min**2))
        )
    above_min = energy > minimum + margin
    away_from_barrier = ~split | (np.abs(energy - vertical) > margin)
    libration = above_min & away_from_barrier & (energy < vertical + 2 * gravity - margin)
    rotation = energy > vertical + 2 * gravity + margin
    chart = np.zeros(alpha.size, dtype=np.int64)
    period = np.full(alpha.size, np.nan)
    phase = np.full(alpha.size, np.nan)

    # The turning equation is a cubic in cos(alpha); its physical roots bound each well.
    valid = libration | rotation
    if valid.any():
        a, nu, h, p2 = alpha[valid], velocity[valid], energy[valid], momentum2[valid]
        turns = libration[valid]
        off_center = turns & split[valid] & (h < vertical[valid])
        lo, hi = np.full(a.size, -np.pi), np.full(a.size, np.pi)
        if turns.any():
            companion = np.zeros((turns.sum(), 3, 3))
            scaled = (h[turns] - gravity) / gravity
            companion[:, 0, :] = np.column_stack(
                (
                    -scaled,
                    np.full(scaled.shape, (base + pivot) / pivot),
                    scaled * (base + pivot) / pivot - p2[turns] / (2 * gravity * pivot),
                )
            )
            companion[:, 1, 0] = companion[:, 2, 1] = 1
            roots = np.linalg.eigvals(companion)
            inside = (np.abs(roots.imag) < 1e-10) & (np.abs(roots.real) < 1)
            if np.any(inside.sum(axis=1) != 1 + off_center[turns]):
                raise FloatingPointError("Regular orbit has an inconsistent turning-root count")
            outer = np.arccos(np.min(np.where(inside, roots.real, np.inf), axis=1))
            inner = np.arccos(np.max(np.where(inside, roots.real, -np.inf), axis=1))
            lo[turns], hi[turns] = -outer, outer
            lo[off_center & (a >= 0)] = inner[(off_center & (a >= 0))[turns]]
            hi[off_center & (a < 0)] = -inner[(off_center & (a < 0))[turns]]

        # Integrate the full leg and the measured partial leg on the same quadrature grid.
        fraction = (a - lo) / (hi - lo)
        if np.any((fraction < -1e-9) | (fraction > 1 + 1e-9)):
            raise FloatingPointError("Measured angle lies outside its regular orbit")
        fraction = np.clip(fraction, 0, 1)
        extent = np.stack((np.ones(a.size), fraction), axis=-1)
        extent[turns, 1] = 2 / np.pi * np.arcsin(np.sqrt(fraction[turns]))
        s = extent[..., None] * _QUADRATURE_X
        angle_fraction = np.where(turns[:, None, None], np.sin(np.pi / 2 * s) ** 2, s)
        angle = lo[:, None, None] + (hi - lo)[:, None, None] * angle_fraction
        jacobian = (hi - lo)[:, None, None] * np.where(
            turns[:, None, None], np.pi / 2 * np.sin(np.pi * s), 1
        )
        cosine = np.cos(angle)
        inertia = base + pivot * np.sin(angle) ** 2
        reduced = pivot - (cross * cosine) ** 2 / inertia
        available = h[:, None, None] - gravity * (1 - cosine) - p2[:, None, None] / (2 * inertia)
        # Evaluate H - V relative to the nearer turning root to avoid cancellation.
        near_lower = angle_fraction <= 0.5
        turn = np.where(near_lower, lo[:, None, None], hi[:, None, None])
        offset = (hi - lo)[:, None, None] * np.where(
            near_lower, angle_fraction, -(np.cos(np.pi / 2 * s) ** 2)
        )
        z_turn = np.cos(turn)
        a_turn = base + pivot * np.sin(turn) ** 2
        difference = -2 * np.sin(turn + offset / 2) * np.sin(offset / 2)
        turn_available = difference * (
            gravity - p2[:, None, None] * pivot * (cosine + z_turn) / (2 * inertia * a_turn)
        )
        available = np.where(turns[:, None, None], turn_available, available)
        # Zero-length partial integrals at the turn contribute exactly zero.
        integrable = extent[..., None] > 1e-10
        if np.any((available <= 0) & integrable & (jacobian > 1e-12)):
            raise FloatingPointError("Nonpositive kinetic energy inside a regular phase integral")
        integrand = np.divide(
            jacobian * np.sqrt(reduced),
            np.sqrt(2 * np.where(available > 0, available, 1)),
            out=np.zeros_like(jacobian),
            where=integrable & (available > 0),
        )
        flight, elapsed = (extent * (integrand @ _QUADRATURE_W)).T
        T = np.where(turns, 2 * flight, flight)
        phase[valid] = (2 * np.pi * np.where(nu >= 0, elapsed, T - elapsed) / T) % (2 * np.pi)
        period[valid] = T
        chart[valid] = np.where(
            turns, np.where(off_center, np.where(a >= 0, 2, 3), 1), np.where(nu > 0, 4, 5)
        )
    return chart.reshape(shape), period.reshape(shape), phase.reshape(shape)
