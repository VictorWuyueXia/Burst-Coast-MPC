"""Clock helpers that keep simulation time separate from wall-clock pacing."""

from __future__ import annotations

import time as _time


def realtime(timestep_s: float) -> float:
    """Return a wall-clock pace that matches the simulation deltaT."""

    value = float(timestep_s)
    if value <= 0.0:
        msg = "realtime pace requires a positive timestep"
        raise ValueError(msg)
    return value


def monotonic_s() -> float:
    """Expose monotonic wall-clock seconds for runtime diagnostics."""

    return _time.monotonic()


def sleep_s(duration_s: float) -> None:
    """Sleep only when the requested wall-clock pace is positive."""

    if duration_s > 0.0:
        _time.sleep(duration_s)
