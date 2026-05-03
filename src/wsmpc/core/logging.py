"""Small structured logging helpers for human-readable traces."""

from __future__ import annotations

import logging
from typing import Any


def configure_logging(level: str) -> None:
    """Configure console logging with Rich when available."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    try:
        from rich.logging import RichHandler

        logging.basicConfig(
            level=numeric_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, markup=False)],
            force=True,
        )
    except ImportError:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            force=True,
        )


def log_event(
    logger: logging.Logger,
    level: int,
    *,
    identity: str,
    status: str,
    action: str,
    action_result: str,
    t_index: int | None = None,
    t_sec: float | None = None,
    **fields: Any,
) -> None:
    """Emit one consistent event line with identity, time, action, and result."""

    # Keep the message self-contained so standard logging formatters preserve the details.
    time_text = _format_time(t_index=t_index, t_sec=t_sec)
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.log(
        level,
        "identity=%s status=%s action=%s action_result=%s %s%s",
        identity,
        status,
        action,
        action_result,
        time_text,
        f" {field_text}" if field_text else "",
    )


def _format_time(*, t_index: int | None, t_sec: float | None) -> str:
    """Format optional simulator time without mixing in wall-clock time."""

    if t_index is None and t_sec is None:
        return "t_index=None t_sec=None"
    return f"t_index={t_index} t_sec={0.0 if t_sec is None else t_sec:.6f}"
