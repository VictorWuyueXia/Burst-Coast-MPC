"""Ordered process-pool map for independent MPC candidate solves."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor


def ordered_process_map[InputT, OutputT](
    function: Callable[[InputT], OutputT],
    items: Sequence[InputT],
    *,
    max_workers: int,
) -> list[OutputT]:
    """Evaluate tasks in worker processes while preserving input order."""

    if max_workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(function, items))
