"""Periodic local-search late-stage strategy."""

from __future__ import annotations

from typing import Any


class PeriodicLocalSearch:
    """Runs opportunistic local search while a late-stage box persists."""

    def __init__(self, controller: Any) -> None:
        self._controller = controller

    def maybe_run(self, *, box_id: int, late_stage_state: dict[str, float | int | bool]) -> None:
        self._controller._maybe_run_periodic_local_search(
            box_id=box_id,
            late_stage_state=late_stage_state,
        )
