"""Budget accounting helpers for AGSLS."""

from __future__ import annotations

from ..smoothlife.state import SmoothLifeState


def evaluation_limit(configured_limit: int | None, active_limit: int | None) -> int | None:
    """Resolve the currently active objective evaluation limit."""

    return active_limit if active_limit is not None else configured_limit


def remaining_evaluations(state: SmoothLifeState, configured_limit: int | None, active_limit: int | None) -> int | None:
    """Return remaining objective evaluations, or ``None`` for unlimited."""

    limit = evaluation_limit(configured_limit, active_limit)
    if limit is None:
        return None
    return max(int(limit) - int(state.evaluations), 0)


def max_evaluations_reached(state: SmoothLifeState, configured_limit: int | None, active_limit: int | None) -> bool:
    """Return whether the active objective budget has been consumed."""

    limit = evaluation_limit(configured_limit, active_limit)
    return limit is not None and int(state.evaluations) >= int(limit)


def bounded_evaluation_batch(requested: int, remaining: int | None) -> int:
    """Clamp a requested evaluation batch by remaining budget."""

    if remaining is None:
        return int(requested)
    return max(0, min(int(requested), int(remaining)))
