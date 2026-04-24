"""Budget accounting helpers for AGSLS."""

from __future__ import annotations

import math

from ..smoothlife.state import SmoothLifeState
from .config import AGSLSConfig


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


def effective_zoom_limit(eval_limit: int | None, configured_limit: int, config: AGSLSConfig) -> int:
    """Scale zoom cycles upward for budgets above the configured baseline."""

    baseline = config.zoom_cycles_budget_baseline
    if eval_limit is None or baseline is None or baseline <= 0 or eval_limit < baseline:
        return int(configured_limit)
    ratio = float(eval_limit) / float(baseline)
    increment = int(math.floor(math.log2(ratio)))
    return int(configured_limit) + max(0, increment)
