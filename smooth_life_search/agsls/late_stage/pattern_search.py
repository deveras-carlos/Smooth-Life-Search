"""Pattern-search late-stage exploitation strategy."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core import Basin
from .base import StrategyResult


class PatternSearchExploiter:
    """Refine a late-stage zoom with coordinate pattern search around a seed."""

    def __init__(self, controller: Any) -> None:
        self._controller = controller

    def run(self, basin: Basin, current_bounds: np.ndarray) -> StrategyResult | None:
        return self._controller._run_late_stage_pattern_search(basin, current_bounds)
