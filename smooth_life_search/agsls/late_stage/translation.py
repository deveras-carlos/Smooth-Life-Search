"""Translation-based late-stage zoom strategy."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core import Basin
from .base import StrategyResult


class TranslationZoom:
    """Shift a weak late-stage zoom toward elite basin evidence."""

    def __init__(self, controller: Any) -> None:
        self._controller = controller

    def run(
        self,
        basin: Basin,
        current_bounds: np.ndarray,
        standard_shrink_bounds: np.ndarray,
        trigger_reason: str,
    ) -> StrategyResult | None:
        return self._controller._run_late_stage_translation(
            basin,
            current_bounds,
            standard_shrink_bounds,
            trigger_reason,
        )
