"""Shared interfaces for late-stage AGSLS strategies."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ...core import Basin

StrategyResult = tuple[np.ndarray, dict[str, object]]


class LateStageStrategy(Protocol):
    """Strategy that may refine or shift the next zoom bounds."""

    def run(self, basin: Basin, current_bounds: np.ndarray) -> StrategyResult | None: ...
