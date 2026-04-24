"""Microgrid late-stage exploitation strategy."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core import Basin
from .base import StrategyResult


class MicrogridExploiter:
    """Refine a late-stage zoom by sampling small lattices around elite centers."""

    def __init__(self, controller: Any) -> None:
        self._controller = controller

    def run(self, basin: Basin, current_bounds: np.ndarray) -> StrategyResult | None:
        return self._controller._run_late_stage_microgrid(basin, current_bounds)
