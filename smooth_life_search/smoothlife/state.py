"""State container for SmoothLife simulation/search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SmoothLifeState:
    """Mutable simulator state."""

    field: np.ndarray
    objective_values: np.ndarray
    objective_field: np.ndarray
    evaluated_mask: np.ndarray
    inner_fill: np.ndarray
    outer_fill: np.ndarray
    transition_field: np.ndarray
    support_ema: np.ndarray
    bounds: np.ndarray
    best_point: np.ndarray
    best_value: float
    local_best_point: np.ndarray
    local_best_value: float
    box_best_point: np.ndarray
    box_best_value: float
    step_index: int = 0
    evaluations: int = 0
